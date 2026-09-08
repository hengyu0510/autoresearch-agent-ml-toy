"""一键顺序运行全部（或指定）ML 任务的总入口。

设计目标：`.env` / Kaggle token 配好、竞赛规则已接受之后，用一条命令
跑完整套流程：

    数据缺失时自动下载 -> 逐个任务调用 run_agent（真实迭代）
    -> run_agent 收尾自动生成提交文件 -> 汇总本批结果并写报告

单个任务失败不会中断整批；流程结束后可通过退出码判断是否有失败。

用法：
    python run_all.py
    python run_all.py --brain llm --max-steps 3
    python run_all.py --tasks titanic,house_prices --brain rule --max-steps 3
    python run_all.py --dry-run                  # 只打印将要执行的动作
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.config import AgentConfig, load_env_file
from agent.logger import configure_stdio
from tasks.registry import load_tasks

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_ID_FORMAT = "%Y%m%d_%H%M%S"


@dataclass
class BatchResult:
    """单个任务在本批次内的运行结果（含下载失败、运行失败等分支）。"""

    task_id: str
    config_file: str
    status: str                     # ok | failed | skipped | dry-run
    exit_code: int | None = None
    error: str = ""
    brain: str = ""
    run_dir: str = ""
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键顺序运行全部（或指定）ML 任务。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs-dir", type=str, default=str(PROJECT_ROOT / "configs"),
        help="存放 <task_id>.yaml 的目录；默认读取全部 yaml 任务配置。",
    )
    parser.add_argument(
        "--tasks", action="append", default=None,
        help="逗号分隔的任务 id（例如 titanic,house_prices）；缺省运行全部。",
    )
    parser.add_argument(
        "--brain", choices=["rule", "llm"], default=None,
        help="覆盖每个 config 中的 brain.type；缺省使用各 config 自身配置。",
    )
    parser.add_argument("--max-steps", type=int, default=None,
                        help="覆盖每个 config 的 run.max_steps。")
    parser.add_argument("--min-iterations", type=int, default=None,
                        help="覆盖每个 config 的 run.min_iterations。")
    parser.add_argument("--seed", type=int, default=None,
                        help="覆盖每个 config 的 experiment.seed。")
    parser.add_argument(
        "--runs-root", type=str, default=None,
        help="覆盖运行产物根目录；缺省使用各 config 的 runs_root。"
             "指定后会按 <runs-root>/<task_id>/ 分目录保存。",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="跳过数据检查/自动下载（数据必须已存在于 data/raw）。",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印每个任务将要执行的下载/运行命令，不执行。",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="透传给 run_agent，降低每个任务的终端输出。",
    )
    parser.add_argument(
        "--kaggle-upload", action="store_true",
        help="本地提交校验通过后，再调用 Kaggle CLI 实际上传（默认不上传）。",
    )
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None,
        metavar="RUN_DIR|latest",
        help=(
            "透传给 run_agent：省略参数时每个任务自动续跑各自 runs 下最新 run；"
            "显式路径仅适合 --tasks 单个任务。"
        ),
    )
    return parser.parse_args(argv)


def parse_task_ids(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    ids: list[str] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part and part not in ids:
                ids.append(part)
    return ids


def discover_configs(
    configs_dir: Path, task_ids: list[str] | None
) -> list[Path]:
    """按任务注册表顺序返回配置；可只保留用户指定的子集。"""
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"任务配置目录不存在: {configs_dir}")
    configs = sorted(configs_dir.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"{configs_dir} 下没有 *.yaml 任务配置")

    by_id = {cfg.stem: cfg for cfg in configs}
    if task_ids is None:
        registry_order = [t.id for t in load_tasks()]
        ordered = [by_id[t] for t in registry_order if t in by_id]
        ordered += [cfg for cfg in configs if cfg.stem not in registry_order]
        return ordered

    missing = [tid for tid in task_ids if tid not in by_id]
    if missing:
        available = ", ".join(sorted(by_id))
        raise ValueError(
            f"未知任务 id: {', '.join(missing)}；可用: {available}"
        )
    return [by_id[tid] for tid in task_ids]


def config_agent_runs_root(
    cfg: AgentConfig, task_id: str, args: argparse.Namespace
) -> Path:
    """计算本次 run_agent 实际写入的 runs 根目录。"""
    if args.runs_root:
        root = Path(args.runs_root)
        root = root if root.is_absolute() else (PROJECT_ROOT / root)
        return root / task_id
    root = Path(cfg.run.runs_root)
    return root if root.is_absolute() else (PROJECT_ROOT / root)


def needs_fetch(cfg: AgentConfig, task_id: str) -> bool:
    """与 data/fetch.py 的 skip 判断一致：目录为空/不存在才需要下载。"""
    data_path = cfg.experiment.data_path
    if not data_path:
        return False
    path = Path(data_path)
    dest = path if path.is_absolute() else (PROJECT_ROOT / path)
    return not dest.parent.exists() or not any(dest.parent.iterdir())


def run_cmd(cmd: list[str], *, cwd: Path, dry_run: bool = False) -> int | None:
    """执行命令；dry-run 只打印并返回 None。"""
    display = " ".join(str(c) for c in cmd)
    print(f"[cmd] {display}", flush=True)
    if dry_run:
        return None
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=False,
    )
    return proc.returncode


def build_agent_cmd(
    cfg_path: Path, args: argparse.Namespace
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_agent.py"),
        "--config", str(cfg_path),
    ]
    if args.brain:
        cmd += ["--brain", args.brain]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.min_iterations is not None:
        cmd += ["--min-iterations", str(args.min_iterations)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.runs_root:
        task_runs_root = Path(args.runs_root) / cfg_path.stem
        cmd += ["--runs-root", str(task_runs_root)]
    if args.quiet:
        cmd.append("--quiet")
    if args.kaggle_upload:
        cmd.append("--kaggle-upload")
    if args.resume:
        cmd += ["--resume", args.resume]
    return cmd


def parse_run_id(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, RUN_ID_FORMAT)
    except ValueError:
        return None


def find_new_run_dir(runs_root: Path, after: datetime) -> Path | None:
    """run_agent 成功后返回本批次刚创建的那个运行目录。"""
    if not runs_root.is_dir():
        return None
    candidates: list[tuple[datetime, Path]] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        ts = parse_run_id(child.name)
        # run_agent 的目录名只精确到秒；与批次启动时间（带微秒）比较前先截断。
        if ts is not None and ts >= after.replace(microsecond=0):
            candidates.append((ts, child))
    if not candidates:
        # --resume 续跑的是已存在的旧目录：按 state.jsonl 的 mtime 兜底识别。
        mtime_candidates: list[tuple[float, Path]] = []
        for child in runs_root.iterdir():
            if not child.is_dir():
                continue
            state_file = child / "state.jsonl"
            if not state_file.exists():
                continue
            mtime = state_file.stat().st_mtime
            if mtime >= after.timestamp() - 5.0:
                mtime_candidates.append((mtime, child))
        if not mtime_candidates:
            return None
        mtime_candidates.sort(key=lambda pair: pair[0])
        return mtime_candidates[-1][1]
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def find_latest_run_dir(runs_root: Path) -> Path | None:
    """返回 runs_root 下最近有 state.jsonl 的 run 目录。"""
    if not runs_root.is_dir():
        return None
    choices: list[tuple[float, Path]] = []
    for child in runs_root.iterdir():
        if child.is_dir() and (child / "state.jsonl").exists():
            choices.append((child.stat().st_mtime, child))
    if not choices:
        return None
    choices.sort(key=lambda pair: pair[0])
    return choices[-1][1]


def load_summary(run_dir: Path | None) -> tuple[dict[str, Any], str]:
    if run_dir is None:
        return {}, ""
    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        return {}, f"产物目录缺少 summary.json: {run_dir}"
    try:
        data = json.loads(summary_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"summary.json 解析失败: {exc}"
    if isinstance(data, dict):
        return data, ""
    return {}, "summary.json 格式异常"


def fetch_one_task(
    cfg_path: Path, task_id: str, args: argparse.Namespace
) -> tuple[bool, str]:
    """下载缺失数据；返回 (是否成功, 错误信息)。"""
    cmd = [
        sys.executable,
        "-m",
        "data.fetch",
        "--task",
        task_id,
    ]
    code = run_cmd(cmd, cwd=PROJECT_ROOT, dry_run=args.dry_run)
    if args.dry_run:
        return True, ""
    if code == 0:
        return True, ""
    return False, (
        f"数据下载失败（exit={code}）。请检查 Kaggle 凭证 / 网络 / "
        "是否已在网页端接受竞赛规则；如需跳过下载请加 --no-fetch。"
    )


def fmt_metric(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if abs(value) < 1e7 else f"{value:g}"
    return str(value)


def render_markdown(
    results: list[BatchResult],
    *,
    command: str,
    started_at: datetime,
    finished_at: datetime,
) -> str:
    lines = [
        "# 批量运行报告",
        "",
        f"- 开始时间: {started_at:%Y-%m-%d %H:%M:%S}",
        f"- 结束时间: {finished_at:%Y-%m-%d %H:%M:%S}",
        f"- 命令: `{command}`",
        "",
        "| 任务 | 状态 | brain | exit | stop_reason | 结果 | 提交 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        summary = result.summary
        stop_reason = summary.get("stop_reason") or ""
        target = summary.get("target_metric") or ""
        best_metrics = summary.get("best_metrics") or {}
        target_value = best_metrics.get(target) if target else None
        if target_value is not None:
            score_txt = f"{target}={fmt_metric(target_value)}"
        elif best_metrics:
            score_txt = json.dumps(best_metrics, ensure_ascii=False)
        else:
            score_txt = "-"

        submission = summary.get("submission") or {}
        sub_status = submission.get("status") or "-"
        if submission.get("submission"):
            sub_path = Path(str(submission["submission"])).name
            sub_txt = f"{sub_status} ({sub_path})"
        else:
            sub_txt = str(sub_status)
        if result.error:
            score_txt = result.error[-80:]

        lines.append(
            f"| {result.task_id} | {result.status} | "
            f"{result.brain or '-'} | {result.exit_code if result.exit_code is not None else '-'} | "
            f"{stop_reason or '-'} | {score_txt or '-'} | {sub_txt} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_batch_report(
    results: list[BatchResult],
    *,
    command: str,
    started_at: datetime,
    args: argparse.Namespace,
) -> Path:
    finished_at = datetime.now()
    report_root = PROJECT_ROOT / (args.runs_root or "runs") / "batch"
    report_dir = report_root / started_at.strftime(RUN_ID_FORMAT)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_md = render_markdown(
        results,
        command=command,
        started_at=started_at,
        finished_at=finished_at,
    )
    (report_dir / "BATCH_REPORT.md").write_text(
        report_md, encoding="utf-8"
    )
    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "command": command,
        "results": [
            {
                "task_id": r.task_id,
                "config_file": r.config_file,
                "status": r.status,
                "exit_code": r.exit_code,
                "error": r.error,
                "brain": r.brain,
                "run_dir": r.run_dir,
                "summary": r.summary,
            }
            for r in results
        ],
    }
    (report_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report_dir


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    load_env_file(PROJECT_ROOT)
    args = parse_args(argv)
    task_ids = parse_task_ids(args.tasks)
    try:
        configs = discover_configs(Path(args.configs_dir), task_ids)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", flush=True)
        return 2
    command = " ".join(["python", "run_all.py", *sys.argv[1:]])
    started_at = datetime.now()

    print(f"任务数量: {len(configs)} | 开始时间: {started_at:%H:%M:%S}")
    print(f"配置文件目录: {Path(args.configs_dir).resolve()}")
    print("=" * 60)

    results: list[BatchResult] = []
    for idx, cfg_path in enumerate(configs, start=1):
        task_id = cfg_path.stem
        print(f"\n===== [{idx}/{len(configs)}] {task_id} "
              f"({cfg_path.name}) =====")
        try:
            cfg = AgentConfig.from_yaml(cfg_path)
        except Exception as exc:
            print(f"[error] 配置加载失败: {exc}", flush=True)
            results.append(BatchResult(
                task_id=task_id,
                config_file=str(cfg_path),
                status="failed",
                error=str(exc)[-1500:],
            ))
            continue

        fetch_needed = needs_fetch(cfg, task_id)
        if fetch_needed and not args.no_fetch:
            print(f"[info] {task_id}: 数据目录为空，准备自动下载。", flush=True)
            ok, err = fetch_one_task(cfg_path, task_id, args)
            if args.dry_run:
                results.append(BatchResult(
                    task_id=task_id,
                    config_file=str(cfg_path),
                    status="dry-run",
                    brain=args.brain or cfg.brain.type,
                ))
                continue
            if not ok:
                print(f"[error] {task_id}: {err}", flush=True)
                results.append(BatchResult(
                    task_id=task_id,
                    config_file=str(cfg_path),
                    status="failed",
                    error=err,
                    brain=args.brain or cfg.brain.type,
                ))
                continue
        elif args.dry_run and not fetch_needed:
            print(f"[info] {task_id}: 数据已存在，无需下载。")

        agent_cmd = build_agent_cmd(cfg_path, args)
        runs_root = config_agent_runs_root(cfg, task_id, args)
        if args.dry_run:
            print(f"[info] {task_id}: 将执行 run_agent -> 自动提交 -> 汇总。")
            code = run_cmd(agent_cmd, cwd=PROJECT_ROOT, dry_run=True)
            results.append(BatchResult(
                task_id=task_id,
                config_file=str(cfg_path),
                status="dry-run",
                brain=args.brain or cfg.brain.type,
                exit_code=code,
            ))
            continue

        print(f"[info] {task_id}: runs 根目录 = {runs_root}", flush=True)
        code = run_cmd(agent_cmd, cwd=PROJECT_ROOT)
        run_dir = find_new_run_dir(runs_root, started_at)
        if run_dir is None and code == 0 and args.resume:
            # --resume 遇到已正常完成的 run：run_agent 会无操作退出（rc=0），
            # 此时沿用最近一次 summary 作为该任务结果。
            latest = find_latest_run_dir(runs_root)
            if latest is not None:
                run_dir = latest
        summary, summary_error = load_summary(run_dir)
        if summary:
            brain = summary.get("brain") or args.brain or cfg.brain.type
            run_dir_txt = (
                str(run_dir.relative_to(PROJECT_ROOT))
                if run_dir is not None else ""
            )
        else:
            brain = args.brain or cfg.brain.type
            run_dir_txt = ""

        error = summary_error
        semantic_failed = bool(
            summary
            and (
                summary.get("stop_reason") == "stop_failure"
                or int(summary.get("successful_iterations") or 0) <= 0
                or (summary.get("submission") or {}).get("status") == "failed"
            )
        )
        if code != 0:
            status = "failed"
            error = error or (
                f"run_agent 退出码 {code}；请查看终端输出或上述运行目录日志。"
            )
        elif not summary:
            status = "failed"
            error = error or (
                "run_agent 退出码 0，但未找到/解析到 summary.json，无法确认成功。"
            )
        elif semantic_failed:
            status = "failed"
            stop_reason = summary.get("stop_reason", "-")
            iterations = summary.get("successful_iterations", 0)
            submission_status = (
                (summary.get("submission") or {}).get("status", "-")
            )
            error = (
                f"run_agent 进程正常结束但语义失败: stop_reason={stop_reason}, "
                f"successful_iterations={iterations}, "
                f"submission.status={submission_status}"
            )
        else:
            status = "ok"
        print(f"[done] {task_id}: status={status}, exit={code}, "
              f"stop_reason={summary.get('stop_reason', '-')}", flush=True)
        results.append(BatchResult(
            task_id=task_id,
            config_file=str(cfg_path),
            status=status,
            exit_code=code,
            error=error,
            brain=brain,
            run_dir=run_dir_txt,
            summary=summary,
        ))

    report_dir = write_batch_report(
        results, command=command, started_at=started_at, args=args
    )
    print("\n" + "=" * 60)
    print(render_markdown(
        results,
        command=command,
        started_at=started_at,
        finished_at=datetime.now(),
    ))
    print(f"批量报告: {report_dir / 'BATCH_REPORT.md'}")
    failed = [r for r in results if not r.ok]
    if args.dry_run:
        print("dry-run 完成：未执行任何下载/训练。")
    elif failed:
        print(f"有 {len(failed)} 个任务未成功: "
              f"{', '.join(r.task_id for r in failed)}")
        return 1
    print("全部任务成功完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
