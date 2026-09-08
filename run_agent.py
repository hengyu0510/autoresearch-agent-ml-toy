"""AutoResearch Agent for ML - 唯一主入口。

流程：
    plan -> execute(真实 Python 实验) -> evaluate -> state -> decision
    （继续 / 回退 / 收敛停止 / 失败退出 / 达到预算停止）

用法：
    python run_agent.py --config config.yaml [--brain rule|llm]
                        [--max-steps N] [--min-iterations N] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.config import AgentConfig, load_env_file
from agent.evaluator import Evaluator
from agent.executor import Executor
from agent.logger import Logger, configure_stdio
from agent.planner import build_brain
from agent.state import RunState

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--brain", choices=["rule", "llm"], default=None,
                        help="覆盖 config 中的 brain.type")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--min-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs-root", type=str, default=None)
    parser.add_argument("--quiet", action="store_true",
                        help="控制台只输出关键信息")
    parser.add_argument(
        "--kaggle-upload", action="store_true",
        help="本地提交校验通过后，可选调用 Kaggle CLI 实际上传（默认不访问网络）",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: AgentConfig, args: argparse.Namespace) -> None:
    if args.brain:
        cfg.brain.type = args.brain
        cfg.apply_llm_env_overrides()
    if args.max_steps is not None:
        cfg.run.max_steps = args.max_steps
    if args.min_iterations is not None:
        cfg.run.min_iterations = args.min_iterations
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    if args.runs_root:
        cfg.run.runs_root = args.runs_root
    cfg.apply_llm_env_overrides()
    cfg.validate()


def make_run_dir(cfg: AgentConfig) -> Path:
    root = Path(cfg.run.runs_root)
    root = root if root.is_absolute() else (cfg.project_root / root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def record_failure(
    state: RunState,
    *,
    step: int,
    attempt: int | None,
    message: str,
    error_tail: str,
    cfg: AgentConfig,
    consecutive_failed_steps: int,
) -> int:
    state.append(
        "error",
        step=step,
        attempt=attempt,
        message=message,
        error_tail=error_tail[-3000:],
    )
    consecutive_failed_steps += 1
    if consecutive_failed_steps >= cfg.run.max_consecutive_errors:
        state.append(
            "decision",
            step=step,
            message=(
                f"连续 {consecutive_failed_steps} 步失败，达到阈值 "
                f"{cfg.run.max_consecutive_errors}，按失败退出"
            ),
            decision="stop_failure",
        )
    return consecutive_failed_steps


def should_stop(
    cfg: AgentConfig,
    *,
    successful_rounds: int,
    best_metrics: dict[str, Any] | None,
    improvements: deque[float],
) -> str | None:
    target = cfg.run.target_metric
    higher = cfg.run.higher_is_better
    if (
        best_metrics
        and cfg.run.success_threshold is not None
        and successful_rounds >= cfg.run.min_iterations
    ):
        best_value = float(best_metrics.get(target))
        hit = (
            best_value >= float(cfg.run.success_threshold)
            if higher
            else best_value <= float(cfg.run.success_threshold)
        )
        if hit:
            return "stop_success"
    deltas = list(improvements)
    if (
        successful_rounds >= cfg.run.min_iterations
        and len(deltas) >= cfg.run.convergence_rounds
        and all(
            d < cfg.run.improve_min
            for d in deltas[-cfg.run.convergence_rounds:]
        )
    ):
        return "stop_converged"
    return None


def run_best_submission(
    cfg: AgentConfig,
    run_dir: Path,
    state: RunState,
    *,
    best_params_file: Path | None,
    log: Logger,
) -> dict[str, Any]:
    """用最佳实验快照自动生成提交文件，并做本地格式校验。

    该阶段不依赖决策大脑，是 Agent 正常流程的收尾动作；结果写入
    state.jsonl 与 summary。失败不改变原运行退出原因，只记录错误。
    """
    if not cfg.experiment.submit_script:
        return {"status": "skipped", "reason": "任务未配置 submit_script"}
    if best_params_file is None or not best_params_file.exists():
        return {
            "status": "skipped",
            "reason": "无最佳参数快照（没有成功的实验轮次）",
        }

    script = cfg.submit_script_path
    if not script.exists():
        return {"status": "failed", "reason": f"submit 脚本不存在: {script}"}

    sub_dir = run_dir / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    submission_path = sub_dir / "best_submission.csv"
    metrics_path = sub_dir / "best_submission_metrics.json"

    cmd = [
        sys.executable, str(script),
        "--params", str(best_params_file),
        "--out", str(submission_path),
        "--metrics-out", str(metrics_path),
    ]
    log.info(f"自动提交：用最佳快照生成提交文件 -> {submission_path}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cfg.project_root),
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(cfg.run.timeout_seconds, 600),
            check=False,
        )
    except subprocess.TimeoutExpired:
        message = f"自动提交超时（>{cfg.run.timeout_seconds}s）"
        state.append("submit", message=message, success=False, error_tail=message)
        return {"status": "failed", "reason": message}
    except Exception as exc:
        state.append("submit", message=f"自动提交启动失败: {exc}",
                     success=False, error_tail=str(exc))
        return {"status": "failed", "reason": str(exc)}

    if proc.returncode != 0:
        error_tail = (proc.stderr or proc.stdout or "").strip()[-3000:]
        state.append(
            "submit", message="提交脚本执行失败", success=False,
            error_tail=error_tail, stdout=proc.stdout[-2000:],
        )
        log.error(f"自动提交失败: {error_tail[-500:]}")
        return {"status": "failed", "reason": error_tail}

    submission_metrics = {}
    if metrics_path.exists():
        try:
            submission_metrics = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
        except Exception:
            log.warning("提交指标文件解析失败，忽略")

    # 可选的官方 sample 格式校验
    validation = {"status": "skipped", "output": ""}
    if cfg.experiment.sample_submission:
        sample = cfg.sample_submission_path
        validator = cfg.project_root / "experiments" / "validate_submission.py"
        if not sample.exists() or not validator.exists():
            validation = {
                "status": "failed",
                "output": f"sample/validator 不存在: {sample} / {validator}",
            }
        else:
            vcmd = [
                sys.executable, str(validator),
                "--submission", str(submission_path),
                "--sample", str(sample),
            ]
            if cfg.experiment.submission_id_col:
                vcmd += ["--id-col", cfg.experiment.submission_id_col]
            if cfg.experiment.submission_required_cols:
                vcmd += ["--required", cfg.experiment.submission_required_cols]
            try:
                vproc = subprocess.run(
                    vcmd,
                    cwd=str(cfg.project_root),
                    env={**os.environ, "PYTHONIOENCODING": "utf-8",
                         "PYTHONUTF8": "1"},
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=120,
                    check=False,
                )
                validation = {
                    "status": "ok" if vproc.returncode == 0 else "failed",
                    "output": (vproc.stdout or vproc.stderr or "").strip(),
                }
            except Exception as exc:
                validation = {"status": "failed", "output": str(exc)}

    state.append(
        "submit", message=f"自动提交完成: {submission_path}",
        success=True, submission=str(submission_path.resolve()),
        params_file=str(best_params_file.resolve()),
        metrics=submission_metrics, validation=validation,
    )
    log.info(f"自动提交成功，校验状态={validation.get('status')}，"
             f"输出={submission_path}")
    return {
        "status": "ok",
        "submission": str(submission_path.resolve()),
        "metrics": submission_metrics,
        "validation": validation,
    }


def run_kaggle_upload(
    task_id: str,
    submission_path: Path,
    log: Logger,
) -> dict[str, Any]:
    """可选实际上传：通过 data/kaggle_upload.py 调用 Kaggle CLI。"""
    script = PROJECT_ROOT / "data" / "kaggle_upload.py"
    cmd = [
        sys.executable, str(script),
        "--task", task_id,
        "--submission", str(submission_path),
        "--message",
        f"AutoResearch Agent for ML submission {datetime.now():%Y-%m-%d %H:%M:%S}",
    ]
    log.info(f"Kaggle 上传（可选）: task={task_id}, file={submission_path}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "detail": "Kaggle 上传超时（>300s）"}
    except Exception as exc:
        return {"status": "failed", "detail": f"Kaggle 上传启动失败: {exc}"}
    detail = (proc.stdout or proc.stderr or "").strip()[-2000:]
    if proc.returncode != 0:
        log.error(f"Kaggle 上传失败: {detail[-500:]}")
        return {"status": "failed", "detail": detail}
    log.info(f"Kaggle 上传成功: {detail[-300:]}")
    return {"status": "ok", "detail": detail}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_stdio()
    load_env_file(PROJECT_ROOT)
    cfg = AgentConfig.from_yaml(args.config)
    apply_overrides(cfg, args)

    run_dir = make_run_dir(cfg)
    log = Logger(log_file=run_dir / "agent.log", verbose=not args.quiet)
    log.info(f"加载配置: {Path(args.config).resolve()}")
    log.info(f"大脑: {cfg.brain.type}，目标指标: {cfg.run.target_metric}")
    log.info(f"运行目录: {run_dir}")

    state = RunState(run_dir, log)
    state.append(
        "init",
        step=0,
        message="AutoResearch Agent 启动",
        brain=cfg.brain.type,
        task_description=cfg.experiment.description,
        seed=cfg.experiment.seed,
        max_steps=cfg.run.max_steps,
        min_iterations=cfg.run.min_iterations,
        target_metric=cfg.run.target_metric,
    )

    brain = build_brain(cfg, log)
    executor = Executor(cfg, run_dir, log)
    evaluator = Evaluator(cfg, log)

    best_metrics: dict[str, Any] | None = None
    best_params_file: Path | None = None
    previous_value: float | None = None
    improvements: deque[float] = deque(maxlen=cfg.run.convergence_rounds)
    successful_rounds = 0
    consecutive_failed_steps = 0
    stop_reason = "stop_max_steps"
    last_metrics: dict[str, Any] | None = None

    for step in range(1, cfg.run.max_steps + 1):
        log.info(f"===== 第 {step} 步（共 {cfg.run.max_steps}） =====")

        # 1) 大脑提出下一步修改方案
        try:
            action = brain.propose(step, state)
        except Exception as exc:
            log.error(f"规划失败: {exc}")
            consecutive_failed_steps = record_failure(
                state, step=step, attempt=None,
                message=f"规划失败: {exc}", error_tail=str(exc),
                cfg=cfg, consecutive_failed_steps=consecutive_failed_steps,
            )
            if consecutive_failed_steps >= cfg.run.max_consecutive_errors:
                stop_reason = "stop_failure"
                break
            continue

        if action is None:
            stop_reason = "plan_exhausted"
            state.append(
                "decision", step=step,
                message="候选方案已耗尽，停止本轮运行",
                decision="stop_plan_exhausted",
            )
            break

        state.append(
            "plan", step=step,
            message=f"大脑({action.source})提出方案: {action.description}",
            description=action.description,
            rationale=action.rationale,
            params=action.params,
            source=action.source,
        )

        # 2) 执行实验（失败时按配置重试同一方案）
        result = None
        step_ok = False
        for attempt in range(1, cfg.run.max_attempts_per_step + 1):
            log.info(f"执行实验: {action.description}（尝试 {attempt}）")
            result = executor.run_experiment(
                action, step=step, attempt=attempt
            )
            if result.success:
                last_metrics = result.metrics
                target = cfg.run.target_metric
                metric_value = result.metrics.get(target)
                val_txt = (
                    f"{metric_value:.4f}"
                    if isinstance(metric_value, (int, float))
                    else str(metric_value)
                )
                state.append(
                    "execute", step=step, attempt=attempt, success=True,
                    message=(
                        f"实验运行成功（耗时 {result.duration_sec}s），"
                        f"{target}={val_txt}"
                    ),
                    params=action.params,
                    metrics=result.metrics,
                    duration_sec=result.duration_sec,
                    stdout_log=str(result.stdout_log),
                    stderr_log=str(result.stderr_log),
                )
                step_ok = True
                break
            consecutive_failed_steps = record_failure(
                state, step=step, attempt=attempt,
                message=f"实验运行失败: {action.description}",
                error_tail=result.error_tail,
                cfg=cfg, consecutive_failed_steps=consecutive_failed_steps,
            )
            log.error(f"第 {step} 步尝试 {attempt} 失败，"
                      f"错误尾部: {result.error_tail[-500:]}")

        if not step_ok:
            if consecutive_failed_steps >= cfg.run.max_consecutive_errors:
                stop_reason = "stop_failure"
                break
            # 尝试下一个候选方案（不把失败计入“成功迭代”）
            continue

        successful_rounds += 1

        # 3) 独立评估：决定采纳 / 回退
        try:
            evaluation = evaluator.evaluate(
                result.metrics, best_metrics, previous_value
            )
        except Exception as exc:
            log.error(f"指标评估失败: {exc}")
            consecutive_failed_steps = record_failure(
                state, step=step, attempt=result.attempt,
                message=f"指标评估失败: {exc}", error_tail=str(exc),
                cfg=cfg, consecutive_failed_steps=consecutive_failed_steps,
            )
            if consecutive_failed_steps >= cfg.run.max_consecutive_errors:
                stop_reason = "stop_failure"
                break
            continue

        best_metrics = evaluation.best_metrics
        if evaluation.decision == "accept" and result.params_file is not None:
            best_params_file = result.params_file
        target = cfg.run.target_metric
        current_value = float(result.metrics[target])
        if evaluation.improvement_from_previous is not None:
            improvements.append(evaluation.improvement_from_previous)
        previous_value = current_value

        best_value = (
            float(best_metrics.get(target))
            if best_metrics is not None else None
        )
        state.append(
            "evaluate", step=step,
            message=evaluation.reason,
            decision=evaluation.decision,
            metrics=result.metrics,
            delta_from_previous=evaluation.delta_from_previous,
        )
        state.append(
            "decision", step=step,
            message=(
                f"本轮决策={evaluation.decision}；"
                f"已成功迭代 {successful_rounds} 轮；"
                f"当前最佳 {target}="
                f"{best_value if best_value is not None else 'None':}"
            ),
            decision=evaluation.decision,
            successful_rounds=successful_rounds,
            best_metrics=best_metrics,
        )

        # 4) 终止检查：成功阈值 / 收敛 / 继续
        reason = should_stop(
            cfg,
            successful_rounds=successful_rounds,
            best_metrics=best_metrics,
            improvements=improvements,
        )
        if reason:
            stop_reason = reason
            state.append(
                "decision", step=step,
                message=f"触发终止条件: {reason}",
                decision=reason,
            )
            break

    # 收尾：汇总与可读日志
    submission_result = run_best_submission(
        cfg, run_dir, state,
        best_params_file=best_params_file,
        log=log,
    )
    upload_result: dict[str, Any] | None = None
    if args.kaggle_upload:
        if submission_result.get("status") != "ok":
            upload_result = {
                "status": "skipped",
                "detail": "本地提交未成功，不执行上传",
            }
        else:
            upload_result = run_kaggle_upload(
                Path(args.config).stem,
                Path(submission_result["submission"]),
                log,
            )
    summary = {
        "run_id": run_dir.name,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "brain": cfg.brain.type,
        "config_file": str(Path(args.config).resolve()),
        "max_steps": cfg.run.max_steps,
        "min_iterations": cfg.run.min_iterations,
        "target_metric": cfg.run.target_metric,
        "successful_iterations": successful_rounds,
        "stop_reason": stop_reason,
        "best_metrics": best_metrics,
        "best_params_file": (
            str(best_params_file) if best_params_file is not None else None
        ),
        "last_metrics": last_metrics,
        "state_file": str(state.state_file),
        "submission": submission_result,
        "kaggle_upload": upload_result,
    }
    submission_status = (submission_result or {}).get("status", "skipped")
    upload_status = (upload_result or {}).get("status", "skipped")
    exit_code = 1 if (
        stop_reason == "stop_failure"
        or successful_rounds == 0
        or submission_status == "failed"
        or upload_status == "failed"
    ) else 0
    summary["exit_code"] = exit_code
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    run_log_md = state.render_markdown(
        f"AutoResearch Agent 运行日志 — {run_dir.name}"
    )
    (run_dir / "RUN_LOG.md").write_text(run_log_md, encoding="utf-8")
    state.append(
        "finish",
        message=f"运行结束: {stop_reason}，成功迭代 {successful_rounds} 轮",
        stop_reason=stop_reason,
        successful_iterations=successful_rounds,
        best_metrics=best_metrics,
    )

    log.info("=" * 50)
    log.info(f"停止原因: {stop_reason}")
    log.info(f"成功迭代: {successful_rounds} 轮")
    log.info(f"最佳结果: {json.dumps(best_metrics, ensure_ascii=False)}")
    log.info(f"退出码: {exit_code}")
    log.info(f"产物目录: {run_dir}")
    log.info(f"完整状态: {state.state_file}")
    log.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
