"""Kaggle 数据下载与缓存层。

设计：
    data/raw/<task_id>/       最终使用的原始文件（解压后）
    data/archives/<task_id>/  下载的压缩包缓存（避免重复下载）

用法：
    python -m data.fetch --list
    python -m data.fetch --task facial_keypoints           # 真实下载
    python -m data.fetch --task facial_keypoints --dry-run  # 仅预览
    python -m data.fetch --all

认证：
    需要 kaggle.json（~/.kaggle/kaggle.json）或环境变量
    KAGGLE_USERNAME / KAGGLE_KEY；凭证严禁提交到仓库。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# 允许从仓库根目录以 python -m data.fetch 方式运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import load_env_file  # noqa: E402
from tasks.registry import MLTask, get_task, load_tasks  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
ARCHIVE_ROOT = DATA_ROOT / "archives"


def _has_kaggle_credentials() -> bool:
    new_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    legacy_env_ok = bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return new_token or legacy_env_ok or kaggle_json.exists()


def check_credentials() -> None:
    if _has_kaggle_credentials():
        return
    raise RuntimeError(
        "未检测到 Kaggle 凭证。请任选其一：\n"
        "  1) 新版 token：设置 KAGGLE_API_TOKEN=KGAT_xxx\n"
        "     （Kaggle Settings -> API -> Generate New Token）；\n"
        "  2) 旧版：在 ~/.kaggle/kaggle.json 放置凭证，或设置\n"
        "     KAGGLE_USERNAME 与 KAGGLE_KEY。\n"
        "注意：大部分竞赛还需先在 Kaggle 网页端接受规则。"
    )


def _kaggle_download_cmd(task: MLTask, dest: Path) -> list[str]:
    if task.source.provider == "kaggle_competition":
        return ["kaggle", "competitions", "download",
                "-c", task.source.slug, "-p", str(dest)]
    if task.source.provider == "kaggle_dataset":
        return ["kaggle", "datasets", "download",
                "-d", task.source.slug, "-p", str(dest)]
    raise NotImplementedError(
        f"暂不支持 source.provider={task.source.provider!r}"
    )


def extract_archives(archive_dir: Path, dest: Path) -> list[str]:
    extracted: list[str] = []
    processed: set[Path] = set()
    pending: list[Path] = sorted(archive_dir.glob("*.zip"))

    while pending:
        archive = pending.pop(0)
        if archive in processed:
            continue
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        processed.add(archive)
        extracted.append(archive.name)
        for nested in sorted(dest.rglob("*.zip")):
            if nested not in processed and nested not in pending:
                pending.append(nested)

    # Remove nested archives copied into the raw dir; only the outer
    # archives under data/archives/ are kept as the download cache.
    resolved_dest = dest.resolve()
    resolved_raw = RAW_ROOT.resolve()
    if resolved_raw not in resolved_dest.parents and resolved_dest != resolved_raw:
        raise ValueError(f"Refusing to clean non-raw directory: {dest}")
    for nested in dest.rglob("*.zip"):
        nested.unlink()
    return extracted


def fetch_task(task: MLTask, *, force: bool = False, dry_run: bool = False) -> None:
    dest = RAW_ROOT / task.id
    archive_dir = ARCHIVE_ROOT / task.id

    if dest.exists() and any(dest.iterdir()) and not force:
        print(f"[skip] {task.id}: 原始数据已存在于 {dest}（--force 可强制重下）")
        return

    dest.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    if force and dest.exists():
        resolved = dest.resolve()
        raw_root = RAW_ROOT.resolve()
        if raw_root not in resolved.parents or resolved == raw_root:
            raise ValueError(f"拒绝删除非 data/raw 下的目录: {dest}")
        shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        check_credentials()
    cmd = _kaggle_download_cmd(task, archive_dir)
    print(f"[{'dry-run' if dry_run else 'run'}] "
          + " ".join(str(c) for c in cmd))

    if dry_run:
        if not _has_kaggle_credentials():
            print("[warn] 尚未检测到 Kaggle 凭证；真实下载前需配置")
        return

    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if "403" in detail:
            url = (f"https://www.kaggle.com/competitions/{task.source.slug}"
                   if task.source.provider == "kaggle_competition"
                   else f"https://www.kaggle.com/datasets/{task.source.slug}")
            raise RuntimeError(
                f"任务 {task.id} 下载被拒绝(403)：请先在 Kaggle 网页端"
                f"打开 {url} 并接受规则（Join/Accept Rules）后重试。\n"
                f"若竞赛已结束且无法再接受规则，需要改用数据集镜像源。"
            )
        raise RuntimeError(
            f"任务 {task.id} 下载失败:\n{detail[-1500:]}"
        )
    extracted = extract_archives(archive_dir, dest)
    files = sorted(p.name for p in dest.rglob("*") if p.is_file())
    print(f"[ok] {task.id}: 解压 {len(extracted)} 个压缩包，共 {len(files)} 个文件")
    for name in files:
        print(f"     {name}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    load_env_file(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default=None,
                        help="任务 id，见 --list")
    parser.add_argument("--all", action="store_true",
                        help="下载注册表中的全部任务")
    parser.add_argument("--force", action="store_true",
                        help="清空并重新下载该任务原始数据")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将要执行的下载命令，不访问网络")
    parser.add_argument("--list", action="store_true",
                        help="列出注册表中的任务")
    args = parser.parse_args(argv)

    if args.list:
        for task in load_tasks():
            print(f"{task.id:<22} {task.name}")
            print(f"    domain={task.domain}, metric={task.metric}, "
                  f"source={task.source.provider}:{task.source.slug}")
        return 0

    if args.all:
        tasks = load_tasks()
    elif args.task:
        tasks = (get_task(args.task),)
    else:
        parser.error("请指定 --task <id> 或 --all（可用 --list 查看）")

    for task in tasks:
        fetch_task(task, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
