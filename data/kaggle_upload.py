"""可选：把本地提交文件上传到 Kaggle 竞赛。

默认不执行；只有用户显式传入 --kaggle-upload（run_agent/run_all）或直接
运行本模块时才访问网络：

    python data/kaggle_upload.py --task titanic \
        --submission submissions/titanic_submission.csv

注意：老竞赛是否仍接受提交取决于账号与该比赛当前状态；失败不会影响本地
提交文件与格式校验结果。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import load_env_file  # noqa: E402
from tasks.registry import get_task  # noqa: E402


def upload_submission(
    task_id: str,
    submission_path: str | Path,
    message: str,
) -> dict[str, str]:
    """调用 Kaggle CLI 上传；返回 {status: ok|failed, detail}。"""
    path = Path(submission_path)
    if not path.is_file():
        return {
            "status": "failed",
            "detail": f"提交文件不存在: {path}",
        }
    task = get_task(task_id)
    if task.source.provider != "kaggle_competition":
        return {
            "status": "failed",
            "detail": (
                f"任务 {task_id} 不是 kaggle_competition，无法用 CLI 上传"
            ),
        }
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", task.source.slug,
        "-f", str(path),
        "-m", message,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0:
        return {
            "status": "failed",
            "detail": f"Kaggle 上传失败（exit={proc.returncode}）:\n{output[-2000:]}",
        }
    return {"status": "ok", "detail": output[-2000:] or "Kaggle 已接受提交"}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    load_env_file(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, required=True,
                        help="tasks/tasks.yaml 中的任务 id")
    parser.add_argument("--submission", type=str, required=True,
                        help="要上传的 CSV 路径")
    parser.add_argument("--message", type=str,
                        default="AutoResearch Agent for ML submission")
    args = parser.parse_args(argv)
    result = upload_submission(args.task, args.submission, args.message)
    print(f"[kaggle-upload] status={result['status']}")
    print(result["detail"])
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
