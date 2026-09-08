"""Evaluator：读取指标，独立判断本轮是否有效、是否保留/回退。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AgentConfig
from .logger import Logger


@dataclass
class Evaluation:
    decision: str          # accept | revert
    reason: str
    best_metrics: dict[str, Any] | None
    delta_from_previous: float | None = None
    improvement_from_previous: float | None = None


class Evaluator:
    def __init__(self, cfg: AgentConfig, log: Logger):
        self.cfg = cfg
        self.log = log

    def evaluate(
        self,
        metrics: dict[str, Any],
        best_metrics: dict[str, Any] | None,
        previous_value: float | None,
    ) -> Evaluation:
        target = self.cfg.run.target_metric
        value = metrics.get(target)
        if value is None:
            raise ValueError(f"指标缺少目标字段 {target!r}: {metrics}")
        prev_best_value = (
            best_metrics.get(target) if best_metrics is not None else None
        )
        higher = self.cfg.run.higher_is_better
        delta = (
            float(value) - float(previous_value)
            if previous_value is not None else None
        )
        improvement = (
            None
            if previous_value is None
            else (
                float(value) - float(previous_value)
                if higher
                else float(previous_value) - float(value)
            )
        )
        if best_metrics is None:
            accepted = True
        elif higher:
            accepted = float(value) >= float(prev_best_value)
        else:
            accepted = float(value) <= float(prev_best_value)
        if accepted:
            new_best = metrics
            decision = "accept"
            if prev_best_value is None:
                reason = f"首轮结果，作为当前最佳（{target}={value:.4f}）"
            else:
                cmp_txt = "不低于" if higher else "不高于"
                reason = (
                    f"{target} 达到 {value:.4f}，{cmp_txt}当前最佳 "
                    f"{prev_best_value:.4f}，采纳本轮结果"
                )
        else:
            new_best = best_metrics
            decision = "revert"
            cmp_txt = "低于" if higher else "高于"
            reason = (
                f"{target}={value:.4f} {cmp_txt}当前最佳 {prev_best_value:.4f}，"
                "保留最佳结果并记录本轮回退"
            )
        self.log.info(f"评估决策: {decision} — {reason}")
        return Evaluation(
            decision=decision,
            reason=reason,
            best_metrics=new_best,
            delta_from_previous=delta,
            improvement_from_previous=improvement,
        )
