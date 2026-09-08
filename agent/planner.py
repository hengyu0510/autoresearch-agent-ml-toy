"""Planner：根据任务与当前状态提出下一步可执行的修改方案。

- `rule` 大脑：内置确定性候选序列，无需 API key，可复现演示；
- `llm` 大脑：调用 OpenAI 兼容的 chat/completions 接口，
  让模型基于运行历史提出下一步；失败时可按配置回退到 rule。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import AgentConfig
from .logger import Logger
from .state import RunState


@dataclass
class Action:
    """一次可执行的实验修改方案。"""

    description: str
    params: dict[str, Any]
    rationale: str = ""
    source: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "params": self.params,
            "rationale": self.rationale,
            "source": self.source,
        }


# rule 大脑的确定性候选序列：每轮只做一个有依据的修改。
RULE_CANDIDATES: list[dict[str, Any]] = [
    {
        "description": "基线：Logistic Regression（不做特征缩放）",
        "rationale": "先建立可运行的 baseline，作为后续所有比较的锚点。",
        "params": {
            "model": "logistic_regression",
            "scaler": False,
            "hyperparams": {"C": 1.0, "max_iter": 2000},
        },
    },
    {
        "description": "加入 StandardScaler 后重训 Logistic Regression",
        "rationale": "数值特征量纲差异大，标准化通常能提升线性模型稳定性。",
        "params": {
            "model": "logistic_regression",
            "scaler": True,
            "hyperparams": {"C": 1.0, "max_iter": 2000},
        },
    },
    {
        "description": "尝试 Random Forest（400 棵树并限制深度防过拟合）",
        "rationale": "树模型对量纲不敏感，可捕捉非线性交互，作为模型族对照。",
        "params": {
            "model": "random_forest",
            "scaler": False,
            "hyperparams": {
                "n_estimators": 400,
                "max_depth": 8,
                "min_samples_leaf": 2,
            },
        },
    },
    {
        "description": "尝试 MLP（64-32 隐藏层，标准化特征）",
        "rationale": "验证简单神经网络在该小数据集上是否优于传统模型。",
        "params": {
            "model": "mlp",
            "scaler": True,
            "hyperparams": {
                "hidden_layer_sizes": [64, 32],
                "max_iter": 1000,
                "alpha": 1e-4,
            },
        },
    },
    {
        "description": "对最佳线性模型做正则微调（C=0.05）",
        "rationale": "小数据集上更强的正则化可降低方差，尝试进一步稳定指标。",
        "params": {
            "model": "logistic_regression",
            "scaler": True,
            "hyperparams": {"C": 0.05, "max_iter": 2000},
        },
    },
]


class RuleBrain:
    name = "rule"

    def __init__(self, cfg: AgentConfig, log: Logger):
        self.cfg = cfg
        self.log = log
        self._index = 0
        self.candidates = (
            list(cfg.experiment.rule_candidates)
            if cfg.experiment.rule_candidates
            else RULE_CANDIDATES
        )

    def propose(self, step: int, state: RunState) -> Action | None:
        if self._index >= len(self.candidates):
            return None
        item = self.candidates[self._index]
        self._index += 1
        return Action(
            description=item["description"],
            params=dict(item["params"]),
            rationale=item["rationale"],
            source=self.name,
        )


class LLMBrain:
    name = "llm"
    DEFAULT_BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
    }
    DEFAULT_KEY_ENVS = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    def __init__(
        self,
        cfg: AgentConfig,
        log: Logger,
        fallback: RuleBrain | None = None,
    ):
        self.cfg = cfg
        self.log = log
        self.fallback = fallback
        brain = cfg.brain
        self.provider = (brain.provider or "openai").lower()
        if self.provider not in self.DEFAULT_BASE_URLS:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")
        default_url = self.DEFAULT_BASE_URLS[self.provider]
        self.base_url = (brain.base_url or default_url).rstrip("/")
        self.model = brain.model
        if not self.model:
            raise ValueError(
                "llm 大脑缺少 model，请在 config.yaml brain.model 或 "
                "OPENAI_MODEL / ANTHROPIC_MODEL 环境变量中配置"
            )
        self.temperature = brain.temperature
        self.max_tokens = brain.max_tokens
        self.reasoning_effort = (brain.reasoning_effort or "").strip().lower()
        self.key_env = (
            brain.api_key_env
            or self.DEFAULT_KEY_ENVS[self.provider]
        )
        self.api_key = os.environ.get(self.key_env, "")

    @property
    def system_prompt(self) -> str:
        return (
            "你是自动迭代机器学习实验的 Agent。每次只做一个有依据的小步"
            "修改，基于验证集指标决策。只输出一个合法的 JSON 对象，不要"
            "Markdown 代码块，不要任何前缀、解释或额外文字。"
        )

    def _allowed_models(self) -> list[str]:
        names: list[str] = []
        for cand in self.cfg.experiment.rule_candidates or []:
            params = cand.get("params") or {}
            model = params.get("model")
            if model and model not in names:
                names.append(model)
        if not names:
            names = ["logistic_regression", "random_forest", "mlp"]
        return names

    def propose(self, step: int, state: RunState) -> Action | None:
        if not self.api_key:
            raise RuntimeError(
                f"缺少环境变量 {self.key_env}（LLM 大脑需要 API key）"
            )
        last_error: Exception | None = None
        for attempt in range(1, 3):  # 允许一次自动重试（空响应/截断 JSON 偶发）
            try:
                return self._propose_remote(step, state)
            except Exception as exc:
                self.log.error(
                    f"LLM 大脑调用失败（第 {attempt} 次）: {exc}"
                )
                last_error = exc
        if last_error is not None:
            if self.fallback is not None and self.cfg.brain.fallback_to_rule:
                self.log.warning(
                    f"重试后仍失败，回退到 rule 大脑提出下一步方案: "
                    f"{last_error}"
                )
                return self.fallback.propose(step, state)
            raise last_error
        return None

    def _build_user_prompt(self, state: RunState) -> str:
        runs = state.successful_runs()
        target = self.cfg.run.target_metric
        test_key = target.replace("val_", "test_", 1)
        history = []
        for i, e in enumerate(runs[-6:], start=1):
            m = e.get("metrics", {})
            params = (e.get("params") or {})
            model = params.get("model", "?")
            val_txt = (
                f"{m[target]:.4f}"
                if isinstance(m.get(target), (int, float))
                else str(m.get(target))
            )
            test_txt = (
                f"{m[test_key]:.4f}"
                if isinstance(m.get(test_key), (int, float))
                else str(m.get(test_key))
            )
            history.append(
                f"轮次 {i}: model={model}, "
                f"{target}={val_txt}, {test_key}={test_txt}"
            )
        history_txt = "\n".join(history) if history else "（暂无已完成实验）"
        return (
            f"任务：{self.cfg.experiment.description}\n"
            f"当前步数：{len(runs) + 1}\n"
            f"已完成的实验：\n{history_txt}\n"
            f"本任务 train.py 仅支持以下模型名：{self._allowed_models()}\n"
            "params.model 必须取上述列表中的值，不要提出白名单外的新模型。\n"
            "请基于以上状态提出下一步唯一的修改方案，并以 JSON 返回："
            '{"description": str, "rationale": str, '
            '"params": {"model": "<该任务 train.py 支持的模型名>", '
            '"scaler": bool, "hyperparams": {...}}}。'
            "不要输出 JSON 以外的内容。"
        )

    def _propose_remote(self, step: int, state: RunState) -> Action:
        if self.provider == "anthropic":
            return self._call_anthropic(state)
        return self._call_openai_compatible(state)

    def _call_openai_compatible(self, state: RunState) -> Action:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_user_prompt(state)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if not content:
            raise ValueError(
                "DeepSeek 返回空 content；"
                f"finish_reason="
                f"{body['choices'][0].get('finish_reason')!r}"
            )
        return self._content_to_action(content)

    def _call_anthropic(self, state: RunState) -> Action:
        if self.reasoning_effort:
            self.log.warning(
                "reasoning_effort 暂不映射到 Anthropic Messages API，已忽略"
            )
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": self.system_prompt,
            "messages": [
                {"role": "user", "content": self._build_user_prompt(state)},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        parts = body.get("content") or []
        content = "\n".join(
            p.get("text", "") for p in parts if p.get("type") == "text"
        )
        if not content:
            raise ValueError(f"Anthropic 返回空文本: {body}")
        return self._content_to_action(content)

    def _content_to_action(self, content: str) -> Action:
        action = self._parse_action(content)
        self.log.info(f"LLM 提出方案: {action.description}")
        return action

    @staticmethod
    def _parse_action(content: str) -> Action:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().lower().startswith("```json"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"LLM 输出不是合法 JSON: {content[:200]}")
        data = json.loads(text[start:end + 1])
        params = data.get("params")
        if not isinstance(params, dict) or "model" not in params:
            raise ValueError(f"LLM 返回缺少合法 params: {data}")
        return Action(
            description=str(data.get("description", "LLM 提出的修改")),
            params=params,
            rationale=str(data.get("rationale", "")),
            source="llm",
        )


def build_brain(cfg: AgentConfig, log: Logger):
    if cfg.brain.type == "llm":
        return LLMBrain(cfg, log, fallback=RuleBrain(cfg, log))
    return RuleBrain(cfg, log)
