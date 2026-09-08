"""Planner：根据任务与当前状态提出下一步可执行的修改方案。

- `rule` 大脑：内置确定性候选序列，无需 API key，可复现演示；
- `llm` 大脑：调用 OpenAI 兼容的 chat/completions 接口，
  让模型基于运行历史提出下一步；失败时可按配置回退到 rule。
"""

from __future__ import annotations

import json
import math
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

    def checkpoint_state(self) -> dict[str, Any]:
        return {"type": self.name, "rule_index": self._index}

    def restore_state(self, data: dict[str, Any] | None) -> None:
        if not data:
            return
        index = data.get("rule_index")
        if isinstance(index, int):
            self._index = max(0, min(index, len(self.candidates)))

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

    def reflect(
        self,
        *,
        state: RunState,
        step: int,
        decision: str,
        reason: str,
        description: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        """rule 大脑不调用 LLM 反思，返回 None（由调用方写确定性 insight）。"""
        return None


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

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "fallback_rule": (
                self.fallback.checkpoint_state() if self.fallback else None
            ),
        }

    def restore_state(self, data: dict[str, Any] | None) -> None:
        if data and self.fallback:
            self.fallback.restore_state(data.get("fallback_rule"))

    @property
    def system_prompt(self) -> str:
        return (
            "你是自动迭代机器学习实验的 Agent。每次只做一个有依据的小步"
            "修改，基于验证集指标决策。只输出一个合法的 JSON 对象，不要"
            "Markdown 代码块，不要任何前缀、解释或额外文字。"
        )

    @property
    def reflect_system_prompt(self) -> str:
        return (
            "你是机器学习实验的反思助手。请基于最近一轮实验结果与实验笔记，"
            "分析该结果意味着什么、哪些方向已被证伪、下一步值得验证什么假设。"
            "只输出一个 JSON 对象，字段为 diagnosis/conclusion/hypothesis，"
            "全部是字符串，不要 Markdown 代码块或额外文字。"
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

    def _allowed_top_level_keys(self) -> set[str]:
        """LLM 可以写哪些顶层 params 字段（model/scaler/hyperparams + 任务示例字段）。"""
        keys = {"model", "scaler", "hyperparams"}
        for cand in self.cfg.experiment.rule_candidates or []:
            params = cand.get("params") or {}
            keys.update(k for k in params if isinstance(k, str))
        return keys

    def _sanitize_llm_params(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """程序级校验/清洗 LLM 返回的 params，失败时给出可读错误。"""
        if not isinstance(params, dict):
            raise ValueError(f"params 必须是对象: {params!r}")
        model = params.get("model")
        if model not in self._allowed_models():
            raise ValueError(
                f"模型不在白名单: {model!r}；可用: {self._allowed_models()}"
            )

        allowed_keys = self._allowed_top_level_keys()
        stripped = [k for k in params if k not in allowed_keys]
        cleaned = {k: v for k, v in params.items() if k in allowed_keys}
        if stripped:
            self.log.warning(
                f"LLM params 包含非白名单顶层字段 {stripped}，已忽略。"
            )

        if "scaler" in cleaned:
            scaler = cleaned["scaler"]
            if isinstance(scaler, str):
                lowered = scaler.strip().lower()
                if lowered in ("true", "1", "yes"):
                    scaler = True
                elif lowered in ("false", "0", "no"):
                    scaler = False
            if not isinstance(scaler, bool):
                raise ValueError(f"scaler 必须是 bool，收到: {scaler!r}")
            cleaned["scaler"] = scaler

        hyperparams = cleaned.get("hyperparams") or {}
        if not isinstance(hyperparams, dict):
            raise ValueError(f"hyperparams 必须是对象: {hyperparams!r}")
        self._validate_hyperparams(model, hyperparams)

        # 任务级可选字段（仅当 rule 候选示例中出现时允许）。
        if "max_rows" in cleaned:
            value = cleaned["max_rows"]
            if value is not None:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    raise ValueError(f"max_rows 必须是整数或 null: {value!r}")
                if not 0 < value <= 1_000_000:
                    raise ValueError(f"max_rows 超出允许范围: {value}")
                cleaned["max_rows"] = value
        if "pca_components" in cleaned:
            value = cleaned["pca_components"]
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"pca_components 必须是整数: {value!r}"
                )
            if not 1 <= value <= 2048:
                raise ValueError(f"pca_components 超出允许范围: {value}")
            cleaned["pca_components"] = value
        return cleaned

    @staticmethod
    def _validate_hyperparams(
        model: str, hyperparams: dict[str, Any]
    ) -> None:
        """拦截明显越界/非法的超参数，避免把机器拖死后再靠 timeout。"""
        numeric_limits: dict[str, tuple[float, float]] = {
            "alpha": (1e-9, 1e6),
            "C": (1e-9, 1e6),
            "learning_rate": (1e-6, 1.0),
        }
        integer_limits: dict[str, tuple[int, int]] = {
            "n_estimators": (1, 3000),
            "max_depth": (1, 128),
            "max_iter": (1, 20000),
            "min_samples_leaf": (1, 10_000),
            "min_samples_split": (2, 100_000),
            "n_iter_no_change": (1, 10_000),
        }
        for key, value in hyperparams.items():
            if value is None:
                continue
            if key in numeric_limits:
                low, high = numeric_limits[key]
                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ) or not math.isfinite(float(value)) or not low <= float(value) <= high:
                    raise ValueError(f"{key}={value!r} 超出允许范围")
            if key in integer_limits:
                low, high = integer_limits[key]
                if isinstance(value, bool) or not isinstance(value, int) \
                        or not low <= value <= high:
                    raise ValueError(f"{key}={value!r} 超出允许范围")
            if key == "hidden_layer_sizes":
                layers = value if isinstance(value, (list, tuple)) else [value]
                if not layers or len(layers) > 4:
                    raise ValueError(
                        f"hidden_layer_sizes 层数需在 1..4: {value!r}"
                    )
                for size in layers:
                    if isinstance(size, bool) or not isinstance(size, int) \
                            or not 1 <= size <= 2048:
                        raise ValueError(
                            f"hidden_layer_sizes 含非法值: {value!r}"
                        )
            if key == "max_features" and model == "random_forest":
                # sklearn >=1.4 不再接受 RandomForest 的 "auto"
                if isinstance(value, str) and value.lower() == "auto":
                    hyperparams[key] = "sqrt"
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and not math.isfinite(float(value)):
                raise ValueError(f"{key}={value!r} 不是有限数值")

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

        def _fmt(value: Any) -> str:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"{value:.4f}"
            return str(value)

        plan_by_step: dict[int, dict[str, Any]] = {}
        execute_by_step: dict[int, dict[str, Any]] = {}
        evaluate_by_step: dict[int, dict[str, Any]] = {}
        decision_by_step: dict[int, dict[str, Any]] = {}
        for e in state.entries:
            step = e.get("step")
            if not isinstance(step, int):
                continue
            phase = e.get("phase")
            if phase == "plan":
                plan_by_step[step] = e
            elif phase == "execute" and e.get("success") is True:
                execute_by_step[step] = e
            elif phase == "evaluate":
                evaluate_by_step[step] = e
            elif phase == "decision":
                decision_by_step[step] = e

        recent_steps = list(evaluate_by_step)[-6:]
        history: list[str] = []
        for idx, step in enumerate(recent_steps, start=1):
            e = evaluate_by_step[step]
            m = e.get("metrics") or {}
            params = (execute_by_step.get(step) or {}).get("params") or {}
            model = params.get("model", "?")
            val_txt = _fmt(m.get(target))
            test_txt = _fmt(m.get(test_key))
            decision = e.get("decision", "?")
            decision_msg = (decision_by_step.get(step) or {}).get(
                "message", ""
            )
            rationale = (plan_by_step.get(step) or {}).get(
                "rationale", ""
            )
            history.append(
                f"轮次 {idx}（step {step}）: model={model}, "
                f"{target}={val_txt}, {test_key}={test_txt}, "
                f"决策={decision}\n"
                f"  上一轮理由: {str(rationale)[:240]}\n"
                f"  决策说明: {str(decision_msg)[:240]}"
            )
        history_txt = "\n".join(history) if history else "（暂无已完成实验）"

        insights = state.events("insight")[-8:]
        insight_txt = "\n".join(
            f"- step {e.get('step')} [{e.get('generator', '?')}] "
            f"diagnosis={e.get('diagnosis', '')}；"
            f"conclusion={e.get('conclusion', '')}；"
            f"hypothesis={e.get('hypothesis', '')}"
            for e in insights
        ) if insights else "（暂无实验笔记）"

        errors = [e for e in state.entries if e.get("phase") == "error"]
        error_txt = "\n".join(
            f"- step {e.get('step')}: {e.get('message', '')} "
            f"| {str(e.get('error_tail', ''))[:240]}"
            for e in errors[-3:]
        ) if errors else "（暂无失败记录）"

        return (
            f"任务：{self.cfg.experiment.description}\n"
            f"当前步数：{len(runs) + 1}\n"
            f"已完成的实验：\n{history_txt}\n"
            f"实验笔记（含结论/待验证假设，已证伪方向不要重复）：\n"
            f"{insight_txt}\n"
            f"最近失败/错误：\n{error_txt}\n"
            f"本任务 train.py 仅支持以下模型名：{self._allowed_models()}\n"
            "params.model 必须取上述列表中的值，不要提出白名单外的新模型。\n"
            "请基于以上状态提出下一步唯一的修改方案，并以 JSON 返回："
            '{"description": str, "rationale": str, '
            '"params": {"model": "<该任务 train.py 支持的模型名>", '
            '"scaler": bool, "hyperparams": {...}}}。'
            "不要输出 JSON 以外的内容。"
        )

    def _propose_remote(self, step: int, state: RunState) -> Action:
        content = self._chat_once(
            user_text=self._build_user_prompt(state),
            system_text=self.system_prompt,
            max_tokens=self.max_tokens,
        )
        return self._content_to_action(content)

    def reflect(
        self,
        *,
        state: RunState,
        step: int,
        decision: str,
        reason: str,
        description: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        """对上一轮结果做一次轻量 LLM 反思；失败不致命，返回 None。"""
        if not self.api_key:
            return None
        user_text = self._build_reflection_prompt(
            state, step=step, decision=decision, reason=reason,
            description=description, metrics=metrics,
        )
        try:
            content = self._chat_once(
                user_text=user_text,
                system_text=self.reflect_system_prompt,
                max_tokens=min(4096, self.max_tokens),
            )
            insight = self._parse_reflection(content)
        except Exception as exc:
            self.log.warning(
                f"LLM 反思失败，使用确定性 insight 兜底: {exc}"
            )
            return None
        self.log.info(
            f"LLM 反思（step {step}）: {insight.get('conclusion', '')[:160]}"
        )
        return insight

    def _build_reflection_prompt(
        self,
        state: RunState,
        *,
        step: int,
        decision: str,
        reason: str,
        description: str,
        metrics: dict[str, Any],
    ) -> str:
        target = self.cfg.run.target_metric
        metric_txt = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, (int, float))
            else f"{k}={v}"
            for k, v in (metrics or {}).items()
            if not isinstance(v, (dict, list))
        )
        insights = state.events("insight")[-6:]
        insight_txt = "\n".join(
            f"- step {e.get('step')}: diagnosis={e.get('diagnosis', '')}；"
            f"conclusion={e.get('conclusion', '')}；"
            f"hypothesis={e.get('hypothesis', '')}"
            for e in insights
        ) if insights else "（暂无）"
        return (
            f"任务：{self.cfg.experiment.description}\n"
            f"当前 step={step}，决策={decision}\n"
            f"本轮方案描述：{description}\n"
            f"评估说明：{reason}\n"
            f"本轮指标：{metric_txt}\n"
            f"实验笔记：\n{insight_txt}\n"
            "请输出 JSON："
            '{"diagnosis": "为什么这轮会这样", '
            '"conclusion": "本轮结论/哪些方向被证伪", '
            '"hypothesis": "下一步值得验证的假设"}'
        )

    @staticmethod
    def _parse_reflection(content: str) -> dict[str, Any]:
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
            raise ValueError(f"反思输出不是合法 JSON: {content[:200]}")
        data = json.loads(text[start:end + 1])
        insight = {
            "diagnosis": str(data.get("diagnosis", "")).strip(),
            "conclusion": str(data.get("conclusion", "")).strip(),
            "hypothesis": str(data.get("hypothesis", "")).strip(),
        }
        if not all(insight.values()):
            raise ValueError(f"反思 JSON 缺少 diagnosis/conclusion/hypothesis: {data}")
        return insight

    def _chat_once(
        self,
        *,
        user_text: str,
        system_text: str,
        max_tokens: int,
    ) -> str:
        """OpenAI 兼容 / Anthropic 的统一 chat 调用。"""
        if self.provider == "anthropic":
            if self.reasoning_effort:
                self.log.warning(
                    "reasoning_effort 暂不映射到 Anthropic Messages API，已忽略"
                )
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "system": system_text,
                "messages": [{"role": "user", "content": user_text}],
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
            return content

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
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
                "模型返回空 content；"
                f"finish_reason="
                f"{body['choices'][0].get('finish_reason')!r}"
            )
        return content

    def _content_to_action(self, content: str) -> Action:
        action = self._parse_action(content)
        action.params = self._sanitize_llm_params(action.params)
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
