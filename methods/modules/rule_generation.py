"""
RuleGeneration — 训练集统计摘要 + Qwen vLLM 客户端 + 规则生成编排。

大模型只离线调用一次（A1），在训练前生成 RuleBank 并缓存。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from methods.modules.rule_bank import (
    RuleBank, Rule, RuleMetadata, validate_llm_output, sha256_file, now_iso,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 第 1 部分：训练集统计摘要
# ─────────────────────────────────────────────────────────────────────────────

NUMERICAL_STAT_FIELDS = [
    "Amount", "TimeDiff", "SenderHistCount",
    "SenderHistAmountSum", "SenderHistAmountMean", "Time",
]
CATEGORICAL_STAT_FIELDS = ["Type", "Target"]
ALL_STAT_FIELDS = NUMERICAL_STAT_FIELDS + CATEGORICAL_STAT_FIELDS


@dataclass
class DatasetStatistics:
    """仅含训练集的字段统计摘要。"""
    numerical: dict = field(default_factory=dict)
    categorical: dict = field(default_factory=dict)
    positive_rate: float = 0.0
    train_size: int = 0

    def to_dict(self) -> dict:
        return {
            "numerical": self.numerical,
            "categorical": self.categorical,
            "positive_rate": self.positive_rate,
            "train_size": self.train_size,
        }


def compute_training_statistics(
    processed: pd.DataFrame,
    train_idx: np.ndarray,
    numerical_fields: Optional[List[str]] = None,
    categorical_fields: Optional[List[str]] = None,
) -> DatasetStatistics:
    """仅从 train_idx 定位的行生成统计摘要。

    Parameters
    ----------
    processed:
        AML_gtan_processed.csv 的全量 DataFrame。
    train_idx:
        训练集行索引。
    numerical_fields:
        需要分位数统计的数值字段列表。
    categorical_fields:
        需要频率统计的分类字段列表。

    Returns
    -------
    DatasetStatistics（仅包含训练集信息）。
    """
    nf = numerical_fields or NUMERICAL_STAT_FIELDS
    cf = categorical_fields or CATEGORICAL_STAT_FIELDS

    train_df = processed.iloc[train_idx].copy()
    stats = DatasetStatistics(
        positive_rate=float(train_df["Labels"].mean()),
        train_size=len(train_df),
    )

    # 数值字段 — 按正常/风险分组
    for field in nf:
        if field not in train_df.columns:
            logger.warning("Numerical field '%s' not found in training data", field)
            continue
        all_series = train_df[field].dropna().astype(float)
        normal = train_df.loc[train_df["Labels"] == 0, field].dropna().astype(float)
        risk = train_df.loc[train_df["Labels"] == 1, field].dropna().astype(float)
        stats.numerical[field] = {
            "all": _percentile_stats(all_series),
            "normal": _percentile_stats(normal) if len(normal) else None,
            "risk": _percentile_stats(risk) if len(risk) else None,
        }

    # 分类字段
    for field in cf:
        if field not in train_df.columns:
            logger.warning("Categorical field '%s' not found in training data", field)
            continue
        stats.categorical[field] = _categorical_stats(train_df[field])

    return stats


def _percentile_stats(s: pd.Series) -> dict:
    if len(s) == 0:
        return {}
    q = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.995]
    percentiles = s.quantile(q).to_dict()
    return {
        "count": int(s.count()),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "skew": round(float(s.skew()), 4),
        **{f"p{int(p*100)}".replace(".", "_"): round(float(v), 4)
           for p, v in zip(q, percentiles.values())},
    }


def _categorical_stats(s: pd.Series) -> list:
    vc = s.value_counts()
    total = len(s)
    return [
        {"value": str(val), "count": int(cnt), "ratio": round(cnt / total, 4)}
        for val, cnt in vc.head(30).items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 第 2 部分：Qwen vLLM 客户端
# ─────────────────────────────────────────────────────────────────────────────


class QwenRuleGenerator:
    """Qwen vLLM API 客户端，用于离线生成规则。

    Parameters
    ----------
    api_url:
        vLLM 服务的完整 URL（含 /v1/completions）。
    model:
        --served-model-name 参数指定的模型名。
    timeout:
        HTTP 请求超时（秒）。
    """

    def __init__(
        self,
        api_url: str = "http://localhost:23333/v1/completions",
        model: str = "qwen",
        timeout: int = 120,
    ):
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    def generate_rules(self, prompt: str, **gen_kwargs) -> Optional[str]:
        """调用 Qwen 生成规则。

        Parameters
        ----------
        prompt:
            完整 prompt 文本。
        **gen_kwargs:
            生成参数 override（temperature, max_tokens, top_p 等）。

        Returns
        -------
        原始响应文本，或 None（调用失败）。
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": gen_kwargs.get("max_tokens", 4096),
            "temperature": gen_kwargs.get("temperature", 0.3),
            "top_p": gen_kwargs.get("top_p", 0.95),
        }
        try:
            resp = requests.post(
                self.api_url, json=payload, timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("choices", [{}])[0].get("text", "")
            return raw.strip() if raw else None
        except requests.RequestException as exc:
            logger.error("Qwen vLLM call failed: %s", exc)
            return None

    def health_check(self) -> bool:
        """检查 vLLM 服务是否存活。"""
        base = self.api_url.rstrip("/")
        health_url = base.rsplit("/v1", 1)[0] + "/health"
        try:
            resp = requests.get(health_url, timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# 第 3 部分：生成编排
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_RULES_DATA = [
    {
        "rule_id": "F001",
        "name": "fallback_high_amount",
        "category": "amount",
        "description": "金额显著高于训练集均值（硬编码 fallback）",
        "conditions": {"operator": "AND", "clauses": [
            {"field": "Amount", "operator": ">", "value": 50000.0},
        ]},
        "risk_score": 0.30,
        "base_confidence": 0.20,
        "business_explanation": "大额交易是洗钱典型特征，高于 50000 的交易需额外关注",
        "limitations": "静态阈值，未自适应数据分布",
    },
    {
        "rule_id": "F002",
        "name": "fallback_rapid_tx",
        "category": "temporal",
        "description": "交易间隔极短（硬编码 fallback）",
        "conditions": {"operator": "AND", "clauses": [
            {"field": "TimeDiff", "operator": "<", "value": 1.0},
        ]},
        "risk_score": 0.25,
        "base_confidence": 0.15,
        "business_explanation": "同一账户短时间内连续交易可能为分散交易（structuring）",
        "limitations": "静态阈值，未区分交易类型",
    },
]


def generate_rulebank(
    stats: DatasetStatistics,
    prompt_path: str,
    output_path: str,
    llm_client: QwenRuleGenerator,
    encoding_map: Optional[Dict[str, Dict[str, int]]] = None,
    data_fingerprint: str = "",
    prompt_fingerprint: str = "",
    prompt_version: str = "1.0",
    max_rules: int = 20,
    vllm_params: Optional[dict] = None,
    stats_ref_keys: Optional[set] = None,
) -> RuleBank:
    """完整生成管线：构建 prompt → 调用 LLM → 校验 → YAML 写出。

    如果 LLM 调用或校验失败会自动降级到 fallback 规则集。

    Parameters
    ----------
    stats:
        由 compute_training_statistics() 生成的训练集统计摘要。
    prompt_path:
        prompt 模板文件的路径。
    output_path:
        输出的 RuleBank YAML 路径。
    llm_client:
        Qwen vLLM 客户端实例。
    encoding_map:
        {field_name: {str_value: int_id}}，用于字符串条件预校验。
    data_fingerprint:
        训练集的 SHA-256 指纹。
    prompt_fingerprint:
        prompt 模板的 SHA-256 指纹。
    prompt_version:
        prompt 版本号。
    max_rules:
        LLM 输出规则数量上限。
    vllm_params:
        传递给 LLM 的生成参数。
    stats_ref_keys:
        合法的统计引用键集合。如果为 None，从 stats 自动推导。

    Returns
    -------
    加载了最终规则的 RuleBank 实例（已写入 output_path）。
    """
    vllm_params = vllm_params or {"temperature": 0.3, "max_tokens": 4096, "top_p": 0.95}

    # 1. 读取 prompt 模板
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # 2. 构建 prompt
    field_schema_table = _build_field_schema_table()
    allowed_display = ", ".join(sorted(ALL_STAT_FIELDS))
    prompt = prompt_template.format(
        max_rules=max_rules,
        field_schema_table=field_schema_table,
        statistics_json=json.dumps(stats.to_dict(), indent=2, ensure_ascii=False),
        positive_rate=stats.positive_rate,
        allowed_fields_display=allowed_display,
        few_shot_examples=_few_shot_examples(),
    )

    # 3. 调用 LLM
    raw_text = llm_client.generate_rules(prompt, **vllm_params)

    # 4. 校验
    ref_keys = stats_ref_keys or _build_ref_keys(stats)
    if raw_text:
        report = validate_llm_output(raw_text, ref_keys, max_rules=max_rules)
    else:
        report = {"valid": False, "errors": ["LLM returned empty response"],
                  "rules": []}

    # 5. 决定规则来源
    if report.get("valid"):
        rules = report["rules"]
        generation_log = {
            "source": "llm",
            "llm_raw_output": raw_text,
            "validation_report": {k: v for k, v in report.items() if k != "rules"},
            "generated_at": now_iso(),
        }
        logger.info("LLM generated %d rules (%d rejected)",
                    len(rules), report.get("rejected_count", 0))
    else:
        logger.warning("LLM rule generation failed: %s; using fallback",
                       report.get("errors", "unknown"))
        rules = _build_fallback_rules(ref_keys)
        generation_log = {
            "source": "fallback",
            "reason": report.get("errors", ["LLM unavailable or output invalid"]),
            "llm_raw_output": raw_text,
            "generated_at": now_iso(),
        }

    # 6. 构建 RuleBank
    bank = RuleBank(
        generated_at=now_iso(),
        prompt_version=prompt_version,
        prompt_fingerprint=prompt_fingerprint,
        data_fingerprint=data_fingerprint,
        source_model="Qwen",
        source_model_fingerprint="",
        vllm_endpoint=llm_client.api_url,
        train_size=stats.train_size,
        positive_rate=stats.positive_rate,
        rulebank_version=_next_version(output_path),
        generation_params=vllm_params,
        rules=rules,
    )

    # 7. 原子写入
    bank.save(output_path)

    # 8. 保存生成日志
    log_path = output_path.replace(".yaml", "_generation_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(generation_log, f, ensure_ascii=False, indent=2)

    return bank


def _build_field_schema_table() -> str:
    lines = [
        "| 字段名 | 类型 | 含义 |",
        "|--------|------|------|",
        "| Amount | float | 交易金额（原始，未裁剪） |",
        "| TimeDiff | float | 同一 Sender 上一笔交易间隔 |",
        "| SenderHistCount | int | Sender 历史交易笔数（不含当前笔） |",
        "| SenderHistAmountSum | float | Sender 历史累计金额（不含当前笔） |",
        "| SenderHistAmountMean | float | Sender 历史平均金额（不含当前笔） |",
        "| Time | int | 交易发生的天序号（0–199） |",
        "| Type | str | 交易类型名称（如 TRANSFER） |",
        "| Target | str | 收款方编码 |",
    ]
    return "\n".join(lines)


def _few_shot_examples() -> str:
    return """```json
{
  "rules": [
    {
      "rule_id": "R001",
      "name": "high_amount_rapid_sequence",
      "category": "velocity",
      "description": "高金额 + 极短时间间隔",
      "conditions": {
        "operator": "AND",
        "clauses": [
          {"field": "Amount", "operator": ">", "value_ref": "Amount_all_p99", "description": "金额超过训练集 99 分位"},
          {"field": "TimeDiff", "operator": "<", "value_ref": "TimeDiff_all_p5", "description": "交易间隔小于训练集 5 分位"}
        ]
      },
      "risk_score": 0.75,
      "base_confidence": 0.60,
      "business_explanation": "同一发送方短时间内高频大额交易，可能为结构化洗钱行为（structuring）",
      "limitations": "合法的大额批量支付可能误报",
      "tags": ["high_severity"]
    }
  ]
}
```"""


def _build_ref_keys(stats: DatasetStatistics) -> set:
    keys = set()
    for field, groups in stats.numerical.items():
        for group_name, values in (("all", groups),):
            if not isinstance(values, dict):
                continue
            for k in values:
                if k in ("count", "mean", "std", "min", "max", "skew"):
                    keys.add(f"{field}_{group_name}_{k}")
                elif k.startswith("p"):
                    keys.add(f"{field}_{group_name}_{k}")
    return keys


def _build_fallback_rules(ref_keys: set) -> List[Rule]:
    rules: List[Rule] = []
    for item in FALLBACK_RULES_DATA:
        rule_item = {**item}
        cond = rule_item.pop("conditions", {})
        rules.append(Rule(
            rule_id=rule_item["rule_id"],
            name=rule_item["name"],
            category=rule_item.get("category", ""),
            active=True,
            description=rule_item.get("description", ""),
            conditions_operator=cond.get("operator", "AND"),
            clauses=cond.get("clauses", []),
            risk_score=rule_item["risk_score"],
            base_confidence=rule_item["base_confidence"],
            metadata=RuleMetadata(
                business_explanation=rule_item.get("business_explanation", ""),
                limitations=rule_item.get("limitations", ""),
                generated_by="fallback",
                prompt_version="",
                review_status="approved",
            ),
        ))
    return rules


def _next_version(output_path: str) -> str:
    """根据已存在的文件推断下一个版本号。"""
    base = os.path.dirname(output_path)
    name = os.path.basename(output_path)
    prefix = name.split("_v")[0] if "_v" in name else name.replace(".yaml", "")
    existing = [f for f in os.listdir(base) if f.startswith(prefix)] if os.path.isdir(base) else []
    max_n = 0
    for f in existing:
        parts = f.split("_v")
        if len(parts) > 1:
            try:
                n = int(parts[-1].replace(".yaml", "").split("_")[0])
                max_n = max(max_n, n)
            except ValueError:
                pass
    return f"v{max_n + 1}.0.0"


def load_or_generate_rulebank(
    rulebank_path: str,
    prompt_path: str,
    processed: pd.DataFrame,
    train_idx: np.ndarray,
    llm_client: Optional[QwenRuleGenerator] = None,
    encoding_map: Optional[Dict[str, Dict[str, int]]] = None,
    data_fingerprint: str = "",
    force: bool = False,
    **gen_kwargs,
) -> RuleBank:
    """加载已有 RuleBank 或生成新的。

    如果 ``force=True`` 或文件不存在，调用 ``generate_rulebank()``。
    否则直接加载已有 YAML。
    """
    if not force and os.path.exists(rulebank_path):
        logger.info("Loading existing RuleBank from %s", rulebank_path)
        return RuleBank.load(rulebank_path)

    logger.info("Generating RuleBank → %s", rulebank_path)
    stats = compute_training_statistics(processed, train_idx)
    if llm_client is None:
        llm_client = QwenRuleGenerator()
    return generate_rulebank(
        stats=stats,
        prompt_path=prompt_path,
        output_path=rulebank_path,
        llm_client=llm_client,
        encoding_map=encoding_map,
        data_fingerprint=data_fingerprint,
        **gen_kwargs,
    )
