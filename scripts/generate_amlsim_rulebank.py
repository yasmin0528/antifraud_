"""AMLSIM RuleBank 生成 — 加载 AMLSIM 训练数据，调用 LLM 生成规则，保存 YAML。

Usage:
    # 先启动 vLLM 服务，然后：
    python scripts/generate_amlsim_rulebank.py

    # 指定输出路径
    python scripts/generate_amlsim_rulebank.py --output config/rulebank/amlsim_rulebank_v1.yaml

    # 强制重新生成（即使 YAML 已存在）
    python scripts/generate_amlsim_rulebank.py --force
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import numpy as np
import pandas as pd

from datasets.amlsim import load_amlsim_data
from methods.modules.rule_generation import (
    DatasetStatistics, QwenRuleGenerator,
)
from methods.modules.rule_bank import (
    RuleBank, Rule, RuleMetadata, validate_llm_output,
    sha256_file, now_iso,
)


# ── 统计计算 ──────────────────────────────────────────────────────────


def _percentile_stats(s: pd.Series) -> dict:
    if len(s) == 0:
        return {}
    q = [0.25, 0.50, 0.75, 0.95, 0.99]
    percentiles = s.quantile(q).to_dict()
    return {
        "count": int(s.count()),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        **{f"p{int(p*100)}": round(float(v), 4)
           for p, v in zip(q, percentiles.items())},
    }


def _categorical_stats(s: pd.Series) -> list:
    vc = s.value_counts()
    total = len(s)
    return [
        {"value": str(val), "count": int(cnt), "ratio": round(cnt / total, 4)}
        for val, cnt in vc.head(10).items()
    ]


def compute_stats(processed: pd.DataFrame, train_idx: np.ndarray) -> dict:
    """返回 {numerical: ..., categorical: ..., positive_rate: ..., train_size: ...}"""
    train_df = processed.iloc[train_idx]
    stats = {
        "numerical": {},
        "categorical": {},
        "positive_rate": float(train_df["Labels"].mean()),
        "train_size": len(train_df),
    }
    # 数值字段
    for field, col in [
        ("Amount", "AmountPaid"),
        ("AmountReceived", "AmountReceived"),
        ("LogAmountPaid", "LogAmountPaid"),
        ("LogAmountReceived", "LogAmountReceived"),
        ("TimeDiff", "TimeDiff"),
        ("Time", "Time"),
        ("CrossBank", "CrossBank"),
        ("TimeHour", "TimeHour"),
        ("TimeDayOfWeek", "TimeDayOfWeek"),
    ]:
        if col not in train_df.columns:
            continue
        s = train_df[col].dropna().astype(float)
        normal = train_df.loc[train_df["Labels"] == 0, col].dropna().astype(float)
        risk = train_df.loc[train_df["Labels"] == 1, col].dropna().astype(float)
        stats["numerical"][field] = {
            "all": _percentile_stats(s),
            "normal": _percentile_stats(normal) if len(normal) else None,
            "risk": _percentile_stats(risk) if len(risk) else None,
        }
    # 分类字段
    for field in ["PaymentFormat", "CurrencyPaid", "CurrencyReceived", "FromBank", "ToBank"]:
        if field not in train_df.columns:
            continue
        stats["categorical"][field] = _categorical_stats(train_df[field])
    return stats


# ── Prompt 构建 ───────────────────────────────────────────────────────

FIELD_SCHEMA_TABLE = """| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Amount | numerical | 交易金额 | 500.0 |
| AmountReceived | numerical | 收款金额 | 1000.0 |
| LogAmountPaid | numerical | 交易金额的对数变换 | 6.21 |
| LogAmountReceived | numerical | 收款金额的对数变换 | 6.91 |
| TimeDiff | numerical | 同一账户相邻交易的时间间隔（秒） | 0.0 |
| Time | numerical | 交易时间戳（Unix 秒） | 1600000000 |
| CrossBank | numerical | 是否跨行（0=同行, 1=跨行） | 1.0 |
| TimeHour | numerical | 交易小时（0-23） | 14 |
| TimeDayOfWeek | numerical | 星期几（0=周一, 6=周日） | 3 |
| PaymentFormat | categorical | 支付方式（编码值） | 0 |
| CurrencyPaid | categorical | 支付币种（编码值） | 0 |
| CurrencyReceived | categorical | 收款币种（编码值） | 0 |
| FromBank | categorical | 发送方银行（编码值） | 0 |
| ToBank | categorical | 收款方银行（编码值） | 0 |"""

ALLOWED_FIELDS_DISPLAY = "Amount, AmountReceived, LogAmountPaid, LogAmountReceived, TimeDiff, Time, CrossBank, TimeHour, TimeDayOfWeek, PaymentFormat, CurrencyPaid, CurrencyReceived, FromBank, ToBank"


def build_prompt(stats: dict, max_rules: int) -> str:
    stats_json = json.dumps(stats, indent=2, ensure_ascii=False)
    return f"""# Role
你是 AML（反洗钱）风控规则专家。任务是分析 AMLSIM 仿真交易数据，生成可区分洗钱与正常交易的结构化规则。

# Task
生成最多 {max_rules} 条结构化候选规则。

**核心方法：**
- 比较每个字段在 normal 与 risk 组间的分布差异（p50, p75, p95, mean）
- 优先使用 **2-3 条件 AND 组合**，捕捉多维度叠加异常

# Business Context
- Structuring：拆分为多笔小额 → TimeDiff 短
- Rapid movement：短时间内多渠道转移 → CrossBank=1 + TimeDiff 短
- 深夜交易 → TimeHour 异常（如 0-5 点）
- 跨行交易 → CrossBank=1

# Field Schema
{FIELD_SCHEMA_TABLE}

# Training Set Statistics（重点比较 normal 与 risk 组的差异）
{stats_json}

# Positive Rate: {stats['positive_rate']:.6f}

# Constraints
1. 可用字段：{ALLOWED_FIELDS_DISPLAY}
2. 操作符：>, <, >=, <=, ==, between, in, not_in
3. 阈值可引用统计键如 "Amount_all_p95"、"TimeDiff_risk_p50"，或标量常数
4. 禁止字段：Labels, TX_ID, FromAccount, ToAccount
5. 每条规则至少 2 个条件子句
6. 输出为严格 JSON，字段名大小写敏感
7. Rule ID 从 R001 开始编号

# Output Format
```json
{{"rules": [
  {{"rule_id": "R001", "name": "example_rule", "category": "amount",
    "description": "...",
    "conditions": {{"operator": "AND", "clauses": [
      {{"field": "TimeDiff", "operator": "<", "value_ref": "TimeDiff_normal_p25", "description": "..."}},
      {{"field": "CrossBank", "operator": "==", "value": 1.0, "description": "跨行"}}
    ]}},
    "risk_score": 0.65, "base_confidence": 0.55,
    "business_explanation": "...", "limitations": "..."
  }}
]}}
```"""


# ── 主流程 ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate AMLSIM RuleBank from Qwen LLM")
    parser.add_argument("--output", default="config/rulebank/amlsim_rulebank_v1.yaml")
    parser.add_argument("--api-url", default="http://localhost:23333/v1/chat/completions")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--max-rules", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing YAML")
    cli_args = parser.parse_args()

    output_path = os.path.join(_PROJECT, cli_args.output)
    if os.path.exists(output_path) and not cli_args.force:
        print(f"[skip] {output_path} exists, use --force to regenerate")
        return

    # 1. 加载 AMLSIM 数据
    print("[1/4] Loading AMLSIM training data…")
    args = {
        "dataset": "amlsim", "amlsim_variant": "HI-Small",
        "data_path": "data/AMLSIM", "method": "rgtan_mpfc_vta",
        "seed": 2023, "split_mode": "temporal",
        "amlsim_train_days": 14, "amlsim_val_days": 2,
        "device": "cpu", "nei_att_heads": {"amlsim": 1}, "ca1_k": 10,
    }
    feat_df, labels, train_idx, test_idx, g, cat_features, neigh_features = \
        load_amlsim_data(args)
    processed = pd.read_csv(args["_amlsim_processed_path"])
    data_fingerprint = sha256_file(args["_amlsim_processed_path"])
    print(f"  total={len(processed)}, train={len(train_idx)}, "
          f"positive_rate={labels.iloc[train_idx].mean():.6f}")

    # 2. 统计
    print("[2/4] Computing statistics…")
    stats = compute_stats(processed, train_idx)
    print(f"  {len(stats['numerical'])} numerical + "
          f"{len(stats['categorical'])} categorical fields")

    # 3. 调用 LLM
    print(f"[3/4] Calling Qwen LLM ({cli_args.api_url})…")
    client = QwenRuleGenerator(
        api_url=cli_args.api_url, model=cli_args.model)
    prompt = build_prompt(stats, cli_args.max_rules)
    raw = client.generate_rules(prompt,
                                temperature=cli_args.temperature,
                                max_tokens=cli_args.max_tokens,
                                top_p=cli_args.top_p)
    if not raw:
        print("[FAIL] LLM returned empty response")
        sys.exit(1)

    # 构建 ref_keys 供校验用
    ref_keys = set()
    for field in stats.get("numerical", {}):
        for group in ("all", "normal", "risk"):
            grp = stats["numerical"][field].get(group) or {}
            for k in grp:
                ref_keys.add(f"{field}_{group}_{k}")
    ref_keys.discard("count"); ref_keys.discard("std")

    report = validate_llm_output(raw, ref_keys,
                                 max_rules=cli_args.max_rules)
    if not report.get("valid"):
        print(f"[FAIL] Validation: {report.get('errors', ['unknown'])}")
        sys.exit(1)

    rules_raw = report["rules"]
    print(f"  LLM generated {len(rules_raw)} rules")

    # 4. 构建 RuleBank 对象
    print("[4/4] Saving RuleBank…")
    rule_objects = []
    for r in rules_raw:
        clauses = r.get("conditions", {}).get("clauses", [])
        cond_op = r.get("conditions", {}).get("operator", "AND")
        meta_raw = r.pop("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
        rule_objects.append(Rule(
            rule_id=r.get("rule_id", ""),
            name=r.get("name", ""),
            category=r.get("category", ""),
            active=r.get("active", True),
            description=r.get("description", ""),
            tags=r.get("tags", []),
            conditions_operator=cond_op,
            clauses=clauses,
            risk_score=float(r.get("risk_score", 0.5)),
            base_confidence=float(r.get("base_confidence", 0.5)),
            metadata=RuleMetadata(**{
                k: v for k, v in meta_raw.items()
                if k in RuleMetadata.__dataclass_fields__
            }),
        ))

    bank = RuleBank(
        generated_at=now_iso(),
        prompt_version="1.0",
        prompt_fingerprint="amlsim_v1",
        data_fingerprint=data_fingerprint,
        source_model=cli_args.model,
        train_size=stats["train_size"],
        positive_rate=stats["positive_rate"],
        rulebank_version="v1.0.0",
        rules=rule_objects,
    )
    bank.save(output_path)
    print(f"  Saved → {output_path}")
    print(f"  Rules: {len(bank.active_rules())} active / {len(bank.rules)} total")


if __name__ == "__main__":
    main()
