"""
RuleBank — 规则数据类、YAML 序列化、场白名单、输出校验管线。

单一模块覆盖：
1. 数据结构：Rule, RuleBank, RuleMetadata
2. ALLOWED_FIELDS / ALLOWED_OPERATORS / FORBIDDEN_FIELDS
3. LLM 原始输出 → JSON 解析 → Schema 校验 → 场白名单 → 去重
4. YAML 序列化 / 反序列化（原子写入）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 场白名单（与审计结论一致）
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_FIELDS = frozenset({
    "Amount", "TimeDiff",
    "SenderHistCount", "SenderHistAmountSum", "SenderHistAmountMean",
    "Time", "Type", "Target",
})

ALLOWED_OPERATORS = frozenset({
    ">", "<", ">=", "<=", "==", "between", "in", "not_in",
})

FORBIDDEN_FIELDS = frozenset({
    "IS_FRAUD", "Labels", "AlertID", "TX_ID",
    "Source", "SenderPrevFraudCount",
})

RULE_CATEGORIES = frozenset({
    "velocity", "amount", "frequency", "pattern",
    "network", "temporal", "behavioral", "composite",
})

# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RuleMetadata:
    """单条规则的可变元数据（审核状态、命中统计等）。"""
    business_explanation: str = ""
    limitations: str = ""
    generated_by: str = ""
    prompt_version: str = ""
    review_status: str = "pending"       # pending | approved | rejected | disabled
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    reviewed_at: Optional[str] = None
    hit_count_train: Optional[int] = None
    precision_train: Optional[float] = None


@dataclass
class Rule:
    """一条结构化规则。"""
    rule_id: str                    # R001, R002 …
    name: str                       # snake_case
    category: str
    active: bool = True
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # 条件
    conditions_operator: str = "AND"     # AND | OR
    clauses: List[dict] = field(default_factory=list)

    # 评分
    risk_score: float = 0.5
    base_confidence: float = 0.5

    # 元数据
    metadata: RuleMetadata = field(default_factory=RuleMetadata)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metadata"] = asdict(self.metadata)
        return d


@dataclass
class RuleBank:
    """版本化规则库。"""
    format_version: str = "1.0"
    generated_at: str = ""
    prompt_version: str = "1.0"
    prompt_fingerprint: str = ""
    data_fingerprint: str = ""
    source_model: str = "Qwen"
    source_model_fingerprint: str = ""
    vllm_endpoint: str = ""
    train_size: int = 0
    positive_rate: float = 0.0
    rulebank_version: str = "v1.0.0"
    generation_params: dict = field(default_factory=lambda: {
        "temperature": 0.3, "max_tokens": 4096, "top_p": 0.95,
    })
    rules: List[Rule] = field(default_factory=list)

    # ── 加载 / 保存 ─────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str) -> "RuleBank":
        """从 YAML 文件加载 RuleBank。"""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    def save(self, path: str) -> str:
        """原子写入 YAML。返回文件的 SHA-256。"""
        import yaml
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=os.path.basename(path) + "_", suffix=".yaml",
            dir=os.path.dirname(path) or ".",
        )
        os.close(fd)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._to_dict(), f, allow_unicode=True,
                               sort_keys=False, default_flow_style=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return sha256_file(path)

    # ── 序列化核心 ──────────────────────────────────────────────────────

    @classmethod
    def _from_dict(cls, raw: dict) -> "RuleBank":
        rules_raw = raw.pop("rules", [])
        bank = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        for r in rules_raw:
            meta_raw = r.pop("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
            clauses = []
            cond = r.pop("conditions", {})
            cond_op = cond.get("operator", "AND")
            if isinstance(cond.get("clauses"), list):
                clauses = cond["clauses"]

            bank.rules.append(Rule(
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
        return bank

    def _to_dict(self) -> dict:
        d = asdict(self)
        d["rules"] = []
        for r in self.rules:
            rd = r.to_dict()
            rd["conditions"] = {
                "operator": rd.pop("conditions_operator", "AND"),
                "clauses": rd.pop("clauses", []),
            }
            d["rules"].append(rd)
        return d

    # ── 便捷方法 ────────────────────────────────────────────────────────

    def active_rules(self) -> List[Rule]:
        return [r for r in self.rules if r.active]

    def get(self, rule_id: str) -> Optional[Rule]:
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 校验管线
# ─────────────────────────────────────────────────────────────────────────────


def _extract_json(raw_text: str) -> str:
    """Extract JSON from LLM response that may contain markdown code fences."""
    # Remove leading/trailing whitespace
    text = raw_text.strip()
    # Try finding JSON inside ```json ... ``` blocks
    for marker in ('```json', '```JSON', '```'):
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) == 2:
                candidate = parts[1]
                # Find the closing ```
                end = candidate.rfind('```')
                if end != -1:
                    candidate = candidate[:end]
                candidate = candidate.strip()
                # Quick validation: starts with { or [
                if candidate.startswith('{') or candidate.startswith('['):
                    return candidate
    return text


def validate_llm_output(
    raw_text: str,
    stats_ref_keys: set,
    *,
    max_rules: int = 25,
    max_clauses: int = 8,
) -> dict:
    """完整校验管线：LLM raw text → 清洗后的 Rule 列表。

    返回 dict:
      - valid: bool
      - rules: list[Rule] （通过校验的规则）
      - rejected_count: int
      - warnings: list[str]
      - errors: list[str]
    """
    report: dict = {"valid": False, "rules": [], "rejected_count": 0,
                    "warnings": [], "errors": []}

    # Step 1 – JSON 解析（先提取 markdown 代码块中的 JSON）
    json_candidate = _extract_json(raw_text)
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError as exc:
        report["errors"].append(f"JSON parse failed: {exc}")
        return report

    if not isinstance(parsed, dict) or "rules" not in parsed:
        report["errors"].append("Top-level object must contain a 'rules' array")
        return report

    rules_raw = parsed["rules"]
    if not isinstance(rules_raw, list):
        report["errors"].append("'rules' must be a JSON array")
        return report

    if len(rules_raw) > max_rules:
        report["warnings"].append(
            f"Truncating {len(rules_raw)} → {max_rules} rules"
        )
        rules_raw = rules_raw[:max_rules]

    # Step 2 – 逐条校验
    accepted: List[Rule] = []
    for idx, item in enumerate(rules_raw):
        rule, errs = _validate_rule_item(item, stats_ref_keys, max_clauses)
        if rule is not None:
            accepted.append(rule)
        else:
            report["rejected_count"] += 1
            for e in errs:
                logger.warning("Rule[%d] rejected: %s", idx, e)

    # Step 3 – 去重
    seen: set = set()
    unique: List[Rule] = []
    for r in accepted:
        sig = json.dumps({"op": r.conditions_operator, "clauses": r.clauses},
                         sort_keys=True)
        if sig in seen:
            report["rejected_count"] += 1
            logger.warning("Duplicate rule skipped: %s", r.name)
            continue
        seen.add(sig)
        unique.append(r)

    report["valid"] = (not report["errors"]) and len(unique) > 0
    report["rules"] = unique
    return report


def _validate_rule_item(item: Any, stats_ref_keys: set,
                        max_clauses: int) -> Tuple[Optional[Rule], List[str]]:
    """校验单条规则。返回 (Rule, []) 或 (None, [errors])。"""
    errs: List[str] = []

    if not isinstance(item, dict):
        return None, ["Rule is not a JSON object"]

    rid = str(item.get("rule_id", ""))
    if not re.match(r"^R\d{3,}$", rid):
        errs.append(f"rule_id '{rid}' must match R<digits≥3>")

    name = str(item.get("name", ""))
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{2,63}$", name):
        errs.append(f"name '{name}' must be 3–64 alphanumeric/underscore chars")

    cat = str(item.get("category", ""))
    if cat not in RULE_CATEGORIES:
        errs.append(f"category '{cat}' not in {sorted(RULE_CATEGORIES)}")

    # 风险 / 置信度
    risk = _clamp(item.get("risk_score"), 0.0, 1.0, "risk_score", errs)
    conf = _clamp(item.get("base_confidence"), 0.0, 1.0, "base_confidence", errs)

    # 条件
    cond = item.get("conditions", {})
    cond_op = cond.get("operator", "AND") if isinstance(cond, dict) else "AND"
    if cond_op not in ("AND", "OR"):
        errs.append(f"conditions.operator must be AND or OR, got {cond_op}")

    clauses_raw = cond.get("clauses", []) if isinstance(cond, dict) else []
    if not isinstance(clauses_raw, list) or len(clauses_raw) < 1:
        errs.append("conditions.clauses must be a non-empty array")

    clauses = clauses_raw[:max_clauses]
    for ci, c in enumerate(clauses):
        _validate_clause(c, ci, stats_ref_keys, errs)

    # description / business_explanation
    desc = str(item.get("description", ""))
    explanation = str(item.get("business_explanation", ""))
    limitations = str(item.get("limitations", ""))

    active = bool(item.get("active", True))
    tags = item.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    if errs:
        return None, errs

    meta = RuleMetadata(
        business_explanation=explanation,
        limitations=limitations,
        generated_by="Qwen",
        prompt_version="",
        review_status="pending",
    )
    return Rule(
        rule_id=rid, name=name, category=cat,
        active=active, description=desc, tags=tags,
        conditions_operator=cond_op, clauses=clauses,
        risk_score=risk, base_confidence=conf,
        metadata=meta,
    ), []


def _validate_clause(c: Any, ci: int, stats_ref_keys: set,
                     errs: List[str]) -> None:
    if not isinstance(c, dict):
        errs.append(f"clause[{ci}] is not a JSON object")
        return
    field = c.get("field")
    if field not in ALLOWED_FIELDS:
        errs.append(f"clause[{ci}].field '{field}' not allowed")
    if field in FORBIDDEN_FIELDS:
        errs.append(f"clause[{ci}].field '{field}' is forbidden")
    op = c.get("operator")
    if op not in ALLOWED_OPERATORS:
        errs.append(f"clause[{ci}].op '{op}' not allowed")
    if c.get("value") is None and c.get("value_ref") is None:
        errs.append(f"clause[{ci}] needs 'value' or 'value_ref'")
    vref = c.get("value_ref")
    if vref is not None and vref not in stats_ref_keys:
        errs.append(f"clause[{ci}].value_ref '{vref}' unknown")


def _clamp(val, lo, hi, label, errs) -> float:
    try:
        v = float(val)
    except (TypeError, ValueError):
        errs.append(f"{label}: non-numeric {val}")
        return lo
    if v < lo or v > hi:
        logger.warning("%s %.4f clamped to [%s, %s]", label, v, lo, hi)
        return max(lo, min(hi, v))
    return v


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
