"""Tests for RuleBank, RuleEngine, and RuleGeneration modules."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from methods.modules.rule_bank import (
    RuleBank, Rule, RuleMetadata, validate_llm_output, sha256_file,
    ALLOWED_FIELDS, FORBIDDEN_FIELDS, ALLOWED_OPERATORS,
)
from methods.modules.rule_engine import RuleEngine, AggregationStrategy
from methods.modules.rule_generation import (
    compute_training_statistics,
    DatasetStatistics,
    NUMERICAL_STAT_FIELDS,
    CATEGORICAL_STAT_FIELDS,
    ALL_STAT_FIELDS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — RuleBank / validation
# ═══════════════════════════════════════════════════════════════════════════

def test_allowed_fields_are_audited():
    """ALLOWED_FIELDS 必须与已审计的 F1 字段完全一致。"""
    expected = frozenset({
        "Amount", "TimeDiff",
        "SenderHistCount", "SenderHistAmountSum", "SenderHistAmountMean",
        "Time", "Type", "Target",
    })
    assert ALLOWED_FIELDS == expected


def test_forbidden_fields_do_not_overlap_allowed():
    assert ALLOWED_FIELDS.isdisjoint(FORBIDDEN_FIELDS)


def test_validate_llm_output_valid_json():
    stats_ref = {"Amount_all_p99", "TimeDiff_all_p5", "SenderHistCount_all_p50"}
    raw = json.dumps({
        "rules": [{
            "rule_id": "R001",
            "name": "high_amount",
            "category": "amount",
            "description": "test",
            "conditions": {
                "operator": "AND",
                "clauses": [
                    {"field": "Amount", "operator": ">", "value_ref": "Amount_all_p99",
                     "value": 50000.0, "description": "high"},
                ]
            },
            "risk_score": 0.75,
            "base_confidence": 0.6,
            "business_explanation": "Large transaction risk",
        }]
    })
    report = validate_llm_output(raw, stats_ref)
    assert report["valid"]
    assert len(report["rules"]) == 1
    assert report["rules"][0].rule_id == "R001"


def test_validate_llm_output_rejects_forbidden_field():
    stats_ref = set()
    raw = json.dumps({
        "rules": [{
            "rule_id": "R001",
            "name": "bad_rule",
            "category": "amount",
            "description": "leak",
            "conditions": {"operator": "AND", "clauses": [
                {"field": "IS_FRAUD", "operator": "==", "value": 1},
            ]},
            "risk_score": 0.5,
            "base_confidence": 0.5,
            "business_explanation": "leak",
        }]
    })
    report = validate_llm_output(raw, stats_ref)
    assert not report["valid"]
    assert report["rejected_count"] == 1


def test_validate_llm_output_rejects_unknown_field():
    stats_ref = set()
    raw = json.dumps({
        "rules": [{
            "rule_id": "R001",
            "name": "bad_field",
            "category": "amount",
            "description": "unknown",
            "conditions": {"operator": "AND", "clauses": [
                {"field": "UNKNOWN_FIELD", "operator": ">", "value": 100},
            ]},
            "risk_score": 0.5,
            "base_confidence": 0.5,
            "business_explanation": "test",
        }]
    })
    report = validate_llm_output(raw, stats_ref)
    assert not report["valid"]
    assert report["rejected_count"] == 1


def test_validate_llm_output_clamps_risk_score():
    stats_ref = set()
    raw = json.dumps({
        "rules": [{
            "rule_id": "R001", "name": "clamp_test",
            "category": "amount", "description": "clamp",
            "conditions": {"operator": "AND", "clauses": [
                {"field": "Amount", "operator": ">", "value": 100},
            ]},
            "risk_score": 1.5,
            "base_confidence": -0.1,
            "business_explanation": "clamp",
        }]
    })
    report = validate_llm_output(raw, stats_ref)
    assert report["valid"]
    assert report["rules"][0].risk_score == 1.0
    assert report["rules"][0].base_confidence == 0.0


def test_validate_llm_output_dedup():
    stats_ref = {"Amount_all_p99"}
    rule = {
        "rule_id": "R001", "name": "duplicate",
        "category": "amount", "description": "dup",
        "conditions": {"operator": "AND", "clauses": [
            {"field": "Amount", "operator": ">", "value_ref": "Amount_all_p99",
             "value": 50000, "description": "h"},
        ]},
        "risk_score": 0.5, "base_confidence": 0.5,
        "business_explanation": "dup",
    }
    raw = json.dumps({"rules": [rule, dict(rule, rule_id="R002")]})
    report = validate_llm_output(raw, stats_ref)
    assert report["valid"]
    assert len(report["rules"]) == 1  # dedup
    assert report["rejected_count"] == 1


def test_rulebank_yaml_round_trip():
    bank = RuleBank(
        rulebank_version="v1.0.0",
        train_size=1000,
        positive_rate=0.05,
        rules=[Rule(rule_id="R001", name="test_rule", category="amount",
                     conditions_operator="AND",
                     clauses=[{"field": "Amount", "operator": ">", "value": 100}],
                     risk_score=0.5, base_confidence=0.5,
                     metadata=RuleMetadata(business_explanation="test"))],
    )
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        tmppath = f.name
    try:
        bank.save(tmppath)
        loaded = RuleBank.load(tmppath)
        assert loaded.rulebank_version == "v1.0.0"
        assert loaded.train_size == 1000
        assert len(loaded.rules) == 1
        assert loaded.rules[0].rule_id == "R001"
        assert loaded.rules[0].clauses[0]["field"] == "Amount"
    finally:
        os.unlink(tmppath)


def test_rulebank_active_rules():
    bank = RuleBank(rules=[
        Rule(rule_id="R001", name="active_rule", category="amount",
             active=True, risk_score=0.5, base_confidence=0.5),
        Rule(rule_id="R002", name="inactive_rule", category="amount",
             active=False, risk_score=0.5, base_confidence=0.5),
    ])
    assert len(bank.active_rules()) == 1
    assert bank.active_rules()[0].rule_id == "R001"


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — RuleEngine
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_rulebank():
    return RuleBank(rules=[
        Rule(rule_id="R001", name="high_amount", category="amount",
             conditions_operator="AND",
             clauses=[{"field": "Amount", "operator": ">",
                       "value": 100, "description": "gt 100"}],
             risk_score=0.8, base_confidence=0.7),
        Rule(rule_id="R002", name="low_time_diff", category="temporal",
             conditions_operator="AND",
             clauses=[{"field": "TimeDiff", "operator": "<",
                       "value": 5, "description": "lt 5"}],
             risk_score=0.4, base_confidence=0.3),
    ])


def test_rule_engine_evaluate_shape(sample_rulebank):
    engine = RuleEngine(sample_rulebank, device=torch.device("cpu"))
    batch = {
        "Amount": torch.tensor([[50.0], [150.0], [200.0]]),
        "TimeDiff": torch.tensor([[1.0], [10.0], [3.0]]),
    }
    score, conf = engine.evaluate(batch)
    assert score.shape == (3, 1)
    assert conf.shape == (3, 1)
    assert (score >= 0).all() and (score <= 1).all()
    assert (conf >= 0).all() and (conf <= 1).all()


def test_rule_engine_no_hit(sample_rulebank):
    engine = RuleEngine(sample_rulebank, device=torch.device("cpu"))
    batch = {
        "Amount": torch.tensor([[10.0]]),
        "TimeDiff": torch.tensor([[100.0]]),
    }
    score, conf = engine.evaluate(batch)
    assert score.item() == 0.0
    assert conf.item() == 0.0


def test_rule_engine_single_hit(sample_rulebank):
    engine = RuleEngine(sample_rulebank, device=torch.device("cpu"))
    batch = {
        "Amount": torch.tensor([[200.0]]),   # R001 hits
        "TimeDiff": torch.tensor([[100.0]]),  # R002 doesn't
    }
    score, conf = engine.evaluate(batch)
    assert score.item() > 0.0
    assert conf.item() > 0.0


def test_rule_engine_empty_bank():
    bank = RuleBank(rules=[])
    engine = RuleEngine(bank, device=torch.device("cpu"))
    batch = {"Amount": torch.randn(5, 1)}
    score, conf = engine.evaluate(batch)
    assert (score == 0).all()
    assert (conf == 0).all()


def test_rule_engine_string_field():
    bank = RuleBank(rules=[
        Rule(rule_id="R001", name="type_test", category="pattern",
             conditions_operator="AND",
             clauses=[{"field": "Type", "operator": "==",
                       "value": "TRANSFER"}],
             risk_score=0.6, base_confidence=0.5),
    ])
    emap = {"Type": {"TRANSFER": 0, "CASH_OUT": 1}}
    engine = RuleEngine(bank, encoding_map=emap, device=torch.device("cpu"))
    batch = {"Type": torch.tensor([[0], [1], [0]])}
    score, _ = engine.evaluate(batch)
    assert (score[0] > 0) and (score[1] == 0) and (score[2] > 0)


def test_rule_engine_aggregation_max(sample_rulebank):
    engine = RuleEngine(sample_rulebank,
                         aggregation=AggregationStrategy.MAX,
                         device=torch.device("cpu"))
    batch = {
        "Amount": torch.tensor([[200.0]]),   # R001 hits (0.8)
        "TimeDiff": torch.tensor([[1.0]]),   # R002 hits (0.4)
    }
    score, _ = engine.evaluate(batch)
    assert pytest.approx(score.item(), abs=1e-5) == 0.8


def test_rule_engine_aggregation_noisy_or(sample_rulebank):
    engine = RuleEngine(sample_rulebank,
                         aggregation="noisy_or",
                         device=torch.device("cpu"))
    batch = {
        "Amount": torch.tensor([[200.0]]),    # 0.8
        "TimeDiff": torch.tensor([[1.0]]),    # 0.4
    }
    score, _ = engine.evaluate(batch)
    expected = 1.0 - (1.0 - 0.8) * (1.0 - 0.4)
    assert pytest.approx(score.item(), abs=1e-5) == expected


def test_rule_engine_missing_field():
    bank = RuleBank(rules=[
        Rule(rule_id="R001", name="missing_test", category="amount",
             conditions_operator="AND",
             clauses=[{"field": "Amount", "operator": ">", "value": 100}],
             risk_score=0.5, base_confidence=0.5),
    ])
    engine = RuleEngine(bank, device=torch.device("cpu"))
    score, _ = engine.evaluate({})  # empty batch → no field
    assert score.item() == 0.0


def test_rule_engine_and_or_semantics():
    bank = RuleBank(rules=[
        # AND
        Rule(rule_id="R001", name="and_rule", category="composite",
             conditions_operator="AND",
             clauses=[{"field": "Amount", "operator": ">", "value": 100},
                      {"field": "TimeDiff", "operator": "<", "value": 10}],
             risk_score=1.0, base_confidence=1.0),
        # OR
        Rule(rule_id="R002", name="or_rule", category="composite",
             conditions_operator="OR",
             clauses=[{"field": "Amount", "operator": ">", "value": 1000},
                      {"field": "TimeDiff", "operator": "<", "value": 1}],
             risk_score=1.0, base_confidence=1.0),
    ])
    engine = RuleEngine(bank, device=torch.device("cpu"))
    # AND: Amount > 100 AND TimeDiff < 10 → True
    batch1 = {"Amount": torch.tensor([[200.0]]), "TimeDiff": torch.tensor([[5.0]])}
    s1, _ = engine.evaluate(batch1)
    assert s1[0, 0] == 1.0

    # AND: Amount > 100 but TimeDiff >= 10 → False
    batch2 = {"Amount": torch.tensor([[200.0]]), "TimeDiff": torch.tensor([[20.0]])}
    s2, _ = engine.evaluate(batch2)
    assert s2[0, 0] == 0.0

    # OR: neither Amount > 1000 nor TimeDiff < 1 → False
    batch3 = {"Amount": torch.tensor([[500.0]]), "TimeDiff": torch.tensor([[5.0]])}
    s3, _ = engine.evaluate(batch3)
    assert s3[0, 0] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — Statistics computation
# ═══════════════════════════════════════════════════════════════════════════

def _sample_processed():
    return pd.DataFrame({
        "Amount": [10.0, 20.0, 30.0, 100.0, 200.0, 500.0],
        "TimeDiff": [1.0, 2.0, 3.0, 10.0, 20.0, 60.0],
        "SenderHistCount": [0, 1, 2, 0, 1, 5],
        "SenderHistAmountSum": [0.0, 10.0, 30.0, 0.0, 100.0, 300.0],
        "SenderHistAmountMean": [0.0, 10.0, 15.0, 0.0, 100.0, 60.0],
        "Time": [1, 2, 3, 50, 100, 150],
        "Type": ["A", "A", "B", "B", "A", "C"],
        "Target": ["X", "Y", "X", "Y", "Z", "Z"],
        "Labels": [0, 0, 0, 1, 1, 1],
    })


def test_compute_training_statistics_uses_only_train_idx():
    df = _sample_processed()
    train_idx = np.array([0, 1, 2])  # 3 normal samples
    stats = compute_training_statistics(df, train_idx)
    assert stats.train_size == 3
    assert stats.positive_rate == 0.0
    assert "Amount" in stats.numerical
    assert stats.numerical["Amount"]["all"]["count"] == 3
    assert stats.numerical["Amount"]["normal"]["count"] == 3
    assert stats.numerical["Amount"].get("risk") is None or \
           stats.numerical["Amount"]["risk"]["count"] == 0
    assert "Type" in stats.categorical


def test_compute_training_statistics_positives():
    df = _sample_processed()
    train_idx = np.array([3, 4, 5, 0])  # 3 positives + 1 normal
    stats = compute_training_statistics(df, train_idx)
    assert stats.train_size == 4
    assert stats.positive_rate == 0.75
    assert stats.numerical["Amount"]["all"]["count"] == 4
    assert stats.numerical["Amount"]["normal"]["count"] == 1
    assert stats.numerical["Amount"]["risk"]["count"] == 3


def test_compute_training_statistics_categorical():
    df = _sample_processed()
    train_idx = np.array([0, 1, 2, 3, 4, 5])
    stats = compute_training_statistics(df, train_idx)
    type_stats = stats.categorical["Type"]
    assert isinstance(type_stats, list)
    assert len(type_stats) <= 3
    # A appears 3 times, B 2 times, C 1 time
    type_map = {e["value"]: e["count"] for e in type_stats}
    assert type_map.get("A", 0) == 3


def test_compute_training_statistics_percentile_keys():
    df = _sample_processed()
    train_idx = np.array([0, 1, 2, 3, 4, 5])
    stats = compute_training_statistics(df, train_idx)
    entry = stats.numerical["Amount"]["all"]
    for key in ("count", "mean", "std", "min", "max", "p50", "p95"):
        assert key in entry, f"Missing key: {key}"


def test_dataset_statistics_to_dict():
    stats = DatasetStatistics(
        numerical={"Amount": {"all": {"count": 100, "mean": 50.0}}},
        categorical={"Type": [{"value": "A", "count": 50, "ratio": 0.5}]},
        positive_rate=0.1, train_size=100,
    )
    d = stats.to_dict()
    assert d["positive_rate"] == 0.1
    assert d["train_size"] == 100


# ═══════════════════════════════════════════════════════════════════════════
# Part 4 — sha256_file
# ═══════════════════════════════════════════════════════════════════════════

def test_sha256_file():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("a,b,c\n1,2,3\n")
        path = f.name
    try:
        digest = sha256_file(path)
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
    finally:
        os.unlink(path)
