"""Standalone RuleBank generation — call Qwen LLM to produce a versioned RuleBank YAML.

Usage:
    # Qwen 服务开启时 → 调用 LLM 生成规则
    python scripts/generate_rulebank.py

    # 指定输出路径
    python scripts/generate_rulebank.py --output config/rulebank/aml_rulebank_v2.yaml
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import pandas as pd

from methods.rgtan.rgtan_main import loda_rgtan_data
from methods.modules.rule_generation import (
    compute_training_statistics,
    QwenRuleGenerator,
    generate_rulebank,
)
from methods.modules.rule_bank import sha256_file


def _base_args() -> dict:
    return {
        "dataset": "aml",
        "data_path": "../AMLdataset.csv",
        "test_size": 0.2,
        "val_size": 0.2,
        "seed": 2023,
        "split_mode": "sender_account",
        "method": "rgtan_mpfc",
        "results_dir": "results",
        "device": "cpu",
        "nei_att_heads": {"aml": 1},
    }


def main():
    parser = argparse.ArgumentParser(description="Generate RuleBank YAML from Qwen LLM")
    parser.add_argument("--output", default="config/rulebank/aml_rulebank_v1.yaml")
    parser.add_argument("--prompt", default="prompts/rule_generation_v1.txt")
    parser.add_argument("--api-url", default="http://localhost:23333/v1/chat/completions")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--max-rules", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--force", action="store_true", help="Overwrite existing YAML")
    args = parser.parse_args()

    # 1. Load AML data (side effect: populates args with _aml_processed_path etc.)
    print("[1/4] Loading AML training data…")
    cli_args = _base_args()
    feat_df, labels, train_idx, test_idx, g, cat_features, neigh_features = \
        loda_rgtan_data(cli_args)
    processed = pd.read_csv(cli_args["_aml_processed_path"])
    print(f"  train_size={len(train_idx)}  positive_rate={labels.iloc[train_idx].mean():.4f}")

    # 2. Compute training statistics (train_idx only)
    print("[2/4] Computing training statistics…")
    stats = compute_training_statistics(processed, train_idx)
    print(f"  {len(stats.numerical)} numerical + {len(stats.categorical)} categorical fields")

    # 3. Generate RuleBank via Qwen LLM
    print(f"[3/4] Calling Qwen LLM ({args.api_url})…")
    data_fingerprint = sha256_file(cli_args["_aml_processed_path"])
    prompt_fingerprint = sha256_file(args.prompt)

    bank = generate_rulebank(
        stats=stats,
        prompt_path=args.prompt,
        output_path=args.output,
        llm_client=QwenRuleGenerator(api_url=args.api_url, model=args.model),
        data_fingerprint=data_fingerprint,
        prompt_fingerprint=prompt_fingerprint,
        prompt_version="1.0",
        max_rules=args.max_rules,
        vllm_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
        },
    )

    # 4. Report
    log_path = args.output.replace(".yaml", "_generation_log.json")
    import json
    with open(log_path) as f:
        log = json.load(f)

    print(f"[4/4] Done → {args.output}")
    print(f"  rules: {len(bank.active_rules())} active / {len(bank.rules)} total")
    print(f"  source: {log.get('source', '?')}  model: {bank.source_model}")
    print(f"  version: {bank.rulebank_version}")


if __name__ == "__main__":
    main()
