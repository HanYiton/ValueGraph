#!/usr/bin/env python3
"""
用 GPT（默认 gpt-5.4）对 MT-CSD / RumourEval19 的文本做立场分类（零样本，仅文本）。
支持断点续跑；默认只预测每条样本的 text 列（叶子评论 / reply_text）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / "prepared"
RESULTS = ROOT / "results"
ENV_CANDIDATES = [
    ROOT / ".env",
    Path(__file__).resolve().parents[1]
    / "llm-misinformation"
    / "experiment"
    / "detection_script"
    / ".env",
]

MT_LABELS = ("favor", "against", "none")
RUMOUR_LABELS = ("support", "deny", "query", "comment")

PROMPT_MT = """You are annotating stance in a social-media discussion about the topic: {topic}.

Classify the STANCE of the following comment toward the discussion target.
Use exactly one label:
- favor: supports/agrees with the target stance
- against: opposes/disagrees with the target stance
- none: neutral, unrelated, or unclear stance

Comment:
{text}

Reply with exactly one word: favor, against, or none."""

PROMPT_RUMOUR = """You are annotating stance toward a rumor in social media.

Classify the STANCE of the following reply (SDQC scheme).
Use exactly one label:
- support: supports/agrees the rumor is true
- deny: refutes or disagrees with the rumor
- query: asks for evidence or clarification
- comment: neutral or unrelated to rumor veracity

Reply text:
{text}

Reply with exactly one word: support, deny, query, or comment."""


def load_dotenv() -> None:
    for p in ENV_CANDIDATES:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_mt_label(raw: str) -> str:
    t = raw.strip().lower()
    for lab in MT_LABELS:
        if re.search(rf"\b{lab}\b", t):
            return lab
    return "none"


def parse_rumour_label(raw: str) -> str:
    t = raw.strip().lower()
    for lab in RUMOUR_LABELS:
        if re.search(rf"\b{lab}\b", t):
            return lab
    return "comment"


def chat_complete(client: OpenAI, model: str, user_content: str, use_max_tokens: bool) -> str:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.0,
    }
    if use_max_tokens:
        kwargs["max_tokens"] = 32
    else:
        kwargs["max_completion_tokens"] = 32
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def build_prompt(row: pd.Series, text_mode: str) -> str:
    text = str(row["text"]) if text_mode == "leaf" else str(row.get("conversation_text") or row["text"])
    text = text[:12000]
    if row["dataset"] == "MT-CSD":
        return PROMPT_MT.format(topic=row.get("topic", ""), text=text)
    return PROMPT_RUMOUR.format(text=text)


def run_split(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    out_csv: Path,
    use_max_tokens: bool,
    text_mode: str,
    sleep_s: float,
    max_retries: int,
) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_csv.is_file():
        prev = pd.read_csv(out_csv, dtype=str)
        for _, r in prev.iterrows():
            if pd.notna(r.get("pred_label")) and str(r.get("pred_label")).strip():
                done[str(r["instance_id"])] = r.to_dict()

    records: list[dict] = []
    for _, row in df.iterrows():
        iid = str(row["instance_id"])
        if iid in done:
            records.append(done[iid])
            continue

        prompt = build_prompt(row, text_mode)
        raw = ""
        pred = ""
        err = ""
        for attempt in range(1, max_retries + 1):
            try:
                raw = chat_complete(client, model, prompt, use_max_tokens)
                pred = (
                    parse_mt_label(raw)
                    if row["dataset"] == "MT-CSD"
                    else parse_rumour_label(raw)
                )
                err = ""
                break
            except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as e:
                err = str(e)
                time.sleep(min(60, 2**attempt))
        rec = row.to_dict()
        rec["raw_response"] = raw
        rec["pred_label"] = pred
        rec["error"] = err
        records.append(rec)
        done[iid] = rec

        partial = pd.DataFrame(records)
        partial.to_csv(out_csv, index=False)
        print(f"[{out_csv.name}] {len(records)}/{len(df)} id={iid} pred={pred}")
        time.sleep(sleep_s)

    return pd.DataFrame(records)


def metrics_for(result_df: pd.DataFrame, name: str) -> dict:
    sub = result_df[result_df["pred_label"].astype(str).str.len() > 0].copy()
    if sub.empty:
        return {"name": name, "n": 0}
    y_true = sub["gold_label"].astype(str).str.lower()
    y_pred = sub["pred_label"].astype(str).str.lower()
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "name": name,
        "n": int(len(sub)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MT-CSD", "RumourEval19"],
        choices=["MT-CSD", "RumourEval19"],
    )
    parser.add_argument("--split", default="test", help="train|valid|test|all")
    parser.add_argument("--openai_model", default="gpt-5.4")
    parser.add_argument("--openai_use_max_tokens", action="store_true")
    parser.add_argument("--text_mode", choices=["leaf", "conversation"], default="leaf")
    parser.add_argument("--sleep_s", type=float, default=0.3)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 条（调试）")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("请设置 OPENAI_API_KEY（可在 LLMbaseline/.env 或 detection_script/.env）")

    client = OpenAI(api_key=api_key)
    all_df = pd.read_csv(PREPARED / "all_datasets.csv", dtype=str)
    splits = None if args.split == "all" else [args.split]
    if splits:
        all_df = all_df[all_df["split"].isin(splits)]

    all_metrics: dict = {"model": args.openai_model, "split": args.split, "text_mode": args.text_mode}
    for ds in args.datasets:
        df = all_df[all_df["dataset"] == ds].reset_index(drop=True)
        if args.limit > 0:
            df = df.head(args.limit)
        tag = f"{ds.replace('-', '').lower()}_{args.split}"
        out_csv = RESULTS / f"gpt54_{tag}_predictions.csv"
        result_df = run_split(
            df,
            client,
            args.openai_model,
            out_csv,
            args.openai_use_max_tokens,
            args.text_mode,
            args.sleep_s,
            args.max_retries,
        )
        m = metrics_for(result_df, ds)
        all_metrics[ds] = {k: v for k, v in m.items() if k != "report"}
        if "report" in m:
            print(f"\n=== {ds} ===\n{m['report']}")

    metrics_path = RESULTS / f"gpt54_metrics_{args.split}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
