#!/usr/bin/env python3
"""
用 GPT（默认 gpt-5.4）对 TwiBot 用户聚合推文做 human/bot 检测（零样本，仅文本）。
默认只评估 split=test 且有 gold label 的用户；支持断点续跑。
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
TWIBOT = ROOT / "twitterbot"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
ENV_CANDIDATES = [
    ROOT / ".env",
    Path(__file__).resolve().parents[1]
    / "llm-misinformation"
    / "experiment"
    / "detection_script"
    / ".env",
]

LABELS = ("human", "bot")

PROMPT = """You are detecting whether a Twitter/X account is operated by a human or a bot.

Read the following tweets posted by ONE account (may be truncated). Classify the account type.
Use exactly one label:
- human: likely a real person
- bot: likely automated, spam, or bot-like

Tweets from this account:
{text}

Reply with exactly one word: human or bot."""


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


def parse_label(raw: str) -> str:
    t = raw.strip().lower()
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", t):
            return lab
    return ""


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


def build_user_dataset(
    parquet_path: Path,
    split: str,
    max_chars: int,
    limit: int,
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=["author_id", "split", "text", "label"])
    df = df[df["split"] == split].copy()
    df = df[df["label"].notna()].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.lower()

    rows: list[dict] = []
    for author_id, grp in df.groupby("author_id", sort=True):
        texts = grp["text"].tolist()
        combined = "\n---\n".join(texts)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n...[truncated]"
        rows.append(
            {
                "author_id": author_id,
                "split": split,
                "gold_label": grp["label"].iloc[0],
                "n_tweets": len(texts),
                "text_chars": len(combined),
                "text": combined,
            }
        )

    out = pd.DataFrame(rows).sort_values("author_id").reset_index(drop=True)
    if limit > 0:
        out = out.head(limit)
    return out


def run_classification(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    out_csv: Path,
    use_max_tokens: bool,
    sleep_s: float,
    max_retries: int,
) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_csv.is_file():
        prev = pd.read_csv(out_csv, dtype=str)
        for _, r in prev.iterrows():
            if pd.notna(r.get("pred_label")) and str(r.get("pred_label")).strip():
                done[str(r["author_id"])] = r.to_dict()

    records: list[dict] = []
    for _, row in df.iterrows():
        aid = str(row["author_id"])
        if aid in done:
            records.append(done[aid])
            continue

        prompt = PROMPT.format(text=row["text"])
        raw = ""
        pred = ""
        err = ""
        for attempt in range(1, max_retries + 1):
            try:
                raw = chat_complete(client, model, prompt, use_max_tokens)
                pred = parse_label(raw)
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
        done[aid] = rec

        partial = pd.DataFrame(records)
        partial.to_csv(out_csv, index=False)
        print(f"[{out_csv.name}] {len(records)}/{len(df)} author_id={aid} gold={row['gold_label']} pred={pred}")
        time.sleep(sleep_s)

    return pd.DataFrame(records)


def metrics_for(result_df: pd.DataFrame) -> dict:
    sub = result_df[result_df["pred_label"].astype(str).str.len() > 0].copy()
    if sub.empty:
        return {"n": 0}
    y_true = sub["gold_label"].astype(str).str.lower()
    y_pred = sub["pred_label"].astype(str).str.lower()
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n": int(len(sub)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default=str(TWIBOT / "filtered_sampled_twibot_no_missing_text_with_label.parquet"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--openai_model", default="gpt-5.4")
    parser.add_argument("--openai_use_max_tokens", action="store_true")
    parser.add_argument("--max_chars", type=int, default=12000)
    parser.add_argument("--sleep_s", type=float, default=0.3)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 个用户（调试）")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("请设置 OPENAI_API_KEY（可在 LLMbaseline/.env 或 detection_script/.env）")

    client = OpenAI(api_key=api_key)
    user_df = build_user_dataset(Path(args.parquet), args.split, args.max_chars, args.limit)
    print(f"Users to classify ({args.split}, labeled): {len(user_df)}")
    print(user_df["gold_label"].value_counts())

    out_csv = RESULTS / f"gpt54_twibot_{args.split}_predictions.csv"
    result_df = run_classification(
        user_df,
        client,
        args.openai_model,
        out_csv,
        args.openai_use_max_tokens,
        args.sleep_s,
        args.max_retries,
    )

    m = metrics_for(result_df)
    all_metrics = {
        "model": args.openai_model,
        "split": args.split,
        "max_chars": args.max_chars,
        "dataset": "TwiBot",
        **{k: v for k, v in m.items() if k != "report"},
    }
    if "report" in m:
        print(f"\n=== TwiBot ({args.split}) ===\n{m['report']}")

    metrics_path = RESULTS / f"gpt54_twibot_metrics_{args.split}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Predictions -> {out_csv}")
    print(f"Metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
