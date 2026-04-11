#!/usr/bin/env python3
"""LLM-as-a-Judge — 원본 vs SAGA 응답 품질 비교 평가.

Gemini 3 Flash로 턴별 품질을 5점 척도로 평가.
Usage:
    python benchmarks/judge.py --replay benchmarks/results/replay5.json --output benchmarks/results/judge5.json
"""

import argparse
import asyncio
import json
import os
import sys
import time

JUDGE_SYSTEM = """당신은 RP(역할극) 응답 품질을 평가하는 전문 심사관입니다.

사용자의 입력(user_input)에 대해 두 개의 AI 응답을 비교 평가합니다:
- **Original**: 원본 AI 응답
- **SAGA**: SAGA 시스템을 통한 AI 응답

각 응답을 아래 5개 지표로 독립 평가하세요 (1~5점).
점수 기준: 1=매우 나쁨, 2=나쁨, 3=보통, 4=좋음, 5=매우 좋음

## 평가 지표

1. **character_consistency** (캐릭터 일관성): 캐릭터의 성격, 말투, 행동 패턴이 일관적인가
2. **narrative_coherence** (서사 연속성): 이전 대화 맥락을 적절히 반영하고 논리적으로 이어지는가
3. **writing_quality** (문체 품질): 묘사력, 몰입감, 문장의 자연스러움
4. **context_utilization** (컨텍스트 활용): 설정, 배경, 이전 사건 정보를 잘 활용하는가
5. **response_relevance** (응답 적절성): 사용자 입력에 대한 반응이 적합하고 자연스러운가

## 출력 형식

JSON만 반환 (마크다운, 설명 금지):
{
  "original": {
    "character_consistency": N,
    "narrative_coherence": N,
    "writing_quality": N,
    "context_utilization": N,
    "response_relevance": N
  },
  "saga": {
    "character_consistency": N,
    "narrative_coherence": N,
    "writing_quality": N,
    "context_utilization": N,
    "response_relevance": N
  },
  "brief_note": "한 줄 코멘트 (어떤 점에서 차이가 있는지)"
}"""

JUDGE_USER_TEMPLATE = """## 이전 대화 맥락 (최근 3턴)
{context}

---

## 현재 Turn {turn}

### User Input
{user_input}

### Original Response
{original}

### SAGA Response
{saga}

위 두 응답을 이전 맥락을 고려하여 5개 지표로 각각 평가하세요. JSON만 반환."""


async def judge_turn(turn_data: dict, context_turns: list[dict], api_key: str, model: str = "gemini-3-flash-preview") -> dict | None:
    """Judge a single turn using Gemini, with prior context."""
    import httpx

    user_input = turn_data.get("user_input", "")
    original = turn_data.get("original_assistant", "")
    saga = turn_data.get("saga_response", "")

    if not original or not saga:
        return None

    # Build context from previous turns
    context_block = ""
    if context_turns:
        ctx_lines = []
        for ct in context_turns:
            ctx_lines.append(f"[Turn {ct.get('turn', '?')} User]: {ct.get('user_input', '')[:500]}")
            ctx_lines.append(f"[Turn {ct.get('turn', '?')} Assistant]: {ct.get('original_assistant', '')[:500]}")
        context_block = "\n".join(ctx_lines)

    prompt = JUDGE_USER_TEMPLATE.format(
        turn=turn_data.get("turn", "?"),
        user_input=user_input[:2000],
        original=original[:3000],
        saga=saga[:3000],
        context=context_block,
    )

    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    api_url,
                    json={
                        "contents": [
                            {"role": "user", "parts": [{"text": JUDGE_SYSTEM + "\n\n" + prompt}]},
                        ],
                        "generationConfig": {
                            "temperature": 0.1,
                            "maxOutputTokens": 2048,
                            "responseMimeType": "application/json",
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]

                # Parse JSON (robust: try direct, then extract from markdown)
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    # Try extracting JSON from markdown code block
                    import re
                    m = re.search(r'\{[\s\S]*\}', text)
                    if m:
                        result = json.loads(m.group())
                    else:
                        raise

                result["turn"] = turn_data.get("turn", 0)
                return result

            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"  [Judge] Turn {turn_data.get('turn', '?')} error: {e}")
                return None
    return None


async def run_judge(args):
    """Run LLM judge on all turns."""
    with open(args.replay, encoding="utf-8") as f:
        data = json.load(f)

    turns = data.get("turns", [])
    replay_turns = [t for t in turns if t.get("type") == "replay" and t.get("original_assistant") and t.get("saga_response")]

    print(f"\n{'='*60}")
    print(f"  LLM-as-a-Judge Quality Evaluation")
    print(f"{'='*60}")
    print(f"  Replay file:  {args.replay}")
    print(f"  Total turns:  {len(turns)}")
    print(f"  Judgeable:    {len(replay_turns)} (replay with both responses)")
    print(f"  Model:        {args.model}")
    print(f"{'='*60}\n")

    api_key = args.google_api_key
    if not api_key:
        print("  [ERROR] --google-api-key required")
        sys.exit(1)

    results = []
    batch_size = 3  # concurrent requests to avoid rate limit

    for i in range(0, len(replay_turns), batch_size):
        batch = replay_turns[i:i + batch_size]
        tasks = []
        for t in batch:
            # Find index of this turn in replay_turns for context
            t_idx = replay_turns.index(t)
            ctx = replay_turns[max(0, t_idx - 3):t_idx]
            tasks.append(judge_turn(t, ctx, api_key, args.model))
        batch_results = await asyncio.gather(*tasks)

        for r in batch_results:
            if r:
                results.append(r)
                turn = r.get("turn", "?")
                o = r.get("original", {})
                s = r.get("saga", {})
                o_avg = sum(o.values()) / len(o) if o else 0
                s_avg = sum(s.values()) / len(s) if s else 0
                note = r.get("brief_note", "")[:60]
                print(f"  Turn {turn:3d} | Original: {o_avg:.1f} | SAGA: {s_avg:.1f} | {note}")

        # Rate limit pause
        if i + batch_size < len(replay_turns):
            await asyncio.sleep(1)

    # Aggregate
    if not results:
        print("\n  No results to aggregate.")
        return

    metrics = ["character_consistency", "narrative_coherence", "writing_quality", "context_utilization", "response_relevance"]

    print(f"\n{'='*60}")
    print(f"  Aggregate Results ({len(results)} turns)")
    print(f"{'='*60}")
    print(f"  {'Metric':<25s} {'Original':>10s} {'SAGA':>10s} {'Diff':>10s}")
    print(f"  {'-'*55}")

    orig_totals = {m: [] for m in metrics}
    saga_totals = {m: [] for m in metrics}

    for r in results:
        for m in metrics:
            o_val = r.get("original", {}).get(m)
            s_val = r.get("saga", {}).get(m)
            if o_val is not None:
                orig_totals[m].append(o_val)
            if s_val is not None:
                saga_totals[m].append(s_val)

    overall_orig = []
    overall_saga = []

    for m in metrics:
        o_avg = sum(orig_totals[m]) / len(orig_totals[m]) if orig_totals[m] else 0
        s_avg = sum(saga_totals[m]) / len(saga_totals[m]) if saga_totals[m] else 0
        diff = s_avg - o_avg
        sign = "+" if diff >= 0 else ""
        print(f"  {m:<25s} {o_avg:>10.2f} {s_avg:>10.2f} {sign}{diff:>9.2f}")
        overall_orig.append(o_avg)
        overall_saga.append(s_avg)

    o_total = sum(overall_orig) / len(overall_orig)
    s_total = sum(overall_saga) / len(overall_saga)
    d_total = s_total - o_total
    sign = "+" if d_total >= 0 else ""
    print(f"  {'-'*55}")
    print(f"  {'OVERALL':<25s} {o_total:>10.2f} {s_total:>10.2f} {sign}{d_total:>9.2f}")
    print(f"{'='*60}")

    # Save
    report = {
        "source": args.replay,
        "model": args.model,
        "judged_turns": len(results),
        "aggregate": {},
        "per_turn": results,
    }
    for m in metrics:
        report["aggregate"][m] = {
            "original": round(sum(orig_totals[m]) / len(orig_totals[m]), 2) if orig_totals[m] else 0,
            "saga": round(sum(saga_totals[m]) / len(saga_totals[m]), 2) if saga_totals[m] else 0,
        }
    report["aggregate"]["overall"] = {
        "original": round(o_total, 2),
        "saga": round(s_total, 2),
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  [OUTPUT] Saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge RP 품질 평가")
    parser.add_argument("--replay", required=True, help="replay result JSON path")
    parser.add_argument("--google-api-key", default=os.environ.get("GOOGLE_API_KEY", ""),
                        help="Google API key (default: $GOOGLE_API_KEY)")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Judge model")
    parser.add_argument("--output", default="", help="Output JSON path")

    args = parser.parse_args()
    asyncio.run(run_judge(args))


if __name__ == "__main__":
    main()
