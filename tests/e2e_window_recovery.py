"""E2E: 슬라이딩 윈도우 캐시 복구 실서버 테스트.

실행 전 SAGA 서버가 --reset-db로 떠있어야 함.
"""
import asyncio
import httpx
import json
import sys

BASE = "http://localhost:8000"
AUTH = {"Authorization": "Bearer saga-test-key-2026", "X-SAGA-Session-ID": "e2e-window-test"}
MODEL = "claude-haiku-4-5-20251001"


def make_messages(system: str, turns: list[tuple[str, str]]) -> list[dict]:
    """Build messages list from system + (user, assistant) pairs."""
    msgs = [{"role": "system", "content": system}]
    for user, assistant in turns:
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


async def send_chat(client: httpx.AsyncClient, messages: list[dict], extra_user: str) -> dict:
    """Send a chat completion request."""
    msgs = messages + [{"role": "user", "content": extra_user}]
    resp = await client.post(
        f"{BASE}/v1/chat/completions",
        headers={**AUTH, "Content-Type": "application/json"},
        json={"model": MODEL, "messages": msgs, "max_tokens": 50},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


async def get_cost(client: httpx.AsyncClient) -> dict:
    resp = await client.get(f"{BASE}/api/cost", headers=AUTH, timeout=10)
    resp.raise_for_status()
    return resp.json()


async def main():
    system = "You are a fantasy RPG narrator. Keep responses short (1-2 sentences)."

    # 10턴짜리 대화 히스토리 구성
    full_history = [
        ("마을에 도착했다. 여관을 찾자.", "마을 광장을 지나 '붉은 사슴' 여관을 발견합니다."),
        ("여관에 들어가서 주인에게 말을 걸자.", "여관주인 하렌이 맥주를 닦으며 인사합니다."),
        ("이 근처에 던전이 있다고 들었는데.", "하렌이 고개를 끄덕이며 북쪽 폐광을 가리킵니다."),
        ("동료를 구하고 싶은데 누가 있을까?", "구석에 앉은 엘프 궁수 리아나가 눈에 띕니다."),
        ("리아나에게 말을 걸자.", "리아나가 흥미를 보이며 함께 가겠다고 합니다."),
        ("장비를 구매하고 출발하자.", "대장간에서 검과 방패를 구입하고 북쪽으로 향합니다."),
        ("폐광 입구에 도착했다.", "어둡고 습한 폐광 입구. 안에서 으르렁거리는 소리가 들립니다."),
        ("조심히 안으로 들어가자.", "리아나가 화살을 장전하고 앞장섭니다. 첫 번째 갈림길이 나타납니다."),
        ("왼쪽 길로 가자.", "왼쪽 통로를 따라가니 고블린 3마리가 모닥불 주위에 있습니다."),
        ("기습 공격!", "리아나의 화살이 고블린 하나를 쓰러뜨립니다. 나머지 둘이 무기를 듭니다."),
    ]

    print("=" * 60)
    print("E2E: 슬라이딩 윈도우 캐시 복구 테스트")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # ── Step 1: 전체 히스토리로 대화 (캐시 구축) ──
        print("\n[Step 1] 10턴 전체 히스토리로 요청...")
        msgs_full = make_messages(system, full_history)
        r1 = await send_chat(client, msgs_full, "전투 결과는?")
        print(f"  응답: {r1['choices'][0]['message']['content'][:80]}")
        print(f"  usage: input={r1['usage']['prompt_tokens']} output={r1['usage']['completion_tokens']}")
        print(f"  cache: read={r1['usage'].get('cache_read_input_tokens', 0)} create={r1['usage'].get('cache_creation_input_tokens', 0)}")

        # 잠시 대기 (Sub-B가 에피소드 저장할 시간)
        print("\n  Sub-B 처리 대기 (3초)...")
        await asyncio.sleep(3)

        # ── Step 2: 같은 히스토리로 한번 더 (캐시 hit 확인) ──
        print("\n[Step 2] 동일 히스토리로 재요청 (캐시 hit 기대)...")
        r2 = await send_chat(client, msgs_full, "고블린을 모두 처치했나?")
        print(f"  응답: {r2['choices'][0]['message']['content'][:80]}")
        cache_read = r2['usage'].get('cache_read_input_tokens', 0)
        cache_create = r2['usage'].get('cache_creation_input_tokens', 0)
        print(f"  cache: read={cache_read} create={cache_create}")
        if cache_read > 0:
            print("  ✅ 캐시 hit 확인!")
        else:
            print("  ⚠️ 캐시 hit 없음 (토큰 수 부족일 수 있음)")

        # ── Step 3: 앞쪽 5턴 잘라서 보내기 (슬라이딩 윈도우 시뮬) ──
        print("\n[Step 3] 앞쪽 5턴 잘라냄 (슬라이딩 윈도우 시뮬레이션)...")
        trimmed_history = full_history[5:]  # turn 6~10만
        msgs_trimmed = make_messages(system, trimmed_history)
        r3 = await send_chat(client, msgs_trimmed, "리아나는 괜찮아?")
        print(f"  응답: {r3['choices'][0]['message']['content'][:80]}")
        print(f"  usage: input={r3['usage']['prompt_tokens']} output={r3['usage']['completion_tokens']}")
        print(f"  cache: read={r3['usage'].get('cache_read_input_tokens', 0)} create={r3['usage'].get('cache_creation_input_tokens', 0)}")

        # ── Step 4: 다시 같은 잘린 히스토리 (복구된 BP로 캐시 hit 기대) ──
        print("\n[Step 4] 잘린 히스토리 재요청 (복구된 BP로 캐시 hit 기대)...")
        r4 = await send_chat(client, msgs_trimmed, "다음 방으로 이동하자.")
        cache_read_4 = r4['usage'].get('cache_read_input_tokens', 0)
        print(f"  응답: {r4['choices'][0]['message']['content'][:80]}")
        print(f"  cache: read={cache_read_4} create={r4['usage'].get('cache_creation_input_tokens', 0)}")
        if cache_read_4 > 0:
            print("  ✅ 윈도우 복구 후 캐시 hit!")
        else:
            print("  ⚠️ 캐시 hit 없음 (요약 블록 크기 부족일 수 있음)")

        # ── Step 5: 비용 집계 확인 ──
        print("\n[Step 5] 비용 집계 확인...")
        cost = await get_cost(client)
        print(f"  총 호출: {cost['total_calls']}회")
        print(f"  총 입력 토큰: {cost['total_input_tokens']}")
        print(f"  총 출력 토큰: {cost['total_output_tokens']}")
        print(f"  총 캐시 읽기: {cost['total_cache_read_tokens']}")
        print(f"  총 비용: ${cost['total_cost_usd']}")
        print(f"  캐시 절감: ${cost['total_savings_usd']}")
        print(f"  절감률: {cost['cache_savings_percent']}%")
        print(f"  모델별:")
        for m in cost.get("by_model", []):
            print(f"    {m['model']}: {m['calls']}회, ${m['cost_usd']}")

    print("\n" + "=" * 60)
    print("E2E 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
