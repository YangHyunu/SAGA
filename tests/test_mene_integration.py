"""MENE Integration Tests — 각 컴포넌트 자동 검증 스크립트.

사용법: python3 tests/test_mene_integration.py
서버가 실행 중이어야 합니다 (python3 -m mene)
"""
import asyncio
import httpx
import json
import sys
import time

BASE = "http://localhost:8000"
TIMEOUT = 90.0

results = []

def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


async def test_server_status():
    """1. 서버 상태 확인"""
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get(f"{BASE}/api/status")
            data = r.json()
            report("서버 상태", data.get("status") == "running", f"version={data.get('version')}")
        except Exception as e:
            report("서버 상태", False, str(e))


async def test_chat_completion():
    """2. 기본 Chat Completion (현재 narration 모델)"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        try:
            t0 = time.time()
            r = await c.post(f"{BASE}/v1/chat/completions", json={
                "model": "test",
                "messages": [
                    {"role": "system", "content": "당신은 판타지 RP 내레이터입니다. 응답 마지막에 ```state 블록을 추가하세요."},
                    {"role": "user", "content": "나는 마을 광장에 서 있다. 주위를 둘러본다."}
                ],
                "max_tokens": 1024
            })
            elapsed = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                has_content = len(content) > 50
                report("Chat Completion", has_content, f"{elapsed:.1f}초, {len(content)}자")
            else:
                report("Chat Completion", False, f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            report("Chat Completion", False, str(e))


async def test_session_apis():
    """3. 세션 API 테스트"""
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get(f"{BASE}/api/sessions")
            sessions = r.json()
            report("세션 목록", isinstance(sessions, list), f"{len(sessions)}개 세션")

            if sessions:
                sid = sessions[0]["id"]
                # State
                r2 = await c.get(f"{BASE}/api/sessions/{sid}/state")
                report("세션 상태", r2.status_code == 200)
                # Graph
                r3 = await c.get(f"{BASE}/api/sessions/{sid}/graph")
                graph = r3.json()
                has_chars = "Characters" in graph.get("graph_summary", "")
                report("그래프 조회", has_chars, f"session={sid}")
                # Cache
                r4 = await c.get(f"{BASE}/api/sessions/{sid}/cache")
                report("캐시 조회", r4.status_code == 200)
                # Turns
                r5 = await c.get(f"{BASE}/api/sessions/{sid}/turns")
                turns = r5.json()
                report("턴 로그", isinstance(turns, list), f"{len(turns)}턴 기록")
            else:
                report("세션 상태", False, "세션 없음")
        except Exception as e:
            report("세션 API", False, str(e))


async def test_state_block_parsing():
    """4. State block 파싱 테스트 (로컬, 서버 불필요)"""
    from mene.utils.parsers import parse_state_block, strip_state_block

    # 정상 케이스
    test1 = """나레이션 텍스트입니다.

```state
location: 어둠의 숲
location_moved: true
hp_change: -10
items_gained: [마법 검]
items_lost: []
npc_met: [에르겐]
mood: tense
event_trigger: null
notes: 첫 번째 전투
```"""
    parsed = parse_state_block(test1)
    report("State 정상 파싱", parsed is not None and parsed.get("location") == "어둠의 숲")
    report("State HP 추출", parsed is not None and parsed.get("hp_change") == -10)
    report("State NPC 추출", parsed is not None and "에르겐" in parsed.get("npc_met", []))

    # Strip 테스트
    stripped = strip_state_block(test1)
    report("State block 제거", "```state" not in stripped and "나레이션" in stripped)

    # 백틱 2개 케이스
    test2 = """텍스트

``state
location: 마을
location_moved: false
mood: calm
``"""
    parsed2 = parse_state_block(test2)
    report("백틱 2개 파싱", parsed2 is not None and parsed2.get("location") == "마을")

    # state block 없는 케이스
    test3 = "일반 텍스트만 있는 응답"
    parsed3 = parse_state_block(test3)
    report("State 없는 응답", parsed3 is None)


async def test_context_builder_timing():
    """5. Context Builder 성능 테스트"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        try:
            # 짧은 메시지로 Context Builder만 측정 (LLM 호출 포함이지만 타이밍 로그로 확인)
            t0 = time.time()
            r = await c.post(f"{BASE}/v1/chat/completions", json={
                "model": "test",
                "messages": [
                    {"role": "system", "content": "테스트"},
                    {"role": "user", "content": "안녕"}
                ],
                "max_tokens": 256
            })
            elapsed = time.time() - t0
            report("전체 응답 시간", r.status_code == 200, f"{elapsed:.1f}초")
        except Exception as e:
            report("Context Builder", False, str(e))


async def test_memory_search():
    """6. 메모리 검색 API"""
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get(f"{BASE}/api/memory/search", params={"q": "어둠의 숲"})
            report("메모리 검색", r.status_code == 200)
        except Exception as e:
            report("메모리 검색", False, str(e))


async def main():
    print("=" * 60)
    print("MENE Integration Test Suite")
    print("=" * 60)

    print("\n[1] 서버 상태")
    await test_server_status()

    print("\n[2] State Block 파싱 (로컬)")
    await test_state_block_parsing()

    print("\n[3] Chat Completion")
    await test_chat_completion()

    print("\n[4] 세션 API")
    await test_session_apis()

    print("\n[5] 응답 시간")
    await test_context_builder_timing()

    print("\n[6] 메모리 검색")
    await test_memory_search()

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    print(f"결과: {passed}/{total} 통과")
    if passed < total:
        print("\n실패 항목:")
        for name, p, detail in results:
            if not p:
                print(f"  - {name}: {detail}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
