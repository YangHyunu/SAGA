"""디렉터 벤치 드라이버 (EVAL2 §2·§3·§4). 기본 80턴, 파일럿은 짧게.

디렉터가 유저(렌) 역할로 전 구간을 자연스럽게 진행하며 매 턴 사실을 추출해
원장에 쌓는다. 지식갱신 이벤트(값 변경) 2개는 디렉터 지시문으로 강제 발생.
--probe-every(기본 10)턴마다 발화 하나가 가시 창 밖으로 evict된 과거 사실을
슬며시 화제로 되짚는다 — 시험조 금지, 유형은 회전
(recall/relation/false/update/recent). 채점은 오라클+judge 이중, 미스는
저장/활용 실패 분해. --turns로 줄여 파일럿을 돌린다.
--runs N 반복 (진행·필러만 변동, report2가 mean±std 집계).

vanilla·dreaming 변형은 트림 없이 전체 히스토리를 보낸다 — 프로브는 트림 창
기준으로 뽑히므로, vanilla가 틀리면 그건 정보 부재가 아니라
lost-in-the-middle이다. dreaming은 창 관리(압축)가 프록시 책임이라 벤치가
미리 자르지 않는다. hypa는 RisuAI HypaV3(뮈토스 하이파 V5 설정)를 그대로
돌려 스스로 요약·절단하므로 벤치의 token_trim을 쓰지 않는다 — 요약 블록은
프리셋 memory 카드(시스템 프롬프트 한가운데) 자리에 실린다.

와이어는 뮈토스 6.2 프리셋을 preset2wire로 조립 — 매 요청 check_wire_shape
게이트. 모델·엔드포인트는 env로 오버라이드 (확정 설정: 나레이터 V4 Pro,
디렉터 Gemini 3 Flash, judge Sonnet 4.5).

usage: python3 -m benchmarks.eval.run2 <preset.risup> <card.json> dreaming \
           --session v2a --runs 3
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import re
import shutil
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import httpx
import tiktoken

_ENC = tiktoken.get_encoding("o200k_base")

from benchmarks.eval.director import (Ledger, LlmFn, extract_facts,
                                      make_false_premise, make_probe,
                                      probe_plan)
from benchmarks.eval.fidelity import check_wire_shape
from benchmarks.eval.preset2wire import assemble, decode_risup, reformat
from benchmarks.eval.scoring import decompose_miss, judge_pass, oracle_pass
from benchmarks.eval import hypa

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "dreaming_data"
EVAL_DIR = DATA / "eval"
PROXY = os.environ.get("DREAMING_EVAL_PROXY", "http://127.0.0.1:8790")
UPSTREAM = os.environ.get("DREAMING_EVAL_UPSTREAM",
                          "https://openrouter.ai/api/v1")
MODEL = os.environ.get("DREAMING_EVAL_MODEL", "deepseek/deepseek-v4-pro")
JUDGE_MODEL = os.environ.get("DREAMING_EVAL_JUDGE",
                             "anthropic/claude-sonnet-4.5")
DIRECTOR_MODEL = os.environ.get("DREAMING_EVAL_DIRECTOR",
                                "google/gemini-3-flash-preview")
HYPA_EXPORT = os.environ.get(
    "DREAMING_EVAL_HYPA_EXPORT",
    str(pathlib.Path.home() / "Downloads" / "뮈토스6.2"
        / "🏺뮈토스 프롬프트 하이파" / "hypaV3_export_뮈토스 하이파 V5.json"))
TURNS = 80
PROBE_EVERY = 10              # 이 간격마다 발화 하나가 과거를 슬며시 되짚는다
# maxContext — RisuAI의 단일 토큰 풀 (index.svelte.ts:614-618). trim·hypa 공용.
# 원 프리셋 200K 대비 4.4× 축소라 hypa의 memoryTokens 선점도 78,000 → 17,550이다.
MAX_CONTEXT = 45000
UPDATE_EVENTS = (12, 28)      # 지식갱신 강제 턴
# NPC 등장은 당채련 하나로 고정 — 런 간 같은 사건 축이라 비교 가능하다.
# 로어 7엔트리는 항상 주입되므로(키워드 게이팅 없음) 활성화 문제가 아니라
# 장면 유도 문제다: 이름을 직접 불러 나레이터가 꺼내게 한다. 이 이름이
# 디렉터 카드 지식 선취 금지의 유일한 예외. 파일럿 50/80턴 NPC 0명 실측.
NPC_NAME = "당채련"
NPC_EVENT_TURN = 40           # 0-기준 (표시 T41)에 첫 유도
NPC_EVENT_RETRY = 44          # 이때까지 미등장이면 다시 유도
# 캡처에서 RisuAI가 실제로 보낸 값이 4000이다 (capture-mythos req-001).
# 실측 완성 평균은 771토큰이라 캡이 물리지 않는다 — 절단은 기억 실패로
# 오인되는 교란이라 finish_reason을 턴마다 기록해 0%임을 증명한다.
MAX_TOKENS = 4000
# 프로바이더가 거부(NSFW 등)를 반복하면 리롤 비용만 태운다 — 런 전체
# 누적 리롤이 이 값에 닿으면 결과를 저장하고 런을 중단한다.
MAX_RUN_REROLLS = 10

# 확정 토글: RP 모드·한국어·성인 지침 ON·중립 렌더링 프리필 ON, 나머지 기본
# select은 옵션 인덱스 문자열: response_language 1=🇰🇷 한국어, execution_mode 0=💬 RP.
# 나머지는 불리언. 미설정 전역변수는 "null"이라(preset2wire.UNSET) 실행 모드는
# 반드시 명시해야 tis:: 분기가 걸린다.
TOGGLES = {"mythos_response_language": "1",           # 🇰🇷 한국어
           "mythos_execution_mode": "0",              # 💬 RP
           "mythos_user_persona_usage": "0",          # 🙋 사용 — 안 켜면 슬롯이 통째로 빈다
           "mythos_bot_structure": "0",               # 💬 캐릭터 중심
           # select 토글은 미설정이면 사이드바 SelectInput이 바인딩하며 첫 옵션
           # 인덱스를 써 넣는다. 프리셋의 templateDefaultVariables는 여기 안
           # 먹는다 — getChatVar 전용이고 tis는 getGlobalChatVar를 본다
           # (chatVar.svelte.ts:15 vs 35, parser.svelte.ts:1284).
           "mythos_user_character_authorship": "0",   # 🛡️ 보호 — 캡처 req-005 확인
           "mythos_input_authority": "0",             # 🔨 사실 확정
           "mythos_prose_register": "0",              # 🤷 미지정
           "mythos_narrative_pov": "0",               # 🤷 자율
           "mythos_narrative_pacing": "0",            # 🤷 자율
           "mythos_response_length_band": "0",        # 🤷 미지정
           "mythos_size_scenario": "0",               # 🤷 미지정
           "mythos_genre_ero": "1",
           "mythos_mature_content_guidance": "1",
           "mythos_domain_neutral_rendering_prefill": "1"}


def _key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DREAMING_UPSTREAM_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no DREAMING_UPSTREAM_KEY in .env")


def _mk_llm(model: str, temperature: float) -> LlmFn:
    client = httpx.Client(base_url=UPSTREAM, timeout=120,
                          headers={"Authorization": f"Bearer {_key()}"})

    def call(system: str, user: str) -> str:
        r = client.post("/chat/completions", json={
            "model": model, "max_tokens": 400, "temperature": temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]})
        r.raise_for_status()
        data = r.json()
        u = data.get("usage") or {}
        call.cost += u.get("cost") or 0.0      # 부대비용도 잰다 — 나레이터만
        call.calls += 1                        # 재면 총비용을 과소보고한다
        return data["choices"][0]["message"]["content"] or ""
    call.cost, call.calls = 0.0, 0
    return call


def make_judge_llm() -> LlmFn:
    return _mk_llm(JUDGE_MODEL, 0.0)


def make_director_llm() -> LlmFn:
    return _mk_llm(DIRECTOR_MODEL, 0.7)


def _count(text: str) -> int:
    # RisuAI reverse_proxy 기본 토크나이저와 동일 (tokenizer.ts:105-133 →
    # o200k_base). len/2.5 근사는 한국어를 ~40% 과소평가해 12K 예산에서
    # eviction이 아예 안 일어났다 (파일럿 실측 18,917 vs 근사 11,816).
    return len(_ENC.encode(text))


def token_trim(history: List[Dict], budget: int,
               count_fn: Callable[[str], int] = _count
               ) -> Tuple[List[Dict], int]:
    """토큰 예산 기반 트림 — 메시지 단위 FIFO (index.svelte.ts:1143-1154).

    RisuAI는 페어 정렬 없이 chats[0]부터 하나씩 제거한다 — greeting도
    이 큐의 일부라 예산 판정에 포함되고, 남는 첫 메시지가 assistant일 수
    있다. 반환: (윈도우, win_start). win_start는 "이 턴 번호부터의 사실이
    창내" 의미 — 창의 첫 메시지가 턴 k의 user면 win_start=k, 턴 k의
    assistant면(반 잘린 턴) win_start=k+1.
    """
    if not history:
        return history, 0
    total = sum(count_fn(m["content"]) for m in history)
    start = 0
    while total > budget and len(history) - start > 1:
        total -= count_fn(history[start]["content"])
        start += 1
    window = history[start:]
    if start == 0:
        return window, 0
    has_greeting = history[0]["role"] == "assistant"
    offset = start - (1 if has_greeting else 0)
    turn, half = divmod(offset, 2)
    return window, turn + half


def probe_schedule(total: int, every: int = PROBE_EVERY) -> List[Optional[str]]:
    """턴별 프로브 배치 — every턴마다 1번, 나머지는 전부 자연 진행.

    진행/평가 구간 구분 없이 렌이 total턴을 자연스럽게 보내고, every턴째
    발화만 과거를 슬며시 화제로 끌어들인다. 유형은 회전 — 30턴 파일럿이면
    recall·relation·false 3개, 80턴 본런이면 회전 두 바퀴째까지 돈다.
    """
    rotation = ["recall", "relation", "false", "update", "recent"]
    out: List[Optional[str]] = [None] * total
    k = 0
    for i in range(every - 1, total, every):
        out[i] = rotation[k % len(rotation)]
        k += 1
    return out


# 풀 히스토리를 그대로 보내는 변형. dreaming은 창 관리(압축)가 프록시 책임이라
# 벤치가 미리 자르면 프록시가 기억해야 할 턴을 아예 못 본다 — night2에서
# dreaming이 "trim 3회차"가 된 원인 중 하나.
_FULL_HISTORY = ("vanilla", "dreaming")


def wire_history(variant: str, history: List[Dict],
                  window: List[Dict]) -> List[Dict]:
    """변형별 전송 히스토리 — 트림 여부 단일 결정점."""
    return history if variant in _FULL_HISTORY else window


def build_wire(preset: Dict, card: Dict, window: List[Dict],
               memory: str = "") -> List[Dict]:
    """뮈토스 조립 + reformater.

    memory는 hypa 요약 블록 — 프리셋의 memory 카드(promptTemplate[35] 'Past
    Summary') 자리에 들어간다. 즉 chat 히스토리보다 앞, 시스템 프롬프트
    한가운데다 (index.svelte.ts:1429-1443). 캐시 파괴 병리의 구조적 원인이다.
    """
    msgs = assemble(preset, TOGGLES, window, memory=memory,
                    card={"description": card.get("description", ""),
                          "persona": card.get("persona", ""),
                          "lore": card.get("lore", []),
                          "system_prompt": card.get("system_prompt", ""),
                          "replace_globalnote":
                              card.get("replace_globalnote", ""),
                          "authornote": card.get("authornote", ""),
                          "post_everything": card.get("post_everything", "")},
                    char_name=card.get("name", ""),
                    user_name=card.get("user_name", ""))
    # 캡처 확인: Custom API 경로는 hasFullSystemPrompt + requiresAlternateRole
    # 해제로 동작한다 — 중간 system(req-005의 2534자)도, 연속 user(req-006)도
    # 그대로 실린다. 둘 다 끄지 않으면 우리 와이어만 다른 모양이 된다.
    return reformat(msgs, fold_mid_system=False, alternate=False)


def _call_upstream(variant: str, session: str, key: str,
                   msgs: List[Dict]) -> Dict:
    """일시 오류(5xx·타임아웃)는 재시도 — 한 번의 502가 100턴 런을 죽였다
    (night2-drm 실측: 프록시 업스트림 ReadTimeout → 502 → 즉사)."""
    for attempt in range(3):
        try:
            return _call_upstream_once(variant, session, key, msgs)
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            if (isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code < 500):
                raise                          # 4xx는 우리 잘못 — 즉시 전파
            if attempt == 2:
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("unreachable")


def _call_upstream_once(variant: str, session: str, key: str,
                        msgs: List[Dict]) -> Dict:
    t0 = time.time()
    if variant == "dreaming":
        r = httpx.post(PROXY + "/v1/chat/completions", timeout=300,
                       headers={"x-dreaming-session-id": session},
                       json={"model": MODEL, "max_tokens": MAX_TOKENS,
                             "messages": msgs})
    else:
        r = httpx.post(UPSTREAM + "/chat/completions", timeout=300,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "max_tokens": MAX_TOKENS,
                             "messages": msgs, "usage": {"include": True}})
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {})
    cached = det.get("cached_tokens", 0) or u.get("prompt_cache_hit_tokens", 0)
    choice = d["choices"][0]
    # content는 None일 수 있다 (프로바이더 필터 등) — 빈 응답은 리롤 게이트가
    # language_drift로 걷어내도록 ""로 강제한다 (실측: pilot80b가 None에 죽음)
    return {"reply": choice["message"]["content"] or "",
            "finish": choice.get("finish_reason", ""),
            "prompt": u.get("prompt_tokens", 0), "cached": cached,
            "completion": u.get("completion_tokens", 0),
            "cost": u.get("cost", 0.0), "sec": round(time.time() - t0, 1)}


# 실유저가 리롤로 걷어내는 응답 — 남겨두면 디렉터가 캐릭터 대사를 지어내며
# 사칭하기 시작한다 (파일럿50 T3 거부 → T4 디렉터가 소연 대사 작성).
_REFUSAL_MARKS = ("죄송합니다만", "처리할 수 없습니", "수행할 수 없습니",
                  "I cannot", "I can't", "I'm not able to")
_HANGUL = re.compile(r"[가-힣]")

_LOOP_LOOKBACK = 3      # 직전 몇 개 응답과 비교할지
_LOOP_RATIO = 0.97      # 이 이상이면 사실상 동일 (실측: 972자/1159자 완전일치)


def reply_flaw(reply: str, prior_replies: Sequence[str] = ()) -> str:
    """리롤 사유. 정상이면 빈 문자열.

    한글 비율 임계 0.3: 파일럿 실측에서 병리 턴(영어 드리프트·프리셋 지시문
    에코)은 전부 0.09 이하, 정상 턴은 전부 0.64 이상 — 사이가 비어 있다.
    loop: 직전 lookback개 응답과 SequenceMatcher ratio>=0.97 — 실측(trim
    런 T85=T86, T91=T92) 완전 동일 응답 재현 방지.
    """
    if any(m in reply for m in _REFUSAL_MARKS):
        return "refusal"
    if len(_HANGUL.findall(reply)) / max(len(reply), 1) < 0.3:
        return "language_drift"
    for prior in prior_replies[-_LOOP_LOOKBACK:]:
        if difflib.SequenceMatcher(None, reply, prior).ratio() >= _LOOP_RATIO:
            return "loop"
    return ""


def reroll_until_clean(call: Callable[[], Dict],
                        prior_replies: Sequence[str] = (),
                        max_rerolls: int = 2) -> Tuple[Dict, List[str]]:
    """flaw 있으면 재호출 최대 max_rerolls회. 반환: (최종 st, 시도별 flaw 이력).

    flaw_history[0]은 첫 시도, 이후는 리롤 시도 순 — 폐기된 세대의 사유도
    남긴다 (이전엔 최종 flaw만 남아 리롤 원인 분석이 불가능했다).
    prior_replies는 직전 턴 응답들 — 중복 응답(loop) 판정에 쓴다.
    """
    st = call()
    flaw = reply_flaw(st["reply"], prior_replies)
    flaw_history = [flaw]
    rerolls = 0
    while flaw and rerolls < max_rerolls:
        st2 = call()
        st2["cost"] += st["cost"]
        st = st2
        rerolls += 1
        flaw = reply_flaw(st["reply"], prior_replies)
        flaw_history.append(flaw)
    st["rerolls"], st["flaw"], st["flaw_history"] = rerolls, flaw, flaw_history
    return st, flaw_history


def abort_reroll_count(flaw_history: Sequence[str]) -> int:
    """중단 게이트(MAX_RUN_REROLLS)에 누적할 리롤 수.

    마지막 항목은 최종 상태지 리롤이 아니므로 제외한다. "loop"은 거부
    반복(비용 소각)과 다른 병리라 게이트 오탐을 막기 위해 카운트에서 뺀다.
    """
    return sum(1 for f in flaw_history[:-1] if f != "loop")


_DIRECT_SYS = ("너는 RP에서 유저(1인칭{user}) 역할을 연기한다. 작품 "
               "설정과 직전 장면에 자연스럽게 이어지는 유저 발화 하나만 출력. "
               "3문장 이내, 메타 발언 금지. 상대는 연상이자 신비한 존재다 — "
               "정중한 존댓말을 쓴다 (반말 금지). 상대 캐릭터의 대사나 "
               "행동을 네가 대신 쓰지 마라 — 유저 자신의 말과 행동만.\n"
               "[작품 설정]은 배경 이해용이다 — 대화에서 아직 드러나지 않은 "
               "정보(호칭·직함·이름·과거사·신체 특징)를 네가 먼저 입에 올리지 "
               "마라. 상대가 말해주기 전까지 모르는 사람으로 산다. "
               "(파일럿 실측: '신녀님' 호칭을 대화에 나온 적 없는데 선취했다)\n"
               "장면에 새 인물이 등장하면 실제 유저처럼 호기심을 갖고 "
               "상호작용하라 — 등장한 인물을 이유 없이 무시하거나 서둘러 "
               "퇴장시키지 마라. (실측: 등장한 조연을 한 턴 만에 흘려보냈다)")
_UPDATE_BEAT = ("이번 발화에서 이전에 언급된 수치나 소지품 상태를 명확히 바꾸는 "
                "행동을 한다 (지불, 획득, 분실 중 하나). 새 값이 드러나게.")
# 5턴마다 이야기를 미는 지시 — 없으면 디렉터가 같은 장면을 맴돈다
# (30턴 파일럿 실측: 한 장소 하룻밤에서 정체). 회전이라 런마다 결이 달라진다.
_BEATS = ("장면이나 장소를 바꾸는 행동을 한다 — 밖으로 나가자고 하거나, "
          "다른 공간으로 옮기거나, 산책을 청한다.",
          "새로운 화제나 작은 사건을 꺼낸다 — 마을, 소문, 상대의 과거, "
          "앞으로의 계획 중 하나.",
          "시간을 흘려보낸다 — 다음 날이나 몇 시간 뒤로 넘어갔음이 발화에 "
          "드러나게 한다.",
          "가벼운 갈등이나 의견 차이를 만든다 — 금방 풀 수 있는 수준으로.",
          "구체적인 수치·이름·약속이 나올 만한 행동을 한다 — 거래, 날짜 잡기, "
          "무언가를 세거나 값을 치르기.")


_NPC_BEAT = (f"{NPC_NAME}(이)가 자연스럽게 장면에 합류할 상황을 만든다 — "
             f"찾아가거나, 우연히 마주치거나, 이름을 언급하며 소식을 묻는다. "
             f"'{NPC_NAME}'이라는 이름은 말해도 되지만 그 외 설정은 지어내지 "
             f"마라. 장면의 중심은 계속 위지소연이다 — {NPC_NAME}은 곁가지로만.")


def pick_beat(i: int, npc_due: bool = False) -> str:
    """턴 i의 필러 지시. NPC 이벤트 > UPDATE_EVENTS > 5턴 주기 비트 > 평서."""
    if npc_due:
        return _NPC_BEAT
    if i in UPDATE_EVENTS:
        return _UPDATE_BEAT
    if i % 5 == 4:
        return _BEATS[(i // 5) % len(_BEATS)]
    return "자연스럽게 이어간다."


def recent_dialogue(history: List[Dict], pairs: int = 3) -> str:
    """디렉터에게 주는 최근 대화 — 직전 응답 하나만 주면 맥락 없이 맴돈다."""
    tail = history[-(pairs * 2):]
    return "\n\n".join(
        f"[{'렌' if m['role'] == 'user' else '캐릭터'}]\n{m['content'][-600:]}"
        for m in tail)


def _load_json(path: str) -> Dict:
    return json.loads(pathlib.Path(path).read_text())


def hypa_in_window(fact_turn: int, kept_start_msg: int,
                   has_greeting: bool) -> bool:
    """hypa가 실제로 보낸 창에 턴 fact_turn의 발화가 남아 있는가.

    hypa는 턴이 아니라 **메시지 인덱스**로 자른다 (chats.slice(startIdx),
    hypav3.ts:934). greeting이 있으면 턴 t의 user 메시지는 인덱스 1+2t다.
    """
    return (1 if has_greeting else 0) + 2 * fact_turn >= kept_start_msg


def run_once(preset_path: str, card_path: str, variant: str, session: str,
             run_no: int, max_context: int, reroll_at: List[int],
             edit_at: List[int], ttl_wait: bool,
             total_turns: int = TURNS,
             probe_every: int = PROBE_EVERY) -> Dict:
    preset = decode_risup(preset_path)
    card = _load_json(card_path)
    key = _key()
    director = make_director_llm()
    judge = make_judge_llm()
    ledger = Ledger()
    history: List[Dict] = []
    if card.get("greeting"):
        history.append({"role": "assistant", "content": card["greeting"]})
    last_reply = card.get("greeting") or "(첫 장면)"
    few_shot = "\n".join(f"- {s}" for s in card.get("style_examples", [])[:8])
    uname = card.get("user_name", "")
    dir_sys = _DIRECT_SYS.format(user=f", 이름 {uname}" if uname else "·무명")
    if few_shot:
        dir_sys += f"\n[실제 유저 발화 예시 — 문체 참고]\n{few_shot}"
    turns, probes = [], []
    total_rerolls, aborted = 0, ""
    sched = probe_schedule(total_turns, probe_every)
    # hypa 상태 — 요약은 세션 안에서 누적된다 (data.summaries).
    hypa_S = hypa.load_hypa_settings(HYPA_EXPORT) if variant == "hypa" else None
    hypa_data: Dict = {"summaries": []}
    hypa_state = EVAL_DIR / f"hypa-state-{session}.json"
    hypa_cost0 = hypa.SUMMARY_COST
    hypa_truncated0 = hypa.SUMMARY_TRUNCATED
    # 프리셋+카드 고정 비용 — RisuAI가 히스토리 앞에 먼저 태우는 몫
    # (index.svelte.ts:614-618). 빈 히스토리 와이어로 실측한다.
    fixed_tokens = (sum(hypa.tok_chat(m) for m in build_wire(preset, card, []))
                    if variant == "hypa" else 0)
    # trim 예산 = 공유 풀 - 프리셋/카드 고정 비용 - 응답 예약(maxResponse+50)
    # (index.svelte.ts:614,618) — hypa와 같은 실측 패턴, 카운터만 o200k.
    trim_budget = (max_context
                   - sum(_count(m["content"])
                        for m in build_wire(preset, card, []))
                   - MAX_TOKENS - 50) if variant == "trim" else max_context
    kept_start_msg = 0

    for i in range(total_turns):
        ptype = sched[i]
        fact, wrong = None, ""
        t_dir = time.time()
        if ptype:
            plan = probe_plan(ledger, i, {ptype: 1})
            if plan:
                _, fact = plan[0]
        if fact is not None and ptype == "false":
            utext, wrong = make_false_premise(director, fact,
                                              scene=last_reply, style=few_shot)
        elif fact is not None:
            utext = make_probe(director, fact,
                               scene=last_reply, style=few_shot)
        else:
            ptype = None                       # eligible 없으면 필러로 강등
            ctx = recent_dialogue(history) or f"[캐릭터]\n{last_reply[-600:]}"
            # T41에 무조건 한 번, T45까지는 히스토리에 이름 없을 때만 재유도
            npc_due = (i == NPC_EVENT_TURN
                       or (NPC_EVENT_TURN < i <= NPC_EVENT_RETRY
                           and not any(NPC_NAME in m["content"]
                                       for m in history)))
            utext = director(
                dir_sys + f"\n[작품 설정]\n{card.get('description', '')[:2000]}",
                f"[최근 대화]\n{ctx}\n[지시]\n{pick_beat(i, npc_due)}")
        dir_sec = round(time.time() - t_dir, 1)

        history.append({"role": "user", "content": utext})
        memory_text = None
        if variant == "hypa":
            # hypa는 token_trim이 아니라 자기 slice로 자른다 — window/win_start
            # 는 이 분기에서 안 읽힌다. 매 턴 전체 히스토리 토큰 카운트를
            # 태우는 데드워크라 건너뛴다.
            (memory_text, use_window, kept_start_msg, hypa_data,
             herr) = hypa.hypa_step(history, fixed_tokens, hypa_S, hypa_data,
                                    hypa._summarize_call, max_context,
                                    MAX_TOKENS)
            hypa_state.parent.mkdir(parents=True, exist_ok=True)
            hypa_state.write_text(json.dumps(hypa_data, ensure_ascii=False,
                                             indent=1))
            if herr:
                # 원본도 요청 자체를 실패시킨다 (hypav3.ts:263-274) — 병리 재현이지
                # 하네스 버그가 아니다. 부분 결과를 저장하고 중단한다.
                turns.append({"turn": i, "user": utext, "hypa_error": herr,
                              "cost": 0.0})
                aborted = f"hypa 요약 불가 T{i + 1}: {herr}"
                break
        else:
            window, win_start = token_trim(history, trim_budget)
            use_window = wire_history(variant, history, window)
        msgs = build_wire(preset, card, use_window, memory=memory_text or "")
        bad = check_wire_shape(msgs)
        if bad:
            raise SystemExit(f"와이어 형태 위반 T{i + 1}: {bad}")
        # 품질 리롤: 거부·언어 드리프트·중복 응답(loop)은 실유저가 리롤로
        # 걷어내는 응답이다. 남기면 이후 턴 전체가 오염된다 (디렉터 사칭·
        # 영어 고착·자기표절).
        prior_replies = [m["content"] for m in history[-6:]
                          if m["role"] == "assistant"]
        st, flaw_history = reroll_until_clean(
            lambda: _call_upstream(variant, session, key, msgs), prior_replies)
        total_rerolls += abort_reroll_count(flaw_history)
        history.append({"role": "assistant", "content": st["reply"]})
        last_reply = st["reply"]

        if i in reroll_at:                     # 리롤: 동일 요청 재전송
            st2 = _call_upstream(variant, session, key, msgs)
            history[-1] = {"role": "assistant", "content": st2["reply"]}
            last_reply = st2["reply"]
            st["cost"] += st2["cost"]
            st["reply"] = st2["reply"]         # 기록·추출은 히스토리와 같게
        if i in edit_at:                       # 수정: user 텍스트 바꿔 재전송
            history[-2]["content"] = utext + " (아니, 정정할게.)"
            if variant == "hypa":
                # 같은 턴의 재전송 — 요약을 다시 돌리지 않는다 (memo는 인덱스
                # 기반이라 편집에도 불변, 창도 그대로다). token_trim은 여기서
                # 안 쓰이는 데드워크라 건너뛴다.
                use_window = history[:-1][kept_start_msg:]
            else:
                window, _ = token_trim(history[:-1], trim_budget)
                use_window = wire_history(variant, history[:-1], window)
            msgs2 = build_wire(preset, card, use_window,
                               memory=memory_text or "")
            st3 = _call_upstream(variant, session, key, msgs2)
            history[-1] = {"role": "assistant", "content": st3["reply"]}
            last_reply = st3["reply"]
            st["cost"] += st3["cost"]
            st["reply"] = st3["reply"]

        t_ext = time.time()
        if ptype is None:
            ledger.add(extract_facts(director, utext, st["reply"], i))
        ext_sec = round(time.time() - t_ext, 1)

        turns.append({"turn": i, "user": utext, **st,
                      "sec_director": dir_sec, "sec_extract": ext_sec,
                      "ptype": ptype})
        if fact is not None:
            o = oracle_pass(st["reply"], fact.value, wrong_value=wrong,
                            char_name=card.get("name", ""))
            j = judge_pass(judge, ptype, fact.text, fact.value, utext,
                           st["reply"], wrong_value=wrong)
            miss = "-"
            # judge 파싱 실패(None)는 미스가 아니다 — 원인 분해에서 뺀다
            if j["pass"] is False and variant == "dreaming":
                miss = decompose_miss(DATA, session, fact)
            probes.append({"turn": i, "ptype": ptype, "fact": fact.text,
                           "value": fact.value, "wrong": wrong,
                           "question": utext,
                           "reply": st["reply"], "oracle": o,
                           "judge": j["pass"], "why": j["why"],
                           "miss_cause": miss,
                           "distance_turns": i - fact.turn,
                           # 나레이터가 실제 본 창 기준 — 창내 실패=LITM,
                           # 창밖 실패=eviction. 풀 히스토리 변형은 항상 창내
                           # (dreaming은 프록시가 압축했을 수 있어 상한값이다).
                           # hypa는 token_trim이 아니라 자기 slice로 자른다.
                           "in_window": (
                               variant in _FULL_HISTORY
                               or (hypa_in_window(fact.turn, kept_start_msg,
                                                  bool(card.get("greeting")))
                                   if variant == "hypa"
                                   else fact.turn >= win_start))})
        if variant == "dreaming" and i in (total_turns // 3,
                                           2 * total_turns // 3):
            time.sleep(12)                     # 꿈 트리거 (유휴 Dreamer)
        if ttl_wait and i % 10 == 9:
            time.sleep(305)                    # TTL 5m 만료 재현 (옵션)
        if total_rerolls >= MAX_RUN_REROLLS:
            aborted = (f"누적 리롤 {total_rerolls}회 (T{i + 1}) — "
                       f"프로바이더 거부 반복, 런 중단")
            break

    passed = sum(1 for p in probes if p["judge"] is True)
    unparsed = sum(1 for p in probes if p["judge"] is None)
    result = {"variant": variant, "session": session, "run": run_no,
              "model": MODEL, "turns": turns, "probes": probes,
              "ledger": ledger.to_rows(),
              "totals": {"probes": len(probes), "judge_pass": passed,
                         "judge_unparsed": unparsed,
                         "oracle_pass": sum(1 for p in probes if p["oracle"]),
                         # 절단은 기억 실패로 오인된다 — 0인지 매 런 확인한다
                         "truncated": sum(1 for t in turns
                                          if t.get("finish") == "length"),
                         "rerolls": sum(t.get("rerolls", 0) for t in turns),
                         # 리롤 2회로도 못 걷어낸 병리 턴 — 0이어야 한다
                         "flawed": sum(1 for t in turns if t.get("flaw")),
                         "cost": round(sum(t["cost"] for t in turns), 4),
                         "cost_director": round(director.cost, 4),
                         "director_calls": director.calls,
                         "cost_judge": round(judge.cost, 4),
                         "judge_calls": judge.calls,
                         # hypa 요약 콜 — 모듈 전역 누적이라 런 시작 대비 증분
                         "cost_hypa": round(hypa.SUMMARY_COST - hypa_cost0, 4),
                         "hypa_truncated": hypa.SUMMARY_TRUNCATED - hypa_truncated0,
                         "aborted": aborted}}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"v2-{session}-run{run_no}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("card")
    ap.add_argument("variant",
                    choices=("dreaming", "vanilla", "trim", "hypa"))
    ap.add_argument("--session", required=True)
    ap.add_argument("--runs", type=int, default=1)
    # --trim-tokens는 하위호환 별칭 — 같은 단일 풀(maxContext)이다
    ap.add_argument("--max-context", "--trim-tokens", dest="max_context",
                    type=int, default=MAX_CONTEXT)
    ap.add_argument("--turns", type=int, default=TURNS)
    ap.add_argument("--probe-every", type=int, default=PROBE_EVERY)
    ap.add_argument("--reroll-at", default="18,33")
    ap.add_argument("--edit-at", default="25")
    ap.add_argument("--ttl-wait", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    reroll = [int(x) for x in args.reroll_at.split(",") if x]
    edit = [int(x) for x in args.edit_at.split(",") if x]
    for n in range(args.runs):
        sess = f"{args.session}-r{n}"
        d = DATA / sess
        if d.exists():
            if not args.reset:
                raise SystemExit(f"{d} 이미 있음 — --reset")
            shutil.rmtree(d)
        r = run_once(args.preset, args.card, args.variant, sess, n,
                     args.max_context, reroll, edit, args.ttl_wait,
                     args.turns, args.probe_every)
        t = r["totals"]
        grand = (t["cost"] + t.get("cost_director", 0) + t.get("cost_judge", 0)
                 + t.get("cost_hypa", 0))
        print(f"[run{n}] {t['judge_pass']}/{t['probes']} "
              f"나레이터 ${t['cost']} + 디렉터 ${t.get('cost_director', 0)} "
              f"+ judge ${t.get('cost_judge', 0)} "
              f"+ hypa ${t.get('cost_hypa', 0)} = ${round(grand, 4)}",
              flush=True)
        if t.get("aborted"):
            # 부분 결과는 이미 저장됨 — 비정상 종료로 상위 스크립트에 알린다
            raise SystemExit(f"런 중단: {t['aborted']}")


if __name__ == "__main__":
    main()
