# summary
P1 4건 모두 benchmarks/eval/run2.py·director.py·report2.py·viewer.py 안에서 기존 순수 함수 패턴(token_trim, reply_flaw, probe_plan, window_split처럼 run_once 밖으로 뺀 테스트 가능 함수)을 그대로 따라가면 구현 가능하다. (1) value-in-window는 probes.append 직전(run2.py:442-452)에 새 헬퍼로 필드 하나 추가 + report2.aggregate에 별도 집계 함수 신설, 기존 window_split은 손대지 않아 test_report_splits_by_window(758줄) 등 기존 테스트 불변. (2)(3) 리롤 게이트(run2.py:393-403)는 reply_flaw만 있고 시도별 이력이 없다 — reroll_until_clean이라는 순수 함수로 추출하면서 flaw_history를 같이 얻고, 동시에 reply_flaw에 prior_replies 파라미터(디폴트 빈 튜플)를 추가해 SequenceMatcher 기반 "loop" 판정을 넣는 것이 최소 변경이다(두 항목이 같은 코드 블록을 건드리므로 반드시 같이 구현). 기존 reply_flaw 단항 호출 테스트 2건은 디폴트 인자로 그대로 통과한다. (4)는 코드 로직 변경이 아니라 전부 프롬프트 문자열(_PROBE_SYS, _EXTRACT_SYS) 수정 + (b)만 검증용 순수 함수 _probe_mentions_fact_object 신설 — 실측 T49 드리프트 사례("저고리"→"옷감")로 재현 가능한 테스트를 만들 수 있음을 확인했다. 기존 72개 테스트는 모두 기존 어서션 문자열이 그대로 남아 있는 한 깨지지 않는다(치환이 아니라 append 방식으로 편집해야 함).

# spec

# P1 하네스 수정 4건 — 구현 스펙 (2026-08-09 야간 분석 근거)

근거 문서: `dreaming_data/eval/night2-deep-analysis.md`(백로그 표, P1 항목 3~6), `dreaming_data/eval/night2-summary.md`. 아래 4건은 그 백로그의 정확한 코드 매핑이다. **본 출력은 스펙이며, 실제 파일은 편집하지 않았다.**

공통 원칙: 이 코드베이스는 `run_once` 안의 로직을 순수 함수로 빼서 개별 테스트하는 패턴을 이미 쓰고 있다(`token_trim`, `probe_schedule`, `reply_flaw`, `pick_beat`, `recent_dialogue`가 전부 이렇게 테스트됨). 4건 모두 이 패턴을 따른다.

---

## 1. in_window 값-생존 보강

**파일**: `benchmarks/eval/run2.py` (기록 지점) + `benchmarks/eval/report2.py` (집계 지점)

**위치**: `run2.py:442-452`의 `probes.append(...)` 블록, `report2.py:18-93`의 `window_split`/`aggregate`.

**문제**: `in_window`은 `fact.turn >= win_start`만 본다(run2.py:450-452). 원본 턴이 evict돼도 이후 서사가 같은 값을 반복 언급하면 실제로는 "안 잊었다"는 착시가 생긴다. 실측: retrieval 런 창밖 4/4 전부 통과했는데, "렌" 452회·"15년" 10회가 창 안에 살아있었을 뿐 검색 발췌 때문이 아니었다(night2-deep-analysis.md:14).

**전후 코드 — run2.py**:

```python
# BEFORE (run2.py:449-452)
                           # 나레이터가 실제 본 창 기준 — 창내 실패=LITM,
                           # 창밖 실패=eviction. vanilla는 항상 창내.
                           "in_window": (variant == "vanilla"
                                         or fact.turn >= win_start)})

# AFTER
                           # 나레이터가 실제 본 창 기준 — 창내 실패=LITM,
                           # 창밖 실패=eviction. vanilla는 항상 창내.
                           "in_window": (variant == "vanilla"
                                         or fact.turn >= win_start),
                           # 원본 턴은 evict돼도 서사 반복으로 값이 창 안에
                           # 남을 수 있다 (실측: retrieval "렌" 452회,
                           # "15년" 10회 — night2-deep-analysis.md). 문자열
                           # 완전일치만 본다 — "250"/"이백오십" 같은 한글
                           # 표기 변형은 놓친다 (알려진 한계, 과소탐지 방향).
                           "value_in_window": any(
                               fact.value in m["content"]
                               for m in use_window)})
```

`use_window`는 이미 388행에서 `use_window = history if variant == "vanilla" else window`로 정의돼 있어 재사용만 하면 된다 — 새 헬퍼 함수도 불필요.

**전후 코드 — report2.py** (기존 `window_split`은 건드리지 않고 병렬 함수 신설 — `test_report_splits_by_window`가 `window_split`의 정확한 반환 형태를 검증하므로):

```python
# AFTER (report2.py, window_split 옆에 추가)
def value_survival(probes: List[Dict]):
    """창밖(evicted) 프로브 중 value_in_window True 비율 — 통과가 진짜
    기억 때문인지 서사 반복 때문인지 가늠하는 오염 지표.

    (오염 의심 pass/n, 창밖 전체 pass/n). 구 JSON은 value_in_window 부재라
    보수적으로 False(생존 증거 없음) 취급한다.
    """
    out = [p for p in probes
           if not p.get("in_window") and p["judge"] is not None]
    survived = [p for p in out if p.get("value_in_window")]

    def _p(xs):
        return (sum(1 for p in xs if p["judge"] is True), len(xs))
    return _p(survived), _p(out)
```

`aggregate()`(33-93행) 안, 기존 `inw, outw = window_split(...)` 바로 아래에 `vs_survived, vs_out = value_survival(...)`를 추가하고 반환 dict에 `"value_survival": {"survived": vs_survived, "out": vs_out}` 키 하나를 얹는다. `render()`에는 "창밖 통과 중 값 생존 오염 의심: x/y" 한 줄만 추가.

**viewer.py**: 프로브 카드(136-157행)의 "창내/창밖" 배지 옆(144행)에 `p.get("value_in_window")`가 True면 "값 생존" 배지 하나 추가 — 렌더 전용, 로직 없음.

**새 테스트 케이스 초안**:
```python
def test_value_survival_flags_evicted_but_repeated_value():
    from benchmarks.eval.report2 import value_survival
    probes = [
        {"judge": True, "in_window": False, "value_in_window": True},   # 오염 의심
        {"judge": False, "in_window": False, "value_in_window": False}, # 진짜 evict 실패
        {"judge": True, "in_window": True, "value_in_window": True},    # 창내 — 분모 제외
    ]
    survived, out = value_survival(probes)
    assert survived == (1, 1) and out == (1, 2)

def test_value_survival_defaults_missing_key_to_not_survived():
    from benchmarks.eval.report2 import value_survival
    probes = [{"judge": True, "in_window": False}]   # 구 JSON — value_in_window 없음
    survived, out = value_survival(probes)
    assert survived == (0, 0) and out == (1, 1)
```

**기존 테스트 영향**: `window_split`을 그대로 두므로 `test_report_splits_by_window`(758-767행) 불변. `test_aggregate_*` 계열(480-513, 667-675행)은 `value_in_window` 키가 없는 `_res()` 픽스처를 쓰는데, `value_survival`이 `.get()`으로 안전하게 처리하므로 `aggregate()` 반환 dict에 새 키가 하나 추가될 뿐 기존 키 값은 그대로라 깨지지 않는다.

---

## 2. 리롤 사유 기록 (flaw_history)

**파일**: `benchmarks/eval/run2.py`

**위치**: `run2.py:396-403`의 리롤 while 루프. **3번(중복 게이트)과 같은 블록을 건드리므로 두 항목은 한 커밋에서 같이 구현해야 한다** — 아래 코드는 2번과 3번을 합친 최종형이다.

**문제**: `flaw` 변수가 매 시도 덮어써져 `st["flaw"]`에는 최종 결과만 남는다. 폐기된 세대(리롤로 버려진 응답)가 왜 버려졌는지 기록이 없다.

**전후 코드**:

```python
# BEFORE (run2.py:393-404)
        st = _call_upstream(variant, session, key, msgs)
        # 품질 리롤: 거부·언어 드리프트는 실유저가 리롤로 걷어내는 응답이다.
        # 남기면 이후 턴 전체가 오염된다 (디렉터 사칭·영어 고착).
        rerolls, flaw = 0, reply_flaw(st["reply"])
        while flaw and rerolls < 2:
            st2 = _call_upstream(variant, session, key, msgs)
            st2["cost"] += st["cost"]
            st = st2
            rerolls += 1
            flaw = reply_flaw(st["reply"])
        st["rerolls"], st["flaw"] = rerolls, flaw
        total_rerolls += rerolls

# AFTER — reroll_until_clean으로 추출 (run2.py 상단, token_trim 근처에 신설)
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

# run_once 호출부 (393-404행 대체)
        prior_replies = [m["content"] for m in history[-6:]
                         if m["role"] == "assistant"]
        st, _ = reroll_until_clean(
            lambda: _call_upstream(variant, session, key, msgs), prior_replies)
        total_rerolls += st["rerolls"]
```

`turns.append({"turn": i, "user": utext, **st, ...})`(430-432행)는 그대로 둔다 — `st`에 `flaw_history` 키가 이미 있으므로 스프레드로 자동 전파된다.

**import 추가**: `from typing import ... Sequence`(31행 typing import에 `Sequence` 추가).

**새 테스트 케이스 초안**:
```python
def test_reroll_records_flaw_history_for_each_attempt():
    from benchmarks.eval.run2 import reroll_until_clean
    replies = iter([
        {"reply": "죄송합니다만, 처리 못 합니다.", "cost": 0.1},
        {"reply": "The house was quiet in the night.", "cost": 0.1},
        {"reply": "정상적인 한국어 산문 응답입니다. 안녕하세요.", "cost": 0.1},
    ])
    st, hist = reroll_until_clean(lambda: next(replies))
    assert hist == ["refusal", "language_drift", ""]
    assert st["rerolls"] == 2 and st["flaw"] == ""
    assert st["flaw_history"] == hist

def test_reroll_stops_at_max_with_flaw_history_full_of_same_cause():
    from benchmarks.eval.run2 import reroll_until_clean
    def call():
        return {"reply": "죄송합니다만, 처리할 수 없습니다.", "cost": 0.0}
    st, hist = reroll_until_clean(call, max_rerolls=2)
    assert st["rerolls"] == 2 and len(hist) == 3
    assert all(h == "refusal" for h in hist)
```

**기존 테스트 영향**: `reply_flaw` 단항 호출 테스트 2건(709-721, 728-732행)은 `prior_replies` 디폴트가 `()`라 그대로 통과. `run_once`를 직접 호출하는 테스트는 없으므로(무거운 통합 테스트라 회피돼 있음) 호출부 변경의 영향을 받는 기존 테스트는 없다.

---

## 3. 중복 응답 게이트 (loop)

**파일**: `benchmarks/eval/run2.py`

**위치**: `reply_flaw` 함수(261-271행) — 2번의 `reroll_until_clean`과 함께 구현.

**문제**: trim 런에서 완전 동일 응답 2쌍 실측(T91=T92, 972자 / T85=T86, 1159자 — 직접 JSON 검증 완료). 현 `reply_flaw`는 응답 하나만 보고 판정해 이 병리를 못 잡는다.

**전후 코드**:
```python
# BEFORE (run2.py:261-271)
def reply_flaw(reply: str) -> str:
    """리롤 사유. 정상이면 빈 문자열.

    한글 비율 임계 0.3: 파일럿 실측에서 병리 턴(영어 드리프트·프리셋 지시문
    에코)은 전부 0.09 이하, 정상 턴은 전부 0.64 이상 — 사이가 비어 있다.
    """
    if any(m in reply for m in _REFUSAL_MARKS):
        return "refusal"
    if len(_HANGUL.findall(reply)) / max(len(reply), 1) < 0.3:
        return "language_drift"
    return ""

# AFTER
import difflib   # 파일 상단 import 블록에 추가

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
```

호출부는 2번 스펙의 `reroll_until_clean`에 이미 `prior_replies` 인자로 배선돼 있다(`history[-6:]`에서 assistant 메시지만 필터 — 3쌍 = lookback 3과 일치).

**새 테스트 케이스 초안**:
```python
def test_reply_flaw_catches_near_duplicate_response():
    from benchmarks.eval.run2 import reply_flaw
    prior = ['소연은 찻잔을 내려놓으며 조용히 고개를 끄덕였다. "하룻밤 정도는 괜찮다."']
    dup = '소연은 찻잔을 내려놓으며 조용히 고개를 끄덕였다. "하룻밤 정도는 괜찮다."'
    assert reply_flaw(dup, prior) == "loop"

def test_reply_flaw_ignores_dissimilar_prior_replies():
    from benchmarks.eval.run2 import reply_flaw
    prior = ["전혀 다른 내용의 응답입니다."]
    assert reply_flaw("소연은 찻잔을 내려놓으며 웃었다.", prior) == ""

def test_reroll_until_clean_triggers_on_loop_and_recovers():
    from benchmarks.eval.run2 import reroll_until_clean
    prior = ["동일한 응답 본문입니다 반복 테스트."]
    replies = iter([
        {"reply": "동일한 응답 본문입니다 반복 테스트.", "cost": 0.0},
        {"reply": "이번엔 다른 내용의 새 응답이다.", "cost": 0.0},
    ])
    st, hist = reroll_until_clean(lambda: next(replies), prior)
    assert hist == ["loop", ""] and st["rerolls"] == 1
```

**기존 테스트 영향**: `reply_flaw(text)` 단항 호출 테스트(709-721, 728-732행)는 `prior_replies` 디폴트 `()`라 루프 자체가 안 돌아 영향 없음. `_LOOP_LOOKBACK`/`_LOOP_RATIO`는 신규 상수라 충돌 없음.

**설계 메모(양쪽 검토)**: `difflib.SequenceMatcher`는 O(n·m)이라 972~1159자 응답 3개 비교는 무시할 비용. "완전일치"만 볼지 "유사도 0.97"까지 볼지는 트레이드오프 — 완전일치만 보면 사소한 공백 차이로 새는 loop를 놓치고(실측 사례는 마침 완전일치라 안전), 0.97까지 보면 캐릭터의 반복 모티프(예: "처음" 표현 반복 — trim 런 특색으로 이미 관찰됨, night2-deep-analysis.md:26)를 오탐할 위험이 있다. 0.97은 실측 완전일치 사례를 보수적으로 잡으면서 일반적인 모티프 반복(문장 일부만 겹침)은 비켜가는 값으로 제안하되, 야간 재실행 로그로 오탐률 확인이 필요하다.

---

## 4. 프로브 품질 4종

**파일**: `benchmarks/eval/director.py`

### (a) deixis 금지

**위치**: `_PROBE_SYS`(70-79행). **실측**: retrieval 런 T19 — "...아까 처음 들어왔을 때 나도 모르게 고개를 한참 들고 올려다봤던 것 같은데... 그게 누구였더라." (fact.kind=relation, value="위지소연" — 대화만으론 "그게"가 무엇을 가리키는지 알 수 없음).

```python
# AFTER — _PROBE_SYS 금지 목록에 한 줄 추가 (기존 "금지:" 문단 안, append만)
    "금지: '기억해?' 같은 시험조·퀴즈조, 핵심값을 직접 말하는 것, 뜬금없는 "
    "회상 도입, 그 일의 시점을 단정하는 표현('방금'·'아까'·'어제' 등 — "
    "언제 있었던 일인지 모르는 채로 말한다), 지시대상이 모호한 표현("
    "'그게 누구였더라'·'그거 뭐였지' 등 — 무엇/누구를 가리키는지 이 발화"
    "만 보고 알 수 없는 질문. 사실의 대상을 최소한으로 특정할 수 있는 "
    "실마리를 함께 준다).\n"
```

### (b) 디렉터 값 드리프트 방지

**위치**: `make_probe`(115-117행). **실측**: T49 — fact.text="소연은 분홍색이 섞인 한복 저고리를 입고 있다"(value="분홍색")인데 발화가 "저고리" 대신 "옷감"을 언급 — 대상 명사 치환.

**양쪽 설계**:
- **프롬프트 강화만**: `_PROBE_SYS`에 "사실 문장에 나온 대상은 원문 명사를 그대로 쓴다(동의어·상위어로 바꿔 부르지 마라)" 추가. 비용 0이지만 LLM 패러프레이즈 성향상 재발 가능성이 남는다 — T49 자체가 이미 "존댓말 유지" 같은 다른 지침은 잘 지켰는데도 이 지침 부재로 드리프트했다는 보장이 없어 효과가 검증되지 않는다.
- **생성 후 검증(권장)**: fact.text에서 조사를 뗀 핵심 명사가 발화에 하나도 없으면 `drift_suspected` 플래그를 확률로 기록(하드 차단·재시도는 작은 LLM 기준 무한루프 위험이 있어 최소 구현에서는 로깅만 한다).

```python
# AFTER — director.py에 신설 (make_probe 위)
_PARTICLES = ("에게서", "에서", "으로", "이라서", "이지만", "하고", "까지",
             "부터", "이나", "라도", "은", "는", "이", "가", "을", "를",
             "의", "와", "과", "도", "만", "에", "로")
_STOPWORDS = {"있다", "한다", "했다", "됐다"}


def _strip_particle(word: str) -> str:
    for p in _PARTICLES:
        if word.endswith(p) and len(word) > len(p):
            return word[:-len(p)]
    return word


def _probe_mentions_fact_object(fact: DirFact, utext: str) -> bool:
    """생성된 프로브가 사실의 핵심 대상 명사를 담았는지 대략 확인.

    완벽한 개체명 인식이 아니라 명백한 대상 치환(실측: '저고리'→'옷감')만
    걸러낸다 — 정밀도 낮음, 야간 로그로 재보정 필요.
    """
    words = {_strip_particle(w) for w in re.findall(r"[가-힣]{2,}", fact.text)}
    words -= _STOPWORDS
    words = {w for w in words if w not in fact.value and fact.value not in w}
    return not words or any(w in utext for w in words)


def make_probe(llm: LlmFn, fact: DirFact, scene: str = "",
               style: str = "") -> str:
    utext = llm(_PROBE_SYS, _probe_user(fact, scene, style)).strip()
    return utext   # drift_suspected 플래그는 호출부(run2.probes.append)가
                   # _probe_mentions_fact_object(fact, utext)로 기록한다
```

호출부(run2.py:442-452 probes.append)에 `"drift_suspected": not _probe_mentions_fact_object(fact, utext)` 필드 추가 — 채점을 바꾸지 않고 리포트에서 필터링 가능하게만 한다.

### (c) 시제 앵커

**위치**: `_PROBE_SYS`(70-79행). 시점 단정(방금/아까/어제) 금지와 별개로, "그때/처음에" 같은 막연한 과거 지시어를 반드시 하나 포함하도록 요구해 (a)의 지시대상 모호성도 같이 줄인다.

```python
# AFTER — 허용 문단에 추가
    "허용: 지나가는 혼잣말('그게 뭐였더라…'), 관련된 행동이나 상황 언급으로 "
    "상대의 말을 끌어내기, 부드러운 되물음. 발화에는 '그때'·'처음에' 같은 "
    "막연한 과거 지시어를 하나 반드시 포함한다 — 언제인지 단정하지 않으면서 "
    "과거의 일임은 분명히 한다.\n"
```

### (d) 자기 외모 문항 배제

**위치**: `_EXTRACT_SYS`(16-22행). **실측**: vanilla "회색 눈동자" 151회/98턴, dreaming 99회 — 상시 노출되는 캐릭터 신체 특징을 사실로 추출하면 프로브 자체가 무의미해진다(항상 창내 생존).

```python
# BEFORE (director.py:16-22)
_EXTRACT_SYS = (
    "너는 RP 대화 감독관이다. 방금 턴에서 나중에 기억력 시험에 쓸 수 있는 "
    "구체적 사실만 추출한다. 한 줄에 하나, 형식: kind|핵심값|한 문장 서술.\n"
    "kind는 exact(숫자·고유명사·시각), relation(인물 관계·호칭), "
    "event(약속·사건) 중 하나. 핵심값은 응답에 그대로 나올 법한 명사형 "
    "단어(이름·품명·숫자·장소·시각)여야 한다 — '~하기로 함' 같은 문장형 "
    "값 금지. 추출할 게 없으면 빈 출력. 다른 말 금지.")

# AFTER
_EXTRACT_SYS = (
    "너는 RP 대화 감독관이다. 방금 턴에서 나중에 기억력 시험에 쓸 수 있는 "
    "구체적 사실만 추출한다. 한 줄에 하나, 형식: kind|핵심값|한 문장 서술.\n"
    "kind는 exact(숫자·고유명사·시각), relation(인물 관계·호칭), "
    "event(약속·사건) 중 하나. 핵심값은 응답에 그대로 나올 법한 명사형 "
    "단어(이름·품명·숫자·장소·시각)여야 한다 — '~하기로 함' 같은 문장형 "
    "값 금지. 캐릭터 자신의 외모·신체 특징(눈동자 색·머리색·체형 등)은 "
    "추출하지 않는다 — 나레이션에 상시 노출돼 시험이 무의미하다. "
    "추출할 게 없으면 빈 출력. 다른 말 금지.")
```

**새 테스트 케이스 초안 (4종 통합)**:
```python
def test_probe_prompt_forbids_ambiguous_referent():
    from benchmarks.eval.director import _PROBE_SYS
    assert "지시대상" in _PROBE_SYS

def test_probe_mentions_fact_object_catches_real_drift_case():
    from benchmarks.eval.director import DirFact, _probe_mentions_fact_object
    fact = DirFact(fid="x", kind="exact", value="분홍색",
                   text="소연은 분홍색이 섞인 한복 저고리를 입고 있다.", turn=10)
    drifted = "그때 그 옷감 색이 참 곱다고 생각했는데..."      # 실측 T49 재현
    faithful = "그때 그 저고리 색이 참 곱다고 생각했는데..."
    assert _probe_mentions_fact_object(fact, drifted) is False
    assert _probe_mentions_fact_object(fact, faithful) is True

def test_probe_prompt_requires_vague_past_anchor():
    from benchmarks.eval.director import _PROBE_SYS
    assert "그때" in _PROBE_SYS and "처음에" in _PROBE_SYS

def test_extract_prompt_excludes_self_appearance_facts():
    from benchmarks.eval.director import _EXTRACT_SYS
    assert "외모" in _EXTRACT_SYS and "신체 특징" in _EXTRACT_SYS
```

**기존 테스트 영향**: 모두 append 방식 편집이라 `test_probe_prompt_forbids_time_anchoring`("방금"/"시점"), `test_director_prompts_use_polite_speech`("존댓말" 포함·"반말 채팅체" 미포함), `test_extract_prompt_demands_noun_values`("명사형"/"문장형"), `test_probe_gets_scene_and_style_context`("슬며시"/"시험조") 전부 기존 부분 문자열이 그대로 남아 통과한다. `make_probe` 시그니처는 안 바꾸므로(내부에서 flag 계산 안 함, 호출부가 별도로 계산) `test_probe_gets_scene_and_style_context`(635-647행)의 `seen["sys"], seen["user"]` 캡처 로직도 영향 없음.

---

## 구현 순서 권고

1. Fix 2+3 (같은 코드 블록) → `reroll_until_clean` + `reply_flaw(prior_replies)` 동시 커밋
2. Fix 1 (run2.py 기록 + report2.py 집계, 독립적)
3. Fix 4 (director.py 프롬프트 4종, 코드 로직 없이 순수 텍스트+헬퍼 1개, 가장 저위험)
4. 전체 `python3 -m pytest -q tests/test_eval_v2.py` → 72 + 신규(약 14개) 전부 PASS 확인 후 커밋


# claims
- [fix1-기록지점] in_window은 fact.turn(원본 턴)이 win_start 이상인지만 본다 — 서사 반복으로 값이 살아남는 경우를 못 잡는다
  근거: benchmarks/eval/run2.py:450-452 `"in_window": (variant == "vanilla" or fact.turn >= win_start)`
- [fix1-실측근거] retrieval 런 창밖 4/4 통과가 검색 때문이 아니라 서사 반복 — "렌" 창내 452회, "15년" 10회
  근거: dreaming_data/eval/night2-deep-analysis.md:14 `retrieval 7/9의 창밖 4/4는 검색이 아니라 서사 자체 반복(narrative rehearsal): "렌"은 창내 452회, "15년" 10회`
- [fix1-집계지점] report2.window_split은 in_window/judge만 보는 순수 함수이며 기존 테스트가 이 시그니처를 정확히 검증 중 — 손대면 깨진다
  근거: benchmarks/eval/report2.py:18-30, tests/test_eval_v2.py:758-767 `def test_report_splits_by_window(): ... inw, out = window_split(probes)`
- [fix2-현재구조] 리롤 루프가 flaw 변수를 매 시도 덮어써 최종 flaw만 st["flaw"]에 남는다 — 폐기 세대의 사유는 유실
  근거: benchmarks/eval/run2.py:396-403 `rerolls, flaw = 0, reply_flaw(st["reply"])\n        while flaw and rerolls < 2: ... st["rerolls"], st["flaw"] = rerolls, flaw`
- [fix2-전파경로] turns.append이 st를 **st로 스프레드하므로 st에 flaw_history 키만 추가하면 별도 배선 없이 턴 기록에 자동 포함된다
  근거: benchmarks/eval/run2.py:430-432 `turns.append({"turn": i, "user": utext, **st, ...})`
- [fix3-실측근거] trim 런에서 972자·1159자 완전 동일 응답 쌍이 실측으로 확인됨(T91=T92, T85=T86)
  근거: dreaming_data/eval/v2-night2-trim-r0-run0.json turns[85]==turns[86] (1159자), turns[91]==turns[92] (972자) — python3 검증 완료
- [fix3-훅지점] reply_flaw는 reply 문자열 하나만 받는 순수 함수라 직전 응답들과 비교할 방법이 현재 없다
  근거: benchmarks/eval/run2.py:261-271 `def reply_flaw(reply: str) -> str:`
- [fix4a-현황] _PROBE_SYS는 시점 단정(방금/아까/어제)만 금지하고 지시대상 모호성("그게 누구였더라")은 금지하지 않는다
  근거: benchmarks/eval/director.py:70-79 `금지: ... 그 일의 시점을 단정하는 표현('방금'·'아까'·'어제' 등 ...)`
- [fix4a-실측근거] retrieval 런 T19 프로브가 실제로 지시대상 모호 위반 — "그게 누구였더라"
  근거: dreaming_data/eval/v2-night2-ret-r0-run0.json probes[turn=19].question `...아까 처음 들어왔을 때 나도 모르게 고개를 한참 들고 올려다봤던 것 같은데... 그게 누구였더라.`
- [fix4b-실측근거] T49 프로브가 fact 대상(저고리)을 다른 명사(옷감)로 바꿔치기해 발화 — 디렉터 값 드리프트 실제 사례
  근거: dreaming_data/eval/v2-night2-ret-r0-run0.json probes[turn=49] fact=`소연은 분홍색이 섞인 한복 저고리를 입고 있다` question=`...예전에 네가 골라줬던 옷감 색처럼...`
- [fix4b-생성지점] make_probe는 llm 호출 결과를 그대로 반환하며 사후 검증이 없다
  근거: benchmarks/eval/director.py:115-117 `def make_probe(llm, fact, scene="", style=""): return llm(_PROBE_SYS, _probe_user(fact, scene, style)).strip()`
- [fix4c-근거] night2-deep-analysis 백로그 표가 시제 앵커 강제를 P1로 명시 — T19/T49/T69/T89
  근거: dreaming_data/eval/night2-deep-analysis.md:53 `| 6 | 프로브 품질: deixis 금지·디렉터 값 드리프트 검증·시제 앵커·자기외모 문항 제거 | T19/T49/T69/T89 | P1 |`
- [fix4d-실측근거] vanilla·dreaming 런 모두 캐릭터 외모(회색 눈동자)가 극단적으로 반복 노출 — 자기외모 사실은 상시 생존해 프로브가 무의미해짐
  근거: dreaming_data/eval/night2-deep-analysis.md:19 `표현 고착 극심 — "회색 눈동자" 151회/98턴, 고정 응답 템플릿` / :40 `"회색 눈동자" 99회`
- [fix4d-생성지점] _EXTRACT_SYS는 명사형 값만 요구할 뿐 캐릭터 자기 신체 특징 배제 규칙이 없다
  근거: benchmarks/eval/director.py:16-22 `kind는 exact(...), relation(...), event(...) 중 하나. 핵심값은 ... 추출할 게 없으면 빈 출력.`
- [테스트기준선] 현재 test_eval_v2.py는 72개 테스트로 확인됨 — 4건 구현 후 이 숫자가 줄면 안 된다
  근거: pytest --collect-only 출력 `72 tests collected`

# open_questions
fix4b의 _probe_mentions_fact_object는 실측 1건(T49)만으로 튜닝된 휴리스틱이다 — 야간 재실행에서 오탐/누락률을 별도로 측정해야 임계 로직(조사 제거 방식)을 신뢰할 수 있다.
fix3의 loop 판정 임계(0.97)와 lookback(3)은 trim 런 T85=T86/T91=T92 두 사례만 근거로 정했다 — retrieval/dreaming처럼 서사가 크게 움직이는 변형에서 정상적인 모티프 반복(예: '처음' 모티프)을 오탐하지 않는지 확인이 필요하다.
fix1의 value_in_window가 True인 창밖-통과 프로브를 리포트에서 어떻게 처리할지(그대로 pass로 셀지, 별도 '오염 의심'으로 분리 표기해 judge_pass 집계에서 제외할지)는 스펙에서 로깅만 제안했고 채점 정책 변경 여부는 사용자 결정이 필요하다.
fix4d(자기 외모 배제)를 _EXTRACT_SYS 프롬프트 수정으로 할지 probe 단계 필터로 할지 — 스펙은 프롬프트 수정을 택했지만, LLM이 지침을 무시하고 추출할 가능성에 대한 2차 방어선(probe_plan 단계에서 fact.text가 '눈동자'/'머리' 등 신체 키워드를 포함하면 스킵)을 추가할지는 야간 로그로 재발 여부를 본 뒤 결정하는 편이 안전하다.