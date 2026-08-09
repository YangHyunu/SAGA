import math
import sys
import types

import httpx
import pytest

from benchmarks.eval import hypa

EXPORT = ("/Users/yanghyeon-u/Downloads/뮈토스6.2/🏺뮈토스 프롬프트 하이파/"
          "hypaV3_export_뮈토스 하이파 V5.json")

S = hypa.load_hypa_settings(EXPORT)


def test_hypa_settings_load_from_real_export():
    s = hypa.load_hypa_settings(EXPORT)
    assert s.max_chats_per_summary == 8 and s.query_chat_count == 3
    assert s.memory_tokens_ratio == 0.39
    assert s.recent_memory_ratio == 0.6 and s.similar_memory_ratio == 0.4
    assert len(s.summarization_prompt) > 19000
    assert s.summary_chunk_separator == "\\n\\n"   # export 부재 → 코드 기본값


def test_random_ratio_is_exactly_zero():
    s = hypa.load_hypa_settings(EXPORT)
    assert 1 - s.recent_memory_ratio - s.similar_memory_ratio == 0.0  # IEEE754


def test_to_risu_chats_memo_stable_under_content_edit():
    hist = [{"role": "assistant", "content": "인사"},
            {"role": "user", "content": "원문"},
            {"role": "assistant", "content": "답"}]
    before = [c["memo"] for c in hypa.to_risu_chats(hist)]
    hist[1]["content"] = "원문 (아니, 정정할게.)"     # edit_at 재현
    after = [c["memo"] for c in hypa.to_risu_chats(hist)]
    assert before == after == ["m0", "m1", "m2"]


# --- Task 5: 요약 배칭 · summarize ---

def _chat(role, content, memo=None):
    return {"role": role, "content": content, "memo": memo}


def test_batch_planning_respects_query_chat_count_floor():
    # 마지막 queryChatCount(3)개는 절대 배치에 안 들어간다 (hypav3.ts:293-296).
    # 배치 1개로 target(1980) 아래까지 내려가 에러 없이 끝나는 경로.
    chats = [_chat("user", f"u{i}" * 200, memo=f"m{i}") for i in range(6)]
    batches, _, new_start, err = hypa.plan_batches(chats, 0, 2010, 2000, S)
    assert err is None and new_start == 3
    got = [c["memo"] for b in batches for c in b]
    assert got == ["m0", "m1", "m2"]
    assert not set(got) & {"m3", "m4", "m5"}


def test_batch_planning_floor_returns_error_and_drops_batches():
    # 바닥에 닿았는데 여전히 max_ctx 초과 → 요청 자체가 실패한다
    # (hypav3.ts:263-274). 원본은 계획된 배치를 통째로 버리고 요약을 안 부른다.
    chats = [_chat("user", f"u{i}" * 200, memo=f"m{i}") for i in range(6)]
    batches, remaining, _, err = hypa.plan_batches(chats, 0, 10**9, 100, S)
    assert batches == []                   # 에러면 요약 비용을 쓰지 않는다
    assert err == (f"Cannot summarize further: {remaining} > 100, "
                   f"but minimum 3 messages required.")
    assert remaining == 10**9 - sum(hypa.tok_chat(c) for c in chats[:3])


def test_batch_planning_charges_skipped_chat_tokens():
    # 스킵 챗 토큰도 차감 — 원본 버그 보존 (hypav3.ts:313, §7 #2)
    skip = _chat("system", "[Start a new chat]" * 50, memo="NewChat")
    real = [_chat("user", f"u{i}" * 200, memo=f"m{i}") for i in range(6)]
    b_with, after_with, _, e1 = hypa.plan_batches([skip] + real, 0, 2010, 2000, S)
    b_without, after_without, _, e2 = hypa.plan_batches(list(real), 0, 2010, 2000, S)
    assert e1 is None and e2 is None
    # skip 챗이 배치엔 없어도 그 토큰만큼 current가 더 줄어든다
    assert after_with < after_without
    assert [c["memo"] for c in b_with[0]] == ["m0", "m1", "m2"]


def test_batch_scan_range_exceeds_max_chats_when_skipping():
    # maxChatsPerSummary는 '담긴 개수' 상한 — 스킵이 많으면 스캔은 더 나간다
    # (§7 #3, legacy와 다름)
    skips = [_chat("system", "x", memo="NewChat") for _ in range(5)]
    real = [_chat("user", f"u{i}" * 200, memo=f"m{i}") for i in range(11)]
    batches, _, new_start, err = hypa.plan_batches(skips + real, 0, 4010, 4000, S)
    assert err is None
    assert len(batches[0]) == 8            # 담긴 건 8개
    assert new_start == 13                 # 스캔은 스킵 5 + 유효 8 = 13칸


def test_no_batches_when_current_below_max_ctx():
    # summarizationMode = current_tokens > max_ctx (hypav3.ts:253).
    # target(990)은 넘지만 max_ctx는 안 넘으므로 아무것도 요약하지 않는다.
    # 조기 반환이 없으면 4토큰짜리 배치 하나가 잡혀버린다.
    chats = [_chat("user", "a", memo=f"m{i}") for i in range(4)]
    assert hypa.plan_batches(chats, 0, 995, 1000, S) == ([], 995, 0, None)


def test_over_summarization_guard_stops_before_undershooting_target():
    # 배치 하나로 target(990) 아래까지 내려가버리면 그 배치는 포기한다
    # (hypav3.ts:352-362). 첫 배치(32토큰)로 1030 → 998까지만 내리고,
    # 다음 배치(84토큰)는 914로 과요약이라 안 담는다.
    small = [_chat("user", "a", memo=f"m{i}") for i in range(8)]
    big = _chat("user", "긴 요약 대상 " * 20, memo="m8")
    tail = [_chat("user", "t", memo=f"m{9 + i}") for i in range(3)]
    first = sum(hypa.tok_chat(c) for c in small)
    batches, remaining, new_start, err = hypa.plan_batches(
        small + [big] + tail, 0, 1000 + first - 2, 1000, S)
    assert err is None
    assert len(batches) == 1               # 가드 없으면 big까지 2개가 된다
    assert remaining == 998 and new_start == 8


def test_summarize_message_order_is_user_then_system():
    # parseChatML null 폴백: [user=원문, system=프롬프트] — system이 뒤
    seen = {}

    def send(messages):
        seen["m"] = messages
        return "요약"

    hypa.summarize(send, [_chat("user", "안녕", memo="m0")], S, cache_dir=None)
    assert seen["m"][0]["role"] == "user"
    assert seen["m"][0]["content"].startswith("user: 안녕")   # role: content 라인
    assert seen["m"][1]["role"] == "system"
    assert "<hypa_memory_extraction>" in seen["m"][1]["content"]


def test_summarize_strips_thoughts_block():
    out = hypa.summarize(lambda m: "<Thoughts>사고</Thoughts>실제 요약",
                         [_chat("user", "x", memo="m0")], S, cache_dir=None)
    assert out == "실제 요약"


def test_summarize_disk_cache_skips_second_call(tmp_path):
    # 리런 결정론: 같은 배치는 디스크 캐시에서 나온다 (temperature 0.0과 2중)
    batch = [_chat("user", "캐시 대상", memo="m0")]
    calls = []

    def send(messages):
        calls.append(messages)
        return "캐시된 요약"

    first = hypa.summarize(send, batch, S, cache_dir=tmp_path)
    second = hypa.summarize(send, batch, S, cache_dir=tmp_path)
    assert first == second == "캐시된 요약"
    assert len(calls) == 1
    assert (tmp_path / f"{hypa.summary_cache_key(batch, S)}.txt").exists()


def test_summarize_rejects_empty_response():
    with pytest.raises(hypa.HypaError):
        hypa.summarize(lambda m: "  ", [_chat("user", "x", memo="m0")], S,
                       cache_dir=None)


def test_cache_key_changes_with_summarizer_model(monkeypatch):
    # 캐시는 디스크에 영구 잔류한다 — 모델을 바꾸면 키도 바뀌어야 한다
    from benchmarks.eval import run2
    batch = [_chat("user", "모델 민감도", memo="m0")]
    before = hypa.summary_cache_key(batch, S)
    monkeypatch.setattr(run2, "DIRECTOR_MODEL", "some/other-model")
    assert hypa.summary_cache_key(batch, S) != before


def test_truncated_summary_is_not_cached(tmp_path):
    # max_tokens=8192 이탈의 부작용 — 잘린 요약을 캐시에 쓰면 이후 리런이 전부 오염
    batch = [_chat("user", "절단 대상", memo="m0")]

    def send(messages):
        hypa.mark_truncated()              # finish_reason == "length"
        return "잘린 요약"

    before = hypa.SUMMARY_TRUNCATED
    out = hypa.summarize(send, batch, S, cache_dir=tmp_path)
    assert out == "잘린 요약"
    assert hypa.SUMMARY_TRUNCATED == before + 1
    assert not (tmp_path / f"{hypa.summary_cache_key(batch, S)}.txt").exists()


def _resp(status, body=None):
    req = httpx.Request("POST", "https://up/chat/completions")
    return httpx.Response(status, json=body or {}, request=req)


_OK_BODY = {"choices": [{"message": {"content": "요약"}, "finish_reason": "stop"}],
            "usage": {"cost": 0.0}}


def test_summarize_call_retries_transient_5xx(monkeypatch):
    # 502 한 방이 100턴 런을 죽인다 — run2._call_upstream과 같은 재시도 정책
    from benchmarks.eval import run2
    seq = [_resp(502), _resp(503), _resp(200, _OK_BODY)]
    monkeypatch.setattr(run2, "_key", lambda: "k")
    monkeypatch.setattr(hypa.httpx, "post", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr(hypa.time, "sleep", lambda s: None)
    assert hypa._summarize_call([{"role": "user", "content": "x"}]) == "요약"
    assert seq == []


def test_summarize_call_flags_length_truncation(monkeypatch):
    # 프로덕션 경로가 finish_reason == "length"를 실제로 잡아내는지
    from benchmarks.eval import run2
    body = {"choices": [{"message": {"content": "잘림"}, "finish_reason": "length"}],
            "usage": {"cost": 0.0}}
    monkeypatch.setattr(run2, "_key", lambda: "k")
    monkeypatch.setattr(hypa.httpx, "post", lambda *a, **k: _resp(200, body))
    before = hypa.SUMMARY_TRUNCATED
    assert hypa._summarize_call([{"role": "user", "content": "x"}]) == "잘림"
    assert hypa.SUMMARY_TRUNCATED == before + 1


def test_summarize_call_pins_params_and_accumulates_cost(monkeypatch):
    # max_tokens/temperature/모델/usage.include가 바뀌면 여기서 잡는다 —
    # usage.include 없으면 OpenRouter가 cost를 안 실어 SUMMARY_COST가 항상 0
    from benchmarks.eval import run2
    captured = {}

    def post(url, **kw):
        captured["json"] = kw["json"]
        return _resp(200, {"choices": [{"message": {"content": "요약"},
                                        "finish_reason": "stop"}],
                           "usage": {"cost": 0.0123}})

    monkeypatch.setattr(run2, "_key", lambda: "k")
    monkeypatch.setattr(hypa.httpx, "post", post)
    monkeypatch.setattr(hypa, "SUMMARY_COST", 0.0)
    assert hypa._summarize_call([{"role": "user", "content": "x"}]) == "요약"
    body = captured["json"]
    assert body["model"] == hypa._director_model()
    assert body["max_tokens"] == 8192
    assert body["temperature"] == 0.0
    assert body["usage"] == {"include": True}
    assert hypa.SUMMARY_COST == pytest.approx(0.0123)


def test_summarize_call_reraises_4xx_immediately(monkeypatch):
    from benchmarks.eval import run2
    calls = []
    monkeypatch.setattr(run2, "_key", lambda: "k")
    monkeypatch.setattr(hypa.httpx, "post",
                        lambda *a, **k: calls.append(1) or _resp(400))
    monkeypatch.setattr(hypa.time, "sleep", lambda s: None)
    with pytest.raises(httpx.HTTPStatusError):
        hypa._summarize_call([{"role": "user", "content": "x"}])
    assert len(calls) == 1                 # 재시도 없이 즉시 전파


# --- Task 6: 선택 4단계 · 임베딩 ---

def _summ(text):
    return {"text": text, "chatMemos": [], "isImportant": False}


_KEYWORDS = ("산", "바다", "숲", "성")


def _keyword_embed(texts):
    """내용별로 **구별되는** 벡터를 주는 페이크 임베더 — 키워드 one-hot.

    모든 텍스트에 같은 벡터를 주면 코사인 정렬·가중 CC·RRF가 전부 degenerate
    no-op이 되어 similar 랭킹 로직이 통째로 테스트를 빠져나간다.
    키워드가 하나도 없는 텍스트는 영벡터 → _cosine이 0.0 (결정론적).
    """
    return [[1.0 if kw in t else 0.0 for kw in _KEYWORDS] for t in texts]


def _stok(s):
    """선택 단계가 세는 요약 토큰 — 요약 뒤에 구분자가 붙는다 (hypav3.ts:512)."""
    return hypa.tok_chat({"content": s["text"] + hypa.SUMMARY_SEP})


def _avail_fitting(summaries, fits):
    """similar 예산이 `fits`만 딱 담는 available_tokens.

    random_ratio==0 흡수(§7 #7)로 similar 예산 ≈ available이 된다. 또한 **최신
    요약이 recent 예산을 넘겨야** 후보가 전부 unused인 채로 similar까지 내려온다
    — similar 랭킹을 보려면 필수인 전제라 여기서 못 박는다.
    """
    avail = sum(_stok(s) for s in fits) + 1
    assert math.floor(avail * S.recent_memory_ratio) < _stok(summaries[-1])
    return avail


def test_selection_break_vs_continue_semantics():
    # recent는 예산 초과 시 break — 더 작은 것 안 찾음 (hypav3.ts:553-569, §7 #5)
    small, big = _summ("나"), _summ("가" * 4000)
    # 역순(최신부터) 스캔: big(최신)에서 초과 → break → small(과거)도 못 담음
    sel = hypa.select_summaries([small, big], available_tokens=50,
                                recent_chats=[], S=S, embed=None)
    assert sel == []


def test_similar_absorbs_unused_recent_budget_when_random_zero():
    # random_ratio==0.0 → recent 잔여가 similar 예산에 합산 (hypav3.ts:596-607, §7 #7)
    # recent가 최신 1개만 담고 남긴 예산을 similar가 흡수해 2개째를 담는다
    old = _summ("옛날 요약.\n\n산에 갔다.")
    new = _summ("최근 요약")
    recent = [{"role": "user", "content": "산에 갔던 얘기", "memo": "m9"}]
    sel = hypa.select_summaries(
        [old, new], available_tokens=hypa.tok_chat(
            {"content": new["text"] + "\n\n"}) * 4,     # recent 몫으론 1개만
        recent_chats=recent, S=S, embed=_keyword_embed)
    assert old in sel and new in sel                    # 잔여 흡수로 둘 다


def test_similar_starves_without_absorption():
    # 흡수가 없으면(=similar 몫만으로는) old가 안 들어간다 — 위 테스트의 대조군
    old = _summ("옛날 요약.\n\n산에 갔다.")
    new = _summ("최근 요약")
    avail = hypa.tok_chat({"content": new["text"] + "\n\n"}) * 4
    assert hypa.tok_chat({"content": old["text"] + "\n\n"}) > int(avail * 0.4)


def test_selection_returns_chronological_order():
    # 검색 순위가 아니라 원래 순서로 재정렬 (hypav3.ts:871-874, §7 #8)
    a, b, c = _summ("A" * 10), _summ("B" * 10), _summ("C" * 10)
    sel = hypa.select_summaries([a, b, c], available_tokens=10**6,
                                recent_chats=[], S=S, embed=None)
    assert sel == [a, b, c]


def test_important_break_stops_at_first_overflow():
    # important도 예산 초과 시 break — 뒤의 작은 중요 요약을 건너뛰지 않는다 (§7 #5)
    small, big, later = _summ("중요 1"), _summ("가" * 4000), _summ("중요 3")
    for s in (small, big, later):
        s["isImportant"] = True
    sel = hypa.select_summaries(
        [small, big, later],
        available_tokens=_stok(small) + _stok(later) + 1,   # later도 담길 크기
        recent_chats=[], S=S, embed=None)
    assert sel == [small]              # continue였다면 [small, later]


def test_similar_ranks_by_weighted_query_similarity():
    # 쿼리 가중치 (idx+1)/(n(n+1)/2)/len(subs) — 최신 챗이 더 무겁다
    # (hypav3.ts:676-695). 가중치를 무시하면 두 요약이 동점이 되어 순서가 뒤집힌다.
    forest, mtn = _summ("숲에 갔다.\n\n숲이 깊었다."), _summ("산에 갔다.\n\n산이 높았다.")
    blocker = _summ("성에 대한 잡담. " * 30)      # 최신·거대 → recent가 아무것도 못 담음
    summaries = [forest, mtn, blocker]
    recent = [{"role": "user", "content": "숲 이야기가 궁금해"},    # 가중치 1/3
              {"role": "user", "content": "산 이야기가 궁금해"}]    # 가중치 2/3
    sel = hypa.select_summaries(
        summaries, available_tokens=_avail_fitting(summaries, [mtn]),
        recent_chats=recent, S=S, embed=_keyword_embed)
    assert sel == [mtn]                # 가중치 무시 시 숲이 먼저 → []


def test_similar_folds_chunks_to_parent_without_duplicates():
    # 청크 순위를 부모 요약 단위로 접는다 (RRF) — 같은 요약이 두 번 담기면 안 된다
    sea = _summ("바다에 갔다.\n\n바다가 넓었다.")
    mtn = _summ("산에 갔다.\n\n산이 높았다.")
    forest = _summ("숲에 갔다.\n\n숲이 깊었다.")
    blocker = _summ("성에 대한 잡담. " * 30)
    summaries = [sea, mtn, forest, blocker]
    recent = [{"role": "user", "content": "숲 이야기가 궁금해"},
              {"role": "user", "content": "산 이야기가 궁금해"}]
    sel = hypa.select_summaries(
        summaries, available_tokens=_avail_fitting(summaries, [mtn, forest]),
        recent_chats=recent, S=S, embed=_keyword_embed)
    # 랭킹 산 > 숲 > (무관) — 시간순으로 되돌려 [mtn, forest]. 접기 없이 청크의
    # meta를 그대로 쓰면 산 청크 2개 때문에 [mtn, mtn]이 된다.
    assert sel == [mtn, forest]


def test_similar_prefers_higher_cosine_on_score_tie():
    # 가중 점수가 동점이면 코사인이 높은 쪽이 먼저 삽입된다 (내림차순 정렬 →
    # simple_cc 삽입 순서). 오름차순이면 순서가 뒤집힌다.
    mtn, sea = _summ("산에 갔다."), _summ("바다에 갔다.")
    summaries = [mtn, sea]
    # 한 챗의 두 문단 → 서브쿼리 2개가 동일 가중치(0.5) → 두 요약 점수 동점
    recent = [{"role": "user", "content": "산 이야기\n\n바다 이야기"}]
    sel = hypa.select_summaries(
        summaries, available_tokens=_avail_fitting(summaries, [mtn]),
        recent_chats=recent, S=S, embed=_keyword_embed)
    assert sel == [mtn]


def test_similar_break_skips_smaller_later_candidate():
    # similar도 break — 예산 초과한 상위 요약에서 멈추고, 뒤의 더 작은 요약을
    # 찾아가지 않는다 (hypav3.ts:596-607, §7 #5)
    tiny, mtn = _summ("성."), _summ("산에 갔다.")
    forest_big = _summ("숲 이야기. " * 40)      # 랭킹 2위이면서 예산 초과 + 최신
    summaries = [tiny, mtn, forest_big]
    recent = [{"role": "user", "content": "숲 이야기가 궁금해"},
              {"role": "user", "content": "산 이야기가 궁금해"}]
    sel = hypa.select_summaries(
        summaries, available_tokens=_avail_fitting(summaries, [mtn, tiny]),
        recent_chats=recent, S=S, embed=_keyword_embed)
    assert sel == [mtn]                # continue였다면 [tiny, mtn]


def test_similar_splits_summary_into_chunks_by_separator():
    # 요약은 summary_chunk_separator로 청크 분할된 뒤 랭킹된다 (hypav3.ts:105-116).
    # 분할하면 multi가 청크 3개의 RRF 합산으로 single을 이긴다.
    single = _summ("산.")
    multi = _summ("산길 안내.\n\n바다 얘기.\n\n숲 얘기.")
    summaries = [single, multi]
    recent = [{"role": "user", "content": "산 이야기"}]
    sel = hypa.select_summaries(
        summaries, available_tokens=_avail_fitting(summaries, [multi]),
        recent_chats=recent, S=S, embed=_keyword_embed)
    assert sel == [multi]              # 분할 없이 통짜 1청크면 single이 이겨 [single]


def test_rrf_folds_chunk_ranks_into_parent():
    ranked = ["c1@s1", "c2@s2", "c3@s1"]   # s1 청크 2개 상위
    out = hypa.child_to_parent_rrf(ranked, key=lambda ch: ch.split("@")[1])
    assert out[0] == "s1"                   # 1/61 + 1/63 > 1/62


def test_rrf_k_constant_is_60():
    # k=60 고정 (hypav3.ts:1874-1893). k가 작으면 상위 1개가 하위 2개를 이겨
    # 대소가 뒤집힌다: k=60이면 1/63+1/64 > 1/61, k=1이면 1/4+1/5 < 1/2
    ranked = ["a@s1", "b@s2", "c@s3", "d@s3"]
    out = hypa.child_to_parent_rrf(ranked, key=lambda ch: ch.split("@")[1])
    assert out[0] == "s3"


def test_embed_uses_512_token_truncation(monkeypatch):
    # 스펙 §6 대체안 A — 원본(transformers.js)은 model_max_length=512로 자르는데
    # sentence-transformers 기본은 256(sentence_bert_config.json)이라 한국어
    # 문단(344자 ≈ wordpiece 494토큰)의 뒷부분이 조용히 잘린다.
    seen = {}

    class _FakeST:
        def __init__(self, name):
            seen["name"] = name
            self.max_seq_length = 256          # 실제 기본값

        def encode(self, texts, normalize_embeddings=True):
            seen["max_seq_length"] = self.max_seq_length
            return [[1.0] for _ in texts]

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    monkeypatch.setattr(hypa, "_EMBED_MODEL", None)
    hypa.embed(["한국어 문단"])
    assert seen["name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert seen["max_seq_length"] == 512


def test_hypa_step_subtracts_max_response_before_trigger():
    # §7 #1: maxResponse 차감 누락이면 조기 발동. 경계값으로 고정한다.
    # current(차감 후) = 50 + preset + hist 가 max_ctx 바로 아래면 미발동.
    hist = [{"role": "user", "content": "가" * 40}]
    hist_tk = sum(hypa.tok_chat(c) for c in hypa.to_risu_chats(hist))
    ctx = 50 + 1000 + hist_tk + 1
    mem, kept, start, data, err = hypa.hypa_step(hist, 1000, S, {"summaries": []},
                                                 None, ctx, max_response=4000)
    assert mem is None and err is None and kept == hist and start == 0


def test_hypa_step_reservation_self_sustains():
    # 요약 1개면 이후 매 턴 memoryTokens 선점 (§7 #11) — 짧은 히스토리에도
    # should_reserve=True 경로로 들어가 memory_text가 계속 나온다
    data = {"summaries": [{"text": "요약", "chatMemos": ["m0"],
                           "isImportant": False}]}
    hist = [{"role": "user", "content": "짧다"}]
    mem, _, _, _, err = hypa.hypa_step(hist, 1000, S, data, None, 10**6, 4000)
    assert err is None and mem is not None and "요약" in mem


def test_hypa_step_errors_when_floor_reached():
    # 남은 챗 ≤ queryChatCount인데 여전히 초과 → 에러 (hypav3.ts:263-274)
    hist = [{"role": "user", "content": "가" * 400} for _ in range(3)]
    mem, _, _, _, err = hypa.hypa_step(hist, 10**6, S, {"summaries": []},
                                       None, 100, 4000)
    assert err is not None and "minimum" in err and mem is None


def test_hypa_step_summarizes_prefix_and_slices_history(monkeypatch):
    # 요약된 원본은 삭제가 아니라 slice로 잘려나가고(hypav3.ts:934), 그 memo가
    # 다음 턴 start_idx 앵커가 된다 (hypav3.ts:214-229).
    monkeypatch.setattr(hypa, "summarize", lambda send, batch, s: "요약본")
    hist = [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"본문{i} 이야기가 이어진다. " * 20} for i in range(8)]
    # 1500: 예약(585) 포함 1터짐 → 배치 1개(불가침 3개 제외한 앞 5개)로 target 아래
    mem, kept, start, data, err = hypa.hypa_step(
        hist, 0, S, {"summaries": []}, None, 1500, 4000)
    assert err is None
    assert mem.startswith("<Past Events Summary>") and "요약본" in mem
    assert data["summaries"][0]["chatMemos"] == ["m0", "m1", "m2", "m3", "m4"]
    assert start == 5 and kept == hist[5:]


def test_hypa_step_resumes_after_last_summarized_memo(monkeypatch):
    # 다음 턴: 이전 요약의 마지막 memo 뒤부터만 센다 — 앵커가 깨지면 이미
    # 요약된 앞부분을 또 요약한다.
    monkeypatch.setattr(hypa, "summarize", lambda send, batch, s: "요약본2")
    hist = [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"본문{i} 이야기가 이어진다. " * 20} for i in range(9)]
    data = {"summaries": [{"text": "이전 요약", "chatMemos": ["m0", "m1", "m2"],
                           "isImportant": False}]}
    _, kept, start, data, err = hypa.hypa_step(hist, 0, S, data, None, 10**6, 4000)
    assert err is None and start == 3 and kept == hist[3:]
    assert len(data["summaries"]) == 1          # 예산 여유 → 추가 요약 없음


def test_hypa_step_records_selection_in_metrics():
    # 비차단 정리 #2: 요약 선택 결과가 data["metrics"]에 실려야 run2가 매 턴
    # 저장하는 hypa-state json으로 사후 분석(캐시 흔들림)이 가능하다.
    data = {"summaries": [{"text": "요약", "chatMemos": ["m0"],
                           "isImportant": False}]}
    hist = [{"role": "user", "content": "짧다"}]
    mem, _, _, data, err = hypa.hypa_step(hist, 1000, S, data, None, 10**6, 4000)
    assert err is None and mem is not None
    assert data["metrics"]["selectedCount"] == 1
    assert data["metrics"]["selectedChatMemos"] == ["m0"]


def test_rrf_tie_keeps_insertion_order():
    # 동점이면 첫 등장 순서 유지 — JS Map 삽입순 대응 (inv-agent2 §2 주의)
    out2 = hypa.child_to_parent_rrf([], key=lambda c: c)
    assert out2 == []
    scores_equal = hypa.simple_cc([[("a", 1.0)], [("b", 1.0)]], [0.5, 0.5])
    assert scores_equal == ["a", "b"]       # 동점 → 삽입 순서
