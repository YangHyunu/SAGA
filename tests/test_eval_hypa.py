from benchmarks.eval import hypa

EXPORT = ("/Users/yanghyeon-u/Downloads/뮈토스6.2/🏺뮈토스 프롬프트 하이파/"
          "hypaV3_export_뮈토스 하이파 V5.json")


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
