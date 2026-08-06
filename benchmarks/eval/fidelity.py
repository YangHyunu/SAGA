"""와이어 충실도 검사 — 합성 요청이 실캡처와 같은 형태인지 매 요청 기계 검증.

실캡처(corpus5, 81건 전수) 실측 형태:
  [선두 system 1개] (assistant 인사) user assistant … user [꼬리 system 1개]
  · 중간 system 없음, 역할 연속 중복 없음, 미해석 매크로 없음
  · greeting은 트림(~56메시지) 이후 사라지므로 고정 프리픽스로 보지 않는다
  · 형식은 OpenAI-compat (content=평문 문자열, system이 messages 안에 위치)

위반이 하나라도 나오면 러너는 즉시 중단한다 — "같다고 믿는" 대신 "매번 확인한다".
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from typing import Dict, List, Optional

MACRO = re.compile(r"\{\{[^}]{1,80}\}\}")


def _text(msg: Dict) -> str:
    c = msg.get("content", "")
    if isinstance(c, list):                    # Anthropic 블록이 섞여 들어온 경우
        return "".join(b.get("text", "") for b in c)
    return c or ""


def check_wire_shape(msgs: List[Dict]) -> List[str]:
    """실캡처 형태 위반 목록 (빈 리스트 = 통과)."""
    out: List[str] = []
    if not msgs:
        return ["메시지가 비어 있음"]

    if msgs[0]["role"] != "system":
        out.append("선두가 system이 아님")
    if any(not isinstance(m.get("content"), str) for m in msgs):
        out.append("content가 평문 문자열이 아님 (OpenAI-compat 위반)")
    if any("cache_control" in m for m in msgs):
        out.append("클라이언트 요청에 cache_control 있음 (마킹은 프록시 몫)")

    # 꼬리 system은 마지막 1개만 허용 — 그 앞은 반드시 user
    body = msgs[1:]
    if body and body[-1]["role"] == "system":
        body = body[:-1]
        if not body or body[-1]["role"] != "user":
            out.append("꼬리 system 앞이 user가 아님")
    elif body and body[-1]["role"] != "user":
        out.append("마지막 메시지가 user가 아님")

    if any(m["role"] == "system" for m in body):
        out.append("중간에 system이 있음 (선두/꼬리 외 금지)")

    for i in range(1, len(body)):
        if body[i]["role"] == body[i - 1]["role"]:
            out.append(f"역할 연속 중복: index {i} ({body[i]['role']})")
            break

    for i, m in enumerate(msgs):
        found = MACRO.search(_text(m))
        if found:
            out.append(f"미해석 매크로: index {i} {found.group()[:40]}")
            break
    return out


def _leading_trailing(msgs: List[Dict]) -> tuple:
    lead = _text(msgs[0]) if msgs and msgs[0]["role"] == "system" else ""
    tail = _text(msgs[-1]) if len(msgs) > 1 and msgs[-1]["role"] == "system" else ""
    return lead, tail


def _sig(text: str) -> Optional[Dict]:
    if not text:
        return None
    return {"sha256": hashlib.sha256(text.encode()).hexdigest()[:16], "len": len(text)}


def corpus_signature(corpus_dir: str) -> Dict:
    """실캡처 디렉터리에서 선두/꼬리 system 시그니처 추출 (req-*.json)."""
    sigs: Dict[str, Dict] = {"leading": {}, "trailing": {}, "n": 0}
    for path in sorted(glob.glob(os.path.join(corpus_dir, "req-*.json"))):
        with open(path) as f:
            body = json.load(f)
        body = body.get("body") or body.get("request") or body
        msgs = body.get("messages") or []
        if not msgs:
            continue
        sigs["n"] += 1
        lead, tail = _leading_trailing(msgs)
        for key, sig in (("leading", _sig(lead)), ("trailing", _sig(tail))):
            if sig:
                sigs[key][sig["sha256"]] = sigs[key].get(sig["sha256"], 0) + 1
    return sigs


def compare_with_corpus(msgs: List[Dict], signature: Dict) -> List[str]:
    """조립한 요청의 선두/꼬리 system이 캡처 시그니처에 있는지 대조."""
    out: List[str] = []
    lead, tail = _leading_trailing(msgs)
    for key, text in (("leading", lead), ("trailing", tail)):
        known = signature.get(key) or {}
        if not known:
            continue
        sig = _sig(text)
        if sig is None or sig["sha256"] not in known:
            got = sig["sha256"] if sig else "없음"
            out.append(f"{key} system이 캡처와 불일치 (got {got})")
    return out
