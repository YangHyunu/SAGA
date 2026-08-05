"""dreaming/facts.py — Fact 버전 체인 (스펙 §4.1).

갱신은 항상 invalidate-and-append: 기존 레코드는 superseded로 표시만 하고
새 레코드가 supersedes로 링크한다. 유저 편집(user_edited)은 ground truth —
Dreamer는 수정할 수 없다 (스펙 §2.7).
"""

from __future__ import annotations

import uuid
from typing import Tuple

from dreaming.records import Fact, utc_now_iso


def supersede(old: Fact, new: Fact) -> Tuple[Fact, Fact]:
    """old를 무효화하고 new를 체인에 링크한 사본 쌍을 돌려준다."""
    old2 = old.model_copy(update={"status": "superseded"})
    new2 = new.model_copy(update={"supersedes": old.id})
    return old2, new2


def dreamer_can_modify(fact: Fact) -> bool:
    """유저가 편집한 사실은 Dreamer가 덮을 수 없다 (스펙 §2.7)."""
    return not fact.user_edited


def apply_user_edit(fact: Fact, **changes: object) -> Tuple[Fact, Fact]:
    """유저 편집 = 수동 supersede. 새 버전은 user_edited로 보호된다."""
    new = fact.model_copy(
        update={
            **changes,
            "id": uuid.uuid4().hex,
            "supersedes": fact.id,
            "user_edited": True,
            "recorded_at": utc_now_iso(),
        }
    )
    old = fact.model_copy(update={"status": "superseded"})
    return old, new
