"""Ingest LOCOMO conversations through SAGA's Sub-B pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from benchmarks.adapter import LocomoConversation, turns_to_messages

logger = logging.getLogger(__name__)


class LocomoIngestion:
    """Process LOCOMO turns through SAGA's PostTurnExtractor pipeline."""

    def __init__(self, sqlite_db, vector_db, md_cache, llm_client, config):
        from functools import partial
        from saga.agents.post_turn import PostTurnExtractor
        from saga.agents.extractors import narrative_extract

        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.md_cache = md_cache
        self.llm_client = llm_client
        self.config = config

        extract_fn = partial(narrative_extract, llm_client=llm_client, config=config)
        self.post_turn = PostTurnExtractor(
            sqlite_db=sqlite_db,
            vector_db=vector_db,
            md_cache=md_cache,
            llm_client=llm_client,
            config=config,
            extract_fn=extract_fn,
        )

        self.stats = {"turns_processed": 0, "errors": 0, "elapsed_sec": 0.0}

    async def ingest_conversation(
        self,
        conv: LocomoConversation,
        session_id: str | None = None,
        resume_from: int = 0,
    ) -> str:
        """Process all turns of a LOCOMO conversation through Sub-B.

        Returns the session_id used.
        """
        if session_id is None:
            session_id = f"locomo-{conv.sample_id}"

        # Initialize session
        existing = await self.sqlite_db.get_session(session_id)
        if not existing:
            await self.sqlite_db.create_session(
                session_id=session_id,
                name=f"LOCOMO {conv.sample_id}",
            )
            cache_dir = self.md_cache.get_session_dir(session_id)
            os.makedirs(cache_dir, exist_ok=True)

        messages = turns_to_messages(conv.all_turns, conv.speaker_a, conv.speaker_b)
        start = time.time()

        for i, turn in enumerate(conv.all_turns):
            if i < resume_from:
                continue

            turn_number = i + 1

            # Determine user_input and assistant_output
            if turn.speaker == conv.speaker_a:
                user_input = turn.text
                assistant_output = ""
            else:
                # For assistant turns, extract_and_update processes the response
                user_input = messages[i - 1]["content"] if i > 0 else ""
                assistant_output = turn.text

            # Only run Sub-B on assistant turns (response extraction)
            if assistant_output:
                try:
                    await self.post_turn.extract_and_update(
                        session_id=session_id,
                        response_text=assistant_output,
                        turn_number=turn_number,
                        user_input=user_input,
                    )
                    self.stats["turns_processed"] += 1
                except Exception as e:
                    logger.error(f"[ingestion] Turn {turn_number} error: {e}")
                    self.stats["errors"] += 1

            if (i + 1) % 50 == 0:
                logger.info(f"[ingestion] {conv.sample_id}: {i + 1}/{len(conv.all_turns)} turns")

        self.stats["elapsed_sec"] = time.time() - start
        logger.info(
            f"[ingestion] {conv.sample_id} done: "
            f"{self.stats['turns_processed']} turns, "
            f"{self.stats['errors']} errors, "
            f"{self.stats['elapsed_sec']:.1f}s"
        )
        return session_id

    async def ingest_conversations(
        self,
        conversations: list[LocomoConversation],
        max_conversations: int | None = None,
    ) -> dict[str, str]:
        """Ingest multiple conversations. Returns {sample_id: session_id}."""
        mapping = {}
        convs = conversations[:max_conversations] if max_conversations else conversations

        for conv in convs:
            session_id = await self.ingest_conversation(conv)
            mapping[conv.sample_id] = session_id

        return mapping
