"""Pair ledger: content-hash based session identity + reroll detection.

RisuAI never sends chat identifiers on the wire, and a reroll is a flagless
resend with the tail assistant popped. Community plugins (HAYAKU, FLASHBACK,
WygLore) all converged on content-hash ledgers to recover this information;
this module is the server-side port of that pattern.

Ledger row: (session_id, pair_index, user_hash, assistant_hash, status, turn_number)
Statuses: provisional -> confirmed (assistant seen again in a later request)
                      -> superseded (rerolled/edited, being replaced)
                      -> quarantined (rolled back, may be restored by a swipe-back)
"""
import hashlib
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# How many trailing pairs to use when matching a request to an existing session
_RESOLVE_WINDOW = 6

_WS_RE = re.compile(r"\s+")

ACTIVE_STATUSES = ("provisional", "confirmed")


def hash_text(text: str | None) -> str:
    """Whitespace/NFC-normalized content hash (16 hex chars)."""
    norm = unicodedata.normalize("NFC", text or "").strip()
    norm = _WS_RE.sub(" ", norm)
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def extract_pairs(messages: list[dict]) -> tuple[list[dict], str | None]:
    """Extract (user, assistant) pairs from raw incoming messages.

    Leading assistants (character card first-messages, identical across chats)
    are skipped. Consecutive assistants are merged (autoContinue splits).
    Returns (pairs, last_user_hash) where last_user_hash is the trailing user
    message without an assistant reply — the current turn's input.
    """
    convo = [m for m in messages if m.get("role") in ("user", "assistant")]
    start = 0
    while start < len(convo) and convo[start]["role"] == "assistant":
        start += 1

    pairs: list[dict] = []
    last_user_hash = None
    i = start
    while i < len(convo):
        m = convo[i]
        if m["role"] != "user":
            i += 1
            continue
        uh = hash_text(m.get("content", ""))
        j = i + 1
        asst_parts = []
        while j < len(convo) and convo[j]["role"] == "assistant":
            asst_parts.append(convo[j].get("content", ""))
            j += 1
        if asst_parts:
            pairs.append({
                "index": len(pairs),
                "user_hash": uh,
                "assistant_hash": hash_text("\n".join(asst_parts)),
            })
        elif j >= len(convo):
            last_user_hash = uh
        else:
            # user followed by user (no assistant in between): pair without reply
            pairs.append({"index": len(pairs), "user_hash": uh, "assistant_hash": None})
        i = j
    return pairs, last_user_hash


def _align_offset(chain: list[dict], request_pairs: list[dict]) -> int | None:
    """Find offset o such that request_pairs[k] maps to chain[k + o].

    RisuAI's sliding window truncates the front of history, so request indices
    are relative. Anchor on the last request pair's user_hash, searching the
    chain from the end.
    """
    if not request_pairs or not chain:
        return None
    for rp in reversed(request_pairs):
        for ci in range(len(chain) - 1, -1, -1):
            if chain[ci]["user_hash"] == rp["user_hash"]:
                offset = ci - rp["index"]
                if offset >= 0:
                    return offset
    return None


def classify(chain: list[dict], request_pairs: list[dict],
             last_user_hash: str | None) -> dict:
    """Classify a request against the session's active pair chain.

    chain: active ledger rows ordered by pair_index (one row per index).
    Returns a verdict dict:
      kind: 'new' | 'append' | 'reroll'
      position: chain position of the current trailing user input
      reroll_turn_number: turn to reuse instead of incrementing (reroll only)
      superseded_indices / quarantined_indices: chain positions to mark
      confirm: [(chain_index, client_assistant_hash|None), ...] pairs seen again
      aligned: whether request→chain alignment succeeded (backfill is safe)
    """
    verdict = {
        "kind": "new", "position": 0, "reroll_turn_number": None,
        "superseded_indices": [], "quarantined_indices": [],
        "confirm": [], "aligned": False, "offset": None,
    }
    if not chain:
        # Empty ledger: request pairs (if any) become the baseline at offset 0
        verdict["kind"] = "append" if (last_user_hash or request_pairs) else "new"
        verdict["position"] = len(request_pairs)
        verdict["aligned"] = bool(request_pairs)
        verdict["offset"] = 0 if request_pairs else None
        return verdict

    length = len(chain)
    offset = _align_offset(chain, request_pairs)

    if offset is not None:
        verdict["aligned"] = True
        verdict["offset"] = offset
        position = offset + len(request_pairs)
        for rp in request_pairs:
            ci = offset + rp["index"]
            if 0 <= ci < length and chain[ci]["user_hash"] == rp["user_hash"]:
                verdict["confirm"].append((ci, rp["assistant_hash"]))
    else:
        # No pair overlap: anchor on the trailing user input itself
        position = length
        if last_user_hash:
            for ci in range(length - 1, -1, -1):
                if chain[ci]["user_hash"] == last_user_hash:
                    position = ci
                    break

    if position < length and last_user_hash:
        # Ledger already has a pair at this position: the client dropped its
        # assistant and is resending — a reroll (or an edited user input,
        # which supersedes the old pair the same way).
        target = chain[position]
        verdict["kind"] = "reroll"
        verdict["position"] = position
        verdict["reroll_turn_number"] = target.get("turn_number")
        verdict["superseded_indices"] = [position]
        verdict["quarantined_indices"] = list(range(position + 1, length))
    else:
        verdict["kind"] = "append"
        verdict["position"] = position
    return verdict


class PairLedgerService:
    """DB-backed orchestration around the pure classification logic."""

    def __init__(self, sqlite_db, vector_db=None):
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db

    async def resolve_session(self, pairs: list[dict]) -> str | None:
        """Find the session whose ledger overlaps the request's recent pairs.

        Requires at least one assistant-hash match (generated text is unique)
        or two total matches — single user-hash overlap ("...", "계속") is too
        collision-prone across sessions.
        """
        if not pairs:
            return None
        recent = pairs[-_RESOLVE_WINDOW:]
        user_hashes = [p["user_hash"] for p in recent]
        asst_hashes = [p["assistant_hash"] for p in recent if p["assistant_hash"]]
        candidates = await self.sqlite_db.find_sessions_by_pair_hashes(user_hashes, asst_hashes)
        for cand in candidates:  # ordered by (match score, recency)
            if cand["asst_matches"] >= 1 or cand["total_matches"] >= 2:
                logger.info(
                    f"[PairLedger] Session resolved via pair-chain: {cand['session_id']} "
                    f"(asst={cand['asst_matches']} total={cand['total_matches']})"
                )
                return cand["session_id"]
        return None

    async def get_active_chain(self, session_id: str) -> list[dict]:
        """Active (non-superseded) ledger rows, one per pair_index, ordered."""
        rows = await self.sqlite_db.get_pair_ledger(session_id)
        chain: dict[int, dict] = {}
        for row in rows:  # ordered by pair_index, updated_at ASC
            if row["status"] in ACTIVE_STATUSES:
                chain[row["pair_index"]] = row
            elif row["pair_index"] not in chain and row["status"] == "quarantined":
                chain.setdefault(row["pair_index"], row)
        return [chain[k] for k in sorted(chain)]

    async def analyze_and_apply(self, session_id: str, pairs: list[dict],
                                last_user_hash: str | None) -> dict:
        """Classify the request, then apply status transitions to the ledger,
        turn_log, and episodes. Returns the verdict."""
        chain = await self.get_active_chain(session_id)
        verdict = classify(chain, pairs, last_user_hash)

        if verdict["kind"] == "reroll":
            logger.info(
                f"[PairLedger] Reroll detected: session={session_id} "
                f"position={verdict['position']} turn={verdict['reroll_turn_number']} "
                f"(+{len(verdict['quarantined_indices'])} quarantined)"
            )
        for ci in verdict["superseded_indices"]:
            await self._transition(session_id, chain[ci], "superseded")
        for ci in verdict["quarantined_indices"]:
            await self._transition(session_id, chain[ci], "quarantined")
        for ci, client_asst_hash in verdict["confirm"]:
            row = chain[ci]
            if row["status"] != "confirmed" or (
                client_asst_hash and client_asst_hash != row["assistant_hash"]
            ):
                # Adopt the client's assistant hash: display scripts may have
                # mutated the text; the client's stored version is canonical.
                await self.sqlite_db.update_pair(
                    row["id"], status="confirmed",
                    assistant_hash=client_asst_hash or row["assistant_hash"],
                )
                if row.get("turn_number") is not None:
                    await self.sqlite_db.set_turn_status(
                        session_id, row["turn_number"], "confirmed")
                    if self.vector_db is not None:
                        self.vector_db.set_episode_status(
                            session_id, row["turn_number"], "confirmed")

        if verdict["aligned"]:
            await self._backfill(session_id, chain, pairs, verdict["offset"])
        return verdict

    async def _transition(self, session_id: str, row: dict, status: str):
        await self.sqlite_db.update_pair(row["id"], status=status)
        if row.get("turn_number") is not None:
            await self.sqlite_db.set_turn_status(session_id, row["turn_number"], status)
            if self.vector_db is not None:
                self.vector_db.set_episode_status(session_id, row["turn_number"], status)

    async def _backfill(self, session_id: str, chain: list[dict],
                        pairs: list[dict], offset: int):
        """Insert aligned request pairs missing from the chain (heals holes
        from autoContinue turns, server restarts, etc.)."""
        known = {row["pair_index"] for row in chain}
        for rp in pairs:
            ci = offset + rp["index"]
            if ci >= 0 and ci not in known:
                await self.sqlite_db.insert_pair(
                    session_id, ci, rp["user_hash"], rp["assistant_hash"],
                    status="confirmed", turn_number=None,
                )

    async def record_turn(self, session_id: str, verdict: dict,
                          last_user_hash: str | None, assistant_text: str,
                          turn_number: int):
        """Record the completed turn's pair as provisional."""
        if not last_user_hash:
            return
        await self.sqlite_db.insert_pair(
            session_id, verdict["position"], last_user_hash,
            hash_text(assistant_text), status="provisional",
            turn_number=turn_number,
        )
