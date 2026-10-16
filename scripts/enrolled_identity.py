"""Time-gated session enrollment; uncertain tracks never inherit a participant ID."""

import numpy as np

from .session_identity import (
    CONTINUITY_THRESHOLD,
    MATCH_MARGIN,
    MATCH_THRESHOLD,
    MAX_TEMPLATES,
    MISMATCH_FRAMES,
    SessionIdentity,
)

ENROLL_SECONDS = 5.0
RETURN_SECONDS = 1.0
PENDING_SECONDS = 60.0
MAX_EVIDENCE_GAP = 0.5
MIN_SAMPLES = 8
MAX_PARTICIPANTS = 6


class EnrolledIdentity(SessionIdentity):
    def __init__(self, model, enrollment_seconds=60):
        super().__init__(model)
        self.enrollment_seconds = enrollment_seconds
        self.started = None
        self.templates = {}
        self.pending = {}
        self.last_active = set()
        self.restored_from = {}
        self.mismatches = {}

    def resolve_at(self, timestamp_ms, spatial_ids, features):
        now = timestamp_ms / 1000
        if self.started is None:
            self.started = now
        self.remaining = max(0, self.enrollment_seconds - (now - self.started))
        self.restored_from = {}
        self.reset_pending_ids = set()
        if not hasattr(self, "mismatches"):
            self.mismatches = {}
        active = set(spatial_ids)
        self.mismatches = {k: v for k, v in self.mismatches.items() if k in active}
        self.bindings = {k: v for k, v in self.bindings.items() if k in active}
        self.pending = {k: v for k, v in self.pending.items() if k in active}
        used = set(self.bindings.values())
        results = []
        for spatial, feature in zip(spatial_ids, features):
            feature = self.valid_feature(feature)
            bound = self.bindings.get(spatial)
            if bound is not None and feature is not None:
                score = max(float(feature @ old) for old in self.templates[bound])
                if score >= CONTINUITY_THRESHOLD:
                    self.mismatches[spatial] = 0
                    results.append(bound)
                    continue
                self.mismatches[spatial] = self.mismatches.get(spatial, 0) + 1
                # Keep the binding for recovery, but never route uncertain
                # observations into the established participant state.
                self.reset_pending_ids.add(-spatial)
                if self.mismatches[spatial] < MISMATCH_FRAMES:
                    results.append(-spatial)
                    continue
                del self.bindings[spatial]
                used.discard(bound)
            elif bound is not None:
                self.reset_pending_ids.add(-spatial)
                results.append(-spatial)
                continue
            pending = self.pending.setdefault(
                spatial, dict(start=now, last=now, samples=[], candidate=None)
            )
            matches = self.candidates(feature, used)
            candidate = (
                matches[0][1]
                if matches
                and matches[0][0] >= MATCH_THRESHOLD
                and (len(matches) == 1 or matches[0][0] - matches[1][0] >= MATCH_MARGIN)
                else None
            )
            # A close gallery tie is uncertainty, not evidence of a new person.
            # Clear enrollment evidence so ambiguity cannot seed a new history.
            ambiguous = (
                len(matches) > 1
                and matches[0][0] >= MATCH_THRESHOLD
                and matches[0][0] - matches[1][0] < MATCH_MARGIN
            )
            if ambiguous:
                self.reset_pending_ids.add(-spatial)
                pending.update(
                    start=now, last=now, samples=[], candidate=None,
                    evidence=0.0, previous_valid=False,
                )
                results.append(-spatial)
                continue
            samples = pending["samples"]
            consistent = (
                not samples
                or feature is not None
                and float(feature @ samples[-1]) >= MATCH_THRESHOLD
            )
            # Invalid observations pause evidence; only adjacent valid intervals
            # contribute time. Retain samples at most MAX_EVIDENCE_GAP since
            # the last valid feature, on this spatial track only.
            if feature is None:
                if now - pending["last"] > MAX_EVIDENCE_GAP:
                    self.reset_pending_ids.add(-spatial)
                    pending.update(start=now, samples=[], candidate=None, evidence=0.0)
                pending["previous_valid"] = False
                results.append(-spatial)
                continue
            previous_last = pending["last"]
            previous_valid = pending.get("previous_valid", False)
            reset = (
                now - pending["last"] > MAX_EVIDENCE_GAP
                or candidate != pending["candidate"]
                or not consistent
            )
            if reset:
                self.reset_pending_ids.add(-spatial)
                pending.update(start=now, samples=[], candidate=candidate, evidence=0.0)
            elif previous_valid:
                pending["evidence"] = pending.get("evidence", 0.0) + max(0, now - previous_last)
            pending["previous_valid"] = True
            pending["last"] = now
            if feature is not None:
                pending["samples"].append(feature)
                pending["samples"] = pending["samples"][-120:]
            duration = pending.get("evidence", 0.0)
            required = RETURN_SECONDS if candidate is not None else ENROLL_SECONDS
            if (
                duration >= required
                and len(pending["samples"]) >= MIN_SAMPLES
                and (
                    candidate is not None
                    or self.remaining > 0
                    and len(self.gallery) < MAX_PARTICIPANTS
                )
            ):
                identity = candidate if candidate is not None else self.next_id
                if candidate is None:
                    self.next_id += 1
                    self.gallery[identity] = pending["samples"][0].copy()
                    self.templates[identity] = [
                        s.copy()
                        for s in pending["samples"][
                            :: max(1, len(pending["samples"]) // MAX_TEMPLATES)
                        ][:MAX_TEMPLATES]
                    ]
                self.bindings[spatial] = identity
                self.restored_from[identity] = -spatial
                used.add(identity)
                del self.pending[spatial]
                results.append(identity)
            else:
                results.append(-spatial)
        self.last_active = active
        return results

    @staticmethod
    def valid_feature(feature):
        if feature is None or not np.isfinite(feature).all():
            return None
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else None

    def candidates(self, feature, used=()):
        if feature is None:
            return []
        return sorted(
            (
                (max(float(feature @ old) for old in values), identity)
                for identity, values in self.templates.items()
                if identity not in used
            ),
            reverse=True,
        )

    def confirm(self, spatial, identity, timestamp_ms):
        if spatial not in self.pending:
            raise ValueError("Pending track no longer visible")
        pending = self.pending[spatial]
        if not pending["samples"]:
            raise ValueError("No valid face reference for confirmation")
        if identity in self.bindings.values():
            raise ValueError("Participant already assigned to another visible face")
        if identity == 0:
            if len(self.gallery) >= MAX_PARTICIPANTS:
                raise ValueError("Participant capacity reached")
            identity = self.next_id
            self.next_id += 1
            self.gallery[identity] = pending["samples"][-1].copy()
            self.templates[identity] = [pending["samples"][-1].copy()]
        if identity not in self.gallery:
            raise ValueError("Unknown participant")
        self.bindings[spatial] = identity
        self.restored_from[identity] = -spatial
        del self.pending[spatial]
        return identity
