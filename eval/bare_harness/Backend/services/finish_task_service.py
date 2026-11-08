"""Stateless acknowledgement service for an agent-requested run termination."""

from typing import Any, Dict


class FinishTaskService:
    """Validate and acknowledge a finish request.

    Backend deliberately does not own the model conversation or run lifecycle.
    The bare client closes tool admission after it receives this acknowledgement.
    """

    ALLOWED_REASONS = {"completed", "no_reasonable_next_step"}

    def execute(self, op_name: str, reason: str, summary: str) -> Dict[str, Any]:
        op_name = op_name.strip()
        reason = reason.strip()
        summary = summary.strip()
        if not op_name:
            raise ValueError("op_name must not be empty")
        if reason not in self.ALLOWED_REASONS:
            raise ValueError(
                "reason must be one of: " + ", ".join(sorted(self.ALLOWED_REASONS))
            )
        if not summary:
            raise ValueError("summary must not be empty")
        if len(summary) > 8000:
            raise ValueError("summary must not exceed 8000 characters")
        return {
            "status": "accepted",
            "stop_requested": True,
            "op_name": op_name,
            "reason": reason,
            "summary": summary,
        }
