from dataclasses import dataclass, field


class SolverError(Exception):
    """Raised when preprocessing or solving fails."""


class ValidationError(Exception):
    """Raised when validation produces errors."""

    def __init__(self, issues):
        self.issues = issues
        msgs = [
            i.message
            for i in issues
            if i.severity == "error"
        ]
        super().__init__(
            "; ".join(msgs)
            if msgs
            else "Validation failed"
        )


@dataclass
class Issue:
    severity: str  # "error" or "warning"
    message: str
    row_indices: list[int] = field(
        default_factory=list
    )
