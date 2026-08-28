from dataclasses import dataclass, field


@dataclass
class ColumnSpec:
    name_col: str | None  # full name, passthrough
    label_col: str  # preferred name for display
    score_cols: list[str]  # the 5 survey columns
    balance_col: str | None  # Pronoun; optional
    id_cols: list[str] = field(
        default_factory=list
    )


@dataclass
class DerivedColumn:
    new_name: str
    method: str  # v1: "extract" only
    source_col: str | None = None
    pattern: str | None = None
    strip_from_source: bool = True
    mapping: dict[str, str] = field(
        default_factory=dict
    )


@dataclass
class SolveConfig:
    balance_weight: float = 1.0
    diversity_weight: float = 1.0
    balance_attr_weight: float = 1.0
    isolation_penalty: float = 10.0
    enforce_same_name: bool = True
    time_limit_s: float = 30.0
    seed: int = 0
