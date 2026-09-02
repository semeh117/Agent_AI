"""Local ESCO skill normalization used only by Agent 2 matching.

The normalizer performs conservative label lookup against the downloaded ESCO
CSV. It never uses fuzzy matching: a label is mapped only when its normalized
preferred/alternative label identifies one unambiguous ESCO concept. Modern
technologies missing from ESCO are preserved unchanged for cosine matching.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import unicodedata
from typing import Optional


DEFAULT_ESCO_SKILLS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "skills_en.csv"
)


def normalize_esco_label(value: str) -> str:
    """Normalize superficial label differences without changing meaning."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("&", " and ")
    tokens = re.findall(r"[a-z0-9]+(?:\+\+|#)?", text)
    return " ".join(tokens)


@dataclass(frozen=True)
class EscoConcept:
    concept_uri: str
    preferred_label: str
    skill_type: str


@dataclass(frozen=True)
class NormalizedSkill:
    original: str
    matching_text: str
    concept_uri: Optional[str] = None
    preferred_label: Optional[str] = None
    match_type: str = "unmapped"

    @property
    def mapped(self) -> bool:
        return self.concept_uri is not None

    @property
    def identity_key(self) -> str:
        if self.concept_uri:
            return f"esco:{self.concept_uri}"
        return f"text:{normalize_esco_label(self.original)}"


class EscoSkillNormalizer:
    """In-memory exact index over ESCO preferred and alternative labels."""

    def __init__(self, csv_path: str | Path = DEFAULT_ESCO_SKILLS_PATH):
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"ESCO skills CSV does not exist: {self.csv_path}")

        self._preferred: dict[str, Optional[EscoConcept]] = {}
        self._alternative: dict[str, Optional[EscoConcept]] = {}
        self.concept_count = 0
        self._load()

    @staticmethod
    def _add_unambiguous(
        index: dict[str, Optional[EscoConcept]],
        label: str,
        concept: EscoConcept,
    ) -> None:
        key = normalize_esco_label(label)
        if not key:
            return
        if key not in index:
            index[key] = concept
            return
        existing = index[key]
        if existing is not None and existing.concept_uri != concept.concept_uri:
            index[key] = None

    def _load(self) -> None:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"conceptUri", "preferredLabel", "altLabels"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    "ESCO skills CSV is missing required columns: "
                    + ", ".join(sorted(missing))
                )

            for row in reader:
                if str(row.get("status") or "").casefold() == "deprecated":
                    continue
                concept_uri = str(row.get("conceptUri") or "").strip()
                preferred_label = str(row.get("preferredLabel") or "").strip()
                if not concept_uri or not preferred_label:
                    continue
                concept = EscoConcept(
                    concept_uri=concept_uri,
                    preferred_label=preferred_label,
                    skill_type=str(row.get("skillType") or "").strip(),
                )
                self.concept_count += 1
                self._add_unambiguous(self._preferred, preferred_label, concept)
                for alternative in str(row.get("altLabels") or "").splitlines():
                    self._add_unambiguous(
                        self._alternative,
                        alternative.strip(),
                        concept,
                    )

    def normalize(self, value: str) -> NormalizedSkill:
        original = str(value or "").strip()
        key = normalize_esco_label(original)
        if key in self._preferred:
            concept = self._preferred[key]
            match_type = "preferred"
        elif key in self._alternative:
            concept = self._alternative[key]
            match_type = "alternative"
        else:
            concept = None
            match_type = "unmapped"
        if concept is None:
            return NormalizedSkill(original=original, matching_text=original)
        return NormalizedSkill(
            original=original,
            matching_text=concept.preferred_label,
            concept_uri=concept.concept_uri,
            preferred_label=concept.preferred_label,
            match_type=match_type,
        )


@lru_cache(maxsize=4)
def get_esco_normalizer(
    csv_path: str | Path | None = None,
) -> EscoSkillNormalizer:
    """Return one cached ESCO index for the configured local dataset."""

    configured = csv_path or os.getenv("ESCO_SKILLS_PATH") or DEFAULT_ESCO_SKILLS_PATH
    return EscoSkillNormalizer(configured)
