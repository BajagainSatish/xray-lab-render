"""Offline parsing and deterministic selection for the small NIH pilot set."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# The 14 findings in the NIH release; all are also outputs of densenet121-res224-all.
NIH_FINDING_LABELS = (
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
)
NO_FINDING = "No Finding"
VIEWS = ("PA", "AP", "Lateral")

# One deterministic image per stratum, with distinct patients. The first
# unused image in sorted order wins, so selection does not depend on input order.
PILOT_STRATA = (
    ("no_finding", lambda findings: findings == (NO_FINDING,)),
    ("cardiomegaly", lambda findings: findings == ("Cardiomegaly",)),
    ("effusion", lambda findings: findings == ("Effusion",)),
    ("atelectasis", lambda findings: findings == ("Atelectasis",)),
    ("multilabel", lambda findings: len(findings) >= 2),
)


@dataclass(frozen=True)
class Entry:
    image_id: str
    findings: tuple[str, ...]
    patient_id: str
    view: str
    follow_up: int
    original_width: int
    original_height: int

    @property
    def label_text(self) -> str:
        return "|".join(self.findings)


@dataclass(frozen=True)
class Selection:
    stratum: str
    image_id: str
    entry: Entry


def read_data_entry(path: Path, *, minimum_rows: int = 1) -> dict[str, Entry]:
    """Parse Data_Entry_2017.csv.

    The official header contains commas inside square brackets, so the CSV has
    11 fields while a naive reading suggests 9. Rows are parsed positionally and
    the bracketed fields are mapped explicitly.
    """
    entries: dict[str, Entry] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or header[:2] != ["Image Index", "Finding Labels"]:
            raise ValueError("Unexpected Data_Entry_2017.csv header.")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 11:
                raise ValueError(
                    f"Data_Entry line {line_number}: expected 11 fields, got {len(row)}."
                )
            image_id = row[0].strip()
            if not image_id.endswith(".png") or "/" in image_id:
                raise ValueError(f"Data_Entry line {line_number}: invalid image id {image_id!r}.")
            findings = tuple(part.strip() for part in row[1].split("|"))
            if not findings or any(not part for part in findings):
                raise ValueError(f"{image_id}: empty finding label.")
            if findings == (NO_FINDING,):
                pass
            elif NO_FINDING in findings:
                raise ValueError(f"{image_id}: 'No Finding' must not be combined with findings.")
            else:
                unknown = [part for part in findings if part not in NIH_FINDING_LABELS]
                if unknown:
                    raise ValueError(f"{image_id}: labels outside the NIH release: {unknown}.")
            view = row[6].strip()
            if view not in VIEWS:
                raise ValueError(f"{image_id}: unexpected view position {view!r}.")
            try:
                follow_up = int(row[2])
                original_width = int(row[7])
                original_height = int(row[8])
            except ValueError as exc:
                raise ValueError(f"{image_id}: non-integer numeric field.") from exc
            if follow_up < 0 or original_width <= 0 or original_height <= 0:
                raise ValueError(f"{image_id}: invalid follow-up or acquisition dimensions.")
            entry = Entry(
                image_id=image_id,
                findings=findings,
                patient_id=row[3].strip(),
                view=view,
                follow_up=follow_up,
                original_width=original_width,
                original_height=original_height,
            )
            if image_id in entries:
                raise ValueError(f"Duplicate image id: {image_id}.")
            entries[image_id] = entry
    if len(entries) < minimum_rows:
        raise ValueError(f"Parsed {len(entries)} rows, expected at least {minimum_rows}.")
    return entries


def read_split(path: Path) -> set[str]:
    """Read an official image-id list such as test_list.txt."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    ids = {line for line in lines if line}
    if not ids:
        raise ValueError("Split file is empty.")
    invalid = [image_id for image_id in ids if not image_id.endswith(".png") or "/" in image_id]
    if invalid:
        raise ValueError(f"Split file contains invalid image ids: {invalid[:3]}.")
    return ids


def select_stratified(
    entries: dict[str, Entry],
    allowed: set[str],
    strata=PILOT_STRATA,
    *,
    view: str = "PA",
) -> list[Selection]:
    """Pick one image per stratum from ``allowed``, using distinct patients."""
    candidates = sorted(
        image_id
        for image_id in allowed
        if image_id in entries and entries[image_id].view == view
    )
    used_patients: set[str] = set()
    selections: list[Selection] = []
    for name, predicate in strata:
        chosen = next(
            (
                entries[image_id]
                for image_id in candidates
                if entries[image_id].patient_id not in used_patients
                and predicate(entries[image_id].findings)
            ),
            None,
        )
        if chosen is None:
            raise ValueError(f"No unused {view} candidate for stratum {name!r}.")
        used_patients.add(chosen.patient_id)
        selections.append(Selection(stratum=name, image_id=chosen.image_id, entry=chosen))
    return selections


def select_quota(
    entries: dict[str, Entry],
    allowed: set[str],
    specs: list[tuple[str, object, int, str | None]],
) -> list[Selection]:
    """Pick a fixed number of images per spec, with distinct patients.

    ``specs`` are ``(name, predicate, count, view)``; ``view`` may be None to
    accept any stored view. Selection is deterministic: candidates are sorted by
    image id and the first unused patient is taken.
    """
    candidates = sorted(
        image_id for image_id in allowed if image_id in entries
    )
    used_images: set[str] = set()
    used_patients: set[str] = set()
    selections: list[Selection] = []
    for name, predicate, count, view in specs:
        if count <= 0:
            raise ValueError(f"Stratum {name!r} needs a positive count.")
        picked: list[Entry] = []
        for image_id in candidates:
            if image_id in used_images:
                continue
            entry = entries[image_id]
            if view is not None and entry.view != view:
                continue
            if not predicate(entry.findings):
                continue
            if entry.patient_id in used_patients:
                continue
            picked.append(entry)
            used_images.add(image_id)
            used_patients.add(entry.patient_id)
            if len(picked) == count:
                break
        if len(picked) != count:
            raise ValueError(
                f"Stratum {name!r}: needed {count} unused images, found {len(picked)}."
            )
        selections.extend(Selection(name, entry.image_id, entry) for entry in picked)
    return selections
