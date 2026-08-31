"""Title families, match fan-out expansion, and standing-harvest slot allocation."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

FAMILY_TITLES: dict[str, list[str]] = {
    "software_engineering": [
        "Software Engineer",
        "Backend Engineer",
        "Frontend Engineer",
        "Full Stack Engineer",
        "Mobile Engineer",
        "DevOps Engineer",
        "QA Engineer",
    ],
    "data_ml": [
        "Data Scientist",
        "Machine Learning Engineer",
        "Data Engineer",
    ],
    "product": [
        "Product Manager",
        "Technical Program Manager",
    ],
    "design": [
        "Product Designer",
        "UX Designer",
    ],
    "go_to_market": [
        "Sales Engineer",
        "Customer Success Manager",
        "Marketing Manager",
    ],
    "ops": [
        "Business Analyst",
        "IT Support Specialist",
        "Security Engineer",
    ],
}

DEFAULT_FAMILIES = list(FAMILY_TITLES.keys())

_TITLE_TO_FAMILY: dict[str, str] = {}
for _family, _titles in FAMILY_TITLES.items():
    for _title in _titles:
        _TITLE_TO_FAMILY[_title.lower()] = _family

_TITLE_ALIASES = {
    "swe": "software_engineering",
    "software engineering": "software_engineering",
    "software developer": "software_engineering",
    "backend developer": "software_engineering",
    "frontend developer": "software_engineering",
    "full stack developer": "software_engineering",
    "fullstack engineer": "software_engineering",
    "sre": "software_engineering",
    "site reliability engineer": "software_engineering",
    "ml engineer": "data_ml",
    "machine learning": "data_ml",
    "data science": "data_ml",
    "pm": "product",
    "product management": "product",
    "ux": "design",
    "ui designer": "design",
}

SENIORITY_WORDS = (
    "junior",
    "jr",
    "senior",
    "sr",
    "staff",
    "principal",
    "lead",
    "intern",
    "associate",
    "mid",
    "mid-level",
    "entry",
    "entry-level",
)

ADJACENT_TITLE: dict[str, str] = {
    "backend engineer": "Software Engineer",
    "frontend engineer": "Software Engineer",
    "full stack engineer": "Software Engineer",
    "fullstack engineer": "Software Engineer",
    "machine learning engineer": "Data Scientist",
    "data scientist": "Machine Learning Engineer",
    "data engineer": "Analytics Engineer",
    "devops engineer": "Site Reliability Engineer",
    "product manager": "Technical Product Manager",
    "product designer": "UX Designer",
    "ux designer": "Product Designer",
    "mobile engineer": "iOS Engineer",
}

CHIPS_WEEK = "date_posted:week"

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class SearchQuery:
    q: str
    location: str | None

    def key(self) -> tuple[str, str]:
        return (self.q.strip().lower(), (self.location or "").strip().lower())


def slug_tokens(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower()))


def strip_seniority(title: str) -> str:
    tokens = title.lower().replace("/", " ").split()
    kept = [token for token in tokens if token not in SENIORITY_WORDS]
    return " ".join(kept).strip() or title.lower().strip()


def title_to_family(raw: str) -> str:
    cleaned = strip_seniority(raw)
    if cleaned in _TITLE_TO_FAMILY:
        return _TITLE_TO_FAMILY[cleaned]
    if cleaned in _TITLE_ALIASES:
        return _TITLE_ALIASES[cleaned]
    for alias, family in _TITLE_ALIASES.items():
        if alias in cleaned or cleaned in alias:
            return family
    for title, family in _TITLE_TO_FAMILY.items():
        if title in cleaned or cleaned in title:
            return family
    return cleaned or "unknown"


def has_seniority(title: str) -> bool:
    tokens = set(title.lower().replace("/", " ").split())
    return bool(tokens & set(SENIORITY_WORDS))


def adjacent_title(title: str) -> str | None:
    lowered = title.strip().lower()
    if lowered in ADJACENT_TITLE:
        return ADJACENT_TITLE[lowered]
    stripped = strip_seniority(title)
    if stripped in ADJACENT_TITLE:
        return ADJACENT_TITLE[stripped]
    if "engineer" in lowered:
        return "Software Engineer"
    return None


def seniority_variant(title: str) -> str | None:
    if has_seniority(title):
        stripped = strip_seniority(title)
        return stripped.title() if stripped else None
    return f"Senior {title.strip()}"


def is_remote_location(location: str | None) -> bool:
    if not location:
        return False
    text = location.strip().lower()
    return text in {"remote", "remote-only", "anywhere", "worldwide", "remote (us)", "remote us"}


def is_remote_only(location: str | None, work_modes: list[str] | None) -> bool:
    modes = {mode.strip().lower() for mode in (work_modes or [])}
    if modes and modes <= {"remote"}:
        return True
    return is_remote_location(location)


def expand_queries(
    title: str,
    location: str | None,
    skill: str | None,
    work_modes: list[str] | None = None,
) -> list[SearchQuery]:
    """Build ≤4 distinct (q, location) pairs for a Match / Refresh fan-out."""
    title = (title or "").strip()
    if not title:
        return []

    remote_only = is_remote_only(location, work_modes)
    city = None if remote_only or not location or is_remote_location(location) else location.strip()

    pairs: list[SearchQuery] = []
    if not remote_only:
        pairs.append(SearchQuery(title, city))
    pairs.append(SearchQuery(title, None))

    adjacent = adjacent_title(title)
    if adjacent and adjacent.lower() != title.lower():
        pairs.append(SearchQuery(adjacent, city if not remote_only else None))
    else:
        variant = seniority_variant(title)
        if variant and variant.lower() != title.lower():
            pairs.append(SearchQuery(variant, city if not remote_only else None))

    if skill and skill.strip():
        pairs.append(SearchQuery(f"{title} {skill.strip()}", city if not remote_only else None))

    seen: set[tuple[str, str]] = set()
    unique: list[SearchQuery] = []
    for query in pairs:
        key = query.key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(query)
        if len(unique) >= 4:
            break
    return unique


def allocate_standing_queries(
    demand: dict,
    budget: int = 20,
    min_slots_per_family: int = 1,
    min_slots_per_location: int = 1,
) -> list[SearchQuery]:
    """Demand-weighted 20-slot plan. Never searches a city with zero users."""
    titles = demand.get("titles") or []
    locations = demand.get("locations") or []
    user_count = int(demand.get("user_count") or 0)

    family_counts: dict[str, int] = defaultdict(int)
    family_raw: dict[str, str] = {}
    for row in titles:
        raw = str(row.get("raw") or "").strip()
        if not raw:
            continue
        family = str(row.get("family") or title_to_family(raw))
        count = int(row.get("count") or 1)
        family_counts[family] += count
        family_raw.setdefault(family, raw)

    loc_counts: dict[str | None, int] = defaultdict(int)
    for row in locations:
        normalized = str(row.get("normalized") or "").strip().lower()
        if not normalized:
            continue
        loc_counts[normalized] += int(row.get("count") or 1)

    if user_count <= 0 or not family_counts:
        family_counts = {family: 1 for family in DEFAULT_FAMILIES}
        loc_counts = {None: 1}

    loc_keys: list[str | None] = list(loc_counts.keys()) if loc_counts else [None]
    if loc_keys == []:
        loc_keys = [None]
        loc_counts = {None: 1}

    family_total = sum(family_counts.values()) or 1
    loc_total = sum(loc_counts.values()) or 1
    w_family = {family: count / family_total for family, count in family_counts.items()}
    w_loc = {loc: count / loc_total for loc, count in loc_counts.items()}

    families = sorted(family_counts, key=lambda family: (-w_family[family], family))
    if None not in loc_keys:
        if len(loc_keys) > budget:
            loc_keys = sorted(loc_keys, key=lambda loc: -w_loc.get(loc, 0))[:budget]

        floor_families = list(families)
        while floor_families and (len(floor_families) * min_slots_per_family + len(loc_keys) * min_slots_per_location) > budget:
            floor_families.pop()
        families = floor_families or families[:1]

    pair_scores: dict[tuple[str, str | None], float] = {}
    for family in families:
        for loc in loc_keys:
            pair_scores[(family, loc)] = w_family.get(family, 0) * w_loc.get(loc, 1.0 if loc is None else 0)

    slots: dict[tuple[str, str | None], int] = defaultdict(int)

    if None not in loc_keys:
        for loc in loc_keys:
            best_family = max(families, key=lambda family: pair_scores.get((family, loc), 0))
            slots[(best_family, loc)] += min_slots_per_location
        for family in families:
            best_loc = max(loc_keys, key=lambda loc: pair_scores.get((family, loc), 0))
            slots[(family, best_loc)] += min_slots_per_family
    else:
        for family in families:
            slots[(family, None)] += min_slots_per_family

    used = sum(slots.values())
    remaining = max(budget - used, 0)
    ranked_pairs = sorted(pair_scores, key=lambda pair: (-pair_scores[pair], pair[0], str(pair[1])))
    if remaining and ranked_pairs:
        weights = [pair_scores[pair] for pair in ranked_pairs]
        weight_sum = sum(weights) or 1.0
        raw = [remaining * weight / weight_sum for weight in weights]
        whole = [math.floor(value) for value in raw]
        leftover = remaining - sum(whole)
        remainders = sorted(
            range(len(ranked_pairs)),
            key=lambda index: -(raw[index] - whole[index]),
        )
        for index in remainders[:leftover]:
            whole[index] += 1
        for pair, extra in zip(ranked_pairs, whole):
            slots[pair] += extra

    queries: list[SearchQuery] = []
    seen: set[tuple[str, str]] = set()
    unused_titles: dict[tuple[str, str | None], list[str]] = {}
    leftover = 0
    ordered_slots = sorted(slots.items(), key=lambda item: -item[1])
    for (family, loc), count in ordered_slots:
        titles_for_family = list(FAMILY_TITLES.get(family) or [family_raw.get(family, family)])
        if not titles_for_family:
            titles_for_family = [family.replace("_", " ").title()]
        location = None if loc is None or is_remote_location(loc) else loc
        remaining_titles: list[str] = []
        added = 0
        for title in titles_for_family:
            query = SearchQuery(title, location)
            key = query.key()
            if key in seen:
                continue
            if added < count:
                seen.add(key)
                queries.append(query)
                added += 1
            else:
                remaining_titles.append(title)
        leftover += max(count - added, 0)
        unused_titles[(family, loc)] = remaining_titles
        if len(queries) >= budget:
            break

    if leftover and len(queries) < budget:
        donors = sorted(unused_titles.items(), key=lambda item: -pair_scores.get(item[0], 0))
        while leftover and len(queries) < budget:
            progressed = False
            for pair, titles in donors:
                if not titles or len(queries) >= budget or leftover <= 0:
                    continue
                loc = pair[1]
                location = None if loc is None or is_remote_location(loc) else loc
                title = titles.pop(0)
                query = SearchQuery(title, location)
                key = query.key()
                if key in seen:
                    continue
                seen.add(key)
                queries.append(query)
                leftover -= 1
                progressed = True
            if not progressed:
                break

    return queries[:budget]
