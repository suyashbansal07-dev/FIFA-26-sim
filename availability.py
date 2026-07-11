"""Manual squad-availability layer (injuries / suspensions).

Edit data/availability.json (see data/availability.example.json):

    {"France": [{"player": "Mbappe", "value_share": 0.25, "note": "hamstring"}]}

value_share = the player's share of the team's top-23 market value (0..1).
The summed missing share (capped at 0.5) subtracts from the team's external
strength, so it rides the capped external-prior channel (weight <= 0.15):
a fully missing star XI can dent a team's edge, never dominate the DC ratings.
No file, or an empty file, is a clean no-op.
"""
from __future__ import annotations

import json
from pathlib import Path

MAX_TEAM_SHARE = 0.5
STRENGTH_SCALE = 2.0  # ponytail: heuristic; strength is z-scored so 0.25 share ~= -0.5 z


def load_availability(path_or_paths) -> dict:
    paths = [path_or_paths] if isinstance(path_or_paths, (str, Path)) else list(path_or_paths)
    merged = {}
    for path in map(Path, paths):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        for team, players in data.items():
            by_name = {str(player.get("player", "")).casefold(): player
                       for player in merged.get(team, [])}
            for player in players if isinstance(players, list) else []:
                name = str(player.get("player", "")).casefold()
                previous = by_name.get(name)
                if previous is None or float(player.get("value_share", 0)) > float(previous.get("value_share", 0)):
                    by_name[name] = player
            merged[team] = list(by_name.values())
    return merged


def apply_availability(strength: dict, path_or_paths) -> tuple[dict, dict]:
    """Return (adjusted strength, meta). Unknown teams in the file are reported, not applied."""
    entries = load_availability(path_or_paths)
    if not entries or not strength:
        return strength, {"present": False}
    adjusted = dict(strength)
    applied, unknown = {}, []
    for team, players in entries.items():
        if team not in adjusted:
            unknown.append(team)
            continue
        share = min(MAX_TEAM_SHARE, sum(float(p.get("value_share", 0)) for p in players))
        if share > 0:
            adjusted[team] = adjusted[team] - STRENGTH_SCALE * share
            applied[team] = {"missing_value_share": round(share, 3),
                             "players": [p.get("player", "?") for p in players]}
    return adjusted, {"present": bool(applied), "applied": applied, "unknown_teams": unknown,
                      "scale": STRENGTH_SCALE, "cap": MAX_TEAM_SHARE}
