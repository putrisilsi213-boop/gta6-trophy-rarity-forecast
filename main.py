"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate
at the case's target, as five quantiles (5/25/50/75/95%) per trophy.

Model (deterministic, offline, standard library only, no model calls)
----------------------------------------------------------------------
An earn rate is a ratio, r = E / N: E = players tracked who earned the trophy,
N = players tracked. The two move for different reasons, so they are forecast
separately and divided again at the target:

* N (players tracked) grows as more people buy/play and Exophase picks them up.
  log N is fit by least squares over the last 24 h of distinct reads; its growth
  rate is assumed to decay exponentially (half-life ~2 days: launch influx
  slows), so N_target = N_last * exp(g * TAU * (1 - exp(-H / TAU))).
* E (earners) per trophy is cumulative and monotone. log E is fit the same way
  over the last 24 h (decay half-life ~1.5 days: the reachable players exhaust),
  then capped at a structural week-one ceiling for the trophy's kind (below):
  a chapter-1 trophy can reach most players in a week, a platinum or a
  250-stunt grind cannot. E never falls below its last observed value.
* The median rate is 100 * E_target / N_target. Dilution is automatic: when N
  grows faster than E, the rate falls -- exactly what Exophase's first refresh
  did on launch night (638 -> 3,868 tracked, every rate but the 100% one lower).

Why this beats "the rate stays where it is": a week after launch the player
base is several times larger and the keenest players have pushed through the
story, so early-story rates rise an order of magnitude while grind rates are
diluted down. A flat last-rate forecast misses both moves.

Flat history (no read ever changed, as in the practice cases): there is no
trend to fit, so the launch-night refresh is the prior -- the next refresh
dilutes every rate to roughly half (the one observed refresh took N 638 ->
3,868 while earners grew far slower), except a trophy already at ~100%.

Grounding for the ceilings (baked in as caps, never as the forecast itself):
GTA V's lifetime PSN rates are ~42% for the first story trophy, ~17-29% for
the final mission and 0.1% for the platinum; week-one Exophase rates run
higher than lifetime PSN averages because launch-week buyers are the keenest
players, so the caps sit above those lifetime figures for story trophies and
far below them for grinds. GTA VI pre-sold ~12.1M digital copies (Sensor
Tower, 13 Nov 2026) with 28-35M first-week copies projected, so N is capped
at 3M tracked players -- far above any plausible Exophase count by 26 Nov.

Spread: quantiles are logistic(logit(median) + z_tau * sigma) with sigma
~0.55 logit units, widened when the history is flat, the earner count is tiny,
or the log-earner fit is noisy, and narrowed for a trophy already near 100%.
The pinball loss on the logit scale rewards exactly this: an honest spread
around a median that moves with the player base.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P keeps
this file's directory OFF sys.path, so a module shipped beside main.py imports
only after the sys.path line below. This entry is a single file; there is no
sibling module.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON
    {"schema": "makefun/gta6-trophies-case/v1", "id", "at", "horizon_minutes",
    "target", "levels", "game", "trophies", "history", "notes"}, where
    `trophies` lists the case's trophies {id, name, description, tier, points,
    secret} and `history` every read before `at`, oldest first, each {slot, at,
    players_tracked, trophies_earned, rates}, `rates` in the order of
    `trophies` (null where that read lacked the trophy);
  - write <id>.out to the output directory:
    {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}, one row for
    each of the case's trophies, each id once, no other key, each q five finite
    numbers from 0 to 100 (percent), non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (this entry ships none; kept for
# parity with the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Growth-rate decay half-lives (days): launch influx slows, and a trophy's
# earner pool exhausts faster than the player base grows.
TAU_N_DAYS = 2.0
TAU_E_DAYS = 1.5
# Last-24h window used for the log-linear fits (hours).
FIT_WINDOW_HOURS = 24.0
# Flat-history dilution prior: the one observed Exophase refresh (launch night,
# 638 -> 3,868 tracked) cut every rate but the 100% one to roughly half.
FLAT_DILUTE_SHORT = 0.55  # horizon <= 6 h
FLAT_DILUTE_LONG = 0.40   # horizon > 6 h
# Plausible ceiling for Exophase-tracked GTA VI players by 26 Nov 2026.
N_CAP = 3_000_000


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def logistic(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> float:
    """ISO-8601 UTC timestamp -> epoch seconds."""
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def least_squares_slope(xs, ys):
    """Slope and residual std of y ~ a + b*x by least squares. Returns (b, resid_std)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return b, resid


def decayed_growth(slope_per_day: float, horizon_days: float, tau_days: float) -> float:
    """log-growth over the horizon when the growth rate decays with half-life
    tau: integral of slope * exp(-t / tau) from 0 to horizon."""
    return slope_per_day * tau_days * (1.0 - math.exp(-horizon_days / tau_days))


def trophy_ceiling(trophy: dict) -> float:
    """Week-one ceiling for the share of tracked players who can hold this
    trophy, from what the trophy asks for (name + description + tier).

    Grounded in GTA V's lifetime PSN rates (~42% first story trophy, ~17-29%
    final mission, 0.1% platinum): week-one Exophase rates run higher for
    story (launch buyers are the keenest players) and far lower for grinds
    that need days of play. These are caps only -- the trend fit sets the
    forecast, the cap only stops a runaway exponential.
    """
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    tier = trophy.get("tier", "")
    if tier == "platinum":
        return 0.02
    if "100% completion" in text:
        return 0.03
    if "final mission for" in text:
        return 0.25  # side-character mission strands
    if "complete the final mission" in text:
        return 0.35
    for chapter, cap in (("chapter 1", 0.85), ("chapter 2", 0.75),
                         ("chapter 3", 0.65), ("chapter 4", 0.55)):
        if chapter in text:
            return cap
    grind = (
        ("250 stunts", 0.06),
        ("3 collectible card sets", 0.05),
        ("100 locations", 0.06),
        ("gold medal in all shooting range", 0.05),
        ("top fun drone race at every location", 0.05),
        ("50 challenges", 0.10),
        ("meditate at every spot", 0.08),
        ("rob 5 bank vaults", 0.08),
        ("earn $250,000 laundering", 0.08),
        ("purchase all garage properties", 0.10),
        ("win 20 different races", 0.12),
        ("explore all of leonida", 0.12),
        ("complete 8 mission strands", 0.15),
        ("register 10 vehicles", 0.15),
        ("spend $100,000", 0.15),
        ("discover 100 vehicles", 0.35),
        ("try each type of activity", 0.45),
        ("complete an arcade game", 0.40),
        ("beat cal in a fishing", 0.35),
    )
    for needle, cap in grind:
        if needle in text:
            return cap
    return {"gold": 0.35, "silver": 0.15, "bronze": 0.30}.get(tier, 0.30)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    if not history:
        # No reads at all: say ~1% everywhere with a wide spread, stay valid.
        return {"trophies": [
            {"id": t["id"], "q": [round4(logistic(logit(1.0) + z * 1.0)) for z in Z]}
            for t in trophies
        ]}

    # Distinct snapshots: collapse consecutive identical reads (Exophase holds a
    # page unchanged between refreshes) keeping the first time each state appears.
    snaps = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if not snaps or snaps[-1][0] != key:
            snaps.append((key, parse_time(read["at"])))
    t0 = snaps[0][1]
    times_h = [(ts - t0) / 3600.0 for _, ts in snaps]
    last_n = snaps[-1][0][0]
    last_rates = snaps[-1][0][1]
    flat = len(snaps) == 1

    # --- Forecast N (players tracked) at the target. ---
    window = [i for i, th in enumerate(times_h)
              if times_h[-1] - th <= FIT_WINDOW_HOURS] or list(range(len(times_h)))
    n_pts = [(times_h[i], math.log(max(1, snaps[i][0][0]))) for i in window]
    slope_n_per_day = 0.0
    if len(n_pts) >= 2:
        b, _ = least_squares_slope([p[0] / 24.0 for p in n_pts], [p[1] for p in n_pts])
        slope_n_per_day = max(0.0, b)  # tracked players never shrink
    n_target = last_n * math.exp(
        decayed_growth(slope_n_per_day, horizon_days, TAU_N_DAYS))
    n_target = min(max(n_target, last_n), min(N_CAP, 10.0 * max(1, last_n)))

    rows = []
    for i, trophy in enumerate(trophies):
        # Last non-null rate for this trophy.
        last_rate = None
        for key, _ in reversed(snaps):
            if key[1][i] is not None:
                last_rate = key[1][i]
                break
        if last_rate is None:
            last_rate = 1.0
        e_last = last_rate / 100.0 * last_n
        cap = trophy_ceiling(trophy)

        if flat:
            # No trend ever observed: the launch-night refresh is the prior.
            if last_rate >= 99.5:
                median = 99.9
                sigma = 0.08
            elif last_rate <= 0.0:
                median = 0.02
                sigma = 0.8
            else:
                dilute = FLAT_DILUTE_SHORT if case["horizon_minutes"] <= 360 else FLAT_DILUTE_LONG
                median = max(0.01, last_rate * dilute)
                sigma = 0.8
            rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
            continue

        # --- Forecast E (earners) at the target. ---
        # (th, earners, players) triples, earners cumulative and monotone
        # (2-decimal rounding wobbles smoothed out).
        mono = []
        running = 0.0
        for (n, rates), th in zip([s[0] for s in snaps], times_h):
            r = rates[i]
            if r is None:
                continue
            running = max(running, r / 100.0 * n)
            mono.append((th, running, n))
        e_last = mono[-1][1] if mono else e_last

        slope_e_per_day = 0.0
        resid = 0.0
        fit_pts = [(th, math.log(e)) for th, e, _ in mono
                   if times_h[-1] - th <= FIT_WINDOW_HOURS and e > 0]
        if len(fit_pts) >= 2:
            slope_e_per_day, resid = least_squares_slope(
                [p[0] / 24.0 for p in fit_pts], [p[1] for p in fit_pts])
            slope_e_per_day = max(0.0, slope_e_per_day)
        if e_last <= 0:
            # Nobody holds it yet: a trickle by the target, under the ceiling.
            e_target = 0.0005 * n_target
        else:
            e_logfit = e_last * math.exp(
                decayed_growth(slope_e_per_day, horizon_days, TAU_E_DAYS))
            # Cohort term: the players who arrive by the target earn the trophy
            # at the marginal rate the newest tracked players show (the share
            # of the last 24 h's new players who hold it). For a trophy
            # everyone earns on arrival this keeps the rate at ~100% instead
            # of diluting it away.
            win = [(th, e, n) for th, e, n in mono
                   if times_h[-1] - th <= FIT_WINDOW_HOURS]
            if len(win) >= 2 and win[-1][2] > win[0][2]:
                marg = (win[-1][1] - win[0][1]) / (win[-1][2] - win[0][2])
                marg = min(1.0, max(0.0, marg))
            else:
                marg = last_rate / 100.0
            e_cohort = e_last + marg * max(0.0, n_target - last_n)
            e_target = max(e_logfit, e_cohort)
        # The structural ceiling caps growth, never the level already reached:
        # a trophy above its ceiling (e.g. already at 100%) keeps its level.
        cap_eff = max(cap, last_rate / 100.0)
        e_target = min(max(e_target, e_last), cap_eff * n_target, n_target)

        median = 100.0 * e_target / n_target
        if last_rate >= 99.5 and median >= 99.0:
            sigma = 0.10
        else:
            sigma = 0.55
            if e_last < 20:
                sigma += 0.15  # small-count rounding noise
            sigma += min(0.30, resid)  # noisy log-earner fit
            sigma = min(1.15, sigma)
        median = clamp(median)
        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
    return {"trophies": rows}


def quantiles(median: float, sigma: float):
    """Five quantiles around the median on the logit scale, rounded, monotone."""
    lg = logit(median)
    out = []
    prev = 0.01
    for z in Z:
        q = round4(logistic(lg + z * sigma))
        q = float(q)
        q = min(100.0, max(0.01, q))
        q = max(q, prev) if out else q
        out.append(q)
        prev = q
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
