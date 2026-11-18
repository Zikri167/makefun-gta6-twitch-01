import datetime
import json
import math
import os

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
SPREAD_GRID = [0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.45, 1.7, 2.0]
CONFIGS = ["equal", "platform", "platform_shape", "platform_shape_scale"]
MAX_Q = 1_000_000_000.0
MIN_Q = 1e-6


def parse_ts(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_and_sort(qs):
    out = []
    prev = MIN_Q
    for q in qs:
        if not math.isfinite(q):
            q = prev
        q = max(MIN_Q, min(MAX_Q, q))
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def weighted_quantiles(pairs, levels):
    pairs = sorted(pairs, key=lambda x: x[0])
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        n = len(pairs)
        pairs = [(v, 1.0 / n) for v, _ in pairs]
    else:
        pairs = [(v, w / total_w) for v, w in pairs]

    cws = []
    acc = 0.0
    for _, w in pairs:
        cws.append(acc + 0.5 * w)
        acc += w

    res = []
    n = len(pairs)
    for p in levels:
        if p <= cws[0]:
            res.append(pairs[0][0])
        elif p >= cws[-1]:
            res.append(pairs[-1][0])
        else:
            for i in range(n - 1):
                if cws[i] <= p <= cws[i + 1]:
                    denom = cws[i + 1] - cws[i]
                    t = (p - cws[i]) / denom if denom > 0 else 0.0
                    res.append(pairs[i][0] + t * (pairs[i + 1][0] - pairs[i][0]))
                    break
    return res


def winsorized_quantiles(pairs, levels):
    q25, q50, q75 = weighted_quantiles(pairs, [0.25, 0.5, 0.75])
    iqr = max(0.1, q75 - q25)
    lo = q50 - 2.5 * iqr
    hi = q50 + 2.5 * iqr
    clipped = [(min(hi, max(lo, v)), w) for v, w in pairs]
    return weighted_quantiles(clipped, levels)


def extract_pre_shape(hourly, day_start_dt):
    vals = {}
    for pt in hourly or []:
        try:
            ts = pt.get("from") or pt.get("at")
            v = pt.get("viewers")
            if ts is None or v is None or v <= 0:
                continue
            off = round((parse_ts(ts) - day_start_dt).total_seconds() / 3600)
            if -72 <= off <= -1:
                vals[off] = math.log(max(1.0, float(v)))
        except Exception:
            continue
    if not vals:
        return {}
    mean_v = sum(vals.values()) / len(vals)
    return {k: v - mean_v for k, v in vals.items()}


def shape_similarity(s1, s2):
    common = [k for k in s1 if k in s2]
    if len(common) < 6:
        return 1.0
    mse = sum((s1[k] - s2[k]) ** 2 for k in common) / len(common)
    return math.exp(-mse)


def scale_similarity(log_v1_a, log_v1_b):
    diff = (log_v1_a - log_v1_b) / 1.5
    return math.exp(-0.5 * diff * diff)


def is_console_only(platforms):
    p = (platforms or "").lower()
    return "console-only" in p or ("console" in p and "pc" not in p)


def compute_weight(cfg, c, target_console, target_shape, target_log_v1):
    if cfg == "equal":
        return 1.0
    pw = 2.0 if c["console_only"] == target_console else 1.0
    if cfg == "platform":
        return pw
    sw = shape_similarity(target_shape, c["shape"])
    if cfg == "platform_shape":
        return pw * sw
    scw = scale_similarity(target_log_v1, c["log_v1"])
    return pw * sw * scw


def pinball_loss(y, qs):
    loss = 0.0
    for p, q in zip(LEVELS, qs):
        diff = y - q
        loss += p * diff if diff >= 0 else (p - 1.0) * diff
    return loss


def persistence_forecast(v1):
    log_v1 = math.log(max(1.0, v1))
    hours = []
    for h in range(2, 25):
        spread = 0.15 * math.sqrt(h - 1)
        qs = [math.exp(log_v1 + z * spread) for z in Z_SCORES]
        hours.append({"hour": h, "q": clamp_and_sort(qs)})
    return {"hours": hours}


def parse_comparable(comp):
    try:
        ds = parse_ts(comp["day_start"])
        hourly = comp.get("hourly") or []
        by_off = {}
        for pt in hourly:
            try:
                ts = pt.get("from")
                v = pt.get("viewers")
                if ts is None or v is None or v <= 0:
                    continue
                off = round((parse_ts(ts) - ds).total_seconds() / 3600)
                if 0 <= off <= 23:
                    by_off[off] = float(v)
            except Exception:
                continue
        if 0 not in by_off or by_off[0] <= 0:
            return None
        log_v1 = math.log(max(1.0, by_off[0]))
        log_ratios = {
            off + 1: math.log(max(1.0, by_off[off])) - log_v1
            for off in range(1, 24)
            if off in by_off
        }
        if not log_ratios:
            return None
        return {
            "name": str(comp.get("name", "")),
            "console_only": is_console_only(comp.get("platforms", "")),
            "shape": extract_pre_shape(hourly, ds),
            "log_v1": log_v1,
            "log_ratios": log_ratios,
        }
    except Exception:
        return None


def calibrate_loo(comps):
    if len(comps) < 3:
        return "platform_shape_scale", 1.0
    best_loss = float("inf")
    best_pair = ("platform_shape_scale", 1.0)
    for cfg in CONFIGS:
        losses = [0.0] * len(SPREAD_GRID)
        count = 0
        for i, target in enumerate(comps):
            others = [c for j, c in enumerate(comps) if j != i]
            weights = [
                compute_weight(cfg, c, target["console_only"], target["shape"], target["log_v1"])
                for c in others
            ]
            for h, y in target["log_ratios"].items():
                pairs = [
                    (c["log_ratios"][h], w)
                    for c, w in zip(others, weights)
                    if h in c["log_ratios"]
                ]
                if not pairs:
                    continue
                base_q = winsorized_quantiles(pairs, LEVELS)
                med = base_q[2]
                for idx, s in enumerate(SPREAD_GRID):
                    scaled = [med + s * (q - med) for q in base_q]
                    losses[idx] += pinball_loss(y, scaled)
                count += 1
        if count > 0:
            for idx, s in enumerate(SPREAD_GRID):
                mean_loss = losses[idx] / count
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_pair = (cfg, s)
    return best_pair


def forecast_case(case):
    known = case.get("known") or []
    v1 = None
    if known and isinstance(known, list):
        try:
            val = float(known[0].get("viewers", 0))
            if val > 0 and math.isfinite(val):
                v1 = val
        except Exception:
            v1 = None
    if v1 is None:
        return persistence_forecast(1000.0)

    raw_comps = sorted(
        case.get("comparables") or [],
        key=lambda c: (str(c.get("name", "")), str(c.get("day_start", ""))),
    )
    comps = [c for c in (parse_comparable(rc) for rc in raw_comps) if c is not None]
    if not comps:
        return persistence_forecast(v1)

    try:
        target_ds = parse_ts(case["day_start"])
        target_shape = extract_pre_shape(case.get("history_hourly") or case.get("history"), target_ds)
    except Exception:
        target_shape = {}

    log_v1 = math.log(max(1.0, v1))
    best_cfg, spread_mult = calibrate_loo(comps)
    weights = [compute_weight(best_cfg, c, True, target_shape, log_v1) for c in comps]

    raw_log_qs = {}
    for h in range(2, 25):
        pairs = [
            (c["log_ratios"][h], w)
            for c, w in zip(comps, weights)
            if h in c["log_ratios"]
        ]
        if not pairs:
            spread = 0.15 * math.sqrt(h - 1)
            raw_log_qs[h] = [log_v1 + z * spread for z in Z_SCORES]
        else:
            base_q = winsorized_quantiles(pairs, LEVELS)
            med = base_q[2]
            raw_log_qs[h] = [log_v1 + med + spread_mult * (q - med) for q in base_q]

    hours = []
    for h in range(2, 25):
        if h == 2:
            smoothed = [0.75 * raw_log_qs[2][k] + 0.25 * raw_log_qs[3][k] for k in range(5)]
        elif h == 24:
            smoothed = [0.25 * raw_log_qs[23][k] + 0.75 * raw_log_qs[24][k] for k in range(5)]
        else:
            smoothed = [
                0.2 * raw_log_qs[h - 1][k] + 0.6 * raw_log_qs[h][k] + 0.2 * raw_log_qs[h + 1][k]
                for k in range(5)
            ]
        qs = [math.exp(lq) for lq in smoothed]
        hours.append({"hour": h, "q": clamp_and_sort(qs)})
    return {"hours": hours}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for fname in files:
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname[:-3] + ".out")
        try:
            with open(in_path, "r", encoding="utf-8") as f:
                case = json.load(f)
            result = forecast_case(case)
        except Exception:
            result = persistence_forecast(1000.0)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
