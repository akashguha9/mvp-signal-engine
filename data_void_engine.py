"""
data_void_engine.py
====================
Fallback inference stack for CAL DATA VOID condition.
Integrates with signal_engine.py's infer_signal() as a drop-in upgrade.

When D_t = ∅ OR |D_t| < k:
    Signal_t = α·Historical + β·Proxy + γ·Narrative + δ·Simulation + ε·Prior
    α + β + γ + δ + ε = 1
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from typing import Optional

import numpy as np
import pandas as pd


# ── STRUCTURAL PRIORS ──────────────────────────────────────────────────────────

STRUCTURAL_PRIORS: dict[str, dict] = {
    "oil_up": {"inflation": +1, "equities": -1, "bonds": -1, "gold": +1},
    "oil_down": {"inflation": -1, "equities": +1, "bonds": +1, "gold": -1},
    "war_escalation": {"risk_off": +1, "bonds": +1, "gold": +1, "equities": -1},
    "war_deescalation": {"risk_off": -1, "bonds": -1, "gold": -1, "equities": +1},
    "fed_hike": {"bonds": -1, "equities": -1, "usd": +1, "gold": -1},
    "fed_cut": {"bonds": +1, "equities": +1, "usd": -1, "gold": +1},
    "recession_fear": {"equities": -1, "bonds": +1, "gold": +1, "usd": +1},
    "inflation_high": {"bonds": -1, "equities": -1, "gold": +1, "commodities": +1},
    "election_risk": {"volatility": +1, "equities": -1, "bonds": +0.5},
    "sanctions": {"target_equities": -1, "commodities": +1, "risk_off": +1},
    "default_risk": {"bonds": -1, "equities": -1, "cds": +1},
    "supply_shock": {"inflation": +1, "equities": -1, "commodities": +1},
}

PROXY_MAP: dict[str, list[str]] = {
    "SPY": ["QQQ", "IVV", "VTI", "ES=F"],
    "QQQ": ["SPY", "XLK", "SOXX"],
    "GLD": ["IAU", "GC=F", "SLV"],
    "USO": ["BNO", "CL=F", "XLE"],
    "TLT": ["IEF", "BND", "ZB=F"],
    "EEM": ["VWO", "IEMG", "EWZ"],
    "VXX": ["UVXY", "VIXY", "^VIX"],
    "EWZ": ["EEM", "BRAZIL_ETF"],
    "UNG": ["FCG", "XLE"],
    "BTC": ["ETH", "GBTC", "BITO"],
    "DXY": ["UUP", "EURUSD"],
    "NIKKEI": ["EWJ", "^N225"],
    "DAX": ["EWG", "^GDAXI"],
}

REGIME_PRIORS: dict[str, int] = {
    "DISINFLATIONARY_RISK_ON": +1,
    "INFLATION_SQUEEZE": -1,
    "GROWTH_SCARE": -1,
    "LIQUIDITY_FLOOD": +1,
    "CREDIT_CRUNCH": -1,
    "GEOPOLITICAL_FRAGMENTATION": 0,
    "TREND": +1,
    "MEAN_REVERSION": 0,
    "SHOCK": -1,
    "MANIPULATED": 0,
    "DEAD_ZONE": 0,
    "KALI_YUGA": 0,
}


# ── DATA VOID DETECTION ────────────────────────────────────────────────────────

def detect_data_void(row: pd.Series, min_score: float = 0.3) -> tuple[bool, list[str]]:
    """
    Returns (is_void, reasons).

    DATA VOID when:
      - score is NaN or below threshold
      - lag is NaN
      - all forward returns are missing
    """
    reasons: list[str] = []

    score = row.get("match_score")
    lag = row.get("lead_lag_minutes")

    if pd.isna(score):
        reasons.append("match_score_missing")
    elif score < min_score:
        reasons.append(f"match_score_low ({score:.3f}<{min_score})")

    if pd.isna(lag):
        reasons.append("lead_lag_missing")

    fwd_cols = [c for c in row.index if c.startswith("future_return_")]
    if fwd_cols and all(pd.isna(row.get(c)) for c in fwd_cols):
        reasons.append("future_returns_all_missing")

    return bool(reasons), reasons


# ── LAYER 1: HISTORICAL ANALOGUE ──────────────────────────────────────────────

def layer1_historical(row: pd.Series, df_history: Optional[pd.DataFrame]) -> tuple[float, float]:
    if df_history is None or df_history.empty:
        return 0.0, 0.0

    seed = row.get("seed_label", "")
    vol = row.get("t0_volatility_20d", np.nan)
    trend = row.get("t0_trend_50_200", np.nan)

    analogues = df_history.copy()

    if seed and "seed_label" in analogues.columns:
        analogues = analogues[analogues["seed_label"] == seed]

    if pd.notna(trend) and "t0_trend_50_200" in analogues.columns:
        same_sign = analogues["t0_trend_50_200"].apply(
            lambda x: np.sign(x) == np.sign(trend) if pd.notna(x) else False
        )
        analogues = analogues[same_sign]

    if pd.notna(vol) and "t0_volatility_20d" in analogues.columns and not analogues.empty:
        q25 = analogues["t0_volatility_20d"].quantile(0.25)
        q75 = analogues["t0_volatility_20d"].quantile(0.75)
        if pd.notna(q25) and pd.notna(q75):
            if vol < q25:
                analogues = analogues[analogues["t0_volatility_20d"] < q25]
            elif vol > q75:
                analogues = analogues[analogues["t0_volatility_20d"] > q75]

    if len(analogues) < 3:
        return 0.0, 0.0

    fwd_col = "future_return_1d"
    if fwd_col not in analogues.columns:
        fwd_cols = sorted([c for c in analogues.columns if c.startswith("future_return_")])
        if not fwd_cols:
            return 0.0, 0.0
        fwd_col = fwd_cols[0]

    returns = analogues[fwd_col].dropna()
    if len(returns) < 3:
        return 0.0, 0.0

    direction = np.sign(returns.mean())
    confidence = float((np.sign(returns) == direction).mean()) * 0.60
    return float(direction), float(confidence)


# ── LAYER 2: CROSS-ASSET PROXY ────────────────────────────────────────────────

def layer2_proxy(row: pd.Series) -> tuple[float, float]:
    signals: list[float] = []

    mom20 = row.get("t0_momentum_20d")
    mom60 = row.get("t0_momentum_60d")
    trend = row.get("t0_trend_50_200")

    if pd.notna(mom20):
        signals.append(np.sign(mom20))
    if pd.notna(mom60):
        signals.append(np.sign(mom60))
    if pd.notna(trend):
        signals.append(np.sign(trend))

    if not signals:
        return 0.0, 0.0

    direction = np.sign(np.mean(signals))
    if direction == 0:
        return 0.0, 0.0

    agreement = float((np.array(signals) == direction).mean())
    confidence = agreement * 0.50
    return float(direction), float(confidence)


# ── LAYER 3: NARRATIVE CONTINUATION ───────────────────────────────────────────

def layer3_narrative(row: pd.Series) -> tuple[float, float]:
    drawdown = row.get("t0_drawdown")
    mom20 = row.get("t0_momentum_20d")
    trend = row.get("t0_trend_50_200")

    narrative_score = 0.0
    count = 0

    if pd.notna(drawdown):
        if drawdown < -0.15:
            narrative_score -= 1.0
        elif drawdown > -0.05:
            narrative_score += 0.5
        count += 1

    if pd.notna(mom20):
        narrative_score += np.sign(mom20) * 0.7
        count += 1

    if pd.notna(trend):
        narrative_score += np.sign(trend) * 0.5
        count += 1

    if count == 0:
        return 0.0, 0.0

    avg_score = narrative_score / count
    direction = np.sign(avg_score)
    confidence = min(0.40, abs(avg_score) * 0.40)

    if direction == 0:
        return 0.0, 0.0

    return float(direction), float(confidence)


# ── LAYER 4: PROBABILISTIC SIMULATION ─────────────────────────────────────────

def layer4_simulation(row: pd.Series, n_scenarios: int = 500) -> tuple[float, float]:
    vol = row.get("t0_volatility_20d", 0.02)
    mom20 = row.get("t0_momentum_20d", 0.0)
    trend = row.get("t0_trend_50_200", 0.0)

    if pd.isna(vol):
        vol = 0.02
    if pd.isna(mom20):
        mom20 = 0.0
    if pd.isna(trend):
        trend = 0.0

    drift = np.sign(mom20) * 0.002 + np.sign(trend) * 0.001

    rng = np.random.default_rng(42)
    simulated_returns = rng.normal(drift, vol, n_scenarios)
    positive_fraction = float((simulated_returns > 0).mean())

    if positive_fraction > 0.55:
        direction = 1.0
        confidence = (positive_fraction - 0.5) * 2 * 0.50
    elif positive_fraction < 0.45:
        direction = -1.0
        confidence = (0.5 - positive_fraction) * 2 * 0.50
    else:
        direction = 0.0
        confidence = 0.0

    return float(direction), float(confidence)


# ── LAYER 5: STRUCTURAL PRIOR ─────────────────────────────────────────────────

def layer5_prior(row: pd.Series) -> tuple[float, float]:
    seed = str(row.get("seed_label", "")).lower().replace(" ", "_")
    label = str(row.get("reason", "")).lower().replace(" ", "_")

    matched_priors: list[float] = []

    for trigger, effects in STRUCTURAL_PRIORS.items():
        trigger_norm = trigger.lower().replace(" ", "_")
        if trigger_norm in seed or trigger_norm in label:
            directions = [
                v for k, v in effects.items()
                if k not in {"inflation", "volatility", "risk_off"}
            ]
            if directions:
                matched_priors.append(np.sign(np.mean(directions)))

    if not matched_priors:
        regime = str(row.get("regime", "")).upper()
        for regime_key, direction in REGIME_PRIORS.items():
            if regime_key in regime:
                return float(direction), 0.30
        return 0.0, 0.0

    direction = np.sign(np.mean(matched_priors))
    confidence = min(0.45, len(matched_priors) * 0.15)

    if direction == 0:
        return 0.0, 0.0

    return float(direction), float(confidence)


# ── WEIGHT ALLOCATION ──────────────────────────────────────────────────────────

def allocate_weights(
    l1_conf: float,
    l2_conf: float,
    l3_conf: float,
    l4_conf: float,
    l5_conf: float,
) -> tuple[float, float, float, float, float]:
    raw = np.array([l1_conf, l2_conf, l3_conf, l4_conf, l5_conf], dtype=float)
    total = raw.sum()

    if total <= 0:
        return 0.20, 0.20, 0.20, 0.20, 0.20

    weights = raw / total
    return tuple(float(x) for x in weights)


# ── MASTER VOID INFERENCE ─────────────────────────────────────────────────────

def infer_from_void(
    row: pd.Series,
    df_history: Optional[pd.DataFrame] = None,
    min_confidence_threshold: float = 0.15,
    force_direction: bool = False,
) -> dict:
    l1_dir, l1_conf = layer1_historical(row, df_history)
    l2_dir, l2_conf = layer2_proxy(row)
    l3_dir, l3_conf = layer3_narrative(row)
    l4_dir, l4_conf = layer4_simulation(row)
    l5_dir, l5_conf = layer5_prior(row)

    alpha, beta, gamma, delta, epsilon = allocate_weights(
        l1_conf, l2_conf, l3_conf, l4_conf, l5_conf
    )

    composite = (
        alpha * l1_dir +
        beta * l2_dir +
        gamma * l3_dir +
        delta * l4_dir +
        epsilon * l5_dir
    )

    overall_confidence = (
        alpha * l1_conf +
        beta * l2_conf +
        gamma * l3_conf +
        delta * l4_conf +
        epsilon * l5_conf
    )

    layers_used: list[str] = []
    proxies_used: list[str] = []

    if l1_conf > 0:
        layers_used.append("L1_historical")

    if l2_conf > 0:
        layers_used.append("L2_proxy")
        seed = str(row.get("seed_label", "")).upper()

        matched_proxy: list[str] = []
        for key, vals in PROXY_MAP.items():
            if key in seed:
                matched_proxy.extend(vals[:2])

        proxies_used.extend(matched_proxy)
        proxies_used.extend(["t0_momentum_20d", "t0_momentum_60d", "t0_trend_50_200"])

    if l3_conf > 0:
        layers_used.append("L3_narrative")
    if l4_conf > 0:
        layers_used.append("L4_simulation")
    if l5_conf > 0:
        layers_used.append("L5_prior")

    if force_direction:
        if composite >= 0:
            signal = 1
            direction_str = "UP"
            reason = "void_forced_up" if overall_confidence < min_confidence_threshold else "void_inferred_up"
        else:
            signal = -1
            direction_str = "DOWN"
            reason = "void_forced_down" if overall_confidence < min_confidence_threshold else "void_inferred_down"
    else:
        if overall_confidence < min_confidence_threshold or composite == 0:
            signal = 0
            direction_str = "NEUTRAL"
            reason = "void_inferred_neutral_low_conf"
        elif composite > 0.10:
            signal = 1
            direction_str = "UP"
            reason = "void_inferred_up"
        elif composite < -0.10:
            signal = -1
            direction_str = "DOWN"
            reason = "void_inferred_down"
        else:
            signal = 0
            direction_str = "NEUTRAL"
            reason = "void_inferred_neutral_weak_signal"

    reason_parts = [reason]

    trend = row.get("t0_trend_50_200")
    if pd.notna(trend):
        reason_parts.append("uptrend" if trend > 0 else "downtrend" if trend < 0 else "flattrend")

    vol = row.get("t0_volatility_20d")
    if pd.notna(vol):
        if vol > 0.04:
            reason_parts.append("high_vol")
        elif vol < 0.015:
            reason_parts.append("low_vol")

    reason = "_".join(reason_parts)

    return {
        "data_void_detected": True,
        "fallback_layers_used": layers_used,
        "proxy_signals_used": proxies_used,
        "confidence_score": round(float(overall_confidence), 4),
        "estimated_direction": direction_str,
        "signal": int(signal),
        "reason": reason,
        "composite_score": round(float(composite), 4),
        "alpha": round(float(alpha), 3),
        "beta": round(float(beta), 3),
        "gamma": round(float(gamma), 3),
        "delta": round(float(delta), 3),
        "epsilon": round(float(epsilon), 3),
        "l1_dir": float(l1_dir), "l1_conf": float(l1_conf),
        "l2_dir": float(l2_dir), "l2_conf": float(l2_conf),
        "l3_dir": float(l3_dir), "l3_conf": float(l3_conf),
        "l4_dir": float(l4_dir), "l4_conf": float(l4_conf),
        "l5_dir": float(l5_dir), "l5_conf": float(l5_conf),
    }


# ── STANDARD NON-VOID INFERENCE ───────────────────────────────────────────────

def infer_standard_signal(
    row: pd.Series,
    min_score: float = 0.3,
) -> tuple[int, str]:
    """
    Standard path for rows that are not in DATA VOID.
    Adds tie-break logic so matched rows do not default to neutral too easily.
    """
    score = row.get("match_score")
    lag = row.get("lead_lag_minutes")
    vol = row.get("t0_volatility_20d")
    trend = row.get("t0_trend_50_200")
    mom20 = row.get("t0_momentum_20d")

    signal = 0
    reason = "no_signal"

    if score >= min_score and lag > 30:
        signal = 1
        reason = "news_led"
    elif score >= min_score and lag < -30:
        signal = -1
        reason = "market_led"
    elif score >= min_score:
        tie = 0

        if pd.notna(trend):
            tie += int(np.sign(trend))
        if pd.notna(mom20):
            tie += int(np.sign(mom20))

        if tie > 0:
            signal = 1
            reason = "matched_tiebreak_up"
        elif tie < 0:
            signal = -1
            reason = "matched_tiebreak_down"
        else:
            signal = 0
            reason = "matched_neutral"

    if pd.notna(vol) and vol > 0.04:
        reason += "_high_vol"

    if pd.notna(trend):
        if trend > 0:
            reason += "_uptrend"
        elif trend < 0:
            reason += "_downtrend"

    if pd.notna(mom20):
        if mom20 > 0:
            reason += "_mom_up"
        elif mom20 < 0:
            reason += "_mom_down"

    return int(signal), reason


# ── DROP-IN REPLACEMENT FOR signal_engine.infer_signal() ──────────────────────

def infer_signal_with_void_fallback(
    row: pd.Series,
    df_history: Optional[pd.DataFrame] = None,
    min_score: float = 0.3,
    force_direction: bool = False,
) -> tuple[int, str]:
    is_void, _ = detect_data_void(row, min_score=min_score)

    if not is_void:
        return infer_standard_signal(row=row, min_score=min_score)

    result = infer_from_void(
        row=row,
        df_history=df_history,
        force_direction=force_direction,
    )
    return int(result["signal"]), str(result["reason"])


# ── REPORTING ──────────────────────────────────────────────────────────────────

def format_void_report(result: dict) -> str:
    lines = [
        "━━ DATA VOID INFERENCE REPORT ━━",
        f"  Data Void Detected:    {result['data_void_detected']}",
        f"  Fallback Layers Used:  {', '.join(result['fallback_layers_used']) or 'none'}",
        f"  Proxy Signals Used:    {', '.join(result['proxy_signals_used']) or 'none'}",
        f"  Confidence Score:      {result['confidence_score']:.4f}",
        f"  Estimated Direction:   {result['estimated_direction']}",
        f"  Composite Score:       {result['composite_score']:+.4f}",
        f"  Weights (α,β,γ,δ,ε):  "
        f"{result['alpha']:.2f}, {result['beta']:.2f}, {result['gamma']:.2f}, "
        f"{result['delta']:.2f}, {result['epsilon']:.2f}",
        f"  Signal:                {result['signal']} ({result['reason']})",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(lines)


# ── SELF-TEST ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running data_void_engine self-test...\n")

    row_void = pd.Series({
        "match_score": np.nan,
        "lead_lag_minutes": np.nan,
        "t0_volatility_20d": 0.025,
        "t0_momentum_20d": 0.03,
        "t0_momentum_60d": 0.05,
        "t0_trend_50_200": 0.01,
        "t0_drawdown": -0.08,
        "seed_label": "macro_spy",
    })

    is_void, reasons = detect_data_void(row_void)
    print(f"Test 1 — Complete void: {is_void} | Reasons: {reasons}")
    result = infer_from_void(row_void)
    print(format_void_report(result))
    signal, reason = infer_signal_with_void_fallback(row_void)
    print(f"  → infer_signal_with_void_fallback: signal={signal}, reason={reason}\n")

    row_low = pd.Series({
        "match_score": 0.10,
        "lead_lag_minutes": 15.0,
        "t0_volatility_20d": 0.045,
        "t0_momentum_20d": -0.02,
        "t0_momentum_60d": -0.04,
        "t0_trend_50_200": -0.005,
        "t0_drawdown": -0.20,
        "seed_label": "oil_geopolitics",
        "reason": "war_escalation",
    })

    is_void, reasons = detect_data_void(row_low)
    print(f"Test 2 — Low score void: {is_void} | Reasons: {reasons}")
    result2 = infer_from_void(row_low)
    print(format_void_report(result2))
    signal2, reason2 = infer_signal_with_void_fallback(row_low)
    print(f"  → infer_signal_with_void_fallback: signal={signal2}, reason={reason2}\n")

    row_good_up = pd.Series({
        "match_score": 0.75,
        "lead_lag_minutes": 45.0,
        "t0_volatility_20d": 0.018,
        "t0_momentum_20d": 0.04,
        "t0_momentum_60d": 0.06,
        "t0_trend_50_200": 0.002,
        "t0_drawdown": -0.03,
        "seed_label": "macro_spy",
    })

    is_void3, reasons3 = detect_data_void(row_good_up)
    signal3, reason3 = infer_signal_with_void_fallback(row_good_up)
    print(f"Test 3 — Good data (no void): is_void={is_void3} | Reasons: {reasons3}")
    print(f"  → signal={signal3}, reason={reason3}\n")

    row_good_tiebreak = pd.Series({
        "match_score": 0.62,
        "lead_lag_minutes": 5.0,
        "t0_volatility_20d": 0.012,
        "t0_momentum_20d": 0.03,
        "t0_momentum_60d": 0.02,
        "t0_trend_50_200": 0.004,
        "t0_drawdown": -0.02,
        "seed_label": "macro_spy",
    })

    is_void4, reasons4 = detect_data_void(row_good_tiebreak)
    signal4, reason4 = infer_signal_with_void_fallback(row_good_tiebreak)
    print(f"Test 4 — Good matched tie-break row: is_void={is_void4} | Reasons: {reasons4}")
    print(f"  → signal={signal4}, reason={reason4}")

    print("\n✓ Self-test complete.")