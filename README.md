# MVP Signal Engine

A research pipeline for building and testing a multi-layer event-to-market signal engine.

## Current Status

The pipeline is structurally functional:

- historical seed-index panel is built
- event matching is working
- leakage checks are clean
- timeseries dataset builds correctly
- signals and validation reports generate correctly

However, the model is **not yet stable enough for live trading**.

## Honest State of the Project

### What is working
- data pipeline
- historical joins
- trend normalization
- leakage / timestamp ordering checks
- rolling and walk-forward validation tooling

### What is not yet solved
- stable edge across regimes
- robust rule families that survive strict rolling validation
- live-trading readiness

At the moment, this should be treated as a **research prototype**, not a production trading system.

---

## Pipeline Order

Run the full pipeline in this order:

1. `matcher.py`
2. `lead_lag.py`
3. `event_matcher.py`
4. `build_seed_index_panel.py`
5. `index_mapper.py`
6. `build_timeseries_dataset.py`
7. `signal_engine.py`
8. `backtest_signals.py`
9. `archive_run.py`

Or simply run:

```bash
python run_pipeline.py