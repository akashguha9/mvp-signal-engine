\# MVP Signal Engine



A research pipeline for building and testing a multi-layer event-to-market signal engine.



\## Current Status



The pipeline is now structurally functional:



\- historical seed-index panel is built

\- event matching is working

\- leakage checks are clean

\- timeseries dataset builds correctly

\- signals and validation reports generate correctly



However, the model is \*\*not yet stable enough for live trading\*\*.



\### Honest state of the project



What is working:

\- data pipeline

\- historical joins

\- trend normalization

\- leakage / timestamp ordering checks

\- rolling and walk-forward validation tooling



What is not yet solved:

\- stable edge across regimes

\- robust rule families that survive strict rolling validation

\- live-trading readiness



At the moment, this is best treated as a \*\*research prototype\*\*, not a production trading system.



\---



\## Pipeline Order



Run the full pipeline in this order:



1\. `matcher.py`

2\. `lead\_lag.py`

3\. `event\_matcher.py`

4\. `build\_seed\_index\_panel.py`

5\. `index\_mapper.py`

6\. `build\_timeseries\_dataset.py`

7\. `signal\_engine.py`

8\. `backtest\_signals.py`

9\. `archive\_run.py`



Or simply run:



```bash

python run\_pipeline.py

