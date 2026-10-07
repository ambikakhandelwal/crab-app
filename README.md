# Interactive What-If Graph Simulator

Visualizing cascading effects of changes in real causal event networks (CRAB dataset).

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

The CRAB pairwise file is already in `data/pairwise_causality.jsonl`. To try another file, upload it in the sidebar.

## Files
- app.py: Streamlit UI (4 tabs)
- engine.py: cascade simulation, criticality ranking, all-stories study
- data.py: CRAB loader, graph builder, demo scenarios
- data/pairwise_causality.jsonl: 2,730 scored event pairs, 20 news stories
