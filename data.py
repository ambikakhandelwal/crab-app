import networkx as nx
import pandas as pd


def _edges(rows):
    g = nx.DiGraph()
    for a, b, w in rows:
        g.add_edge(a, b, weight=w)
    return g


DEMOS = {
    "Career": _edges(
        [
            ("Choose AI", "Learn Python", 0.9),
            ("Choose AI", "Take ML Course", 0.9),
            ("Learn Python", "Take ML Course", 0.6),
            ("Learn Python", "Build Web Project", 0.5),
            ("Take ML Course", "Build ML Project", 0.9),
            ("Build ML Project", "Kaggle Rank", 0.5),
            ("Build ML Project", "Internship", 0.8),
            ("Kaggle Rank", "Internship", 0.4),
            ("Build Web Project", "Internship", 0.3),
            ("Internship", "Job Offer", 0.9),
        ]
    ),
    "Disaster": _edges(
        [
            ("Earthquake", "Road Blocked", 0.9),
            ("Earthquake", "Power Outage", 0.8),
            ("Earthquake", "Building Collapse", 0.7),
            ("Road Blocked", "Ambulance Delayed", 0.9),
            ("Building Collapse", "More Injuries", 0.9),
            ("Power Outage", "Hospital Backup Only", 0.7),
            ("Ambulance Delayed", "Hospital Overloaded", 0.6),
            ("More Injuries", "Hospital Overloaded", 0.9),
            ("Hospital Backup Only", "Hospital Overloaded", 0.4),
            ("Hospital Overloaded", "Emergency Response Fails", 0.9),
        ]
    ),
    "Business": _edges(
        [
            ("Supplier Failure", "Production Delay", 0.9),
            ("Port Strike", "Production Delay", 0.5),
            ("Port Strike", "Shipping Delay", 0.8),
            ("Production Delay", "Product Shortage", 0.9),
            ("Shipping Delay", "Product Shortage", 0.6),
            ("Product Shortage", "Sales Drop", 0.9),
            ("Product Shortage", "Customer Complaints", 0.7),
            ("Customer Complaints", "Brand Damage", 0.6),
            ("Sales Drop", "Revenue Loss", 0.9),
            ("Brand Damage", "Revenue Loss", 0.5),
        ]
    ),
}

REQUIRED = {"topic_id", "event_a", "event_b", "score", "event_order_a", "event_order_b"}


def load_crab(src):
    name = getattr(src, "name", str(src)).lower()
    if hasattr(src, "seek"):
        src.seek(0)
    if name.endswith(".csv"):
        df = pd.read_csv(src)
    else:
        df = pd.read_json(src, lines=True)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    keep = [c for c in df.columns if not c.startswith("article")]
    return df[keep]


def crab_graph(part, threshold, main_causes):
    g = nx.DiGraph()
    order = {}
    for r in part.itertuples():
        order[r.event_a] = r.event_order_a
        order[r.event_b] = r.event_order_b
    for effect, grp in part.groupby("event_b"):
        grp = grp[grp["score"] >= threshold].nlargest(main_causes, "score")
        for r in grp.itertuples():
            g.add_edge(r.event_a, r.event_b, weight=r.score / 100)
    g.graph["labels"] = {n: f"E{order[n]}" for n in g.nodes}
    g.graph["order"] = {n: order[n] for n in g.nodes}
    return g


def crab_graphs(df, threshold, main_causes, min_nodes=5):
    out = {}
    for tid, part in df.groupby("topic_id"):
        g = crab_graph(part, threshold, main_causes)
        if g.number_of_nodes() >= min_nodes:
            first = min(g.graph["order"], key=g.graph["order"].get)
            title = first if len(first) <= 55 else first[:54] + "…"
            out[f"Story {tid}: {title}"] = g
    return out
