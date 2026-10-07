import networkx as nx
import pandas as pd


def source_nodes(g):
    cond = nx.condensation(g)
    out = set()
    for c in cond.nodes:
        if cond.in_degree(c) == 0:
            out |= set(cond.nodes[c]["members"])
    return out


def supported(g, sources):
    keep = set()
    for s in sources:
        if s in g:
            keep.add(s)
            keep |= nx.descendants(g, s)
    return keep


def simulate(g, removed, sources=None):
    if nx.is_directed_acyclic_graph(g):
        return simulate_strength(g, removed)
    return simulate_reach(g, removed, sources)


def simulate_strength(g, removed):
    strength = {}
    for n in nx.topological_sort(g):
        if n == removed:
            strength[n] = 0.0
        elif g.in_degree(n) == 0:
            strength[n] = 1.0
        else:
            tot = sum(d.get("weight", 1.0) for _, _, d in g.in_edges(n, data=True))
            got = sum(d.get("weight", 1.0) * strength[u] for u, _, d in g.in_edges(n, data=True))
            strength[n] = got / tot if tot > 0 else 1.0
    lost = {n for n, s in strength.items() if n != removed and s <= 1e-9}
    weakened = {n: 1 - s for n, s in strength.items() if n != removed and 1e-9 < s < 1 - 1e-9}
    return finish(g, removed, lost, weakened)


def simulate_reach(g, removed, sources=None):
    if sources is None:
        sources = source_nodes(g)
    before = supported(g, sources)
    h = g.copy()
    h.remove_node(removed)
    after = supported(h, sources - {removed})
    lost = (before - after) - {removed}
    weakened = {}
    for n in h.nodes:
        if n in lost:
            continue
        tot = sum(d.get("weight", 1.0) for _, _, d in g.in_edges(n, data=True))
        if tot <= 0:
            continue
        gone = sum(
            d.get("weight", 1.0)
            for u, _, d in g.in_edges(n, data=True)
            if u == removed or u in lost
        )
        if gone > 0:
            weakened[n] = gone / tot
    return finish(g, removed, lost, weakened)


def finish(g, removed, lost, weakened):
    status = {}
    for n in g.nodes:
        if n == removed:
            status[n] = "removed"
        elif n in lost:
            status[n] = "lost"
        elif n in weakened:
            status[n] = "weakened"
        else:
            status[n] = "safe"
    lengths = nx.single_source_shortest_path_length(g, removed)
    depth = max([lengths.get(n, 0) for n in list(lost) + list(weakened)] + [0])
    broken = sum(1 for u, v in g.edges if u == removed or v == removed or u in lost or v in lost)
    h = g.copy()
    h.remove_node(removed)
    n = g.number_of_nodes()
    damage = 1 + len(lost) + sum(weakened.values())
    return {
        "status": status,
        "lost": lost,
        "weakened": weakened,
        "depth": depth,
        "broken_edges": broken,
        "components_before": nx.number_weakly_connected_components(g),
        "components_after": nx.number_weakly_connected_components(h),
        "impact_pct": 100 * damage / n if n else 0,
    }


def criticality_table(g):
    sources = source_nodes(g)
    n = g.number_of_nodes()
    bet = nx.betweenness_centrality(g)
    rows = []
    for node in g.nodes:
        r = simulate(g, node, sources)
        rows.append(
            {
                "event": node,
                "impact_pct": r["impact_pct"],
                "cut_off": len(r["lost"]),
                "weakened": len(r["weakened"]),
                "degree": g.in_degree(node) + g.out_degree(node),
                "out_degree": g.out_degree(node),
                "betweenness": bet[node],
            }
        )
    df = pd.DataFrame(rows)
    df["impact_rank"] = df["impact_pct"].rank(ascending=False, method="min").astype(int)
    df["degree_rank"] = df["degree"].rank(ascending=False, method="min").astype(int)
    df["betweenness_rank"] = df["betweenness"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("impact_rank").reset_index(drop=True)


def rank_agreement(df):
    if len(df) < 3 or df["impact_pct"].nunique() < 2:
        return None, None
    a = df["impact_pct"].corr(df["degree"], method="spearman") if df["degree"].nunique() > 1 else None
    b = df["impact_pct"].corr(df["betweenness"], method="spearman") if df["betweenness"].nunique() > 1 else None
    return a, b


def top_k_overlap(df, k=3):
    k = min(k, len(df))
    real = set(df.nsmallest(k, "impact_rank")["event"])
    deg = set(df.nsmallest(k, "degree_rank")["event"])
    bet = set(df.nsmallest(k, "betweenness_rank")["event"])
    return len(real & deg), len(real & bet), k


def rerank(df):
    df = df.copy()
    df["impact_rank"] = df["impact_pct"].rank(ascending=False, method="min").astype(int)
    df["degree_rank"] = df["degree"].rank(ascending=False, method="min").astype(int)
    df["betweenness_rank"] = df["betweenness"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("impact_rank").reset_index(drop=True)


def study(graphs, skip_roots=True):
    rows = []
    for name, g in graphs.items():
        t = criticality_table(g)
        if skip_roots:
            roots = {n for n in g.nodes if g.in_degree(n) == 0}
            if len(t) - len(roots) >= 3:
                t = rerank(t[~t["event"].isin(roots)])
        a, b = rank_agreement(t)
        top = set(t.loc[t["impact_rank"] == 1, "event"])
        rows.append(
            {
                "story": name,
                "events": g.number_of_nodes(),
                "links": g.number_of_edges(),
                "rho_degree": a,
                "rho_betweenness": b,
                "top_matches_degree": bool(top & set(t.loc[t["degree_rank"] == 1, "event"])),
                "top_matches_betweenness": bool(top & set(t.loc[t["betweenness_rank"] == 1, "event"])),
            }
        )
    return pd.DataFrame(rows)
