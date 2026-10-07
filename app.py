import textwrap
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from data import DEMOS, crab_graphs, load_crab
from engine import criticality_table, rank_agreement, rerank, simulate, study, top_k_overlap

st.set_page_config(page_title="What-If Graph Simulator", page_icon="🦋", layout="wide")

DATA_FILE = Path(__file__).parent / "data" / "pairwise_causality.jsonl"
COLORS = {"removed": "#d62728", "lost": "#ff7f0e", "weakened": "#f2c200", "safe": "#2ca02c"}
NAMES = {
    "removed": "Removed event",
    "lost": "Cut off: no causal support left",
    "weakened": "Weakened: lost part of its causal support",
    "safe": "Unaffected",
}


def short(text, n=60):
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def tag(g, n):
    return g.graph.get("labels", {}).get(n, short(n, 30))


def pretty(g, n):
    t = tag(g, n)
    return f"{t}: {short(n, 70)}" if t != short(n, 30) else str(n)


def layout(g):
    if nx.is_directed_acyclic_graph(g):
        for i, gen in enumerate(nx.topological_generations(g)):
            for node in gen:
                g.nodes[node]["layer"] = i
        return nx.multipartite_layout(g, subset_key="layer", align="vertical")
    return nx.spring_layout(g, seed=7)


def draw(g, status=None, weakened=None):
    pos = layout(g)
    fig = go.Figure()
    for u, v, d in g.edges(data=True):
        w = d.get("weight", 1.0)
        dead = status and (status[u] in ("removed", "lost") or status[v] == "removed")
        fig.add_annotation(
            x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.1,
            arrowwidth=max(0.8, 3.5 * min(w, 1.0)),
            arrowcolor="#cccccc" if dead else "#666666",
            standoff=14, startstandoff=14,
        )
    xs, ys, cols, texts, labels = [], [], [], [], []
    for n in g.nodes:
        xs.append(pos[n][0])
        ys.append(pos[n][1])
        s = status[n] if status else None
        cols.append(COLORS[s] if s else "#4c78a8")
        extra = ""
        if weakened and n in weakened:
            extra = f"<br>Lost {weakened[n]*100:.0f}% of its causal support"
        body = textwrap.fill(str(n), 55).replace("\n", "<br>")
        texts.append(f"{body}<br><b>{NAMES[s] if s else ''}</b>{extra}")
        labels.append(tag(g, n))
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers+text", text=labels, textposition="top center",
            hovertext=texts, hoverinfo="text",
            marker=dict(size=24, color=cols, line=dict(width=1.5, color="white")),
        )
    )
    fig.update_layout(
        height=580, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def to_edges(g):
    return tuple((u, v, d.get("weight", 1.0)) for u, v, d in g.edges(data=True))


def from_edges(edges, labels):
    g = nx.DiGraph()
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
    if labels:
        g.graph["labels"] = dict(labels)
    return g


@st.cache_data(show_spinner=False)
def cached_table(edges, labels):
    return criticality_table(from_edges(edges, labels))


@st.cache_data(show_spinner="Running every story...")
def cached_study(items, skip_roots):
    graphs = {name: from_edges(edges, labels) for name, edges, labels in items}
    return study(graphs, skip_roots)


@st.cache_data(show_spinner=False)
def cached_load(file_bytes, name):
    import io

    class F(io.BytesIO):
        pass

    f = F(file_bytes)
    f.name = name
    return load_crab(f)


st.title("🦋 Interactive What-If Graph Simulator")
st.caption("Visualizing cascading effects of changes in real causal event networks (CRAB dataset)")

graphs = {}
is_crab = False
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", ["CRAB dataset (real news)", "Demo scenarios"])
    if source.startswith("CRAB"):
        is_crab = True
        up = st.file_uploader("Use a different CRAB file (optional)", type=["jsonl", "json", "csv"])
        try:
            if up is not None:
                df = cached_load(up.getvalue(), up.name)
            elif DATA_FILE.exists():
                df = load_crab(DATA_FILE)
            else:
                df = None
                st.info("Put pairwise_causality.jsonl inside the data folder, or upload it above.")
        except Exception as ex:
            df = None
            st.error(f"Could not read the file: {ex}")
        if df is not None:
            thr = st.slider("Minimum causality score (0-100)", 0, 95, 50)
            k = st.slider("Main causes kept per event", 1, 5, 2)
            graphs = crab_graphs(df, thr, k)
            st.caption(f"{len(graphs)} stories with 5+ events")
    else:
        graphs = dict(DEMOS)

if not graphs:
    st.warning("No graph to show. Lower the minimum causality score or raise the number of main causes.")
    st.stop()

with st.sidebar:
    st.header("2. Story")
    gname = st.selectbox("Pick one", list(graphs.keys()))

g = graphs[gname]
tab1, tab2, tab3, tab4 = st.tabs(["What-If Simulator", "Critical Events", "All Stories", "How it works"])

with tab1:
    left, right = st.columns([3, 1])
    with right:
        event = st.selectbox("Remove this event", list(g.nodes), format_func=lambda n: pretty(g, n))
        go_btn = st.toggle("Run what-if", value=False)
        st.markdown("**Legend**")
        for kk in ("removed", "lost", "weakened", "safe"):
            st.markdown(f"<span style='color:{COLORS[kk]}'>●</span> {NAMES[kk]}", unsafe_allow_html=True)
        st.caption("Thicker arrow = stronger cause")
    if go_btn:
        r = simulate(g, event)
        with left:
            st.plotly_chart(draw(g, r["status"], r["weakened"]), width="stretch")
        m = st.columns(5)
        m[0].metric("Butterfly impact", f"{r['impact_pct']:.0f}%")
        m[1].metric("Events cut off", len(r["lost"]))
        m[2].metric("Events weakened", len(r["weakened"]))
        m[3].metric("Links broken", r["broken_edges"])
        m[4].metric("Max depth", r["depth"])
        if r["weakened"]:
            st.markdown("**Most weakened events**")
            for n, f in sorted(r["weakened"].items(), key=lambda x: -x[1])[:8]:
                st.progress(min(f, 1.0), text=f"{pretty(g, n)}: lost {f*100:.0f}% of causal support")
        if r["lost"]:
            st.markdown("**Cut off:** " + ", ".join(pretty(g, x) for x in sorted(r["lost"], key=str)))
    else:
        with left:
            st.plotly_chart(draw(g), width="stretch")
        st.info("Switch on **Run what-if** to remove the selected event and watch the cascade.")
    if is_crab:
        with st.expander("Event list (E-numbers follow the story timeline)"):
            order = g.graph["order"]
            for n in sorted(g.nodes, key=lambda x: order[x]):
                st.markdown(f"**{tag(g, n)}**: {n}")

with tab2:
    st.subheader("Which event is the real weak point?")
    st.write("Every event is removed one at a time. The real damage is compared with degree and betweenness.")
    labels = tuple(g.graph.get("labels", {}).items())
    df2 = cached_table(to_edges(g), labels)
    roots = {n for n in g.nodes if g.in_degree(n) == 0}
    skip = st.checkbox("Ignore root causes (starting events) in the ranking", value=True)
    if skip and len(df2) - len(roots) >= 3:
        df2 = rerank(df2[~df2["event"].isin(roots)])
    a, b = rank_agreement(df2)
    ov_d, ov_b, kk = top_k_overlap(df2, 3)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rank match: impact vs degree", "n/a" if a is None else f"{a:.2f}")
    c2.metric("Rank match: impact vs betweenness", "n/a" if b is None else f"{b:.2f}")
    c3.metric(f"Top-{kk} found by degree / betweenness", f"{ov_d} / {ov_b}")
    top = df2.iloc[0]
    st.success(
        f"Most critical event: **{pretty(g, top['event'])}**. Removing it hits {top['impact_pct']:.0f}% of the network "
        f"(degree rank {top['degree_rank']}, betweenness rank {top['betweenness_rank']})."
    )
    miss = df2[(df2["degree_rank"] - df2["impact_rank"]) >= 3]
    if not miss.empty:
        row = miss.iloc[0]
        st.warning(
            f"Degree missed this one: **{pretty(g, row['event'])}** is rank {row['impact_rank']} by real impact "
            f"but rank {row['degree_rank']} by degree."
        )
    top15 = df2.head(15)
    fig = go.Figure(
        go.Bar(
            x=[tag(g, x) for x in top15["event"]][::-1],
            y=list(top15["impact_pct"])[::-1],
            marker_color="#d62728",
            hovertext=[str(x) for x in top15["event"]][::-1],
        )
    )
    fig.update_layout(height=380, yaxis_title="Impact %", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    show = df2.copy()
    show.insert(0, "id", [tag(g, x) for x in show["event"]])
    st.dataframe(
        show[["id", "event", "impact_pct", "impact_rank", "degree", "degree_rank", "betweenness", "betweenness_rank"]].round(3),
        width="stretch",
        hide_index=True,
    )
    st.download_button("Download ranking (CSV)", df2.to_csv(index=False), "critical_events.csv")

with tab3:
    st.subheader("Does degree predict real damage? Across all stories")
    if not is_crab:
        st.info("Switch the source to the CRAB dataset to run this across all news stories.")
    else:
        skip_all = st.checkbox("Ignore root causes", value=True, key="skip_all")
        items = tuple(
            (name, to_edges(gr), tuple(gr.graph.get("labels", {}).items())) for name, gr in graphs.items()
        )
        res = cached_study(items, skip_all)
        n = len(res)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stories analysed", n)
        c2.metric("Avg rank match: degree", f"{res['rho_degree'].mean():.2f}")
        c3.metric("Avg rank match: betweenness", f"{res['rho_betweenness'].mean():.2f}")
        c4.metric("Top event found by degree", f"{int(res['top_matches_degree'].sum())} of {n}")
        st.caption(
            "Rank match = Spearman correlation between real impact and the centrality measure (1 = identical order). "
            f"Betweenness found the single most critical event in {int(res['top_matches_betweenness'].sum())} of {n} stories."
        )
        figc = go.Figure()
        figc.add_trace(go.Bar(name="Degree", x=list(range(1, n + 1)), y=res["rho_degree"]))
        figc.add_trace(go.Bar(name="Betweenness", x=list(range(1, n + 1)), y=res["rho_betweenness"]))
        figc.update_layout(
            barmode="group", height=360, xaxis_title="Story", yaxis_title="Rank match with real impact",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(figc, width="stretch")
        st.dataframe(res.round(2), width="stretch", hide_index=True)
        st.download_button("Download results (CSV)", res.to_csv(index=False), "all_stories.csv")

with tab4:
    st.markdown(
        """
**Model.** Nodes are events. A directed edge A → B means A causes B. The edge weight is the CRAB causality score divided by 100.

**Building graphs from CRAB.** CRAB scores every earlier-later pair of events inside a news story. For each event, the app keeps its strongest causes above the minimum score (the "main causes" slider). Each story becomes one graph, with events numbered E1, E2, ... in timeline order.

**Causal support.** Starting events have support 1. Every other event gets the weighted average of its causes' support. Removing an event sets its support to 0, and the loss flows downstream through the weights.
- 🟠 Cut off: support drops to 0
- 🟡 Weakened: support drops partly
- 🟢 Unaffected: support stays 1

**Butterfly impact** = (1 + cut off events + sum of support lost by weakened events) ÷ all events × 100.

**Critical events** removes each event in turn and compares real impact with degree and betweenness centrality using Spearman rank correlation. **All Stories** repeats this over every story.

**Algorithms used:** topological ordering, weighted propagation on a DAG, BFS/DFS reachability (for graphs with cycles), shortest paths (depth), weakly connected components, degree and betweenness centrality, Spearman rank correlation.
"""
    )
