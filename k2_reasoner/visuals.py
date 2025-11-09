from __future__ import annotations

from typing import Dict, List

import networkx as nx
import plotly.graph_objects as go

from .reasoner import ExposureSummary


def _node_size(value: float) -> float:
    return max(20.0, min(60.0, abs(value) / 50.0))


def _edge_width(value: float) -> float:
    return max(1.0, min(6.0, abs(value) / 100.0))


def build_causal_graph(summary: ExposureSummary) -> go.Figure:
    graph = nx.DiGraph()
    nodes = {
        "Rates": summary.totals["dv01"],
        "Curve": summary.totals["convexity"],
        "Credit": summary.totals["credit_spread_dv01"],
        "FX": summary.totals["fx_delta"],
        "Equity": summary.totals["beta_notional"],
        "PnL": summary.totals["notional"],
    }

    edges = [
        ("Rates", "Curve", summary.totals["convexity"]),
        ("Rates", "FX", summary.totals["fx_delta"]),
        ("Credit", "Equity", summary.totals["beta_notional"]),
        ("Credit", "PnL", summary.totals["credit_spread_dv01"]),
        ("Rates", "PnL", summary.totals["dv01"]),
        ("Equity", "PnL", summary.totals["beta_notional"]),
        ("FX", "PnL", summary.totals["fx_delta"]),
    ]

    for name, value in nodes.items():
        graph.add_node(name, value=value)

    for source, target, weight in edges:
        graph.add_edge(source, target, weight=weight)

    pos = nx.spring_layout(graph, seed=42, k=0.7)

    edge_traces: List[go.Scatter] = []
    for source, target, data in graph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        width = _edge_width(data["weight"])
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color="#9ca3af"),
                hoverinfo="none",
            )
        )

    node_trace = go.Scatter(
        x=[pos[node][0] for node in graph.nodes()],
        y=[pos[node][1] for node in graph.nodes()],
        mode="markers+text",
        marker=dict(
            size=[_node_size(graph.nodes[node]["value"]) for node in graph.nodes()],
            color=["#6366f1", "#0ea5e9", "#22d3ee", "#f97316", "#10b981", "#1f2937"],
            opacity=0.9,
            line=dict(width=2, color="#1f2937"),
        ),
        text=[f"{node}\n{graph.nodes[node]['value']:.0f}" for node in graph.nodes()],
        textposition="bottom center",
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Causal Flow Across Risk Factors",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        template="plotly_white",
    )
    return fig
