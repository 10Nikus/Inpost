"""
Plotly chart builders for the Analytics tab.

Each function returns a ``plotly.graph_objects.Figure`` ready to be
rendered with ``st.plotly_chart()``.
"""

import pandas as pd
import plotly.graph_objects as go


# Consistent color palette
_EV_COLOR = "#2ecc71"
_HYBRID_COLOR = "#3498db"
_DIESEL_COLOR = "#95a5a6"
_BG_COLOR = "rgba(0,0,0,0)"
_TEXT_COLOR = "#e0e0e0"

_COMMON_LAYOUT = dict(
    paper_bgcolor=_BG_COLOR,
    plot_bgcolor=_BG_COLOR,
    font=dict(family="Inter, Helvetica, Arial, sans-serif", color=_TEXT_COLOR),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor=_BG_COLOR),
)


def fleet_allocation_bar_chart(zone_stats: pd.DataFrame) -> go.Figure:
    """Stacked bar chart: EV / Hybrid / Diesel vehicles per zone."""
    fig = go.Figure()
    for col, name, color in [
        ("EV_assigned", "EV", _EV_COLOR),
        ("Hybrid_assigned", "Hybrid", _HYBRID_COLOR),
        ("Diesel_assigned", "Diesel", _DIESEL_COLOR),
    ]:
        fig.add_trace(
            go.Bar(
                x=zone_stats["delivery_zone_id"].astype(str),
                y=zone_stats[col],
                name=name,
                marker_color=color,
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Fleet Allocation per Zone",
        xaxis_title="Delivery Zone",
        yaxis_title="Vehicles Assigned",
        **_COMMON_LAYOUT,
    )
    return fig


def air_quality_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of air quality distribution across lockers."""
    counts = df["air_index_level"].value_counts().reset_index()
    counts.columns = ["level", "count"]

    color_map = {
        "VERY_GOOD": _EV_COLOR,
        "GOOD": "#f1c40f",
        "VERY_BAD": "#e74c3c",
    }
    colors = [color_map.get(lvl, _DIESEL_COLOR) for lvl in counts["level"]]

    fig = go.Figure(
        go.Pie(
            labels=counts["level"],
            values=counts["count"],
            hole=0.45,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=13),
        )
    )
    fig.update_layout(
        title="Air Quality Distribution",
        **_COMMON_LAYOUT,
    )
    return fig


def zone_machine_count_chart(zone_stats: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of machine count per zone, colored by fleet profile."""
    profile_colors = {
        "EV Dominant": _EV_COLOR,
        "Hybrid Mixed": _HYBRID_COLOR,
        "Diesel Dominant": _DIESEL_COLOR,
    }
    colors = [profile_colors.get(p, _DIESEL_COLOR) for p in zone_stats["fleet_profile"]]

    fig = go.Figure(
        go.Bar(
            y=zone_stats["delivery_zone_id"].astype(str),
            x=zone_stats["machine_count"],
            orientation="h",
            marker_color=colors,
            text=zone_stats["fleet_profile"],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Machines per Delivery Zone",
        xaxis_title="Number of Lockers",
        yaxis_title="Zone ID",
        **_COMMON_LAYOUT,
    )
    return fig


def smog_vs_fleet_scatter(zone_stats: pd.DataFrame) -> go.Figure:
    """Bubble chart: avg smog vs cars needed, sized by machine count."""
    profile_colors = {
        "EV Dominant": _EV_COLOR,
        "Hybrid Mixed": _HYBRID_COLOR,
        "Diesel Dominant": _DIESEL_COLOR,
    }
    colors = [profile_colors.get(p, _DIESEL_COLOR) for p in zone_stats["fleet_profile"]]

    fig = go.Figure(
        go.Scatter(
            x=zone_stats["avg_smog"],
            y=zone_stats["cars_needed"],
            mode="markers+text",
            text=zone_stats["delivery_zone_id"].astype(str),
            textposition="top center",
            marker=dict(
                size=zone_stats["machine_count"] / 3,
                color=colors,
                line=dict(width=1, color="white"),
            ),
        )
    )
    fig.update_layout(
        title="Pollution vs Fleet Demand",
        xaxis_title="Average Smog Score",
        yaxis_title="Cars Needed",
        **_COMMON_LAYOUT,
    )
    return fig
