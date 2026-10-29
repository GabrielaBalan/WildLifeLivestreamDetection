"""
Live dashboard for the wildlife detection logs.

    streamlit run dashboard.py

Reads the CSVs main.py is writing and refreshes on a timer, so it can run alongside a live session.
For a shareable, self-contained snapshot use export_dashboard.py instead.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import OUTPUT_DIR, DETECTIONS_CSV, FRAME_SUMMARY_CSV, EVENTS_CSV, VISITS_CSV, VISIT_GAP_SECONDS
from dashboard_data import build_tables, ZONES

PALETTES = {
    "light": {
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "base": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"], "wash": "rgba(42,120,214,0.12)", "zero": "#e6e5de",
        "heat": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "dark": {
        "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "base": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"], "wash": "rgba(57,135,229,0.18)", "zero": "#2a2a28",
        "heat": ["#104281", "#184f95", "#256abf", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"],
    },
}
ZONE_LABELS = {"left": "Left", "center": "Centre", "right": "Right"}
EVENT_LABELS = {
    "activity_started": "Activity started",
    "activity_ended": "Scene emptied",
    "new_species_detected": "New species",
    "multi_species_detected": "Mixed species",
    "large_group_detected": "Large group",
}
VISIT_BINS = [(0, 5, "<5 s"), (5, 15, "5–15 s"), (15, 30, "15–30 s"), (30, 60, "30–60 s"),
              (60, 120, "1–2 min"), (120, 300, "2–5 min"), (300, np.inf, ">5 min")]
LOG_FILES = (DETECTIONS_CSV, FRAME_SUMMARY_CSV, EVENTS_CSV, VISITS_CSV)

st.set_page_config(page_title="Savanna Cam Log", page_icon=":material/monitoring:", layout="wide")


def species_label(species):
    text = species.removesuffix("_like") + "-like" if species.endswith("_like") else species
    return text.replace("_", " ").capitalize()


def fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def default_data_dir():
    return OUTPUT_DIR if os.path.exists(os.path.join(OUTPUT_DIR, FRAME_SUMMARY_CSV)) else "docs"


def log_mtimes(data_dir):
    paths = [os.path.join(data_dir, name) for name in LOG_FILES]
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths)


@st.cache_data(show_spinner=False, max_entries=4)
def load_tables(data_dir, mtimes):
    # mtimes is only part of the cache key, so the tables reload whenever main.py writes
    return build_tables(data_dir)


def style(fig, colors, height):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["ink2"], size=13),
        hoverlabel=dict(font_size=13),
        barcornerradius=4,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=colors["base"], tickfont_color=colors["muted"])
    fig.update_yaxes(gridcolor=colors["grid"], zeroline=False, tickfont_color=colors["muted"])
    return fig


def chart_card(title, caption, fig=None, table=None, note=None):
    with st.container(border=True):
        st.subheader(title)
        st.caption(caption)
        if fig is None and table is None:
            st.info("Nothing in this selection.")
            return
        if fig is None:
            st.dataframe(table, hide_index=True, width="stretch")
            return
        chart_tab, table_tab = st.tabs(["Chart", "Table"])
        with chart_tab:
            st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
            if note:
                st.caption(note)
        with table_tab:
            st.dataframe(table, hide_index=True, width="stretch")


def hbar_figure(labels, values, colors, hover_label, highlight_index=None):
    opacity = [1.0 if highlight_index is None or i == highlight_index else 0.3 for i in range(len(values))]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=colors["series"][0], opacity=opacity),
        text=[f"{v:,}" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(color=colors["ink"], size=12),
        hovertemplate=f"<b>%{{x:,}}</b> {hover_label}<extra>%{{y}}</extra>",
    ))
    style(fig, colors, height=34 * len(values) + 30)
    fig.update_layout(bargap=0.55)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont_color=colors["ink2"])
    fig.update_xaxes(showgrid=True, gridcolor=colors["grid"], showticklabels=False, showline=False)
    return fig


def timeline_figure(minutes, species_minutes, species, colors, single_day):
    df = minutes.sort_values("minute").reset_index(drop=True)
    if species:
        per_minute = species_minutes.set_index("minute")["animals"]
        df["value"] = df["minute"].map(per_minute).fillna(0) / df["frames"]
    else:
        df["value"] = df["animals"] / df["frames"]

    # Break the line between sessions by inserting a gap row
    df["x"] = np.arange(len(df))
    new_session = df["session"].ne(df["session"].shift())
    gaps = df[new_session & (df.index > 0)].assign(value=np.nan, x=lambda d: d["x"] - 0.5)
    line = pd.concat([df, gaps]).sort_values("x")
    isolated = df[new_session & df["session"].ne(df["session"].shift(-1))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=line["x"], y=line["value"], mode="lines", connectgaps=False, fill="tozeroy",
        line=dict(color=colors["series"][0], width=2), fillcolor=colors["wash"],
        customdata=line["minute"].dt.strftime("%a %d %b, %H:%M"),
        hovertemplate="<b>%{y:.2f}</b> per frame<extra>%{customdata}</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=isolated["x"], y=isolated["value"], mode="markers", hoverinfo="skip",
        marker=dict(color=colors["series"][0], size=8),
    ))

    if single_day:
        starts = df[new_session]
        tickvals, ticktext = starts["x"], starts["minute"].dt.strftime("%H:%M")
    else:
        day = df["minute"].dt.date
        starts = df[day.ne(day.shift())]
        tickvals, ticktext = starts["x"], starts["minute"].dt.strftime("%a %d %b")
        for x in starts["x"].iloc[1:]:
            fig.add_vline(x=x - 0.5, line_width=1, line_color=colors["base"])

    style(fig, colors, height=280)
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, range=[-0.5, len(df) - 0.5])
    fig.update_yaxes(rangemode="tozero")
    return fig, df[["minute", "frames", "value"]].rename(columns={"value": "per frame"})


def heatmap_figure(species_minutes, minutes, order, colors):
    counts = (species_minutes.assign(hour=species_minutes["minute"].dt.hour)
              .groupby(["species", "hour"])["detections"].sum()
              .unstack("hour").reindex(index=order, columns=range(24)))
    counts = counts.loc[counts.sum(axis=1) > 0]
    recorded = set(minutes["minute"].dt.hour)
    not_recorded = [h for h in range(24) if h not in recorded]
    counts = counts.fillna(0)
    counts[not_recorded] = np.nan

    # Log scale so the rare species still show up next to thousands of zebra detections
    max_log = np.log1p(np.nanmax(counts.values)) or 1
    z = np.log1p(counts.values) / max_log
    # Exactly 0 gets the neutral "recorded, none seen" colour; anything above runs along the blue ramp
    heat = colors["heat"]
    colorscale = [[0, colors["zero"]], [0.0005, colors["zero"]]] + \
                 [[0.0005 + (1 - 0.0005) * i / (len(heat) - 1), c] for i, c in enumerate(heat)]
    text = counts.map(lambda v: "No footage" if pd.isna(v) else f"{int(v):,} detections").values

    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{h:02d}" for h in range(24)], y=[species_label(s) for s in counts.index],
        colorscale=colorscale, zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
        text=text, hovertemplate="%{y}, %{x}:00<br><b>%{text}</b><extra></extra>",
    ))
    style(fig, colors, height=40 * len(counts) + 50)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont_color=colors["ink2"])
    fig.update_xaxes(side="top", showline=False)

    table = counts.reset_index().melt(id_vars="species", var_name="hour", value_name="detections").dropna()
    table = table[table["detections"] > 0].assign(species=lambda d: d["species"].map(species_label))
    return fig, table


def zones_figure(species_minutes, order, colors):
    zone_columns = [f"zone_{z}" for z in ZONES]
    totals = species_minutes.groupby("species")[zone_columns].sum().reindex(order).dropna()
    totals = totals.loc[totals.sum(axis=1) > 0]
    shares = totals.div(totals.sum(axis=1), axis=0) * 100
    labels = [species_label(s) for s in totals.index]

    fig = go.Figure()
    for zone, color in zip(ZONES, colors["series"]):
        column = f"zone_{zone}"
        fig.add_trace(go.Bar(
            name=ZONE_LABELS[zone], x=shares[column], y=labels, orientation="h", marker_color=color,
            text=[f"{v:.0f}%" if v >= 8 else "" for v in shares[column]], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#ffffff", size=11),
            customdata=totals[column], hovertemplate="<b>%{x:.0f}%</b> of its detections (%{customdata:,})<extra>%{y}, " + ZONE_LABELS[zone].lower() + " third</extra>",
        ))
    style(fig, colors, height=34 * len(totals) + 60)
    fig.update_layout(barmode="stack", bargap=0.5, showlegend=True, barcornerradius=0,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font_color=colors["ink2"]))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont_color=colors["ink2"])
    fig.update_xaxes(showticklabels=False, range=[0, 100])

    table = totals.astype(int).rename(columns=lambda c: ZONE_LABELS[c.removeprefix("zone_")]).reset_index()
    table["species"] = table["species"].map(species_label)
    note = None
    if totals["zone_right"].sum() == 0:
        note = ("Nothing was logged in the right third. Runs before the coordinate fix placed boxes without "
                "scaling them back from the model's 640px input, so they never reached it.")
    return fig, table, note


def visits_figure(visits, colors):
    counts = [int(((visits["duration_seconds"] >= lo) & (visits["duration_seconds"] < hi)).sum()) for lo, hi, _ in VISIT_BINS]
    labels = [label for _, _, label in VISIT_BINS]
    fig = go.Figure(go.Bar(
        x=labels, y=counts, marker_color=colors["series"][0], width=0.4,
        text=counts, textposition="outside", cliponaxis=False, textfont=dict(color=colors["ink"], size=12),
        hovertemplate="<b>%{y}</b> visits<extra>%{x}</extra>",
    ))
    style(fig, colors, height=280)
    return fig, pd.DataFrame({"length": labels, "visits": counts})


def render(data_dir):
    colors = PALETTES["dark" if getattr(st.context.theme, "type", "light") == "dark" else "light"]

    try:
        tables = load_tables(data_dir, log_mtimes(data_dir))
        st.session_state["last_tables"] = tables
    except FileNotFoundError as e:
        st.info(f"{e}")
        return
    except Exception as e:
        # main.py may be halfway through writing a row; keep showing the last good read
        tables = st.session_state.get("last_tables")
        if tables is None:
            st.error(f"Could not read the logs in {data_dir}: {e}")
            return

    minutes, species_minutes = tables["minutes"], tables["species_minutes"]
    if minutes.empty:
        st.info("No frames logged yet. Start main.py and this page fills in as it runs.")
        return

    order = species_minutes.groupby("species")["detections"].sum().sort_values(ascending=False).index.tolist()
    days = sorted(minutes["minute"].dt.date.unique())

    filter_day, filter_species = st.columns([1, 3])
    day = filter_day.selectbox("Day", [None] + days, key="day",
                               format_func=lambda d: "All days" if d is None else d.strftime("%a %d %b %Y"))
    species = filter_species.pills("Species", order, key="species", format_func=species_label,
                                   help="Leave empty for all species")

    in_day = (lambda t: t.dt.date == day) if day else (lambda t: pd.Series(True, index=t.index))
    mins = minutes[in_day(minutes["minute"])]
    sm_day = species_minutes[in_day(species_minutes["minute"])]
    sm = sm_day[sm_day["species"] == species] if species else sm_day
    events = tables["events"][in_day(tables["events"]["minute"])]
    visits = tables["visits"]
    if not visits.empty:
        visits = visits[in_day(visits["visit_start"])]
        visits = visits[visits["species"] == species] if species else visits
    cooc = tables["co_occurrence"][in_day(tables["co_occurrence"]["minute"])]
    if species:
        cooc = cooc[(cooc["species_a"] == species) | (cooc["species_b"] == species)]
    sessions = tables["sessions"][in_day(tables["sessions"]["start"])]

    frames = int(mins["frames"].sum())
    occupied = int(sm["frames_present"].sum() if species else mins["occupied"].sum())
    detections = int(sm["detections"].sum())
    tiles = st.columns(5)
    tiles[0].metric(f"Frames with {species_label(species).lower()}" if species else "Scene occupancy",
                    f"{occupied / frames:.1%}" if frames else "–", help=f"{occupied:,} of {frames:,} frames")
    tiles[1].metric("Recorded", f"{len(mins):,} min", help=f"{mins['session'].nunique()} sessions")
    tiles[2].metric("Detections", f"{detections:,}",
                    help=f"average confidence {sm['confidence_sum'].sum() / detections:.2f}" if detections else None)
    tiles[3].metric("Visits", f"{len(visits):,}",
                    help=f"median {fmt_duration(visits['duration_seconds'].median())}" if len(visits) else None)
    tiles[4].metric("Scene events", f"{int(events['count'].sum()):,}")

    fig, table = timeline_figure(mins, sm, species, colors, single_day=day is not None)
    name = species_label(species).lower() if species else "animals"
    chart_card("Animals in view",
               f"Average {name} in view per frame, for each minute with footage. Gaps between sessions are closed up.",
               fig, table)

    left, right = st.columns(2)
    with left:
        per_species = sm_day.groupby("species")["detections"].sum().reindex(order).dropna()
        per_species = per_species[per_species > 0].astype(int)
        if per_species.empty:
            chart_card("Detections by species", "Boxes summed over all frames.")
        else:
            highlight = per_species.index.get_loc(species) if species in per_species.index else None
            fig = hbar_figure([species_label(s) for s in per_species.index], per_species.tolist(), colors, "detections", highlight)
            chart_card("Detections by species",
                       "Boxes summed over all frames, so one animal seen for a minute counts many times.",
                       fig, per_species.rename(index=species_label).reset_index())
    with right:
        if sm.empty:
            chart_card("Where in the frame", "Share of detections in each third of the watched area.")
        else:
            fig, table, note = zones_figure(sm, order, colors)
            chart_card("Where in the frame", "Share of each species' detections in the left, centre and right third of the watched area.",
                       fig, table, note)

    if not sm.empty:
        fig, table = heatmap_figure(sm, mins, order, colors)
        chart_card("Time of day", "Detections per hour, on a log scale so rare species stay visible. Grey: recorded but none seen; blank: no footage.",
                   fig, table)

    left, right = st.columns(2)
    with left:
        per_type = events.groupby("event_type")["count"].sum().reindex(list(EVENT_LABELS)).dropna().astype(int)
        per_type = per_type[per_type > 0]
        caption = "Scene-level events raised by the live pipeline." + (" These cover all species." if species else "")
        if per_type.empty:
            chart_card("Scene events", caption)
        else:
            fig = hbar_figure([EVENT_LABELS[t] for t in per_type.index], per_type.tolist(), colors, "events")
            chart_card("Scene events", caption, fig, per_type.rename(index=EVENT_LABELS).reset_index())
    with right:
        caption = f"A visit ends once a species is unseen for {VISIT_GAP_SECONDS} s."
        if visits.empty:
            chart_card("Visit length", caption)
        else:
            fig, table = visits_figure(visits, colors)
            chart_card("Visit length", f"{len(visits):,} visits, median {fmt_duration(visits['duration_seconds'].median())}. {caption}", fig, table)

    left, right = st.columns(2)
    with left:
        pairs = cooc.groupby(["species_a", "species_b"])["count"].sum().sort_values(ascending=False).reset_index()
        pairs = pd.DataFrame({
            "pair": pairs["species_a"].map(species_label) + " + " + pairs["species_b"].map(species_label).str.lower(),
            "frames together": pairs["count"],
        })
        with st.container(border=True):
            st.subheader("Seen together")
            st.caption("Frames in which both species were detected at the same time.")
            if pairs.empty:
                st.info("No frames with two species at once in this selection.")
            else:
                st.dataframe(pairs, hide_index=True, width="stretch", column_config={
                    "frames together": st.column_config.ProgressColumn(
                        "Frames together", format="%d", min_value=0, max_value=int(pairs["frames together"].max())),
                })
    with right:
        table = pd.DataFrame({
            "day": sessions["start"].dt.strftime("%a %d %b"),
            "time": sessions["start"].dt.strftime("%H:%M") + "–" + sessions["end"].dt.strftime("%H:%M"),
            "length": ((sessions["end"] - sessions["start"]).dt.total_seconds()).map(fmt_duration),
            "frames": sessions["frames"],
            "animals in view": sessions["occupied"] / sessions["frames"] * 100,
        })
        with st.container(border=True):
            st.subheader("Sessions")
            st.caption("Each continuous stretch of recording.")
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "animals in view": st.column_config.NumberColumn(format="%.1f%%"),
            })


with st.sidebar:
    st.header("Data")
    data_dir = st.text_input("Log folder", value=default_data_dir(), key="data_dir",
                             help="Folder holding the CSVs. main.py writes to output/; archived runs are in docs/.")
    live = st.toggle("Auto-refresh", value=True, key="live")
    refresh_seconds = st.select_slider("Refresh every", options=[5, 10, 30, 60], value=10, key="refresh",
                                       format_func=lambda s: f"{s} s", disabled=not live)
    st.caption("Leave auto-refresh on while main.py is running to watch new detections arrive.")

st.title("Savanna Cam Log")
st.caption(f"Wildlife livestream · YOLO detection + ByteTrack · data from `{data_dir}/`")

st.fragment(run_every=refresh_seconds if live else None)(render)(data_dir)
