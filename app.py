import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_data import agent_name, build_timeline, list_runs, load_json

st.set_page_config(page_title="Emergency Response MAS", page_icon="🚑", layout="wide")

SEV_COLOR = {"immediate": "#E5484D", "delayed": "#F5A524", "minor": "#30A46C", "cleared": "#A0A7B4"}
AMB_COLOR = {"available": "#30A46C", "en_route": "#3E8EF7", "waiting_for_responders": "#F5A524",
             "loading": "#A26BF5", "transporting": "#E5484D"}
TEAM_COLOR = {"available": "#30A46C", "en_route": "#3E8EF7", "on_scene": "#F5A524"}


@st.cache_data
def load(path):
    run = load_json(path)
    world = load_json(f"data/world_v1.0_seed{run['seed']}.json")
    return run, world


@st.cache_resource
def timeline(path):
    run, world = load(path)
    return build_timeline(run, world)


@st.cache_data
def run_info(path):
    r = load_json(path)
    return r["seed"], r["metrics"] | r.get("negotiation", {})

st.sidebar.title("🚑 Emergency Response")
runs = list_runs()
if not runs:
    st.error("No runs found. Run `python run_baseline.py` and `python run_agents.py --mode rule` first.")
    st.stop()
run_path = st.sidebar.selectbox("Run", runs, format_func=lambda p: Path(p).stem,
                                index=next((i for i, p in enumerate(runs) if "agents" in p), 0))
run, world = load(run_path)
tl = timeline(run_path)
T = tl["T"]

if "minute" not in st.session_state:
    st.session_state.minute = min(60, T)
if "playing" not in st.session_state:
    st.session_state.playing = False
if st.session_state.pop("_advance", False):         
    st.session_state.minute = min(T, st.session_state.minute + st.session_state.get("speed", 2))
    if st.session_state.minute >= T:
        st.session_state.playing = False
st.session_state.minute = min(st.session_state.minute, T)
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Play" if not st.session_state.playing else "⏸ Pause", width="stretch"):
    st.session_state.playing = not st.session_state.playing
if c2.button("⟲ Reset", width="stretch"):
    st.session_state.minute, st.session_state.playing = 1, False
st.sidebar.select_slider("Play speed (minutes per step)", options=[1, 2, 5, 10], value=2, key="speed")
st.sidebar.slider("Simulation minute", 1, T, key="minute")
t = st.session_state.minute
st.sidebar.caption(f"Policy: **{run['policy']}** · seed **{run['seed']}**")

msgs = pd.DataFrame(run.get("messages", []))
msgs_now = msgs[msgs["t"] <= t] if not msgs.empty else msgs

st.title("Multi-Agent Emergency Response Coordination")
k = tl["kpis_at"](t)
snap = run["snapshots"][t - 1]
cols = st.columns(6)
cols[0].metric("Minute", f"{t} / {T}")
cols[1].metric("Victims reported", k["reported"])
cols[2].metric("Admitted to hospital", k["served"])
cols[3].metric("Waiting for a unit", k["waiting"])
cols[4].metric("Critical missed so far", k["critical_missed"])
cols[5].metric("Free ambulances", sum(s == "available" for s in snap["ambulances"].values()))

tab_map, tab_res, tab_agents, tab_neg, tab_perf = st.tabs(
    ["🗺️ Incident map", "📊 Resources", "🤖 Agent activity", "🤝 Negotiation history", "🏁 Performance"])

with tab_map:
    left, right = st.columns([3, 2])
    incs = pd.DataFrame(tl["incidents_at"](t))
    fig = go.Figure()
    for h in tl["hospitals"]:
        free = snap["beds_available"][h["hospital_id"]]
        total = snap.get("beds_total", {}).get(h["hospital_id"], h["er_beds"])
        fig.add_trace(go.Scatter(x=[h["location"]["x"]], y=[h["location"]["y"]], mode="markers+text",
                                 marker=dict(symbol="square", size=22, color="#4C5C70"),
                                 text=[f"{h['hospital_id']} ({free}/{total} beds)"], textposition="top center",
                                 name=h["hospital_id"], showlegend=False,
                                 hovertext=f"{h['name']} · {h['kind']}", hoverinfo="text"))
    if not incs.empty:
        for sev, g in incs.groupby("worst"):
            fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers+text", name=f"incident: {sev}",
                                     marker=dict(size=12 + 4 * g["waiting"], color=SEV_COLOR[sev], opacity=0.8,
                                                 line=dict(width=1, color="white")),
                                     text=g["incident_id"], textposition="bottom center",
                                     customdata=g[["type", "waiting", "victims"]],
                                     hovertemplate="%{text}<br>%{customdata[0]}<br>"
                                                   "%{customdata[1]} of %{customdata[2]} waiting<extra></extra>"))
    amb = tl["amb"][t]
    inc_xy = {i["incident_id"]: (i["location"]["x"], i["location"]["y"]) for i in world["incidents"]}
    hosp_xy = {h["hospital_id"]: (h["location"]["x"], h["location"]["y"]) for h in world["hospitals"]}
    for aid, (x, y, status, target) in amb.items():
        goal = inc_xy.get(target) or hosp_xy.get(target)
        if goal:
            fig.add_trace(go.Scatter(x=[x, goal[0]], y=[y, goal[1]], mode="lines", showlegend=False,
                                     line=dict(color=AMB_COLOR[status], width=1, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[v[0] for v in amb.values()], y=[v[1] for v in amb.values()], mode="markers+text",
                             name="ambulances", text=list(amb), textposition="middle right",
                             marker=dict(symbol="triangle-up", size=14, color=[AMB_COLOR[v[2]] for v in amb.values()],
                                         line=dict(width=1, color="black")),
                             hovertext=[f"{a}: {v[2]} {('→ ' + v[3]) if v[3] else ''}" for a, v in amb.items()],
                             hoverinfo="text"))
    teams = tl["teams"][t]
    fig.add_trace(go.Scatter(x=[v[0] for v in teams.values()], y=[v[1] for v in teams.values()], mode="markers",
                             name="responder teams",
                             marker=dict(symbol="diamond", size=12, color=[TEAM_COLOR[v[2]] for v in teams.values()],
                                         line=dict(width=1, color="black")),
                             hovertext=[f"{k_}: {v[2]} {('@ ' + v[3]) if v[3] else ''}" for k_, v in teams.items()],
                             hoverinfo="text"))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.08),
                      xaxis=dict(range=[-0.5, world["grid_km"] + 0.5], title="km", showgrid=True, dtick=2),
                      yaxis=dict(range=[-0.5, world["grid_km"] + 0.5], title="km", showgrid=True, dtick=2,
                                 scaleanchor="x"),
                      title=f"City grid at minute {t}")
    left.plotly_chart(fig, width="stretch")

    right.subheader("Active incidents")
    if incs.empty:
        right.info("No incidents reported yet.")
    else:
        right.dataframe(incs[["incident_id", "type", "victims", "waiting", "worst"]]
                        .rename(columns={"worst": "worst waiting"}), hide_index=True, width="stretch")
        pick = right.selectbox("Read a report", incs["incident_id"])
        right.caption(incs.set_index("incident_id").loc[pick, "report"])
    right.markdown("**Ambulance colours:** " + " · ".join(f"<span style='color:{c}'>■</span> {s.replace('_', ' ')}"
                                                       for s, c in AMB_COLOR.items()), unsafe_allow_html=True)

with tab_res:
    a, b = st.columns(2)
    amb_counts = pd.Series(snap["ambulances"]).value_counts().reindex(AMB_COLOR, fill_value=0)
    a.plotly_chart(px.bar(x=amb_counts.index.str.replace("_", " "), y=amb_counts.values, color=amb_counts.index,
                          color_discrete_map=AMB_COLOR, title="Ambulances by status",
                          labels={"x": "", "y": "ambulances"}).update_layout(showlegend=False, height=320),
                   width="stretch")
    totals = snap.get("beds_total", {h["hospital_id"]: h["er_beds"] for h in world["hospitals"]})
    beds = pd.DataFrame([{"hospital": h["hospital_id"], "in use": totals[h["hospital_id"]] - snap["beds_available"][h["hospital_id"]],
                          "free": snap["beds_available"][h["hospital_id"]]} for h in world["hospitals"]])
    b.plotly_chart(px.bar(beds, x="hospital", y=["in use", "free"], title="ER beds (incl. surge beds opened by staff loans)",
                          color_discrete_sequence=["#E5484D", "#30A46C"]).update_layout(height=320, legend_title=""),
                   width="stretch")

    c, d = st.columns(2)
    if "doctors" in snap:
        rows = []
        for h in snap["doctors"]:
            for role, key in (("doctors", "doctors"), ("nurses", "nurses")):
                busy, total, people = snap[key][h]
                rows.append({"hospital": h, "role": role, "load %": round(100 * busy / total) if total else 0,
                             "on staff": people, "busy slots": busy, "total slots": total})
        staff = pd.DataFrame(rows)
        c.plotly_chart(px.bar(staff, x="hospital", y="load %", color="role", barmode="group", range_y=[0, 100],
                              title="Doctor and nurse workload", hover_data=["on staff", "busy slots", "total slots"],
                              color_discrete_map={"doctors": "#3E8EF7", "nurses": "#A26BF5"})
                       .update_layout(height=320, legend_title=""), width="stretch")
        d.subheader("Medical staff on duty")
        d.dataframe(staff.pivot(index="hospital", columns="role", values="on staff").reset_index(),
                    hide_index=True, width="stretch")
    else:
        st.info("This run was saved before doctor/nurse tracking was added. Re-run it to see staff charts.")
    team_df = pd.DataFrame([{"team": k_, "status": v} for k_, v in snap["teams"].items()])
    d.subheader("Responder teams")
    d.dataframe(team_df, hide_index=True, width="stretch")

    hist = pd.DataFrame([{"minute": s["t"], "waiting victims": s["waiting"],
                          "free ambulances": sum(x == "available" for x in s["ambulances"].values()),
                          "free beds": sum(s["beds_available"].values())} for s in run["snapshots"]])
    fig = px.line(hist, x="minute", y=["waiting victims", "free ambulances", "free beds"], title="Over time")
    fig.add_vline(x=t, line_dash="dash", line_color="gray")
    st.plotly_chart(fig.update_layout(height=320, legend_title=""), width="stretch")

with tab_agents:
    if msgs.empty:
        st.info("This run is the greedy baseline: there are no agents and no messages. "
                "Pick an `agents_...` run in the sidebar.")
    else:
        msgs_now = msgs_now.assign(agent=msgs_now["sender"].map(agent_name))
        cols = st.columns(4)
        for col, name in zip(cols, ["Incident Assessment", "Dispatch", "Resource Allocation", "Hospital Coordination"]):
            mine = msgs_now[msgs_now["agent"] == name]
            with col:
                st.markdown(f"#### {name}")
                if mine.empty:
                    st.caption("idle")
                    continue
                last = mine.iloc[-1]
                busy = last["t"] == t
                st.markdown(f"{'🟢 **acting now**' if busy else '⚪ waiting'} · last at minute {last['t']}")
                st.metric("messages sent", len(mine))
                llm_share = (mine["decided_by"] == "llm").mean()
                st.caption(f"decided by LLM: {llm_share:.0%}")
                for _, m in mine.tail(4).iloc[::-1].iterrows():
                    st.markdown(f"`{m['t']}` **{m['msg_type']}** {m['proposed_action']}  \n"
                                f"<span style='color:gray;font-size:0.8em'>{m['reason']}</span>",
                                unsafe_allow_html=True)
        st.divider()
        pending = msgs_now[(msgs_now["t"] == t) & (msgs_now["msg_type"] == "ADMISSION_REQUEST")]
        st.markdown(f"**Pending / just-made admission requests at minute {t}:** {len(pending)}")
        if not pending.empty:
            st.dataframe(pending[["sender", "recipient", "incident_id", "proposed_action"]],
                         hide_index=True, width="stretch")

with tab_neg:
    if msgs.empty:
        st.info("No negotiation in the baseline run.")
    else:
        counts = msgs_now["msg_type"].value_counts()
        cols = st.columns(7)
        for col, key, label in zip(cols, ["CONFLICT", "ADMISSION_REQUEST", "ACCEPT", "REJECT", "COUNTER",
                                          "STAFF_REQUEST", "STAFF_OFFER"],
                                   ["Conflicts", "Requests", "Accepted", "Rejected", "Counter-offers",
                                    "Staff requests", "Staff loans"]):
            col.metric(label, int(counts.get(key, 0)))
        per_min = (msgs[msgs["msg_type"].isin(["ACCEPT", "REJECT", "COUNTER", "CONFLICT", "STAFF_OFFER",
                                               "STAFF_DECLINE"])]
                   .groupby(["t", "msg_type"]).size().reset_index(name="n"))
        if not per_min.empty:
            fig = px.bar(per_min, x="t", y="n", color="msg_type", title="Negotiation outcomes over time",
                         color_discrete_map={"ACCEPT": "#30A46C", "REJECT": "#E5484D", "COUNTER": "#F5A524",
                                             "CONFLICT": "#A26BF5", "STAFF_OFFER": "#3E8EF7",
                                             "STAFF_DECLINE": "#A0A7B4"})
            fig.add_vline(x=t, line_dash="dash", line_color="gray")
            st.plotly_chart(fig.update_layout(height=300, xaxis_title="minute", yaxis_title="messages"),
                            width="stretch")
        options = sorted(msgs["msg_type"].unique())
        types = st.multiselect("Show message types", options,
                               default=[x for x in ["CONFLICT", "COUNTER", "REJECT", "DECISION", "STAFF_REQUEST",
                                                    "STAFF_OFFER", "STAFF_DECLINE"] if x in options])
        view = msgs_now[msgs_now["msg_type"].isin(types)].iloc[::-1]
        st.dataframe(view[["t", "sender", "recipient", "msg_type", "incident_id", "proposed_action", "reason",
                           "decided_by"]], hide_index=True, width="stretch", height=380)
with tab_perf:
    st.subheader(f"Final results for seed {run['seed']} (incl. allocation conflicts and negotiation)")
    table = pd.DataFrame({Path(p).stem: run_info(p)[1] for p in runs if run_info(p)[0] == run["seed"]})
    st.dataframe(table, width="stretch")

    summary = Path("results/summary.csv")
    if summary.exists():
        st.subheader("Experiments: 3 scenarios × 10 seeds (baseline vs. agents)")
        s = pd.read_csv(summary)
        labels = {"critical_unserved_in_time": "Critical victims missed (lower is better)",
                  "victims_served": "Victims served (higher is better)",
                  "avg_response_min": "Average response time, min (lower is better)",
                  "avg_critical_to_hospital_min": "Critical victim to hospital, min (lower is better)"}
        cols = st.columns(2)
        for i, (metric, title) in enumerate(labels.items()):
            d = s[s["metric"] == metric].melt(id_vars=["scenario"], value_vars=["baseline_mean", "agents_mean"],
                                              var_name="policy", value_name="value")
            d["policy"] = d["policy"].str.replace("_mean", "")
            fig = px.bar(d, x="scenario", y="value", color="policy", barmode="group", title=title,
                         color_discrete_map={"baseline": "#A0A7B4", "agents": "#3E8EF7"}, text_auto=".1f")
            cols[i % 2].plotly_chart(fig.update_layout(height=320, legend_title=""), width="stretch")
        st.dataframe(s, hide_index=True, width="stretch")
    else:
        st.info("Run `python run_experiments.py` to add the experiment charts here.")
if st.session_state.playing and st.session_state.minute < T:
    time.sleep(0.4)
    st.session_state._advance = True
    st.rerun()