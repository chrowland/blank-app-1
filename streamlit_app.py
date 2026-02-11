# streamlit_app.py
# V1: Early-stage New LOB Model Factory (Strategic Finance / FP&A style)
# - Transaction archetype (units x price) with scenario overrides
# - Headcount plan + loaded cost by role
# - P&L + simplified cash
# - Scenario comparison + one-way sensitivities
#
# Run:
#   pip install streamlit pandas numpy openpyxl
#   streamlit run streamlit_app.py

from __future__ import annotations

import io
import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Helpers
# -----------------------------
def month_start(d: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d.year, d.month, 1)


def make_timeline(start_month: pd.Timestamp, horizon_months: int) -> pd.DatetimeIndex:
    start = month_start(start_month)
    return pd.date_range(start=start, periods=int(horizon_months), freq="MS")


def stable_hash(obj) -> str:
    b = repr(obj).encode("utf-8")
    return hashlib.md5(b).hexdigest()


def fmt_money(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"${x:,.0f}"


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{100*x:,.1f}%"


def ensure_session_defaults():
    if "model" not in st.session_state:
        st.session_state.model = default_model()


# -----------------------------
# Model structures
# -----------------------------
SCENARIOS = ["Base", "Upside", "Downside"]

ASSUMPTION_META = [
    # key, display, category, unit, dtype, default
    ("meta.archetype", "Archetype", "meta", "", "scalar", "transaction"),
    ("cash.starting_cash", "Starting cash (optional)", "cash", "$", "scalar", 0.0),
    ("pricing.price_per_unit", "Price per unit", "pricing", "$/unit", "timeseries", 300.0),
    ("pricing.discount_rate", "Discount rate", "pricing", "%", "scalar", 0.05),
    ("demand.new_units", "New units (pre-conversion)", "demand", "count", "timeseries", 1000.0),
    ("demand.conversion_rate", "Conversion rate", "demand", "%", "scalar", 0.85),
    ("varcost.cogs_pct_revenue", "COGS as % of revenue", "variable_cost", "%", "scalar", 0.15),
    ("varcost.processing_pct_revenue", "Processing as % of revenue", "variable_cost", "%", "scalar", 0.02),
    ("varcost.service_cost_per_unit", "Service cost per unit", "variable_cost", "$/unit", "scalar", 3.0),
    ("opex.tools", "Tools / vendors (non-labor)", "opex", "$/mo", "timeseries", 50_000.0),
    ("opex.marketing_non_labor", "Marketing (non-labor)", "opex", "$/mo", "timeseries", 200_000.0),
    ("capital.capex", "Capex", "capital", "$/mo", "timeseries", 0.0),
]

PNL_LINES = [
    ("pnl.revenue", "Revenue", "$"),
    ("pnl.cogs", "COGS", "$"),
    ("pnl.processing", "Processing", "$"),
    ("pnl.service", "Service", "$"),
    ("pnl.variable_costs", "Variable costs (total)", "$"),
    ("pnl.gross_profit", "Gross profit", "$"),
    ("pnl.gross_margin", "Gross margin", "%"),
    ("pnl.labor", "Labor (loaded)", "$"),
    ("pnl.tools", "Tools / vendors", "$"),
    ("pnl.marketing_non_labor", "Marketing (non-labor)", "$"),
    ("pnl.opex_total", "Opex (total)", "$"),
    ("pnl.ebitda", "EBITDA", "$"),
]

CASH_LINES = [
    ("cash.capex", "Capex", "$"),
    ("cash.net_burn", "Net cash burn (simplified)", "$"),
    ("cash.cum_burn", "Cumulative burn", "$"),
    ("cash.ending_cash", "Ending cash", "$"),
]


@dataclass
class Model:
    lob_name: str
    start_month: pd.Timestamp
    horizon_months: int
    assumptions_base: Dict[str, object]  # scalars + timeseries arrays
    assumptions_override: Dict[str, Dict[str, object]]  # scenario -> overrides dict
    headcount_plan: pd.DataFrame  # role, start_month, headcount
    loaded_cost_by_role: pd.DataFrame  # role, loaded_cost_per_month


def default_model() -> Model:
    start = pd.Timestamp(date.today().replace(day=1))
    horizon = 36
    idx = make_timeline(start, horizon)

    base: Dict[str, object] = {}
    for k, _, _, _, dtype, default in ASSUMPTION_META:
        if dtype == "scalar":
            base[k] = default
        elif dtype == "timeseries":
            base[k] = pd.Series([default] * len(idx), index=idx)
        else:
            base[k] = default

    overrides = {s: {} for s in SCENARIOS}
    # Simple default scenario shaping (optional)
    overrides["Upside"]["demand.new_units"] = base["demand.new_units"] * 1.2
    overrides["Upside"]["opex.marketing_non_labor"] = base["opex.marketing_non_labor"] * 1.15
    overrides["Downside"]["demand.new_units"] = base["demand.new_units"] * 0.8
    overrides["Downside"]["opex.marketing_non_labor"] = base["opex.marketing_non_labor"] * 0.9

    headcount_plan = pd.DataFrame(
        {
            "role": ["GM", "Product", "Eng", "Sales", "Ops"],
            "start_month": [idx[0], idx[0], idx[1], idx[2], idx[1]],
            "headcount": [1, 2, 6, 3, 2],
        }
    )
    loaded_cost = pd.DataFrame(
        {
            "role": ["GM", "Product", "Eng", "Sales", "Ops"],
            "loaded_cost_per_month": [35_000, 25_000, 24_000, 22_000, 18_000],
        }
    )

    return Model(
        lob_name="New LOB (Draft)",
        start_month=start,
        horizon_months=horizon,
        assumptions_base=base,
        assumptions_override=overrides,
        headcount_plan=headcount_plan,
        loaded_cost_by_role=loaded_cost,
    )


# -----------------------------
# Assumption resolution
# -----------------------------
def resolve_assumption(model: Model, scenario: str, key: str, timeline: pd.DatetimeIndex):
    """Return a scalar or pd.Series aligned to timeline."""
    base_val = model.assumptions_base.get(key)
    override_val = model.assumptions_override.get(scenario, {}).get(key)

    val = override_val if override_val is not None else base_val

    if isinstance(val, pd.Series):
        # Ensure alignment
        return val.reindex(timeline).fillna(method="ffill").fillna(method="bfill")
    else:
        return val


def resolve_all(model: Model, scenario: str, timeline: pd.DatetimeIndex) -> Dict[str, object]:
    out = {}
    for k, *_ in ASSUMPTION_META:
        out[k] = resolve_assumption(model, scenario, k, timeline)
    return out


# -----------------------------
# Compute engine (Transaction archetype V1)
# -----------------------------
def compute_headcount_costs(model: Model, timeline: pd.DatetimeIndex) -> pd.Series:
    """Monthly fully-loaded labor cost from headcount plan + loaded cost table."""
    plan = model.headcount_plan.copy()
    costs = model.loaded_cost_by_role.set_index("role")["loaded_cost_per_month"].to_dict()

    # Expand plan to monthly headcount by role (assumes step function from start_month onward)
    hc_by_role = {}
    for role in plan["role"].unique():
        s = pd.Series(0.0, index=timeline)
        rows = plan[plan["role"] == role].sort_values("start_month")
        for _, r in rows.iterrows():
            start = month_start(pd.Timestamp(r["start_month"]))
            if start in s.index:
                s.loc[start:] += float(r["headcount"])
            else:
                # if start outside range, ignore
                pass
        hc_by_role[role] = s

    labor = pd.Series(0.0, index=timeline)
    for role, hc in hc_by_role.items():
        labor += hc * float(costs.get(role, 0.0))
    return labor


def compute_financials(model: Model, scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """Return pnl_df, cash_df, kpis, drivers_df."""
    tl = make_timeline(model.start_month, model.horizon_months)
    A = resolve_all(model, scenario, tl)

    # Drivers
    new_units = A["demand.new_units"]
    conv = float(A["demand.conversion_rate"])
    units = new_units * conv

    price = A["pricing.price_per_unit"]
    discount = float(A["pricing.discount_rate"])
    gross_rev = units * price
    revenue = gross_rev * (1.0 - discount)

    # Variable costs
    cogs_pct = float(A["varcost.cogs_pct_revenue"])
    proc_pct = float(A["varcost.processing_pct_revenue"])
    svc_per_unit = float(A["varcost.service_cost_per_unit"])

    cogs = revenue * cogs_pct
    processing = revenue * proc_pct
    service = units * svc_per_unit
    variable_costs = cogs + processing + service
    gross_profit = revenue - variable_costs
    gross_margin = np.where(revenue.values != 0, (gross_profit / revenue).values, np.nan)

    # Opex
    labor = compute_headcount_costs(model, tl)
    tools = A["opex.tools"]
    mkt_nl = A["opex.marketing_non_labor"]
    opex_total = labor + tools + mkt_nl

    ebitda = gross_profit - opex_total

    pnl = pd.DataFrame(
        {
            "pnl.revenue": revenue,
            "pnl.cogs": -cogs,
            "pnl.processing": -processing,
            "pnl.service": -service,
            "pnl.variable_costs": -variable_costs,
            "pnl.gross_profit": gross_profit,
            "pnl.gross_margin": gross_margin,
            "pnl.labor": -labor,
            "pnl.tools": -tools,
            "pnl.marketing_non_labor": -mkt_nl,
            "pnl.opex_total": -opex_total,
            "pnl.ebitda": ebitda,
        },
        index=tl,
    )

    # Cash (simplified): burn is -min(ebitda, 0) + capex
    capex = A["capital.capex"]
    burn = (-np.minimum(ebitda, 0.0)) + capex
    cum_burn = burn.cumsum()
    starting_cash = float(A["cash.starting_cash"])
    ending_cash = starting_cash - cum_burn

    cash = pd.DataFrame(
        {
            "cash.capex": capex,
            "cash.net_burn": burn,
            "cash.cum_burn": cum_burn,
            "cash.ending_cash": ending_cash,
        },
        index=tl,
    )

    # KPIs
    peak_burn = float(cum_burn.max())
    peak_burn_month = cum_burn.idxmax()
    be_month = None
    be_date = None
    be_idx = np.where(ebitda.values >= 0)[0]
    if len(be_idx) > 0:
        be_month = int(be_idx[0] + 1)
        be_date = tl[be_idx[0]]

    kpis = {
        "Revenue (Year 1)": float(revenue.iloc[:12].sum()),
        "Revenue (Year 3)": float(revenue.iloc[24:36].sum()) if len(revenue) >= 36 else float(revenue.sum()),
        "Gross margin (avg)": float(np.nanmean(gross_margin)),
        "EBITDA (Year 1)": float(ebitda.iloc[:12].sum()),
        "Peak cumulative burn": peak_burn,
        "Peak burn month": peak_burn_month,
        "EBITDA break-even month": be_month,
        "EBITDA break-even date": be_date,
    }

    drivers = pd.DataFrame(
        {
            "new_units": new_units,
            "units_after_conversion": units,
            "price_per_unit": price,
            "gross_revenue": gross_rev,
            "revenue_net_discount": revenue,
        },
        index=tl,
    )

    return pnl, cash, kpis, drivers


# -----------------------------
# UI components
# -----------------------------
def kpi_strip(kpis: Dict[str, object]):
    cols = st.columns(6)
    cols[0].metric("Rev Y1", fmt_money(kpis["Revenue (Year 1)"]))
    cols[1].metric("Rev Y3", fmt_money(kpis["Revenue (Year 3)"]))
    cols[2].metric("Gross margin", fmt_pct(kpis["Gross margin (avg)"]))
    cols[3].metric("EBITDA Y1", fmt_money(kpis["EBITDA (Year 1)"]))
    cols[4].metric("Peak burn", fmt_money(kpis["Peak cumulative burn"]))
    be = kpis["EBITDA break-even date"]
    cols[5].metric("EBITDA BE", "-" if be is None else str(pd.Timestamp(be).date()))


def rollup_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "Monthly":
        out = df.copy()
    elif mode == "Quarterly":
        out = df.resample("QS").sum()
    elif mode == "Annual":
        out = df.resample("YS").sum()
    else:
        out = df.copy()
    return out


def export_to_excel(model: Model, scenario: str) -> bytes:
    pnl, cash, kpis, drivers = compute_financials(model, scenario)
    tl = pnl.index

    meta = pd.DataFrame(
        {
            "key": ["lob_name", "scenario", "start_month", "horizon_months"],
            "value": [model.lob_name, scenario, str(model.start_month.date()), model.horizon_months],
        }
    )

    # Assumptions export (base + scenario overrides)
    rows = []
    for k, display, cat, unit, dtype, _ in ASSUMPTION_META:
        base = model.assumptions_base.get(k)
        ov = model.assumptions_override.get(scenario, {}).get(k)
        if dtype == "scalar":
            rows.append(
                {
                    "assumption_key": k,
                    "display_name": display,
                    "category": cat,
                    "unit": unit,
                    "type": dtype,
                    "base_value": float(base) if base is not None else np.nan,
                    "scenario_value": float(ov) if ov is not None else np.nan,
                }
            )
        else:
            # summarize timeseries
            base_y1 = float(base.reindex(tl).iloc[:12].sum())
            scen_series = resolve_assumption(model, scenario, k, tl)
            scen_y1 = float(pd.Series(scen_series, index=tl).iloc[:12].sum())
            rows.append(
                {
                    "assumption_key": k,
                    "display_name": display,
                    "category": cat,
                    "unit": unit,
                    "type": dtype,
                    "base_value": base_y1,
                    "scenario_value": scen_y1,
                }
            )
    assumptions_summary = pd.DataFrame(rows)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        meta.to_excel(writer, index=False, sheet_name="Meta")
        pd.DataFrame({"metric": list(kpis.keys()), "value": list(kpis.values())}).to_excel(
            writer, index=False, sheet_name="KPIs"
        )
        assumptions_summary.to_excel(writer, index=False, sheet_name="Assumptions_Summary")
        drivers.to_excel(writer, sheet_name="Drivers")
        pnl.to_excel(writer, sheet_name="P&L")
        cash.to_excel(writer, sheet_name="Cash")
        model.headcount_plan.to_excel(writer, index=False, sheet_name="Headcount_Plan")
        model.loaded_cost_by_role.to_excel(writer, index=False, sheet_name="Loaded_Costs")

    return buf.getvalue()


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="New LOB Model Factory (V1)", layout="wide")
ensure_session_defaults()
model: Model = st.session_state.model

st.sidebar.title("New LOB Model Factory (V1)")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Assumptions", "Revenue Drivers", "Costs & Headcount", "Financial Statements", "Cash & Runway", "Scenarios", "Sensitivities", "Export"],
)

st.sidebar.divider()
model.lob_name = st.sidebar.text_input("LOB name", value=model.lob_name)
model.start_month = pd.Timestamp(st.sidebar.date_input("Start month", value=model.start_month.date())).replace(day=1)
model.horizon_months = int(st.sidebar.slider("Horizon (months)", 12, 60, int(model.horizon_months), step=12))

scenario = st.sidebar.selectbox("Active scenario", SCENARIOS, index=0)

# compute cache
compute_key = stable_hash((model.lob_name, str(model.start_month), model.horizon_months, model.assumptions_base, model.assumptions_override, model.headcount_plan.to_dict(), model.loaded_cost_by_role.to_dict(), scenario))
if "compute_cache" not in st.session_state or st.session_state.get("compute_key") != compute_key:
    pnl, cash, kpis, drivers = compute_financials(model, scenario)
    st.session_state.compute_cache = (pnl, cash, kpis, drivers)
    st.session_state.compute_key = compute_key
else:
    pnl, cash, kpis, drivers = st.session_state.compute_cache


# -----------------------------
# Pages
# -----------------------------
if page == "Overview":
    st.title("Overview")
    kpi_strip(kpis)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Revenue")
        st.line_chart(pnl["pnl.revenue"])
    with c2:
        st.subheader("EBITDA")
        st.line_chart(pnl["pnl.ebitda"])
    with c3:
        st.subheader("Cumulative burn")
        st.line_chart(cash["cash.cum_burn"])

    st.subheader("What changed across scenarios (vs Base)")
    base_pnl, base_cash, base_kpis, _ = compute_financials(model, "Base")
    comp = []
    for s in SCENARIOS:
        _, _, sk, _ = compute_financials(model, s)
        comp.append(
            {
                "Scenario": s,
                "Rev Y1": sk["Revenue (Year 1)"],
                "Rev Y3": sk["Revenue (Year 3)"],
                "GM avg": sk["Gross margin (avg)"],
                "EBITDA Y1": sk["EBITDA (Year 1)"],
                "Peak burn": sk["Peak cumulative burn"],
                "BE date": None if sk["EBITDA break-even date"] is None else str(pd.Timestamp(sk["EBITDA break-even date"]).date()),
            }
        )
    st.dataframe(pd.DataFrame(comp).set_index("Scenario"), use_container_width=True)

elif page == "Assumptions":
    st.title("Assumptions")
    tl = make_timeline(model.start_month, model.horizon_months)

    left, right = st.columns([1, 2])

    with left:
        cats = sorted(list({c for _, _, c, *_ in ASSUMPTION_META}))
        cat = st.selectbox("Category", ["(all)"] + cats)
        dtype = st.selectbox("Type", ["(all)", "scalar", "timeseries"])
        show_required = st.checkbox("Show only core inputs", value=True)

    # Build a table of assumption defs
    defs = []
    for k, display, category, unit, dt, default in ASSUMPTION_META:
        if show_required and k.startswith("meta."):
            continue
        defs.append({"key": k, "name": display, "category": category, "unit": unit, "type": dt})
    defs_df = pd.DataFrame(defs)

    if cat != "(all)":
        defs_df = defs_df[defs_df["category"] == cat]
    if dtype != "(all)":
        defs_df = defs_df[defs_df["type"] == dtype]

    sel_key = st.selectbox("Select assumption", defs_df["key"].tolist())
    meta_row = [r for r in ASSUMPTION_META if r[0] == sel_key][0]
    _, display, category, unit, dt, default = meta_row

    st.caption(f"**{display}**  •  Category: `{category}`  •  Unit: `{unit}`  •  Type: `{dt}`")

    base_val = model.assumptions_base.get(sel_key)
    scen_overrides = model.assumptions_override.get(scenario, {})
    has_override = sel_key in scen_overrides

    override_toggle = st.toggle("Override in active scenario", value=has_override)

    if dt == "scalar":
        # pick current value
        current = float(resolve_assumption(model, scenario, sel_key, tl))
        new_val = st.number_input("Value", value=float(current), step=0.01, format="%.4f")
        if st.button("Save"):
            if override_toggle and scenario != "Base":
                model.assumptions_override[scenario][sel_key] = float(new_val)
            else:
                # write to base (and remove scenario override if toggle off)
                model.assumptions_base[sel_key] = float(new_val)
                if sel_key in model.assumptions_override.get(scenario, {}) and not override_toggle:
                    model.assumptions_override[scenario].pop(sel_key, None)
            st.success("Saved.")
            st.rerun()

    elif dt == "timeseries":
        series = resolve_assumption(model, scenario, sel_key, tl)
        series = pd.Series(series, index=tl).copy()

        st.write("Edit as monthly time series (you can paste a column of numbers).")
        editor_df = pd.DataFrame({"month": series.index.strftime("%Y-%m"), "value": series.values})
        edited = st.data_editor(editor_df, use_container_width=True, num_rows="fixed", height=400)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Fill forward from first value"):
                v0 = float(edited["value"].iloc[0])
                edited["value"] = v0
                st.session_state._tmp_timeseries = edited
                st.rerun()
        with c2:
            growth = st.number_input("Apply monthly growth rate", value=0.0, step=0.005, format="%.3f")
            if st.button("Apply growth"):
                vals = []
                v = float(edited["value"].iloc[0])
                for i in range(len(edited)):
                    if i == 0:
                        vals.append(v)
                    else:
                        v = v * (1.0 + float(growth))
                        vals.append(v)
                edited["value"] = vals
                st.session_state._tmp_timeseries = edited
                st.rerun()
        with c3:
            if st.button("Reset to Base"):
                base_series = model.assumptions_base[sel_key]
                if isinstance(base_series, pd.Series):
                    edited["value"] = base_series.reindex(tl).values
                    st.session_state._tmp_timeseries = edited
                    st.rerun()
        with c4:
            if st.button("Save series"):
                s = pd.Series(pd.to_numeric(edited["value"], errors="coerce").values, index=tl)
                if override_toggle and scenario != "Base":
                    model.assumptions_override[scenario][sel_key] = s
                else:
                    model.assumptions_base[sel_key] = s
                    if sel_key in model.assumptions_override.get(scenario, {}) and not override_toggle:
                        model.assumptions_override[scenario].pop(sel_key, None)
                st.success("Saved.")
                st.rerun()

        # allow persisting temporary edits across reruns from buttons
        if "_tmp_timeseries" in st.session_state:
            st.info("Temporary edits applied. Re-open the editor by re-selecting the assumption if needed.")
            st.session_state.pop("_tmp_timeseries", None)

elif page == "Revenue Drivers":
    st.title("Revenue Drivers (Transaction archetype)")
    st.caption("This V1 assumes: `units = new_units × conversion_rate`, `revenue = units × price × (1 - discount)`.")

    st.subheader("Drivers")
    st.dataframe(drivers, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Units after conversion")
        st.line_chart(drivers["units_after_conversion"])
    with c2:
        st.subheader("Net revenue")
        st.line_chart(drivers["revenue_net_discount"])

elif page == "Costs & Headcount":
    st.title("Costs & Headcount")

    st.subheader("Loaded costs by role ($/month)")
    costs_edit = st.data_editor(model.loaded_cost_by_role, use_container_width=True, num_rows="dynamic")
    if st.button("Save loaded costs"):
        model.loaded_cost_by_role = costs_edit
        st.success("Saved loaded costs.")
        st.rerun()

    st.subheader("Headcount plan (step function from start_month)")
    st.caption("Each row adds headcount starting in the specified month through the horizon.")
    plan_edit = st.data_editor(model.headcount_plan, use_container_width=True, num_rows="dynamic")
    if st.button("Save headcount plan"):
        # basic cleaning
        plan_edit["start_month"] = plan_edit["start_month"].apply(lambda x: month_start(pd.Timestamp(x)))
        model.headcount_plan = plan_edit
        st.success("Saved headcount plan.")
        st.rerun()

    st.subheader("Labor cost preview")
    tl = make_timeline(model.start_month, model.horizon_months)
    labor = compute_headcount_costs(model, tl)
    st.line_chart(labor)

    st.subheader("Variable + non-labor opex preview")
    st.line_chart(pd.DataFrame({"Tools": -pnl["pnl.tools"], "Marketing NL": -pnl["pnl.marketing_non_labor"]}))

elif page == "Financial Statements":
    st.title("Financial Statements")
    roll = st.radio("View", ["Monthly", "Quarterly", "Annual"], horizontal=True)

    pnl_view = rollup_table(pnl.drop(columns=["pnl.gross_margin"]), roll)
    # For gross margin, recompute from rolled revenue and gross profit
    if roll != "Monthly":
        rev = rollup_table(pnl[["pnl.revenue"]], roll)["pnl.revenue"]
        gp = rollup_table(pnl[["pnl.gross_profit"]], roll)["pnl.gross_profit"]
        gm = (gp / rev).replace([np.inf, -np.inf], np.nan)
        pnl_view["pnl.gross_margin"] = gm

    # reorder and format
    cols = []
    for k, _, _fmt in PNL_LINES:
        if k in pnl_view.columns:
            cols.append(k)
    pnl_view = pnl_view[cols]

    # Display friendly
    display = pnl_view.copy()
    for k, _, unit in PNL_LINES:
        if k in display.columns:
            if unit == "$":
                display[k] = display[k].apply(fmt_money)
            else:
                display[k] = display[k].apply(fmt_pct)

    st.dataframe(display, use_container_width=True)

    st.subheader("Line-item trace (V1: formula + key assumptions)")
    line = st.selectbox("Select line", [k for k, _, _ in PNL_LINES])
    if line == "pnl.revenue":
        st.markdown(
            "- `revenue = (new_units × conversion_rate) × price_per_unit × (1 - discount_rate)`\n"
            "- Key assumptions: `demand.new_units`, `demand.conversion_rate`, `pricing.price_per_unit`, `pricing.discount_rate`"
        )
    elif line in ("pnl.cogs", "pnl.processing", "pnl.service"):
        st.markdown(
            "- `cogs = revenue × cogs_pct_revenue`\n"
            "- `processing = revenue × processing_pct_revenue`\n"
            "- `service = units × service_cost_per_unit`"
        )
    elif line == "pnl.ebitda":
        st.markdown("- `EBITDA = gross_profit - labor - tools - marketing_non_labor`")
    else:
        st.markdown("- V1 trace: see Drivers + Assumptions pages for inputs.")

elif page == "Cash & Runway":
    st.title("Cash & Runway")
    roll = st.radio("View", ["Monthly", "Quarterly", "Annual"], horizontal=True)

    cash_view = rollup_table(cash, roll)
    display = cash_view.copy()
    for k in display.columns:
        display[k] = display[k].apply(fmt_money)

    st.dataframe(display, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Net cash burn")
        st.line_chart(cash["cash.net_burn"])
    with c2:
        st.subheader("Cumulative burn")
        st.line_chart(cash["cash.cum_burn"])

    st.subheader("Runway / Ending cash")
    st.line_chart(cash["cash.ending_cash"])

elif page == "Scenarios":
    st.title("Scenarios")
    st.caption("Compare Base/Upside/Downside. Scenarios are implemented as assumption overrides.")

    rows = []
    for s in SCENARIOS:
        _, _, sk, _ = compute_financials(model, s)
        rows.append(
            {
                "Scenario": s,
                "Rev Y1": sk["Revenue (Year 1)"],
                "Rev Y3": sk["Revenue (Year 3)"],
                "GM avg": sk["Gross margin (avg)"],
                "EBITDA Y1": sk["EBITDA (Year 1)"],
                "Peak burn": sk["Peak cumulative burn"],
                "BE date": None if sk["EBITDA break-even date"] is None else pd.Timestamp(sk["EBITDA break-even date"]).date(),
            }
        )
    df = pd.DataFrame(rows).set_index("Scenario")
    df_disp = df.copy()
    df_disp["Rev Y1"] = df_disp["Rev Y1"].apply(fmt_money)
    df_disp["Rev Y3"] = df_disp["Rev Y3"].apply(fmt_money)
    df_disp["GM avg"] = df_disp["GM avg"].apply(fmt_pct)
    df_disp["EBITDA Y1"] = df_disp["EBITDA Y1"].apply(fmt_money)
    df_disp["Peak burn"] = df_disp["Peak burn"].apply(fmt_money)
    st.dataframe(df_disp, use_container_width=True)

    st.subheader("Quick lever bundles (V1 demo)")
    st.caption("These are macros that modify multiple assumptions in the active scenario.")
    bundle = st.selectbox("Select bundle", ["(none)", "Hiring pace +20%", "Marketing intensity +15% (and +10% units)", "COO constrained (-15% marketing and -10% units)"])
    if st.button("Apply bundle to active scenario"):
        if bundle == "(none)":
            st.info("No bundle selected.")
        else:
            tl = make_timeline(model.start_month, model.horizon_months)
            # bundles operate by creating overrides
            if scenario == "Base":
                st.warning("Bundles are best applied to Upside/Downside/custom scenarios. Switch scenario on sidebar.")
            else:
                if bundle == "Hiring pace +20%":
                    model.headcount_plan["headcount"] = model.headcount_plan["headcount"].astype(float) * 1.2
                elif bundle == "Marketing intensity +15% (and +10% units)":
                    m = resolve_assumption(model, scenario, "opex.marketing_non_labor", tl)
                    u = resolve_assumption(model, scenario, "demand.new_units", tl)
                    model.assumptions_override[scenario]["opex.marketing_non_labor"] = pd.Series(m, index=tl) * 1.15
                    model.assumptions_override[scenario]["demand.new_units"] = pd.Series(u, index=tl) * 1.10
                elif bundle == "COO constrained (-15% marketing and -10% units)":
                    m = resolve_assumption(model, scenario, "opex.marketing_non_labor", tl)
                    u = resolve_assumption(model, scenario, "demand.new_units", tl)
                    model.assumptions_override[scenario]["opex.marketing_non_labor"] = pd.Series(m, index=tl) * 0.85
                    model.assumptions_override[scenario]["demand.new_units"] = pd.Series(u, index=tl) * 0.90
                st.success("Applied bundle.")
                st.rerun()

elif page == "Sensitivities":
    st.title("Sensitivities (one-way)")
    st.caption("V1 runs a one-way sensitivity by scaling one assumption and recomputing outcomes.")

    target_metric = st.selectbox(
        "Metric",
        ["Peak cumulative burn", "Revenue (Year 1)", "Revenue (Year 3)", "EBITDA (Year 1)"],
    )

    candidate_keys = [
        "pricing.price_per_unit",
        "demand.new_units",
        "demand.conversion_rate",
        "pricing.discount_rate",
        "varcost.cogs_pct_revenue",
        "opex.marketing_non_labor",
    ]
    key = st.selectbox("Assumption", candidate_keys)
    span = st.slider("Scale range", 0.5, 1.5, (0.8, 1.2), 0.05)
    steps = st.slider("Steps", 5, 21, 9, 2)

    tl = make_timeline(model.start_month, model.horizon_months)

    multipliers = np.linspace(span[0], span[1], int(steps))
    results = []
    # Baseline metric
    _, _, base_k, _ = compute_financials(model, scenario)
    base_val = base_k[target_metric]

    for m in multipliers:
        # create a temporary override on top of current scenario
        temp = model.assumptions_override.get(scenario, {}).copy()

        current_val = resolve_assumption(model, scenario, key, tl)
        if isinstance(current_val, pd.Series):
            temp[key] = pd.Series(current_val, index=tl) * float(m)
        else:
            temp[key] = float(current_val) * float(m)

        # compute with temp overrides
        saved = model.assumptions_override[scenario]
        model.assumptions_override[scenario] = temp
        try:
            _, _, kpi, _ = compute_financials(model, scenario)
        finally:
            model.assumptions_override[scenario] = saved

        results.append({"multiplier": m, "metric": kpi[target_metric]})

    res = pd.DataFrame(results)
    st.line_chart(res.set_index("multiplier")["metric"])
    st.write(f"Baseline ({scenario}) **{target_metric}**: {fmt_money(base_val) if 'Revenue' in target_metric or 'EBITDA' in target_metric or 'burn' in target_metric else base_val}")

    st.dataframe(res.assign(delta=lambda d: d["metric"] - base_val), use_container_width=True)

elif page == "Export":
    st.title("Export")
    st.caption("Exports current scenario outputs and key inputs to Excel.")

    xlsx = export_to_excel(model, scenario)
    st.download_button(
        "Download Excel",
        data=xlsx,
        file_name=f"{model.lob_name.replace(' ', '_')}_{scenario}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Reset model")
    if st.button("Reset to defaults"):
        st.session_state.model = default_model()
        st.success("Reset.")
        st.rerun()

# Persist edits back
st.session_state.model = model

