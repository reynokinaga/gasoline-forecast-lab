import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import model

st.set_page_config(page_title="Gasoline Forecast Lab", layout="wide")
st.title("Gasoline Forecast Lab")
st.caption("Weekly U.S. and California gasoline prices, 2000 to date. Explore the history, "
           "compare forecasting models, and test oil-price scenarios.")

TARGETS = {"California gasoline ($/gal)": "ca_gas", "U.S. gasoline ($/gal)": "us_gas"}
DRIVERS = {"Brent crude ($/bbl)": "brent", "WTI crude ($/bbl)": "wti"}
VARS = {"ca_gas": "CA gasoline", "us_gas": "US gasoline", "spread": "CA-US spread",
        "brent": "Brent", "wti": "WTI", "refinery_util_padd5": "West Coast refinery util %",
        "refinery_util_us": "US refinery util %", "gas_stocks_padd5": "West Coast gasoline stocks",
        "gas_stocks_us": "US gasoline stocks"}
EVENT_LABELS = {"covid": "COVID", "p66_la_closed": "P66 LA closed", "benicia_closed": "Benicia closed",
                "chevron_es_fire": "Chevron ES fire", "iraq_war": "Iraq war",
                "ukraine_war": "Ukraine war", "iran_war_2026": "Iran war"}


@st.cache_data
def get_df():
    return model.load("weekly_data.csv")


@st.cache_data(show_spinner="Running rolling-origin backtest (about 10 seconds)...")
def get_backtest(target, driver):
    return model.backtest(get_df(), target, driver, horizon=12)


def shade_events(fig, df, lo, hi):
    for e, label in EVENT_LABELS.items():
        on = df[e].astype(bool)
        grp = (on != on.shift()).cumsum()
        for _, sub in df[on].groupby(grp[on]):
            a, b = max(sub.index[0], lo), min(sub.index[-1], hi)
            if a <= b:
                fig.add_vrect(x0=a, x1=b, fillcolor="orange", opacity=0.12, line_width=0,
                              annotation_text=label, annotation_position="top left",
                              annotation_font_size=9)


df = get_df()
last = df.index[-1]

with st.sidebar:
    st.header("Forecast settings")
    t_label = st.selectbox("Variable to forecast", list(TARGETS))
    d_label = st.selectbox("Oil price driver", list(DRIVERS))
    target, driver = TARGETS[t_label], DRIVERS[d_label]
    h = st.slider("Forecast horizon (weeks)", 1, 26, 8)
    models = st.multiselect("Models", ["ARX (oil scenario)", "ARIMA(2,1,0)+drift", "Naive (last value)"],
                            default=["ARX (oil scenario)", "ARIMA(2,1,0)+drift"])
    scen = st.multiselect("Oil scenarios (total % change over horizon)",
                          [-30, -20, -10, 0, 10, 20, 30], default=[-20, 0, 20],
                          format_func=lambda p: f"{p:+d}%")
    focus = st.selectbox("Show 80% band for scenario", scen or [0], format_func=lambda p: f"{p:+d}%")
    history = st.slider("History shown on forecast chart (weeks)", 52, 520, 156, step=26)

tab_x, tab_f, tab_b, tab_d = st.tabs(["Explore history", "Forecast & scenarios",
                                      "Compare models (backtest)", "Diagnostics"])

with tab_x:
    c1, c2 = st.columns([3, 1])
    sel = c1.multiselect("Variables", list(VARS), default=["ca_gas", "us_gas", "brent"],
                         format_func=lambda k: VARS[k])
    scale = c2.radio("Scale", ["Indexed (start = 100)", "Raw values"])
    dmin, dmax = df.index.min().date(), df.index.max().date()
    rng = st.slider("Date range", dmin, dmax, (dt.date(2015, 1, 1), dmax), format="YYYY-MM")
    shade = st.checkbox("Shade events (closures, wars, COVID)", value=True)
    lo, hi = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    sub = df.loc[lo:hi]
    fig = go.Figure()
    for v in sel:
        s = sub[v].dropna()
        if s.empty:
            continue
        y = s / s.iloc[0] * 100 if scale.startswith("Indexed") else s
        fig.add_scatter(x=s.index, y=y, name=VARS[v])
    if shade:
        shade_events(fig, df, lo, hi)
    fig.update_layout(height=500, hovermode="x unified",
                      yaxis_title="Index" if scale.startswith("Indexed") else "Value")
    st.plotly_chart(fig, width="stretch")
    if scale.startswith("Raw") and len({"stocks" in VARS[v] or "util" in VARS[v] for v in sel}) > 1:
        st.caption("Raw values mix very different units; use Indexed to compare shapes.")

with tab_f:
    fig = go.Figure()
    hist = df[target].iloc[-history:]
    fig.add_scatter(x=hist.index, y=hist, name="Actual", line=dict(color="#222"))
    future = pd.date_range(last + pd.Timedelta(weeks=1), periods=h, freq="7D")
    arx = model.ARX(target, driver).fit(df)
    table = {}
    if "ARX (oil scenario)" in models:
        if not scen:
            st.warning("Pick at least one oil scenario in the sidebar.")
        for p in sorted(scen):
            paths = arx.forecast(df, h, model.driver_path(p, h))
            q10, mid, q90 = np.percentile(paths, [10, 50, 90], axis=1)
            table[f"ARX, oil {p:+d}%"] = mid
            if p == focus:
                fig.add_scatter(x=future, y=q90, line=dict(width=0), showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=future, y=q10, fill="tonexty", line=dict(width=0),
                                fillcolor="rgba(31,119,180,0.18)", name=f"80% band (oil {p:+d}%)",
                                hoverinfo="skip")
            fig.add_scatter(x=future, y=mid, name=f"ARX, oil {p:+d}%")
    if "ARIMA(2,1,0)+drift" in models:
        m_, l_, u_ = model.arima_forecast(df, h, target)
        table["ARIMA"] = m_
        fig.add_scatter(x=future, y=m_, name="ARIMA", line=dict(color="#d62728", dash="dash"))
    if "Naive (last value)" in models:
        table["Naive"] = np.full(h, df[target].iloc[-1])
        fig.add_scatter(x=future, y=table["Naive"], name="Naive", line=dict(color="gray", dash="dot"))
    fig.update_layout(yaxis_title="$ / gallon", height=500, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Last {VARS[target]}", f"${df[target].iloc[-1]:.2f}")
    m2.metric("CA-US spread", f"${df.spread.iloc[-1]:.2f}", help=f"Historical max ${df.spread.max():.2f}")
    m3.metric(d_label.split(" (")[0], f"${df[driver].iloc[-1]:.0f}")
    if table:
        out = pd.DataFrame(table, index=future.date).round(2)
        out.index.name = "week"
        st.markdown(f"**Median forecast ($/gal), weeks 1 to {h}**")
        st.dataframe(out, width="stretch")
        st.download_button("Download forecasts (CSV)", out.to_csv(), f"{target}_forecasts.csv", "text/csv")
    st.info("Scenario limits: refinery-closure and war flags are still on at the end of the sample, so "
            "they have no 'after' period to learn from. Forecasts assume they stay on; a reopening "
            "scenario cannot be estimated from these data.")

with tab_b:
    bt = get_backtest(target, driver)
    bt_h = st.slider("Backtest horizon (weeks)", 2, 12, 8)
    metric = st.radio("Accuracy metric", ["MAE", "RMSE", "MAPE %"], horizontal=True)
    acc = model.accuracy(bt[bt.h <= bt_h])
    f2 = go.Figure()
    for name, d in acc.groupby("model"):
        f2.add_scatter(x=d.h, y=d[metric], name=name, mode="lines+markers",
                       line=dict(dash="dot") if "oracle" in name else None)
    f2.update_layout(xaxis_title="Weeks ahead", yaxis_title=metric, height=400)
    st.plotly_chart(f2, width="stretch")
    st.caption("26 rolling origins, 4 weeks apart, all models refit at each origin. 'ARX (oil flat)' uses only "
               "what was known at the origin. 'ARX (oracle oil)' is given the realised oil path: an upper bound "
               "showing how much forecast error comes from not knowing where crude goes.")
    st.markdown("**Look at one forecast origin**")
    origins = sorted(bt.origin.unique())
    o = st.select_slider("Forecast origin", origins, value=origins[-1],
                         format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"))
    sel_o = bt[(bt.origin == o) & (bt.h <= bt_h)]
    o = pd.Timestamp(o)
    ctx = df[target].loc[o - pd.Timedelta(weeks=12): o + pd.Timedelta(weeks=bt_h)]
    f3 = go.Figure()
    f3.add_scatter(x=ctx.index, y=ctx, name="Actual", line=dict(color="#222"))
    for name, d in sel_o.groupby("model"):
        f3.add_scatter(x=[o + pd.Timedelta(weeks=int(k)) for k in d.h], y=d.pred, name=name,
                       line=dict(dash="dot" if "oracle" in name else "solid"))
    f3.add_vline(x=o, line_dash="dash", line_color="gray")
    f3.update_layout(yaxis_title="$ / gallon", height=380, hovermode="x unified")
    st.plotly_chart(f3, width="stretch")

with tab_d:
    from statsmodels.tsa.stattools import adfuller, coint
    sp = df.spread
    yr = sp.groupby(sp.index.year).mean()
    f4 = go.Figure(go.Bar(x=yr.index, y=yr.values))
    f4.update_layout(title="Average CA-US spread by year ($/gal)", height=320)
    st.plotly_chart(f4, width="stretch")
    a, b = st.columns(2)
    a.metric("ADF p-value, CA-US spread", f"{adfuller(sp)[1]:.2f}", help="High = cannot reject a unit root")
    b.metric("Cointegration p, log CA vs log Brent",
             f"{coint(np.log(df.ca_gas), np.log(df.brent))[1]:.2f}")
    st.markdown("The spread ratchets up in steps (about 0.3 before 2015, 1.2 to 1.4 since 2022) and does not "
                "mean-revert, and CA prices are not cointegrated with Brent. So the model works in **weekly log "
                "changes** with oil pass-through lags instead of forecasting price levels.")
    st.write(f"ARX coefficients for the current selection ({t_label}, {d_label}).")
    arx_d = model.ARX(target, driver).fit(df)
    st.dataframe(arx_d.res.summary2().tables[1].round(4))
