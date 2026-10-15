"""Forecast models for weekly gasoline prices (target and oil driver are selectable).

Diagnostics that drove the design: the CA-US spread is non-stationary (ADF p=0.95) and
log CA gas is not cointegrated with log Brent (p=0.68), so we model weekly *log changes*
with oil pass-through lags rather than price levels.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

EVENTS = ["covid", "p66_la_closed", "benicia_closed", "chevron_es_fire",
          "iraq_war", "ukraine_war", "iran_war_2026"]
DRIVER_LAGS, AR_LAGS = 4, 2


def load(path="weekly_data.csv"):
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def _features(df, target, driver):
    X = pd.DataFrame(index=df.index)
    dd = np.log(df[driver]).diff()
    dy = np.log(df[target]).diff()
    for k in range(DRIVER_LAGS + 1):
        X[f"ddriver_l{k}"] = dd.shift(k)
    for k in range(1, AR_LAGS + 1):
        X[f"dy_l{k}"] = dy.shift(k)
    X["d_summer"] = df.summer_blend.diff()
    X["d_july_tax"] = df.july_tax.diff()
    for e in EVENTS:
        X[f"d_{e}"] = df[e].diff()
    return X, dy


class ARX:
    """OLS on dlog(target): AR lags + oil pass-through lags + calendar/event dummies."""

    def __init__(self, target="ca_gas", driver="brent"):
        self.target, self.driver = target, driver

    def fit(self, df):
        X, y = _features(df, self.target, self.driver)
        ok = X.notna().all(axis=1) & y.notna()
        X, y = X[ok], y[ok]
        self.cols = [c for c in X.columns if X[c].std() > 0]
        self.res = sm.OLS(y, sm.add_constant(X[self.cols])).fit()
        self.resid = self.res.resid.values
        return self

    def forecast(self, df, h, driver_dlog, n_sims=400, seed=0):
        """driver_dlog: (h,) future weekly dlog(oil). Returns (h, n_sims) price paths.
        Future dummies are held constant (diffs = 0): no extrapolated events."""
        rng = np.random.default_rng(seed)
        p = self.res.params
        dy_hist = list(np.log(df[self.target]).diff().dropna().values[-AR_LAGS:])
        dd_hist = list(np.log(df[self.driver]).diff().dropna().values[-DRIVER_LAGS:])
        paths = np.zeros((h, n_sims))
        level = np.full(n_sims, np.log(df[self.target].iloc[-1]))
        dy = np.tile(dy_hist[::-1], (n_sims, 1))
        for t in range(h):
            dd = [driver_dlog[t]] + dd_hist[::-1][:DRIVER_LAGS]
            mu = np.full(n_sims, p["const"])
            for c in self.cols:
                if c.startswith("ddriver_l"):
                    mu += p[c] * dd[int(c[-1])]
                elif c.startswith("dy_l"):
                    mu += p[c] * dy[:, int(c[-1]) - 1]
            step = mu + rng.choice(self.resid, n_sims)
            level = level + step
            paths[t] = np.exp(level)
            dy = np.column_stack([step, dy[:, :-1]])
            dd_hist.append(driver_dlog[t])
        return paths


def driver_path(total_pct, h):
    """Spread a total % change in the oil price evenly (in logs) over h weeks."""
    return np.full(h, np.log1p(total_pct / 100) / h)


def arima_forecast(df, h, target="ca_gas"):
    m = ARIMA(np.log(df[target].values), order=(2, 1, 0), trend="t").fit()
    f = m.get_forecast(h)
    mu, se = f.predicted_mean, np.sqrt(f.var_pred_mean)
    return np.exp(mu), np.exp(mu - 1.2816 * se), np.exp(mu + 1.2816 * se)


def backtest(df, target="ca_gas", driver="brent", horizon=12, n_origins=26, step=4):
    """Rolling-origin out-of-sample test; every model is refit at each origin.
    'ARX (oil flat)' uses only information available at the origin; 'ARX (oracle oil)'
    sees the realised oil path, an upper bound showing how much error is oil uncertainty."""
    rows = []
    n = len(df)
    for o in [n - horizon - i * step for i in range(n_origins)][::-1]:
        train, test = df.iloc[:o], df.iloc[o:o + horizon]
        actual = test[target].values
        arx = ARX(target, driver).fit(train)
        real = np.log(test[driver]).diff().to_numpy().copy()
        real[0] = np.log(test[driver].iloc[0] / train[driver].iloc[-1])
        preds = {
            "Naive (last value)": np.full(horizon, train[target].iloc[-1]),
            "ARIMA(2,1,0)+drift": arima_forecast(train, horizon, target)[0],
            "ARX (oil flat)": np.median(arx.forecast(train, horizon, np.zeros(horizon), 200), axis=1),
            "ARX (oracle oil)": np.median(arx.forecast(train, horizon, real, 200), axis=1),
        }
        for name, pr in preds.items():
            for k in range(horizon):
                rows.append((train.index[-1], name, k + 1, actual[k], pr[k]))
    r = pd.DataFrame(rows, columns=["origin", "model", "h", "actual", "pred"])
    r["err"] = r.pred - r.actual
    return r


def accuracy(bt):
    out = bt.groupby(["model", "h"]).apply(lambda d: pd.Series({
        "MAE": d.err.abs().mean(),
        "RMSE": np.sqrt((d.err ** 2).mean()),
        "MAPE %": (d.err.abs() / d.actual).mean() * 100}), include_groups=False)
    return out.reset_index()
