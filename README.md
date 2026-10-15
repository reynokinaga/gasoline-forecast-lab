# Gasoline Forecast Lab

Interactive Streamlit dashboard for weekly U.S. and California gasoline prices (May 2000 to Oct 2026).

## What it does
- **Explore history:** pick any variables (prices, crude, spread, refinery utilization, inventories), choose a date range, index or raw scale, and shade event windows (COVID, Iraq/Ukraine/Iran wars, refinery closures and fire).
- **Forecast & scenarios:** choose the variable to forecast (CA or US gasoline), the oil driver (Brent or WTI), horizon (1-26 weeks), models, and several oil-price scenarios side by side. Download the forecasts as CSV.
- **Compare models (backtest):** rolling-origin out-of-sample test (26 origins, models refit each time) with MAE / RMSE / MAPE by horizon, plus a view of any single forecast origin.
- **Diagnostics:** the evidence behind the model design.

## Method
Diagnostics: the CA-US spread is non-stationary (ADF p = 0.95), stepping up from about $0.3 before 2015 to $1.2-1.4 since 2022, and log CA gas is not cointegrated with log Brent (p = 0.68). So the main model (**ARX**) is an OLS regression of weekly log price changes on 2 own lags, the current and 4 lagged weekly changes in oil, and calendar/event dummies. Forecast intervals come from bootstrapped residuals. Benchmarks: ARIMA(2,1,0) with drift, and a naive last-value forecast.

## Key finding
With oil held flat (the only information available at forecast time), ARX does not beat naive or ARIMA (CA gasoline, 8 weeks ahead: MAPE about 7% for all three). Given the realised oil path, ARX improves to about 5.7%. The model is therefore best used as a scenario tool for "what if crude moves X%", not as a better point forecaster.

## Limitations
Refinery-closure and war flags are still on at the end of the sample, so reopening scenarios cannot be estimated; forecasts assume current conditions persist. Oil enters as a user-chosen scenario, not a forecast.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
