from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
import pandas as pd

from forecasting.direct import DirectAggregatePipeline, DirectConfig
from forecasting.indirect import IndirectDailyPipeline, IndirectConfig
from forecasting import features as F
from forecasting import metrics as M


def ensure_example_csv(path: str) -> None:
    if os.path.exists(path):
        return
    # Make a small synthetic daily dataset with weekly seasonality
    start = datetime(2024, 1, 1)
    days = 120
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        w = d.weekday()
        base = 100
        # weekend boost
        demand = base + (20 if w >= 5 else 0)
        # mild trend
        demand += i * 0.2
        rows.append({'date': d.strftime('%Y-%m-%d'), 'demand': float(demand)})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    csv_path = 'data.csv'
    ensure_example_csv(csv_path)

    df = pd.read_csv(csv_path)
    df = F.add_calendar_features(df, date_col='date')

    test_start = '2024-03-15'
    past_window = 28
    future_window = 7
    step = 7

    selected = F.infer_selected_features(df)
    train_end = pd.to_datetime(test_start) - pd.Timedelta(days=1)

    # Direct (aggregate)
    dcfg = DirectConfig(past_window=past_window, future_window=future_window, step=step, date_col='date')
    dpipe = DirectAggregatePipeline(dcfg)
    dpipe.fit(df, selected, train_end_date=str(train_end.date()))
    dres = dpipe.rolling_origin_backtest(df, test_start_date=test_start)
    d_actuals = dres['actuals']
    d_name = list(dres['predictions'].keys())[0]
    d_preds = dres['predictions'][d_name]
    d_scores = M.evaluate_aggregate(d_actuals, d_preds)

    # Indirect (sum of daily)
    icfg = IndirectConfig(past_window=past_window, future_window=future_window, step=step, date_col='date')
    ipipe = IndirectDailyPipeline(icfg)
    ipipe.fit(df, selected, train_end_date=str(train_end.date()))
    ires = ipipe.rolling_origin_backtest(df, test_start_date=test_start)
    i_actuals = ires['actuals']
    i_name = list(ires['predictions'].keys())[0]
    i_preds = ires['predictions'][i_name]
    i_scores = M.evaluate_aggregate(i_actuals, i_preds)

    print('=== Direct (aggregate) ===')
    print(f'Model: {d_name}')
    for k, v in d_scores.items():
        print(f'{k}: {v:.6f}')

    print('=== Indirect (sum of daily) ===')
    print(f'Model: {i_name}')
    for k, v in i_scores.items():
        print(f'{k}: {v:.6f}')

    # Save predictions for inspection
    pd.DataFrame([
        {
            'period_start': p['start'],
            'period_end': p['end'],
            'actual': y,
            'pred_direct': yhat_d,
            'pred_indirect': yhat_i,
        }
        for p, y, yhat_d, yhat_i in zip(dres['period_infos'], d_actuals, d_preds, i_preds)
    ]).to_csv('predictions.csv', index=False)
    print('Saved predictions to predictions.csv')


if __name__ == '__main__':
    main()

