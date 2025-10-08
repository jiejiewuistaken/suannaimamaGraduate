from __future__ import annotations

import argparse
from typing import List
import pandas as pd

from .direct import DirectAggregatePipeline, DirectConfig
from .indirect import IndirectDailyPipeline, IndirectConfig
from . import features as F
from . import metrics as M


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Forecasting CLI: direct aggregated vs indirect daily pipelines')
    p.add_argument('--data', required=True, help='Path to CSV containing daily data')
    p.add_argument('--date-col', default='date')
    p.add_argument('--demand-col', default='demand')
    p.add_argument('--test-start', required=True, help='Test start date, e.g., 2024-01-01')
    p.add_argument('--past-window', type=int, default=28)
    p.add_argument('--future-window', type=int, default=7)
    p.add_argument('--step', type=int, default=7)
    p.add_argument('--mode', choices=['direct', 'indirect', 'both'], default='both')
    p.add_argument('--save-preds', default='')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    if args.date_col != 'date':
        df = df.rename(columns={args.date_col: 'date'})
    if args.demand_col != 'demand':
        df = df.rename(columns={args.demand_col: 'demand'})

    df = F.add_calendar_features(df, date_col='date')
    # Common
    selected = F.infer_selected_features(df)
    train_end = pd.to_datetime(args.test_start) - pd.Timedelta(days=1)

    if args.mode in ('direct', 'both'):
        dcfg = DirectConfig(past_window=args.past_window, future_window=args.future_window, step=args.step, date_col='date')
        dpipe = DirectAggregatePipeline(dcfg)
        dpipe.fit(df, selected, train_end_date=str(train_end.date()))
        dres = dpipe.rolling_origin_backtest(df, test_start_date=args.test_start)
        d_actuals = dres['actuals']
        d_name = list(dres['predictions'].keys())[0]
        d_preds = dres['predictions'][d_name]
        d_scores = M.evaluate_aggregate(d_actuals, d_preds)
        print('=== Direct (aggregate) ===')
        print(f'Model: {d_name}')
        for k, v in d_scores.items():
            print(f'{k}: {v:.6f}')
        if args.save_preds and args.mode == 'direct':
            rows = []
            for (pinfo, y, yhat) in zip(dres['period_infos'], d_actuals, d_preds):
                rows.append({'period_start': pinfo['start'], 'period_end': pinfo['end'], 'actual': y, 'pred': yhat})
            pd.DataFrame(rows).to_csv(args.save_preds, index=False)
            print(f'Saved predictions to {args.save_preds}')

    if args.mode in ('indirect', 'both'):
        icfg = IndirectConfig(past_window=args.past_window, future_window=args.future_window, step=args.step, date_col='date')
        ipipe = IndirectDailyPipeline(icfg)
        ipipe.fit(df, selected, train_end_date=str(train_end.date()))
        ires = ipipe.rolling_origin_backtest(df, test_start_date=args.test_start)
        i_actuals = ires['actuals']
        i_name = list(ires['predictions'].keys())[0]
        i_preds = ires['predictions'][i_name]
        i_scores = M.evaluate_aggregate(i_actuals, i_preds)
        print('=== Indirect (sum of daily) ===')
        print(f'Model: {i_name}')
        for k, v in i_scores.items():
            print(f'{k}: {v:.6f}')
        if args.save_preds and args.mode == 'indirect':
            rows = []
            for (pinfo, y, yhat) in zip(ires['period_infos'], i_actuals, i_preds):
                rows.append({'period_start': pinfo['start'], 'period_end': pinfo['end'], 'actual': y, 'pred': yhat})
            pd.DataFrame(rows).to_csv(args.save_preds, index=False)
            print(f'Saved predictions to {args.save_preds}')


if __name__ == '__main__':
    main()

