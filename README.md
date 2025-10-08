## Forecasting Pipelines

This branch adds two reproducible forecasting pipelines and a CLI:

- Direct (aggregate): model the H-day aggregated demand directly with rolling-origin backtest.
- Indirect (daily): model daily demand with recursive B-feature filling, then aggregate to H-day windows.

### Data format

CSV with at minimum columns:

- `date`: ISO date
- `demand`: numeric target per day

Optional columns can include calendar/holiday flags; the pipeline will also generate standard calendar features.

### CLI Usage

```bash
python3 -m forecasting.cli --data your_daily.csv --test-start 2024-01-01 --past-window 28 --future-window 7 --step 7 --mode both
```

Outputs WAPE/sMAPE/MAE for both pipelines. Use `--save-preds path.csv` to export period-level predictions for the chosen mode.

