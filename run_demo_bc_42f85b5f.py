import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ab_feature_utils import aggregate_prediction_recursive_bsafe, separate_a_b_features


class DummyPredictor:
    def __init__(self, bias: float = 0.0, scale: float = 1.0):
        self.bias = bias
        self.scale = scale

    def predict_period_aggregate(self, features: dict) -> float:
        # Very simple linear rule on a few keys to keep deterministic
        base = float(features.get('past_demand_mean', 0.0))
        wknd = float(features.get('future_weekend_days', 0.0))
        holi = float(features.get('future_holiday_days', 0.0))
        return max(0.0, self.scale * (base * (1.0 + 0.05 * wknd + 0.08 * holi)) + self.bias)


class SimpleConfig:
    def __init__(self, past_window: int = 14, future_window: int = 7, step: int = 7):
        self.past_window = past_window
        self.future_window = future_window
        self.step = step


def build_synthetic_daily(start_date: str = '2023-01-01', end_date: str = '2023-04-30') -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    rng = np.random.default_rng(123)

    # Base seasonal pattern: weekday higher on Mon-Fri, lower on weekend
    weekday = dates.dayofweek.values
    weekday_multiplier = np.where(weekday < 5, 1.0, 0.8)

    # Simple yearly/monthly component via sine on day index
    t = np.arange(len(dates))
    seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * t / 30.0)

    demand = 100 * weekday_multiplier * seasonal + rng.normal(0, 5, size=len(dates))
    demand = np.clip(demand, 0, None)

    df = pd.DataFrame({'date': dates})
    df['demand'] = demand

    # Time features (subset of A features expected by utils)
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    # Fake holiday flags: every 20th day
    df['is_holiday'] = ((df['date'].dt.day % 20) == 0).astype(int)
    # One-hot weekdays used by aggregate A-features
    weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for i, name in enumerate(weekdays):
        df[f'is_{name}'] = (df['date'].dt.dayofweek == i).astype(int)

    # Additional innocuous features
    df['avg_price'] = 10 + 0.1 * t + rng.normal(0, 0.5, size=len(dates))

    return df


def main() -> None:
    daily_df = build_synthetic_daily()

    # Choose features: include demand and some others to exercise pipeline
    selected_features = [
        'demand', 'avg_price', 'is_weekend', 'is_holiday',
        'is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday', 'is_Saturday', 'is_Sunday',
    ]

    # Train/test split date
    test_start_date = '2023-03-15'

    # Simple predictors bag
    trained_predictors = {
        'DummyRF': DummyPredictor(bias=0.0, scale=1.0),
        'XGBoost': DummyPredictor(bias=2.0, scale=1.02),  # use this as reference distributor
    }

    config = SimpleConfig(past_window=14, future_window=7, step=7)

    result = aggregate_prediction_recursive_bsafe(
        daily_df=daily_df,
        selected_features=selected_features,
        product_name='synthetic',
        config=config,
        trained_predictors=trained_predictors,
        test_start_date=test_start_date,
        verbose=True,
    )

    preds = result['predictions']
    actuals = result['actuals']
    periods = result['period_infos']

    # Print brief summary
    print('Models:', list(preds.keys()))
    print('Num periods:', len(actuals))
    for i, info in enumerate(periods[:5]):
        row = {name: float(preds[name][i]) for name in preds}
        print(f"{i:02d} {info['start'].date()} ~ {info['end'].date()} | actual={actuals[i]:.2f} | preds={row}")


if __name__ == '__main__':
    main()

