import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ========================= 单日：辅助函数 ========================= #

def _rolling_ma_shift(series: pd.Series, window: int) -> pd.Series:
    ma = series.rolling(window=window, min_periods=1).mean()
    return ma.shift(1)


def calculate_adaptive_baseline_for_agg(
    agg_df: pd.DataFrame, actual_col: str, horizon_days: int
) -> pd.Series:
    if horizon_days <= 7:
        return _rolling_ma_shift(agg_df[actual_col], window=13)
    elif 8 <= horizon_days <= 14:
        return _rolling_ma_shift(agg_df[actual_col], window=6)
    else:
        return _rolling_ma_shift(agg_df[actual_col], window=3)


def ensure_stacking_in_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    comparison_like: pd.DataFrame,
    actual_col: str,
    pred_suffix: str = "_forecast",
) -> Dict[str, Dict[str, float]]:
    if any(k.lower() == "stacking" for k in metrics_dict.keys()):
        return metrics_dict
    stack_cols = [
        c
        for c in comparison_like.columns
        if c.replace(pred_suffix, "").lower() == "stacking" and c.endswith(pred_suffix)
    ]
    if not stack_cols:
        return metrics_dict
    stacking_col = stack_cols[0]
    try:
        # 依赖宿主环境中提供的 calculate_streamlined_metrics
        from __main__ import calculate_streamlined_metrics  # type: ignore

        m = calculate_streamlined_metrics(comparison_like[actual_col], comparison_like[stacking_col])
        metrics_dict = dict(metrics_dict)
        metrics_dict["Stacking"] = m
    except Exception:
        pass
    return metrics_dict


def plot_model_heatmap(metrics_dict: Dict[str, Dict[str, float]], title: str, save_path: str) -> None:
    if not metrics_dict:
        return
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    metrics_to_show = [m for m in ["MAE", "WMAPE (%)", "RMSE", "R²"] if m in metrics_df.columns]
    if not metrics_to_show:
        return
    metrics_df = metrics_df[metrics_to_show]
    norm_df = metrics_df.copy()
    ann_df = metrics_df.copy()
    for m in metrics_to_show:
        col = metrics_df[m]
        if m == "R²":
            norm_df[m] = col
            ann_df[m] = col.apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        elif "%" in m:
            norm_df[m] = -col
            ann_df[m] = col.apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        else:
            norm_df[m] = -col
            ann_df[m] = col.apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
    plt.figure(figsize=(10, max(6, 0.4 * len(norm_df))))
    sns.heatmap(norm_df, annot=ann_df, fmt="", cmap="RdYlGn", cbar=False)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_agg_comparison(
    agg_df: pd.DataFrame, actual_col: str, pred_cols: List[str], title: str, save_path: str
) -> None:
    plt.figure(figsize=(20, 12))
    plt.plot(
        agg_df.iloc[:, 0],
        agg_df[actual_col],
        "o-",
        label="实际需求",
        linewidth=3,
        markersize=8,
        color="black",
    )
    for col in pred_cols:
        if col in agg_df.columns:
            plt.plot(
                agg_df.iloc[:, 0],
                agg_df[col],
                "o-",
                label=col.replace("_forecast", "").replace("_prediction", ""),
            )
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_metrics_from_agg(
    agg_df: pd.DataFrame, actual_col: str, pred_suffixes: Tuple[str, ...] = ("_forecast", "_prediction")
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    try:
        from __main__ import calculate_streamlined_metrics  # type: ignore
    except Exception:
        return out
    for c in agg_df.columns:
        if any(c.endswith(suf) for suf in pred_suffixes):
            name = c
            for suf in pred_suffixes:
                name = name.replace(suf, "")
            try:
                out[name] = calculate_streamlined_metrics(agg_df[actual_col], agg_df[c])
            except Exception:
                continue
    return out


def visualize_residual_diagnostics_daily(
    test_results: pd.DataFrame, product_name: str, output_dir: str
) -> None:
    actual = test_results["demand"].values
    candidate_models: List[str] = []
    priority_order = [
        "Stacking",
        "XGBoost",
        "RandomForest",
        "LightGBM",
        "GBM",
        "SVR",
        "LinearRegression",
        "QRF",
        "MLP",
    ]
    for m in priority_order:
        col = f"{m}_prediction"
        if col in test_results.columns:
            candidate_models.append(m)
    if not candidate_models:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["red", "blue", "green", "purple"]
    for idx, m in enumerate(candidate_models[:4]):
        pred = test_results[f"{m}_prediction"].values
        n = min(len(actual), len(pred))
        residuals = actual[:n] - pred[:n]
        axes[0, 0].plot(residuals, "o-", label=m, color=colors[idx % len(colors)], alpha=0.7)
        if len(residuals) > 1:
            autocorr = []
            for i in range(1, min(12, len(residuals))):
                corr = np.corrcoef(residuals[:-i], residuals[i:])[0, 1]
                autocorr.append(corr)
            if autocorr:
                axes[0, 1].plot(
                    range(1, len(autocorr) + 1),
                    autocorr,
                    "o-",
                    label=m,
                    color=colors[idx % len(colors)],
                    alpha=0.7,
                )
        axes[1, 0].hist(residuals, bins=20, alpha=0.5, label=m, color=colors[idx % len(colors)])
    axes[0, 0].set_title("残差时序")
    axes[0, 0].legend()
    axes[0, 1].set_title("残差自相关(简版)")
    axes[0, 1].legend()
    axes[1, 0].set_title("残差分布")
    axes[1, 0].legend()
    axes[1, 1].axis("off")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{product_name}_残差诊断_单日.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def sliding_window_eval_daily(
    test_results: pd.DataFrame, product_name: str, output_dir: str, windows: List[int] = [7, 14, 21, 30]
) -> None:
    if test_results is None or test_results.empty:
        return
    models = [c.replace("_prediction", "") for c in test_results.columns if c.endswith("_prediction")]
    if not models:
        return
    dates = pd.to_datetime(test_results["date"]) if "date" in test_results.columns else pd.RangeIndex(len(test_results))
    actual = test_results["demand"].values
    for w in windows:
        plt.figure(figsize=(16, 9))
        for m in models[:6]:
            pred = test_results.get(f"{m}_prediction")
            if pred is None:
                continue
            pred = pred.values
            n = min(len(actual), len(pred))
            ae = np.abs(actual[:n] - pred[:n])
            rolling_mae = pd.Series(ae).rolling(window=w, min_periods=1).mean()
            plt.plot(dates[:n], rolling_mae, label=f"{m} (MAE)")
        plt.title(f"{product_name} 滚动窗口评估（窗口={w}日）")
        plt.xlabel("日期")
        plt.ylabel("滚动MAE")
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_path = os.path.join(output_dir, f"{product_name}_滑动窗口评估_{w}D.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def run_single_day_postprocessing(all_results: Dict[str, Dict], base_dir: str) -> None:
    try:
        from __main__ import calculate_streamlined_metrics  # type: ignore
    except Exception:
        # 依赖宿主环境定义
        return
    for product, res in all_results.items():
        output_dir = res.get("output_dir", os.path.join(base_dir, product))
        os.makedirs(output_dir, exist_ok=True)

        weekly_cmp = res.get("comparison_df")
        if isinstance(weekly_cmp, pd.DataFrame) and not weekly_cmp.empty:
            wm = res.get("weekly_metrics", {})
            wm = ensure_stacking_in_metrics(wm, weekly_cmp, "actual_weekly_demand", "_forecast")
            res["weekly_metrics"] = wm
            plot_model_heatmap(
                wm,
                f"{product} 模型性能热力图 (7D)",
                os.path.join(output_dir, f"{product}_逐日预测各产品独立指标热力图_7D.png"),
            )

        for horizon, key, title_prefix in [
            (14, "bi14d_results", "14日"),
            (15, "bi15d_results", "15日"),
            (30, "bi30d_results", "30日"),
        ]:
            agg = res.get(key)
            if isinstance(agg, pd.DataFrame) and not agg.empty:
                if "baseline_forecast" not in agg.columns:
                    try:
                        agg = agg.copy()
                        agg["baseline_forecast"] = calculate_adaptive_baseline_for_agg(
                            agg, "actual_period_demand", horizon
                        )
                        res[key] = agg
                    except Exception:
                        pass
                metrics = compute_metrics_from_agg(agg, "actual_period_demand", ("_prediction",))
                existing_metrics = res.get(key.replace("results", "metrics"), {})
                existing_metrics.update(metrics)
                if "baseline_forecast" in agg.columns:
                    try:
                        existing_metrics["Baseline"] = calculate_streamlined_metrics(
                            agg["actual_period_demand"], agg["baseline_forecast"]
                        )
                    except Exception:
                        pass
                if "Stacking" not in existing_metrics and "Stacking_prediction" in agg.columns:
                    try:
                        existing_metrics["Stacking"] = calculate_streamlined_metrics(
                            agg["actual_period_demand"], agg["Stacking_prediction"]
                        )
                    except Exception:
                        pass
                res[key.replace("results", "metrics")] = existing_metrics
                pred_cols = [c for c in agg.columns if c.endswith("_prediction")] + (
                    ["baseline_forecast"] if "baseline_forecast" in agg.columns else []
                )
                plot_agg_comparison(
                    agg.rename(columns={"period_start": "period"}),
                    "actual_period_demand",
                    pred_cols,
                    f"{product} 逐日预测汇总每{title_prefix}需求比较 (含Stacking/基线)",
                    os.path.join(output_dir, f"{product}_逐日预测汇总比较_{horizon}D.png"),
                )
                plot_model_heatmap(
                    existing_metrics,
                    f"{product} 模型性能热力图 ({title_prefix})",
                    os.path.join(output_dir, f"{product}_逐日预测各产品独立指标热力图_{horizon}D.png"),
                )

        test_results = res.get("daily_results")
        if isinstance(test_results, pd.DataFrame) and not test_results.empty:
            visualize_residual_diagnostics_daily(test_results, product, output_dir)
            sliding_window_eval_daily(test_results, product, output_dir, windows=[7, 14, 21, 30])


# ========================= 多日：辅助函数 ========================= #

def add_adaptive_baseline_block(
    comparison_df: pd.DataFrame, actual_col: str, horizon_days: int
) -> pd.DataFrame:
    df = comparison_df.copy()
    if "baseline_forecast" in df.columns:
        return df
    values = df[actual_col]
    if horizon_days <= 7:
        base = values.rolling(window=13, min_periods=1).mean().shift(1)
    elif 8 <= horizon_days <= 14:
        base = values.rolling(window=6, min_periods=1).mean().shift(1)
    else:
        base = values.rolling(window=3, min_periods=1).mean().shift(1)
    df["baseline_forecast"] = base
    return df


def ensure_stacking_and_baseline_in_performance(
    perf: Dict[str, Dict[str, float]], comparison_df: pd.DataFrame, actual_col: str, horizon_days: int
) -> Dict[str, Dict[str, float]]:
    perf = dict(perf) if perf else {}
    try:
        from __main__ import calculate_streamlined_metrics  # type: ignore
    except Exception:
        return perf
    if "Baseline" not in perf and "baseline_forecast" in comparison_df.columns:
        try:
            perf["Baseline"] = calculate_streamlined_metrics(comparison_df[actual_col], comparison_df["baseline_forecast"])
        except Exception:
            pass
    if "Stacking" not in perf:
        stack_col = (
            "Stacking_forecast"
            if "Stacking_forecast" in comparison_df.columns
            else ("Stacking_prediction" if "Stacking_prediction" in comparison_df.columns else None)
        )
        if stack_col is not None:
            try:
                perf["Stacking"] = calculate_streamlined_metrics(comparison_df[actual_col], comparison_df[stack_col])
            except Exception:
                pass
    return perf


def remove_duplicate_plot_files(output_dir: str, product_name: str, cfg_name: str) -> None:
    dup = os.path.join(output_dir, f"{product_name}_{cfg_name}_预测比较.png")
    if os.path.exists(dup):
        try:
            os.remove(dup)
        except Exception:
            pass

