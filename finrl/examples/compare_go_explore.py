from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunSpec:
    name: str
    cwd: str
    go_explore_enabled: bool


def load_recorder(cwd: str) -> np.ndarray:
    path = os.path.join(cwd, "recorder.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing recorder.npy at: {path}")
    arr = np.load(path, allow_pickle=True)
    if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Unexpected recorder.npy shape: {getattr(arr, 'shape', None)}")
    return arr


def plot_training_curves(out_dir: str, baseline: RunSpec, go: RunSpec) -> str:
    import matplotlib.pyplot as plt

    base = load_recorder(baseline.cwd)
    ge = load_recorder(go.cwd)

    # Recorder format (ElegantRL): [step, avgR, stdR, expR, objC, objA, etc]
    def _series(arr: np.ndarray):
        step = arr[:, 0]
        avgR = arr[:, 1]
        expR = arr[:, 3]
        return step, avgR, expR

    b_step, b_avgR, b_expR = _series(base)
    g_step, g_avgR, g_expR = _series(ge)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
    ax[0].plot(b_step, b_avgR, label=f"{baseline.name}")
    ax[0].plot(g_step, g_avgR, label=f"{go.name}")
    ax[0].set_title("Training: avgR vs step")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("avgR")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(b_step, b_expR, label=f"{baseline.name}")
    ax[1].plot(g_step, g_expR, label=f"{go.name}")
    ax[1].set_title("Training: expR vs step")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("expR")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "training_curves.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_backtest_assets(
    *,
    cwd: str,
    start_date: str,
    end_date: str,
    ticker_list: list[str],
    if_vix: bool,
    env_kwargs: dict | None = None,
):
    import torch

    from finrl.config import INDICATORS
    from finrl.meta.data_processor import DataProcessor
    from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

    dp = DataProcessor("yahoofinance")
    data = dp.download_data(ticker_list, start_date, end_date, "1D")
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, INDICATORS)
    if if_vix:
        data = dp.add_vix(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
        "env_kwargs": env_kwargs or {"go_explore_enabled": False},
    }
    env = StockTradingEnv(config=env_config)

    actor_path = os.path.join(cwd, "act.pth")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Missing actor at: {actor_path}")

    device = torch.device("cpu")
    act = torch.load(actor_path, map_location=device)
    act.eval()

    state, _info = env.reset()
    assets = [float(env.initial_total_asset)]

    for _ in range(int(env.max_step)):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a_tensor = act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]
        state, _reward, done, _trunc, _info = env.step(action)
        total_asset = float(env.amount + (env.price_ary[env.day] * env.stocks).sum())
        assets.append(total_asset)
        if done:
            break

    return np.asarray(assets, dtype=np.float64)


def plot_assets(out_dir: str, title: str, base_assets: np.ndarray, ge_assets: np.ndarray) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
    ax.plot(base_assets, label="baseline")
    ax.plot(ge_assets, label="go-explore")
    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("total asset")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, title.lower().replace(" ", "_") + ".png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    from finrl.config import ERL_PARAMS, INDICATORS
    from finrl.config import TEST_END_DATE, TEST_START_DATE, TRADE_END_DATE, TRADE_START_DATE
    from finrl.config import TRAIN_END_DATE, TRAIN_START_DATE
    from finrl.config_tickers import DOW_30_TICKER
    from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
    from finrl.train import train

    out_dir = os.path.join("results", "go_explore_compare")

    baseline = RunSpec(name="baseline (go_explore_enabled=False)", cwd="./compare_runs/baseline", go_explore_enabled=False)
    go = RunSpec(name="go-explore (go_explore_enabled=True)", cwd="./compare_runs/go_explore", go_explore_enabled=True)

    # Keep runs short so you can iterate quickly.
    break_step = int(os.environ.get("FINRL_COMPARE_BREAK_STEP", "20000"))

    # Train baseline
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=StockTradingEnv,
        model_name="ppo",
        cwd=baseline.cwd,
        erl_params=ERL_PARAMS,
        break_step=break_step,
        env_kwargs={"go_explore_enabled": False},
    )

    # Train Go-Explore
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=StockTradingEnv,
        model_name="ppo",
        cwd=go.cwd,
        erl_params=ERL_PARAMS,
        break_step=break_step,
        env_kwargs={"go_explore_enabled": True},
    )

    training_plot = plot_training_curves(out_dir, baseline, go)

    # Testing (backtest)
    base_test_assets = run_backtest_assets(
        cwd=baseline.cwd,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        if_vix=True,
        env_kwargs={"go_explore_enabled": False},
    )
    go_test_assets = run_backtest_assets(
        cwd=go.cwd,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        if_vix=True,
        env_kwargs={"go_explore_enabled": False},
    )
    test_plot = plot_assets(out_dir, "Test Total Asset", base_test_assets, go_test_assets)

    # Trading (backtest trade window)
    base_trade_assets = run_backtest_assets(
        cwd=baseline.cwd,
        start_date=TRADE_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=DOW_30_TICKER,
        if_vix=True,
        env_kwargs={"go_explore_enabled": False},
    )
    go_trade_assets = run_backtest_assets(
        cwd=go.cwd,
        start_date=TRADE_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=DOW_30_TICKER,
        if_vix=True,
        env_kwargs={"go_explore_enabled": False},
    )
    trade_plot = plot_assets(out_dir, "Trade Total Asset", base_trade_assets, go_trade_assets)

    print("Saved plots:")
    print(training_plot)
    print(test_plot)
    print(trade_plot)


if __name__ == "__main__":
    main()
