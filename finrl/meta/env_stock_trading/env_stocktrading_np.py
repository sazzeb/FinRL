from __future__ import annotations

import gymnasium as gym
import numpy as np
import warnings
from numpy import random as rd

from finrl.meta.env_stock_trading.go_explore_archive import (
    GoExploreArchive,
    compute_exposure,
    default_should_consider,
)


class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e6,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
        # Go-Explore style "return then explore" knobs
        go_explore_enabled: bool = False,
        go_explore_warmup_episodes: int = 1,
        go_explore_reset_prob: float = 0.5,
        go_explore_return_mode: str = "checkpoint",  # "checkpoint" | "replay"
        go_explore_archive_max_size: int = 2048,
        go_explore_time_bucket_size: int = 1,
        go_explore_turbulence_bucket_width: float = 0.25,
        go_explore_trend_bucket_width: float = 0.01,
        go_explore_drawdown_bucket_width: float = 0.02,
        go_explore_cash_bucket_width: float = 0.05,
        go_explore_exposure_bucket_width: float = 0.05,
        go_explore_positions_bucket_width: int = 1,
        go_explore_trend_lookback: int = 5,
        go_explore_novelty_coeff: float = 1.0,
        go_explore_uncertainty_coeff: float = 0.25,
        go_explore_risk_coeff: float = 1.0,
        go_explore_sample_beta: float = 1.0,
        # Reward shaping (opt-in)
        reward_mode: str = "delta_asset",  # "delta_asset" | "log_return"
        reward_a: float = 1.0,
        reward_b_turnover: float = 0.1,
        reward_c_drawdown: float = 1.0,

        # Robustification (domain randomization) - opt in
        robustify_enabled: bool = False,
        robustify_buy_cost_pct_range: tuple[float, float] = (1e-3, 3e-3),
        robustify_sell_cost_pct_range: tuple[float, float] = (1e-3, 3e-3),
        robustify_slippage_range: tuple[float, float] = (0.0, 0.0),
    ):
        # Optional: pass env knobs through env_config to support ElegantRL builders that
        # only forward the `config` argument.
        cfg_env_kwargs = config.get("env_kwargs") if isinstance(config, dict) else None
        if isinstance(cfg_env_kwargs, dict):
            for k, v in cfg_env_kwargs.items():
                if k == "go_explore_enabled":
                    go_explore_enabled = v
                elif k == "go_explore_warmup_episodes":
                    go_explore_warmup_episodes = v
                elif k == "go_explore_reset_prob":
                    go_explore_reset_prob = v
                elif k == "go_explore_return_mode":
                    go_explore_return_mode = v
                elif k == "go_explore_archive_max_size":
                    go_explore_archive_max_size = v
                elif k == "go_explore_time_bucket_size":
                    go_explore_time_bucket_size = v
                elif k == "go_explore_turbulence_bucket_width":
                    go_explore_turbulence_bucket_width = v
                elif k == "go_explore_trend_bucket_width":
                    go_explore_trend_bucket_width = v
                elif k == "go_explore_drawdown_bucket_width":
                    go_explore_drawdown_bucket_width = v
                elif k == "go_explore_cash_bucket_width":
                    go_explore_cash_bucket_width = v
                elif k == "go_explore_exposure_bucket_width":
                    go_explore_exposure_bucket_width = v
                elif k == "go_explore_positions_bucket_width":
                    go_explore_positions_bucket_width = v
                elif k == "go_explore_trend_lookback":
                    go_explore_trend_lookback = v
                elif k == "go_explore_novelty_coeff":
                    go_explore_novelty_coeff = v
                elif k == "go_explore_uncertainty_coeff":
                    go_explore_uncertainty_coeff = v
                elif k == "go_explore_risk_coeff":
                    go_explore_risk_coeff = v
                elif k == "go_explore_sample_beta":
                    go_explore_sample_beta = v
                elif k == "reward_mode":
                    reward_mode = v
                elif k == "reward_a":
                    reward_a = v
                elif k == "reward_b_turnover":
                    reward_b_turnover = v
                elif k == "reward_c_drawdown":
                    reward_c_drawdown = v
                elif k == "robustify_enabled":
                    robustify_enabled = v
                elif k == "robustify_buy_cost_pct_range":
                    robustify_buy_cost_pct_range = v
                elif k == "robustify_sell_cost_pct_range":
                    robustify_sell_cost_pct_range = v
                elif k == "robustify_slippage_range":
                    robustify_slippage_range = v

        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]

        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2**-7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks
        )

        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        self.env_name = "StockEnv"
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

        self._next_reset_ckpt = None

        # Go-Explore style archive (promising states)
        self.go_explore_enabled = bool(go_explore_enabled)
        if self.go_explore_enabled:
            warnings.warn(
                "Go-Explore is enabled for StockTradingEnv: training resets may start from archived states (non-standard start-state distribution).",
                RuntimeWarning,
                stacklevel=2,
            )
        self.go_explore_warmup_episodes = int(go_explore_warmup_episodes)
        self.go_explore_reset_prob = float(go_explore_reset_prob)
        self.go_explore_return_mode = str(go_explore_return_mode)
        if self.go_explore_return_mode not in ("checkpoint", "replay"):
            raise ValueError("go_explore_return_mode must be 'checkpoint' or 'replay'")
        self._go_episode_idx = 0
        self._go_started_from_archive = False
        self._go_last_total_asset = None
        self._go_peak_total_asset = None
        self._go_action_history: list[np.ndarray] = []
        self._go_trend_lookback = int(go_explore_trend_lookback)

        self.reward_mode = str(reward_mode)
        if self.reward_mode not in ("delta_asset", "log_return"):
            raise ValueError("reward_mode must be 'delta_asset' or 'log_return'")
        self.reward_a = float(reward_a)
        self.reward_b_turnover = float(reward_b_turnover)
        self.reward_c_drawdown = float(reward_c_drawdown)

        self.robustify_enabled = bool(robustify_enabled)
        self.robustify_buy_cost_pct_range = tuple(float(x) for x in robustify_buy_cost_pct_range)
        self.robustify_sell_cost_pct_range = tuple(float(x) for x in robustify_sell_cost_pct_range)
        self.robustify_slippage_range = tuple(float(x) for x in robustify_slippage_range)
        self._robustify_slippage = 0.0

        # Allow sharing an archive between env instances by passing it through config.
        archive = config.get("go_explore_archive")
        if archive is not None and not isinstance(archive, GoExploreArchive):
            raise TypeError("config['go_explore_archive'] must be a GoExploreArchive or None")

        self.go_explore_archive: GoExploreArchive = archive or GoExploreArchive(
            max_size=go_explore_archive_max_size,
            time_bucket_size=go_explore_time_bucket_size,
            turbulence_bucket_width=go_explore_turbulence_bucket_width,
            trend_bucket_width=go_explore_trend_bucket_width,
            drawdown_bucket_width=go_explore_drawdown_bucket_width,
            cash_bucket_width=go_explore_cash_bucket_width,
            exposure_bucket_width=go_explore_exposure_bucket_width,
            positions_bucket_width=go_explore_positions_bucket_width,
            novelty_coeff=go_explore_novelty_coeff,
            uncertainty_coeff=go_explore_uncertainty_coeff,
            risk_coeff=go_explore_risk_coeff,
            sample_beta=go_explore_sample_beta,
        )

        self._go_visited_cells: set = set()

    def _compute_trend(self, day: int) -> float:
        """Market trend proxy: mean basket return over a short lookback."""
        lb = self._go_trend_lookback
        if lb <= 0 or day <= 0:
            return 0.0
        start = max(0, int(day) - lb)
        if start == int(day):
            return 0.0
        p0 = self.price_ary[start]
        p1 = self.price_ary[int(day)]
        denom = np.where(p0 > 0, p0, 1.0)
        r = (p1 / denom) - 1.0
        return float(np.mean(np.clip(r, -1.0, 1.0)))

    def _compute_drawdown(self, total_asset: float) -> float:
        peak = float(self._go_peak_total_asset) if self._go_peak_total_asset is not None else float(total_asset)
        if peak <= 0:
            return 0.0
        return float(max(0.0, 1.0 - float(total_asset) / peak))

    def _compute_turnover(self, *, prev_stocks: np.ndarray, price: np.ndarray) -> float:
        """Turnover proxy based on change in holdings value (0..inf)."""
        if prev_stocks is None:
            return 0.0
        delta = np.abs(self.stocks - prev_stocks)
        value = float(np.sum(delta * price))
        if self.total_asset is None or float(self.total_asset) <= 0:
            return value
        return float(value / float(self.total_asset))

    def _go_return_to_entry(self, entry) -> tuple[np.ndarray, dict]:
        """Return phase: either restore checkpoint or replay stored actions."""
        if self.go_explore_return_mode == "checkpoint" or not entry.trajectory_actions:
            ckpt = dict(entry.checkpoint)
            ckpt["_go_explore_from_archive"] = True
            self.set_checkpoint(ckpt, reset_metrics=True)
            price = self.price_ary[self.day]
            self._go_started_from_archive = True
            self._go_last_total_asset = float(self.total_asset)
            self._go_peak_total_asset = float(self.total_asset)
            self._go_action_history = []
            return self.get_state(price), {
                "go_explore_started_from_archive": True,
                "go_explore_archive_size": len(self.go_explore_archive),
            }

        # Replay return phase (deterministic exploration setting).
        self.day = 0
        price0 = self.price_ary[self.day]
        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - (
                self.stocks * price0
            ).sum()
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price0).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.episode_return = 0.0
        self._go_started_from_archive = True
        self._go_last_total_asset = float(self.total_asset)
        self._go_peak_total_asset = float(self.total_asset)
        self._go_action_history = []

        # Apply actions without exposing rewards to the learner.
        for a in entry.trajectory_actions:
            if self.day >= self.max_step:
                break
            _s, _r, done, _tr, _info = self.step(np.array(a, copy=True))
            if done:
                break

        price = self.price_ary[self.day]
        return self.get_state(price), {
            "go_explore_started_from_archive": True,
            "go_explore_archive_size": len(self.go_explore_archive),
        }

    def get_checkpoint(self) -> dict:
        return {
            "day": int(self.day),
            "amount": float(self.amount),
            "stocks": np.array(self.stocks, copy=True),
            "stocks_cool_down": np.array(self.stocks_cool_down, copy=True),
            "total_asset": float(self.total_asset),
            "gamma_reward": float(self.gamma_reward),
            "initial_total_asset": float(self.initial_total_asset),
            "episode_return": float(self.episode_return),
            "rng_state": rd.get_state(),
        }

    def set_checkpoint(self, ckpt: dict, reset_metrics: bool = True) -> None:
        day = int(ckpt["day"])
        if day < 0 or day > self.max_step:
            raise ValueError(f"Checkpoint day out of range: {day}")

        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            rd.set_state(rng_state)

        self.day = day
        self.amount = float(ckpt["amount"])
        self.stocks = np.array(ckpt["stocks"], copy=True).astype(np.float32)
        self.stocks_cool_down = np.array(ckpt["stocks_cool_down"], copy=True).astype(
            np.float32
        )
        self.total_asset = float(ckpt["total_asset"])

        if reset_metrics:
            self.initial_total_asset = float(self.total_asset)
            self.gamma_reward = 0.0
            self.episode_return = 0.0
        else:
            self.gamma_reward = float(ckpt.get("gamma_reward", 0.0))
            self.initial_total_asset = float(ckpt.get("initial_total_asset", self.total_asset))
            self.episode_return = float(ckpt.get("episode_return", 0.0))

    def set_next_reset_checkpoint(self, ckpt: dict) -> None:
        self._next_reset_ckpt = ckpt

    def reset(self, *, seed=None, options=None):
        # Episode boundary.
        self._go_episode_idx += 1
        self._go_started_from_archive = False
        self._go_action_history = []

        # Domain randomization for robustification.
        if self.robustify_enabled:
            lo, hi = self.robustify_buy_cost_pct_range
            self.buy_cost_pct = float(lo + (hi - lo) * rd.rand())
            lo, hi = self.robustify_sell_cost_pct_range
            self.sell_cost_pct = float(lo + (hi - lo) * rd.rand())
            lo, hi = self.robustify_slippage_range
            self._robustify_slippage = float(lo + (hi - lo) * rd.rand())
        else:
            self._robustify_slippage = 0.0

        # Highest priority: explicit checkpoint requested by wrapper/caller.
        if self._next_reset_ckpt is not None:
            ckpt = self._next_reset_ckpt
            self._next_reset_ckpt = None
            self.set_checkpoint(ckpt, reset_metrics=True)
            price = self.price_ary[self.day]
            self._go_last_total_asset = float(self.total_asset)
            self._go_peak_total_asset = float(self.total_asset)
            return self.get_state(price), {
                "go_explore_started_from_archive": bool(ckpt.get("_go_explore_from_archive", False)),
                "go_explore_archive_size": len(self.go_explore_archive),
            }

        # Go-Explore: "remember first" then "return to promises".
        if (
            self.go_explore_enabled
            and len(self.go_explore_archive) > 0
            and self._go_episode_idx > self.go_explore_warmup_episodes
            and rd.rand() < self.go_explore_reset_prob
        ):
            entry = self.go_explore_archive.sample()
            if entry is not None:
                self.go_explore_archive.mark_visit(entry.key, step_idx=int(self._go_episode_idx))
                return self._go_return_to_entry(entry)

        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - (
                self.stocks * price
            ).sum()
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        self.episode_return = 0.0

        self._go_last_total_asset = float(self.total_asset)
        self._go_peak_total_asset = float(self.total_asset)
        return self.get_state(price), {
            "go_explore_started_from_archive": False,
            "go_explore_archive_size": len(self.go_explore_archive),
        }

    def step(self, actions):
        actions_raw = np.array(actions, copy=True)
        actions = (actions_raw * self.max_stock).astype(int)

        # Some training loops may call step() even after the env is terminal.
        # Make this safe by returning done=True without advancing beyond max_step.
        if self.day is not None and self.day >= self.max_step:
            price = self.price_ary[self.max_step]
            state = self.get_state(price)
            return state, 0.0, True, False, {
                "go_explore_started_from_archive": self._go_started_from_archive,
                "go_explore_archive_size": len(self.go_explore_archive),
            }

        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1

        prev_stocks = np.array(self.stocks, copy=True)

        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)

            # Slippage: buys pay higher, sells receive lower.
            buy_price = price * (1.0 + self._robustify_slippage)
            sell_price = price * (1.0 - self._robustify_slippage)

            for index in np.where(actions < -min_action)[0]:
                if sell_price[index] > 0:
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    self.stocks[index] -= sell_num_shares
                    self.amount += sell_price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    self.stocks_cool_down[index] = 0

            for index in np.where(actions > min_action)[0]:
                if buy_price[index] > 0:
                    buy_num_shares = min(self.amount // buy_price[index], actions[index])
                    self.stocks[index] += buy_num_shares
                    self.amount -= buy_price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    self.stocks_cool_down[index] = 0

        else:
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()

        # Reward
        if self.reward_mode == "delta_asset":
            reward = (total_asset - self.total_asset) * self.reward_scaling
        else:
            prev_total = float(self.total_asset) if self.total_asset is not None else float(total_asset)
            if prev_total <= 0:
                log_ret = 0.0
            else:
                log_ret = float(np.log(max(float(total_asset), 1e-12) / prev_total))
            turnover = self._compute_turnover(prev_stocks=prev_stocks, price=price)
            dd_prev = self._compute_drawdown(prev_total)
            # Update peak based on *previous* value first.
            if self._go_peak_total_asset is None:
                self._go_peak_total_asset = prev_total
            self._go_peak_total_asset = max(float(self._go_peak_total_asset), prev_total)
            dd_now = self._compute_drawdown(float(total_asset))
            dd_inc = max(0.0, dd_now - dd_prev)
            reward = self.reward_a * log_ret - self.reward_b_turnover * turnover - self.reward_c_drawdown * dd_inc
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step

        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        # Go-Explore: update archive with promising checkpoints.
        if self.go_explore_enabled:
            self._go_action_history.append(actions_raw)
            last_total_asset = float(self._go_last_total_asset) if self._go_last_total_asset is not None else float(total_asset)
            peak_total_asset = float(self._go_peak_total_asset) if self._go_peak_total_asset is not None else float(total_asset)
            peak_total_asset = max(peak_total_asset, float(total_asset))
            self._go_peak_total_asset = peak_total_asset

            stocks_value = float((self.stocks * price).sum())
            exposure = compute_exposure(amount=float(self.amount), stocks_value=stocks_value, total_asset=float(total_asset))
            cash_ratio = 0.0 if float(total_asset) <= 0 else float(max(self.amount, 0.0) / float(total_asset))
            positions = int(np.sum(self.stocks > 0))
            trend = self._compute_trend(int(self.day))
            drawdown = self._compute_drawdown(float(total_asset))
            turbulence = float(self.turbulence_ary[self.day]) if hasattr(self, "turbulence_ary") else 0.0
            turbulence_flag = float(self.turbulence_bool[self.day]) if hasattr(self, "turbulence_bool") else 0.0
            key = self.go_explore_archive.cell_key(
                day=int(self.day),
                turbulence=turbulence,
                trend=trend,
                drawdown=drawdown,
                cash_ratio=cash_ratio,
                exposure=exposure,
                positions=positions,
            )

            self._go_visited_cells.add(key)

            if default_should_consider(
                day=int(self.day),
                total_asset=float(total_asset),
                last_total_asset=float(last_total_asset),
                peak_total_asset=float(peak_total_asset),
                turbulence_flag=float(turbulence_flag),
            ):
                ckpt = self.get_checkpoint()
                self.go_explore_archive.consider(
                    key=key,
                    checkpoint=ckpt,
                    objective=float(total_asset),
                    drawdown=float(drawdown),
                    trajectory_actions=list(self._go_action_history),
                    step_idx=int(self.day),
                )

            self._go_last_total_asset = float(total_asset)

        return state, reward, done, False, {
            "go_explore_started_from_archive": self._go_started_from_archive,
            "go_explore_archive_size": len(self.go_explore_archive),
        }

    def get_go_explore_stats(self) -> dict:
        return {
            "enabled": bool(self.go_explore_enabled),
            "episode_idx": int(self._go_episode_idx),
            "archive_size": int(len(self.go_explore_archive)),
            "started_from_archive": bool(self._go_started_from_archive),
        }

    def get_go_explore_coverage(self) -> dict:
        """Coverage-style metric: how many distinct regime cells were visited."""
        return {
            "visited_cells": int(len(self._go_visited_cells)),
            "archived_cells": int(len(self.go_explore_archive)),
        }

    def get_state(self, price):
        amount = np.array(self.amount * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)

        return np.hstack(
            (
                amount,
                self.turbulence_ary[self.day],
                self.turbulence_bool[self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day],
            )
        )

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
