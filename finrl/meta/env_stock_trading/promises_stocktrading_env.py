from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy import random as rd

from finrl.meta.env_stock_trading.go_explore_archive import (
    GoExploreArchive,
    compute_exposure,
    default_should_consider,
)


@dataclass
class EnvSnapshot:
    day: int
    amount: float
    stocks: np.ndarray
    stocks_cool_down: np.ndarray
    total_asset: float
    rng_state: object | None = None


class PromisesStockTradingEnv:
    def __init__(
        self,
        env,
        *,
        go_explore_enabled: bool = False,
        go_explore_warmup_episodes: int = 1,
        go_explore_reset_prob: float = 0.5,
        go_explore_return_mode: str = "snapshot",  # "snapshot" | "replay" (snapshot supported here)
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
    ):
        self.env = env

        self.go_explore_enabled = bool(go_explore_enabled)
        self.go_explore_warmup_episodes = int(go_explore_warmup_episodes)
        self.go_explore_reset_prob = float(go_explore_reset_prob)
        self.go_explore_return_mode = str(go_explore_return_mode)
        if self.go_explore_return_mode not in ("snapshot", "replay"):
            raise ValueError("go_explore_return_mode must be 'snapshot' or 'replay'")

        self._go_episode_idx = 0
        self._go_last_total_asset = None
        self._go_peak_total_asset = None
        self._go_action_history: list[np.ndarray] = []
        self._go_trend_lookback = int(go_explore_trend_lookback)

        self.go_explore_archive = GoExploreArchive(
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

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _compute_trend(self, day: int) -> float:
        if not hasattr(self.env, "price_ary"):
            return 0.0
        lb = self._go_trend_lookback
        if lb <= 0 or day <= 0:
            return 0.0
        start = max(0, int(day) - lb)
        if start == int(day):
            return 0.0
        p0 = self.env.price_ary[start]
        p1 = self.env.price_ary[int(day)]
        denom = np.where(p0 > 0, p0, 1.0)
        r = (p1 / denom) - 1.0
        return float(np.mean(np.clip(r, -1.0, 1.0)))

    def _compute_drawdown(self, total_asset: float) -> float:
        peak = float(self._go_peak_total_asset) if self._go_peak_total_asset is not None else float(total_asset)
        if peak <= 0:
            return 0.0
        return float(max(0.0, 1.0 - float(total_asset) / peak))

    def _snapshot_to_ckpt(self, snap: EnvSnapshot) -> dict:
        return {
            "day": int(snap.day),
            "amount": float(snap.amount),
            "stocks": np.array(snap.stocks, copy=True),
            "stocks_cool_down": np.array(snap.stocks_cool_down, copy=True),
            "total_asset": float(snap.total_asset),
            "rng_state": snap.rng_state,
        }

    def _ckpt_to_snapshot(self, ckpt: dict) -> EnvSnapshot:
        return EnvSnapshot(
            day=int(ckpt["day"]),
            amount=float(ckpt["amount"]),
            stocks=np.array(ckpt["stocks"], copy=True),
            stocks_cool_down=np.array(ckpt["stocks_cool_down"], copy=True),
            total_asset=float(ckpt["total_asset"]),
            rng_state=ckpt.get("rng_state"),
        )

    def save_snapshot(self) -> EnvSnapshot:
        if hasattr(self.env, "get_checkpoint"):
            ckpt = self.env.get_checkpoint()
            return EnvSnapshot(
                day=int(ckpt["day"]),
                amount=float(ckpt["amount"]),
                stocks=np.array(ckpt["stocks"], copy=True),
                stocks_cool_down=np.array(ckpt["stocks_cool_down"], copy=True),
                total_asset=float(ckpt["total_asset"]),
                rng_state=rd.get_state(),
            )

        return EnvSnapshot(
            day=int(self.env.day),
            amount=float(self.env.amount),
            stocks=np.array(self.env.stocks, copy=True),
            stocks_cool_down=np.array(self.env.stocks_cool_down, copy=True),
            total_asset=float(self.env.total_asset),
            rng_state=rd.get_state(),
        )

    def load_snapshot(self, snap: EnvSnapshot, reset_metrics: bool = True):
        if snap.rng_state is not None:
            rd.set_state(snap.rng_state)

        if hasattr(self.env, "set_checkpoint"):
            ckpt = {
                "day": int(snap.day),
                "amount": float(snap.amount),
                "stocks": np.array(snap.stocks, copy=True),
                "stocks_cool_down": np.array(snap.stocks_cool_down, copy=True),
                "total_asset": float(snap.total_asset),
            }
            self.env.set_checkpoint(ckpt, reset_metrics=reset_metrics)
            price = self.env.price_ary[self.env.day]
            obs = self.env.get_state(price)
            return obs

        self.env.day = int(snap.day)
        self.env.amount = float(snap.amount)
        self.env.stocks = np.array(snap.stocks, copy=True).astype(np.float32)
        self.env.stocks_cool_down = np.array(snap.stocks_cool_down, copy=True).astype(
            np.float32
        )
        self.env.total_asset = float(snap.total_asset)

        if reset_metrics:
            self.env.initial_total_asset = float(self.env.total_asset)
            self.env.gamma_reward = 0.0
            self.env.episode_return = 0.0

        price = self.env.price_ary[self.env.day]
        obs = self.env.get_state(price)
        return obs

    def set_next_reset_snapshot(self, snap: EnvSnapshot, reset_metrics: bool = True):
        if hasattr(self.env, "set_next_reset_checkpoint"):
            ckpt = {
                "day": int(snap.day),
                "amount": float(snap.amount),
                "stocks": np.array(snap.stocks, copy=True),
                "stocks_cool_down": np.array(snap.stocks_cool_down, copy=True),
                "total_asset": float(snap.total_asset),
            }
            self.env.set_next_reset_checkpoint(ckpt)
            return

        self._next_snap = (snap, reset_metrics)

    def reset(self, *, seed=None, options=None):
        self._go_episode_idx += 1
        self._go_action_history = []

        if hasattr(self, "_next_snap") and self._next_snap is not None:
            snap, reset_metrics = self._next_snap
            self._next_snap = None
            obs = self.load_snapshot(snap, reset_metrics=reset_metrics)
            self._go_last_total_asset = float(snap.total_asset)
            self._go_peak_total_asset = float(snap.total_asset)
            return obs, {
                "go_explore_started_from_archive": bool(getattr(snap, "_go_explore_from_archive", False)),
                "go_explore_archive_size": len(self.go_explore_archive),
            }

        if (
            self.go_explore_enabled
            and len(self.go_explore_archive) > 0
            and self._go_episode_idx > self.go_explore_warmup_episodes
            and rd.rand() < self.go_explore_reset_prob
        ):
            entry = self.go_explore_archive.sample()
            if entry is not None:
                self.go_explore_archive.mark_visit(entry.key, step_idx=int(self._go_episode_idx))
                snap = self._ckpt_to_snapshot(entry.checkpoint)
                # Tag for info only.
                setattr(snap, "_go_explore_from_archive", True)
                obs = self.load_snapshot(snap, reset_metrics=True)
                self._go_last_total_asset = float(snap.total_asset)
                self._go_peak_total_asset = float(snap.total_asset)
                return obs, {
                    "go_explore_started_from_archive": True,
                    "go_explore_archive_size": len(self.go_explore_archive),
                }

        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        out = self.env.step(action)
        # Support both gym and gymnasium signatures.
        if len(out) == 4:
            obs, reward, done, info = out
            truncated = False
        else:
            obs, reward, done, truncated, info = out

        if self.go_explore_enabled:
            self._go_action_history.append(np.array(action, copy=True))

            # Pull what we can from env.
            day = int(getattr(self.env, "day", 0))
            amount = float(getattr(self.env, "amount", 0.0))
            total_asset = float(getattr(self.env, "total_asset", 0.0))
            stocks = np.array(getattr(self.env, "stocks", []), copy=True)
            price = None
            if hasattr(self.env, "price_ary") and 0 <= day < len(self.env.price_ary):
                price = np.array(self.env.price_ary[day], copy=False)
            stocks_value = float(np.sum(stocks * price)) if price is not None and stocks.size else 0.0
            exposure = compute_exposure(amount=amount, stocks_value=stocks_value, total_asset=total_asset)
            cash_ratio = 0.0 if total_asset <= 0 else float(max(amount, 0.0) / total_asset)
            positions = int(np.sum(stocks > 0)) if stocks.size else 0
            trend = self._compute_trend(day)

            last_total_asset = float(self._go_last_total_asset) if self._go_last_total_asset is not None else total_asset
            peak_total_asset = float(self._go_peak_total_asset) if self._go_peak_total_asset is not None else total_asset
            peak_total_asset = max(peak_total_asset, total_asset)
            self._go_peak_total_asset = peak_total_asset
            drawdown = self._compute_drawdown(total_asset)

            turbulence = 0.0
            turbulence_flag = 0.0
            if hasattr(self.env, "turbulence_ary"):
                try:
                    turbulence = float(self.env.turbulence_ary[day])
                except Exception:
                    turbulence = 0.0
            if hasattr(self.env, "turbulence_bool"):
                try:
                    turbulence_flag = float(self.env.turbulence_bool[day])
                except Exception:
                    turbulence_flag = 0.0

            key = self.go_explore_archive.cell_key(
                day=day,
                turbulence=turbulence,
                trend=trend,
                drawdown=drawdown,
                cash_ratio=cash_ratio,
                exposure=exposure,
                positions=positions,
            )

            if default_should_consider(
                day=day,
                total_asset=total_asset,
                last_total_asset=last_total_asset,
                peak_total_asset=peak_total_asset,
                turbulence_flag=turbulence_flag,
            ):
                snap = self.save_snapshot()
                ckpt = self._snapshot_to_ckpt(snap)
                self.go_explore_archive.consider(
                    key=key,
                    checkpoint=ckpt,
                    objective=total_asset,
                    drawdown=drawdown,
                    trajectory_actions=list(self._go_action_history),
                    step_idx=day,
                )

            self._go_last_total_asset = total_asset

        if len(out) == 4:
            return obs, reward, done, info
        return obs, reward, done, truncated, info
