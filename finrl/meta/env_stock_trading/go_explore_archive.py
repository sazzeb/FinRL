from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GoExploreCellKey:
    time_bucket: int
    turbulence_bucket: int
    trend_bucket: int
    drawdown_bucket: int
    cash_bucket: int
    exposure_bucket: int
    positions_bucket: int


@dataclass
class GoExploreArchiveEntry:
    key: GoExploreCellKey
    checkpoint: dict[str, Any]
    trajectory_actions: list[np.ndarray] | None
    best_objective: float
    visits: int = 0
    last_visit_step: int = 0

    outcome_n: int = 0
    outcome_mean: float = 0.0
    outcome_m2: float = 0.0
    max_drawdown_seen: float = 0.0


class GoExploreArchive:
    """A lightweight Go-Explore style archive.

    This is intentionally minimal: it stores one best checkpoint per discretized cell
    and provides a biased sampler that prefers less-visited cells.
    """

    def __init__(
        self,
        *,
        max_size: int = 2048,
        time_bucket_size: int = 1,
        turbulence_bucket_width: float = 0.25,
        trend_bucket_width: float = 0.01,
        drawdown_bucket_width: float = 0.02,
        cash_bucket_width: float = 0.05,
        exposure_bucket_width: float = 0.05,
        positions_bucket_width: int = 1,
        novelty_coeff: float = 1.0,
        uncertainty_coeff: float = 0.25,
        risk_coeff: float = 1.0,
        sample_beta: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        if turbulence_bucket_width <= 0:
            raise ValueError("turbulence_bucket_width must be > 0")
        if trend_bucket_width <= 0:
            raise ValueError("trend_bucket_width must be > 0")
        if drawdown_bucket_width <= 0:
            raise ValueError("drawdown_bucket_width must be > 0")
        if cash_bucket_width <= 0:
            raise ValueError("cash_bucket_width must be > 0")
        if exposure_bucket_width <= 0:
            raise ValueError("exposure_bucket_width must be > 0")
        if positions_bucket_width <= 0:
            raise ValueError("positions_bucket_width must be > 0")
        if time_bucket_size <= 0:
            raise ValueError("time_bucket_size must be > 0")

        self.max_size = int(max_size)
        self.time_bucket_size = int(time_bucket_size)
        self.turbulence_bucket_width = float(turbulence_bucket_width)
        self.trend_bucket_width = float(trend_bucket_width)
        self.drawdown_bucket_width = float(drawdown_bucket_width)
        self.cash_bucket_width = float(cash_bucket_width)
        self.exposure_bucket_width = float(exposure_bucket_width)
        self.positions_bucket_width = int(positions_bucket_width)

        self.novelty_coeff = float(novelty_coeff)
        self.uncertainty_coeff = float(uncertainty_coeff)
        self.risk_coeff = float(risk_coeff)
        self.sample_beta = float(sample_beta)
        self._rng = rng if rng is not None else np.random.default_rng()

        self._entries: dict[GoExploreCellKey, GoExploreArchiveEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> list[GoExploreCellKey]:
        return list(self._entries.keys())

    def entries(self) -> list[GoExploreArchiveEntry]:
        return list(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    def cell_key(
        self,
        *,
        day: int,
        turbulence: float,
        trend: float,
        drawdown: float,
        cash_ratio: float,
        exposure: float,
        positions: int,
    ) -> GoExploreCellKey:
        time_bucket = int(int(day) // self.time_bucket_size)
        t_bucket = int(np.floor(float(turbulence) / self.turbulence_bucket_width))
        tr_bucket = int(np.floor(float(trend) / self.trend_bucket_width))
        dd_bucket = int(np.floor(float(max(drawdown, 0.0)) / self.drawdown_bucket_width))
        cash_bucket = int(np.floor(float(np.clip(cash_ratio, 0.0, 1.0)) / self.cash_bucket_width))
        e_bucket = int(np.floor(float(np.clip(exposure, 0.0, 1.0)) / self.exposure_bucket_width))
        pos_bucket = int(int(max(positions, 0)) // self.positions_bucket_width)
        return GoExploreCellKey(
            time_bucket=time_bucket,
            turbulence_bucket=t_bucket,
            trend_bucket=tr_bucket,
            drawdown_bucket=dd_bucket,
            cash_bucket=cash_bucket,
            exposure_bucket=e_bucket,
            positions_bucket=pos_bucket,
        )

    def consider(
        self,
        *,
        key: GoExploreCellKey,
        checkpoint: dict[str, Any],
        objective: float,
        drawdown: float,
        trajectory_actions: list[np.ndarray] | None = None,
        step_idx: int = 0,
    ) -> bool:
        """Insert/update the archive for this cell.

        Update rule (Go-Explore-inspired): keep the entry with the best objective.
        Promise (objective + novelty + uncertainty - risk) is used for sampling.
        """
        objective = float(objective)
        drawdown = float(max(drawdown, 0.0))
        existing = self._entries.get(key)
        if existing is None:
            if len(self._entries) >= self.max_size:
                self._evict_one()
            self._entries[key] = GoExploreArchiveEntry(
                key=key,
                checkpoint=checkpoint,
                trajectory_actions=trajectory_actions,
                best_objective=objective,
                visits=0,
                last_visit_step=int(step_idx),
                outcome_n=1,
                outcome_mean=objective,
                outcome_m2=0.0,
                max_drawdown_seen=drawdown,
            )
            return True

        # Update running outcome stats (for uncertainty bonus).
        self._update_outcome_stats(existing, outcome=objective, drawdown=drawdown)

        if objective > existing.best_objective:
            existing.checkpoint = checkpoint
            existing.best_objective = objective
            if trajectory_actions is not None:
                existing.trajectory_actions = trajectory_actions
            return True

        return False

    def mark_visit(self, key: GoExploreCellKey, *, step_idx: int = 0) -> None:
        entry = self._entries.get(key)
        if entry is not None:
            entry.visits += 1
            entry.last_visit_step = int(step_idx)

    def promise(self, entry: GoExploreArchiveEntry) -> float:
        novelty_bonus = self.novelty_coeff / (1.0 + np.sqrt(float(entry.visits)))

        if entry.outcome_n >= 2:
            var = entry.outcome_m2 / float(entry.outcome_n - 1)
            std = float(np.sqrt(max(var, 0.0)))
        else:
            std = 0.0
        uncertainty_bonus = self.uncertainty_coeff * std

        risk_penalty = self.risk_coeff * float(max(entry.max_drawdown_seen, 0.0))
        return float(entry.best_objective + novelty_bonus + uncertainty_bonus - risk_penalty)

    def sample(self) -> GoExploreArchiveEntry | None:
        if not self._entries:
            return None

        entries = list(self._entries.values())
        promises = np.array([self.promise(e) for e in entries], dtype=np.float64)
        # Stabilize exponentials.
        p0 = promises - promises.max()
        weights = np.exp(self.sample_beta * p0)
        weights = weights / weights.sum()
        idx = int(self._rng.choice(len(entries), p=weights))
        return entries[idx]

    def _evict_one(self) -> None:
        """Evict a cell when capacity is exceeded.

        Strategy: drop the most-visited among the bottom-score quartile.
        This keeps a bias toward novelty while not hoarding poor cells.
        """
        entries = list(self._entries.values())
        if not entries:
            return

        objectives = np.array([e.best_objective for e in entries], dtype=np.float64)
        cutoff = np.quantile(objectives, 0.25)
        candidates = [e for e in entries if e.best_objective <= cutoff]
        if not candidates:
            candidates = entries

        worst = max(candidates, key=lambda e: e.visits)
        self._entries.pop(worst.key, None)

    @staticmethod
    def _update_outcome_stats(entry: GoExploreArchiveEntry, *, outcome: float, drawdown: float) -> None:
        # Welford online variance.
        entry.outcome_n += 1
        delta = float(outcome) - entry.outcome_mean
        entry.outcome_mean += delta / float(entry.outcome_n)
        delta2 = float(outcome) - entry.outcome_mean
        entry.outcome_m2 += delta * delta2
        entry.max_drawdown_seen = float(max(entry.max_drawdown_seen, drawdown))


def default_should_consider(
    *,
    day: int,
    total_asset: float,
    last_total_asset: float,
    peak_total_asset: float,
    turbulence_flag: float,
) -> bool:
    """Heuristic for when to snapshot a state.

    - New peak (good performance)
    - Turbulence/crash region (hard regime)
    - Large move (transition)
    """
    if total_asset >= peak_total_asset:
        return True

    if turbulence_flag >= 0.5:
        return True

    if last_total_asset > 0 and abs(total_asset - last_total_asset) / last_total_asset >= 0.02:
        return True

    return False


def compute_exposure(*, amount: float, stocks_value: float, total_asset: float) -> float:
    if total_asset <= 0:
        return 0.0
    return float(stocks_value / total_asset)
