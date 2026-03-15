"""
regime_model.py
===============
Gaussian HMM wrapper for market regime detection.

The model is trained on three features (Returns, Range, Vol_Change).
After fitting, it automatically identifies which hidden state corresponds
to the Bull, Bear, and Neutral market regimes by ranking states on their
mean return in the original (unscaled) feature space.

Public constants
----------------
BULL, BEAR, NEUTRAL   – regime label strings used across the entire system
REGIME_COLORS         – Plotly-compatible RGBA colour map for chart shading

Public class
------------
RegimeModel           – trains and predicts market regimes
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from config import HMMConfig, DEFAULT_CONFIG

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Regime label constants
# ---------------------------------------------------------------------------

BULL    = "Bull"
BEAR    = "Bear"
NEUTRAL = "Neutral"

# Plotly RGBA fill colours for candlestick background shading
REGIME_COLORS: dict[str, str] = {
    BULL:    "rgba(0, 200, 100, 0.15)",
    BEAR:    "rgba(220, 50,  50,  0.18)",
    NEUTRAL: "rgba(180, 180, 180, 0.08)",
}


# ---------------------------------------------------------------------------
# Regime model
# ---------------------------------------------------------------------------

class RegimeModel:
    """
    Wraps a GaussianHMM to expose human-readable regime labels.

    The model is StandardScaler-normalised internally so that the three
    features (Returns, Range, Vol_Change) contribute equally to the EM fit.
    After fitting, states are ranked by their inverse-transformed mean return:

        bull_state  = argmax(mean_return)   → labelled "Bull"
        bear_state  = argmin(mean_return)   → labelled "Bear"
        all others  = labelled "Neutral"

    Typical usage
    -------------
    >>> model = RegimeModel()
    >>> model.fit(X)                          # X: (n, 3) ndarray, raw (unscaled)
    >>> series = model.predict_series(X, idx) # pd.Series of "Bull"/"Bear"/"Neutral"
    >>> summary = model.state_summary()       # per-state statistics DataFrame
    """

    def __init__(self, cfg: HMMConfig = DEFAULT_CONFIG.hmm):
        self.cfg    = cfg
        self.scaler = StandardScaler()

        self._model:    Optional[hmm.GaussianHMM] = None
        self._state_map: dict[int, str]            = {}

        self.bull_state: Optional[int] = None
        self.bear_state: Optional[int] = None
        self.is_fitted:  bool          = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "RegimeModel":
        """
        Fit the HMM on a raw (unscaled) feature matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 3)
            Columns: [Returns, Range, Vol_Change]

        Returns
        -------
        self  (enables chaining: model.fit(X).predict_series(X, idx))
        """
        if X.shape[0] < self.cfg.n_components * 10:
            raise ValueError(
                f"Too few samples ({X.shape[0]}) for {self.cfg.n_components} HMM states. "
                "Reduce n_components or provide more data."
            )

        # Scale features to zero-mean, unit-variance
        X_scaled = self.scaler.fit_transform(X)

        model = hmm.GaussianHMM(
            n_components=self.cfg.n_components,
            covariance_type=self.cfg.covariance_type,
            n_iter=self.cfg.n_iter,
            random_state=self.cfg.random_state,
            verbose=False,
        )
        model.fit(X_scaled)

        self._model = model
        self._identify_states()
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode the most likely state sequence and map to regime labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 3) — raw (unscaled)

        Returns
        -------
        np.ndarray of str, shape (n_samples,)
            Values are one of: BULL, BEAR, NEUTRAL.
        """
        self._require_fit()
        X_scaled   = self.scaler.transform(X)
        raw_states = self._model.predict(X_scaled)
        return np.array([self._state_map[s] for s in raw_states])

    def predict_series(self, X: np.ndarray, index: pd.Index) -> pd.Series:
        """
        Convenience wrapper — returns a named pd.Series aligned to `index`.

        Parameters
        ----------
        X     : np.ndarray, shape (n_samples, 3)
        index : pd.Index of length n_samples (e.g. DatetimeIndex)
        """
        return pd.Series(self.predict(X), index=index, name="regime")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def state_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame with per-state statistics (un-scaled means).

        Columns: State, Label, Mean_Return, Mean_Range, Mean_Vol_Change
        Sorted by Mean_Return descending so Bull is always the top row.
        """
        self._require_fit()

        # Inverse-transform the HMM means back to original feature scale
        means_scaled = self._model.means_                          # (n, 3)
        means_raw    = self.scaler.inverse_transform(means_scaled) # (n, 3)

        rows = []
        for s in range(self.cfg.n_components):
            rows.append({
                "State":           s,
                "Label":           self._state_map[s],
                "Mean_Return":     round(float(means_raw[s, 0]), 6),
                "Mean_Range":      round(float(means_raw[s, 1]), 4),
                "Mean_Vol_Change": round(float(means_raw[s, 2]), 4),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("Mean_Return", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _identify_states(self) -> None:
        """
        Un-scale HMM means and rank states by mean log-return.
        Highest return → Bull, lowest → Bear, rest → Neutral.
        """
        means_raw = self.scaler.inverse_transform(self._model.means_)
        ret_means = means_raw[:, 0]

        self.bull_state = int(np.argmax(ret_means))
        self.bear_state = int(np.argmin(ret_means))

        for s in range(self.cfg.n_components):
            if s == self.bull_state:
                self._state_map[s] = BULL
            elif s == self.bear_state:
                self._state_map[s] = BEAR
            else:
                self._state_map[s] = NEUTRAL

    def _require_fit(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "RegimeModel.fit(X) must be called before predict(). "
            )
