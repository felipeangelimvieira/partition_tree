"""
Interval-based probability distribution that consumes densities produced by
the Rust `PiecewiseConstantDistribution::pdf_with_intervals` helper. It supports
heterogeneous interval partitions per instance, keeps densities explicitly, and
derives masses and cumulative probabilities for CDF/PPF/sampling.
"""

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from skpro.distributions.base import BaseDistribution
from typing import List

import numpy as np
import pandas as pd
from skpro.distributions.base import BaseDistribution


def _coerce_1d_per_instance(values, n_instances: int, name: str) -> np.ndarray:
    """Coerce scalar / vector / (n,1) inputs to shape (n_instances,)."""
    arr = np.asarray(values, dtype=float)

    if arr.ndim == 0:
        return np.full(n_instances, float(arr), dtype=float)

    if arr.ndim == 1:
        if arr.shape[0] == n_instances:
            return arr.astype(float)
        if arr.shape[0] == 1:
            return np.full(n_instances, float(arr[0]), dtype=float)

    if arr.ndim == 2:
        if arr.shape == (n_instances, 1):
            return arr[:, 0].astype(float)
        if arr.shape == (1, n_instances):
            return arr[0, :].astype(float)

    raise ValueError(
        f"{name} must broadcast to shape ({n_instances},) or ({n_instances}, 1)"
    )


def _truncated_interval_length(low: float, high: float, x: float) -> float:
    """Length of [low, high] intersected with (-inf, x]."""
    if x <= low:
        return 0.0
    if x >= high:
        return high - low
    return x - low


def _abs_interval_integral(a: float, b: float, x0: float) -> float:
    """Return :math:`\int_a^b |u - x_0|\,du`.

    This is the scalar building block for point-energy computations,
    :math:`\mathbb{E}|X - x_0| = Z^{-1}\sum_j f_j \int_{a_j}^{b_j}|u - x_0|\,du`.
    """
    if b <= a:
        return 0.0
    if x0 <= a:
        return 0.5 * ((b - x0) ** 2 - (a - x0) ** 2)
    if x0 >= b:
        return 0.5 * ((x0 - a) ** 2 - (x0 - b) ** 2)
    return 0.5 * ((x0 - a) ** 2 + (b - x0) ** 2)


def _uniform_cross_energy(a: float, b: float, c: float, d: float) -> float:
    """Return :math:`\mathbb{E}|U - V|` for two uniform intervals.

    Here :math:`U \sim \mathrm{Unif}(a, b)` and
    :math:`V \sim \mathrm{Unif}(c, d)` are independent. This is the pairwise
    interval term used in
    :math:`\mathbb{E}|X - X'| = \sum_{i,j} p_i p_j\,\mathbb{E}|U_i - V_j|`.

    The implementation evaluates the exact double integral
    :math:`((b-a)(d-c))^{-1} \int_a^b \int_c^d |u-v|\,dv\,du`
    through an antiderivative of :math:`\int_c^d |x-y|\,dy`.
    """
    wa = b - a
    wb = d - c
    if wa <= 0 or wb <= 0:
        return 0.0

    def _H(x: float) -> float:
        # Antiderivative of G(x) = int_c^d |x - y| dy.
        return (abs(x - c) ** 3 - abs(x - d) ** 3) / 6.0

    return (_H(b) - _H(a)) / (wa * wb)


def _intervals_are_sorted_and_disjoint(intervals) -> bool:
    """Return whether intervals are ordered by `low` and pairwise non-overlapping."""
    prev_high = float("-inf")
    for interval in intervals:
        if interval.low < prev_high:
            return False
        prev_high = interval.high
    return True


def _has_overlapping_intervals(intervals) -> bool:
    """Return whether any pair of intervals has a positive-length overlap."""
    if len(intervals) < 2:
        return False

    sorted_intervals = sorted(
        intervals, key=lambda interval: (interval.low, interval.high)
    )
    prev_high = sorted_intervals[0].high

    for interval in sorted_intervals[1:]:
        if interval.low < prev_high:
            return True
        prev_high = max(prev_high, interval.high)

    return False


def _energy_self_general(intervals, masses: np.ndarray) -> float:
    """Exact self-energy without structural assumptions.

    For normalized interval masses :math:`p_j = m_j / \sum_k m_k`, this
    computes
    :math:`\mathbb{E}|X - X'| = \sum_{i,j} p_i p_j\,\mathbb{E}|U_i - V_j|`,
    where :math:`U_i` and :math:`V_j` are independent uniform draws on the
    corresponding intervals.

    No shortcut is used here: every interval pair is evaluated explicitly via
    :func:`_uniform_cross_energy`, so this is the exact :math:`O(k^2)` fallback
    for overlapping or out-of-order intervals.
    """
    total_mass = masses.sum()
    if total_mass <= 0:
        return np.nan

    probs = masses / total_mass
    val = 0.0
    for ia, iv_a in enumerate(intervals):
        pa = probs[ia]
        if pa <= 0:
            continue

        for ib, iv_b in enumerate(intervals):
            pb = probs[ib]
            if pb <= 0:
                continue

            val += (
                pa
                * pb
                * _uniform_cross_energy(iv_a.low, iv_a.high, iv_b.low, iv_b.high)
            )

    return val


def _energy_self_sorted_disjoint(intervals, masses: np.ndarray) -> float:
    """Exact self-energy in :math:`O(k)` for sorted, disjoint intervals.

    Let :math:`p_j` be the normalized mass and
    :math:`\mu_j = (a_j + b_j)/2` the mean of interval :math:`j`.
    The diagonal term is
    :math:`\mathbb{E}|U_j - U'_j| = (b_j - a_j)/3`.
    For :math:`\ell < j`, disjointness implies :math:`U_j \ge U_\ell`, hence
    :math:`\mathbb{E}|U_j - U_\ell| = \mu_j - \mu_\ell`.

    The shortcut rewrites the full double sum as
    :math:`\sum_j p_j^2 (b_j-a_j)/3 + 2\sum_j p_j(\mu_j C_{j-1} - M_{j-1})`,
    where :math:`C_{j-1} = \sum_{\ell<j} p_\ell` and
    :math:`M_{j-1} = \sum_{\ell<j} p_\ell\mu_\ell`.
    Running cumulative sums of :math:`C` and :math:`M` make the computation
    linear.
    """
    total_mass = masses.sum()
    if total_mass <= 0:
        return np.nan

    probs = masses / total_mass
    val = 0.0
    cumulative_prob = 0.0
    cumulative_prob_mean = 0.0

    for prob, interval in zip(probs, intervals):
        if prob <= 0:
            continue

        mean = 0.5 * (interval.low + interval.high)
        val += prob * prob * interval.measure() / 3.0
        val += 2.0 * prob * (mean * cumulative_prob - cumulative_prob_mean)

        cumulative_prob += prob
        cumulative_prob_mean += prob * mean

    return val


def _build_energy_x_sorted_disjoint_cache(intervals, masses: np.ndarray):
    """Precompute prefix masses and first moments for fast point-energy queries.

    With interval masses :math:`m_j = f_j (b_j-a_j)` and means
    :math:`\mu_j = (a_j+b_j)/2`, the cache stores prefix sums
    :math:`S_j = \sum_{\ell \le j} m_\ell` and
    :math:`T_j = \sum_{\ell \le j} m_\ell\mu_\ell`.
    :func:`_energy_x_sorted_disjoint` uses these arrays together with a binary
    search on interval right endpoints to collapse left and right blocks into
    constant-time prefix-sum differences.
    """
    if masses.sum() <= 0:
        return None

    highs = np.asarray([interval.high for interval in intervals], dtype=float)
    first_moments = masses * np.asarray(
        [0.5 * (interval.low + interval.high) for interval in intervals],
        dtype=float,
    )
    prefix_masses = np.concatenate(([0.0], np.cumsum(masses)))
    prefix_first_moments = np.concatenate(([0.0], np.cumsum(first_moments)))

    return highs, prefix_masses, prefix_first_moments


def _energy_x_general(intervals, densities, norm: float, x0: float) -> float:
    """Exact point-energy without assuming any interval structure.

    This evaluates
    :math:`\mathbb{E}|X - x_0| = Z^{-1}\sum_j f_j \int_{a_j}^{b_j}|u-x_0|\,du`
    with :math:`Z = \sum_j f_j(b_j-a_j)`.

    No shortcut is used here: every interval contributes its exact
    absolute-value integral, so this is the fallback for overlapping or
    out-of-order intervals.
    """
    val = 0.0
    for density, interval in zip(densities, intervals):
        val += float(density) * _abs_interval_integral(interval.low, interval.high, x0)
    return val / norm


def _energy_x_sorted_disjoint(intervals, densities, norm: float, x0: float, cache):
    """Exact point-energy in :math:`O(\log k)` for sorted, disjoint intervals.

    If interval :math:`j` lies completely left of :math:`x_0`, then
    :math:`\int_{a_j}^{b_j}|u-x_0| f_j\,du = x_0 m_j - m_j\mu_j`.
    If it lies completely right of :math:`x_0`, the contribution is
    :math:`m_j\mu_j - x_0 m_j`.
    At most one interval can contain :math:`x_0`; that interval still uses the
    exact integral :math:`f_j\int_{a_j}^{b_j}|u-x_0|\,du`.

    The shortcut binary-searches the interval right endpoints to identify the
    split point and then evaluates the left and right blocks from cached prefix
    sums of :math:`m_j` and :math:`m_j\mu_j`.
    """
    highs, prefix_masses, prefix_first_moments = cache

    left_count = np.searchsorted(highs, x0, side="right")
    left_mass = prefix_masses[left_count]
    left_first_moment = prefix_first_moments[left_count]
    val = x0 * left_mass - left_first_moment

    right_start = left_count
    if left_count < len(intervals) and intervals[left_count].contains(x0):
        interval = intervals[left_count]
        val += float(densities[left_count]) * _abs_interval_integral(
            interval.low, interval.high, x0
        )
        right_start += 1

    right_mass = prefix_masses[-1] - prefix_masses[right_start]
    right_first_moment = prefix_first_moments[-1] - prefix_first_moments[right_start]
    val += right_first_moment - x0 * right_mass

    return val / norm


class Interval:
    """Interval helper with configurable open/closed boundaries."""

    def __init__(
        self,
        low: float,
        high: float,
        lower_closed: bool = True,
        upper_closed: bool = True,
    ):
        self.low = float(low)
        self.high = float(high)
        self.lower_closed = lower_closed
        self.upper_closed = upper_closed

        if self.high < self.low:
            raise ValueError("Interval high must be >= low")

    def measure(self) -> float:
        return float(self.high - self.low)

    def contains(self, x):
        lower_check = (x >= self.low) if self.lower_closed else (x > self.low)
        upper_check = (x <= self.high) if self.upper_closed else (x < self.high)
        return lower_check & upper_check


class IntervalDistribution(BaseDistribution):
    """
    Piecewise-uniform distribution over intervals.

    Parameters
    ----------
    intervals : list[list[tuple]]
        One list of interval-tuples per instance. Each tuple is either
        (low, high) or (low, high, lower_closed, upper_closed).
    pdf_values : list[list[float]]
        Raw densities per interval, one list per instance. They are normalized
        internally using total mass = sum(density * interval_width).
    index : pandas.Index, optional
        Row index for instances.
    columns : pandas.Index, optional
        Column index for pandas alignment.
    """

    _tags = {
        "authors": ["felipeangelimvieira"],
        "capabilities:approx": [],
        "capabilities:exact": [
            "mean",
            "var",
            "energy",
            "pdf",
            "log_pdf",
            "cdf",
            "ppf",
        ],
        "distr:measuretype": "continuous",
        "distr:paramtype": "nonparametric",
        "broadcast_init": "off",
    }

    def __init__(self, intervals, pdf_values, index=None, columns=None):

        self.intervals = intervals
        self.pdf_values = pdf_values

        n_instances = len(intervals)
        if index is None:
            index = pd.RangeIndex(start=0, stop=n_instances, step=1)
        if columns is None:
            columns = pd.Index([0])

        self.index = index
        self.columns = columns

        super().__init__(index=self.index, columns=self.columns)

        def make_interval(x):
            if len(x) == 2:
                return Interval(x[0], x[1])
            if len(x) == 4:
                return Interval(x[0], x[1], x[2], x[3])
            raise ValueError(f"Invalid interval tuple: {x}")

        self._intervals = [list(map(make_interval, ints)) for ints in intervals]
        self._intervals_are_sorted_and_disjoint = [
            _intervals_are_sorted_and_disjoint(intervals_i)
            for intervals_i in self._intervals
        ]

        if len(self._intervals) != len(self.pdf_values):
            raise ValueError("intervals and pdf_values must have the same outer length")

        self._mass = []
        self._normalization_factor = []
        self._energy_x_sorted_disjoint_cache = []

        for i in range(len(self._intervals)):
            intervals_i = self._intervals[i]
            densities_i = self.pdf_values[i]

            if len(intervals_i) != len(densities_i):
                raise ValueError(
                    f"Instance {i}: number of intervals and pdf_values must match"
                )

            if _has_overlapping_intervals(intervals_i):
                raise ValueError(f"Instance {i}: intervals must not overlap")

            masses_i = [
                float(densities_i[j]) * intervals_i[j].measure()
                for j in range(len(intervals_i))
            ]
            self._mass.append(masses_i)
            norm_i = float(sum(masses_i))
            self._normalization_factor.append(norm_i)

            if self._intervals_are_sorted_and_disjoint[i] and norm_i > 0:
                self._energy_x_sorted_disjoint_cache.append(
                    _build_energy_x_sorted_disjoint_cache(
                        intervals_i, np.asarray(masses_i, dtype=float)
                    )
                )
            else:
                self._energy_x_sorted_disjoint_cache.append(None)

    def _pdf(self, x):
        x = _coerce_1d_per_instance(x, len(self.index), "x")
        n_instances = len(self.index)

        pdf_vals = np.zeros(n_instances, dtype=float)
        for i in range(n_instances):
            norm = self._normalization_factor[i]
            if norm <= 0:
                pdf_vals[i] = 0.0
                continue

            for density, interval in zip(self.pdf_values[i], self._intervals[i]):
                if interval.contains(x[i]):
                    pdf_vals[i] = float(density) / norm
                    break

        return pd.DataFrame(pdf_vals, index=self.index, columns=self.columns)

    def _log_pdf(self, x):
        pdf = self._pdf(x).to_numpy()
        with np.errstate(divide="ignore"):
            out = np.log(pdf)
        return pd.DataFrame(out, index=self.index, columns=self.columns)

    def _cdf(self, x):
        x = _coerce_1d_per_instance(x, len(self.index), "x")
        n_instances = len(self.index)

        cdf_vals = np.zeros(n_instances, dtype=float)
        for i in range(n_instances):
            norm = self._normalization_factor[i]
            if norm <= 0:
                cdf_vals[i] = 0.0
                continue

            cdf_val = 0.0
            for density, interval in zip(self.pdf_values[i], self._intervals[i]):
                cdf_val += float(density) * _truncated_interval_length(
                    interval.low, interval.high, x[i]
                )
            cdf_vals[i] = cdf_val / norm

        return pd.DataFrame(cdf_vals, index=self.index, columns=self.columns)

    def _ppf(self, p):
        """Percent point function (inverse CDF) for piecewise uniform distribution."""
        p = _coerce_1d_per_instance(p, len(self.index), "p")
        n_instances = len(self.index)

        ppf_vals = np.zeros(n_instances, dtype=float)
        for i in range(n_instances):
            intervals = self._intervals[i]
            densities = self.pdf_values[i]
            norm = self._normalization_factor[i]

            if not intervals or norm <= 0:
                ppf_vals[i] = np.nan
                continue

            q = float(np.clip(p[i], 0.0, 1.0))

            if q <= 0.0:
                ppf_vals[i] = intervals[0].low
                continue
            if q >= 1.0:
                ppf_vals[i] = intervals[-1].high
                continue

            cumulative = 0.0
            found = False
            for density, interval in zip(densities, intervals):
                mass = float(density) * interval.measure() / norm
                if cumulative + mass >= q - 1e-12:
                    remaining = q - cumulative
                    if density > 0:
                        offset = remaining * norm / float(density)
                        ppf_vals[i] = interval.low + offset
                    else:
                        ppf_vals[i] = interval.low
                    found = True
                    break
                cumulative += mass

            if not found:
                ppf_vals[i] = intervals[-1].high

        return pd.DataFrame(ppf_vals, index=self.index, columns=self.columns)

    def _mean(self):
        """
        Exact mean for piecewise uniform distribution.

        Computes :math:`1/Z \sum_j \int_{a_j}^{b_j}x f_j dx` where
        :math:`Z = \\sum_j f_j (b_j - a_j)`.
        """
        n_instances = len(self.index)
        means = np.zeros(n_instances, dtype=float)

        for i in range(n_instances):
            norm = self._normalization_factor[i]
            if norm <= 0:
                means[i] = np.nan
                continue

            num = 0.0
            for density, interval in zip(self.pdf_values[i], self._intervals[i]):
                a, b = interval.low, interval.high
                num += float(density) * 0.5 * (b**2 - a**2)

            means[i] = num / norm

        return pd.DataFrame(means, index=self.index, columns=self.columns)

    def _var(self):
        """
        Exact variance for piecewise uniform distribution.

        Computes :math:`Var[X] = E[X^2] - E[X]^2` where
        :math:`E[X^2] = 1/Z \\sum_j \\int_{a_j}^{b_j} x^2 f_j dx`.
        """
        n_instances = len(self.index)
        variances = np.zeros(n_instances, dtype=float)
        means = self._mean().to_numpy().reshape(-1)

        for i in range(n_instances):
            norm = self._normalization_factor[i]
            if norm <= 0:
                variances[i] = np.nan
                continue

            second_moment = 0.0
            for density, interval in zip(self.pdf_values[i], self._intervals[i]):
                a, b = interval.low, interval.high
                second_moment += float(density) * (b**3 - a**3) / 3.0
            second_moment /= norm

            variances[i] = max(second_moment - means[i] ** 2, 0.0)

        return pd.DataFrame(variances, index=self.index, columns=self.columns)

    def _energy_x(self, x):
        """Exact point-energy :math:`\mathbb{E}|X - x|` for each instance.

        For a piecewise-uniform law with interval densities :math:`f_j` on
        :math:`[a_j, b_j]`, this computes
        :math:`\mathbb{E}|X-x| = Z^{-1}\sum_j f_j\int_{a_j}^{b_j}|u-x|\,du`,
        where :math:`Z = \sum_j f_j(b_j-a_j)`.

        When an instance's intervals are sorted and pairwise disjoint, the
        method dispatches to the cached prefix-sum shortcut in
        :func:`_energy_x_sorted_disjoint`; otherwise it falls back to the
        general exact integral sum in :func:`_energy_x_general`.
        """
        x = _coerce_1d_per_instance(x, len(self.index), "x")
        n_instances = len(self.index)

        energy = np.zeros(n_instances, dtype=float)
        for i in range(n_instances):
            norm = self._normalization_factor[i]
            if norm <= 0:
                energy[i] = np.nan
                continue

            intervals_i = self._intervals[i]
            densities_i = self.pdf_values[i]
            cache_i = self._energy_x_sorted_disjoint_cache[i]

            if self._intervals_are_sorted_and_disjoint[i] and cache_i is not None:
                energy[i] = _energy_x_sorted_disjoint(
                    intervals_i, densities_i, norm, x[i], cache_i
                )
            else:
                energy[i] = _energy_x_general(intervals_i, densities_i, norm, x[i])

        return energy.reshape(-1, 1)

    def _energy_self(self):
        """Exact self-energy :math:`\mathbb{E}|X - X'|` for each instance.

        For interval masses :math:`p_j`, the general identity is
        :math:`\mathbb{E}|X-X'| = \sum_{i,j} p_i p_j\,\mathbb{E}|U_i - V_j|`,
        where :math:`U_i` and :math:`V_j` are uniform draws on the respective
        intervals.

        When an instance's intervals are sorted and pairwise disjoint, this
        uses the linear-time cumulative-sum shortcut in
        :func:`_energy_self_sorted_disjoint`; otherwise it evaluates the full
        pairwise exact computation in :func:`_energy_self_general`.
        """
        n_instances = len(self.index)
        out = np.zeros((n_instances, 1), dtype=float)

        for i in range(n_instances):
            masses = np.asarray(self._mass[i], dtype=float)
            intervals = self._intervals[i]

            if self._intervals_are_sorted_and_disjoint[i]:
                out[i, 0] = _energy_self_sorted_disjoint(intervals, masses)
            else:
                out[i, 0] = _energy_self_general(intervals, masses)

        return out

    def _match_interval_idx(self, x):
        x = _coerce_1d_per_instance(x, len(self.index), "x")
        idxs = np.full(len(self.index), -1, dtype=int)

        for pos, (intervals, val) in enumerate(zip(self._intervals, x)):
            for idx, interval in enumerate(intervals):
                if interval.contains(val):
                    idxs[pos] = idx
                    break

        return idxs

    def plot(self, ax=None, colors=None, alpha=0.6, **kwargs):
        """Plot the piecewise constant PDF as histogram-like bars."""
        import matplotlib.pyplot as plt
        from itertools import cycle

        if ax is None:
            _, ax = plt.subplots()

        if colors is None:
            colors = plt.cm.get_cmap("tab10").colors
        color_cycle = cycle(colors)

        for i in range(len(self.index)):
            intervals_i = self._intervals[i]
            densities_i = self.pdf_values[i]
            norm_i = self._normalization_factor[i]

            lows = [interval.low for interval in intervals_i]
            widths = [interval.high - interval.low for interval in intervals_i]
            heights = [
                (float(densities_i[j]) / norm_i) if norm_i > 0 else 0.0
                for j in range(len(intervals_i))
            ]

            ax.bar(
                lows,
                heights,
                width=widths,
                align="edge",
                color=next(color_cycle),
                alpha=alpha,
                edgecolor="black",
                label=f"Instance {self.index[i]}",
                **kwargs,
            )

        ax.set_xlabel("x")
        ax.set_ylabel("PDF")
        ax.legend()
        return ax

    def _subset_params(self, rowidx, colidx, coerce_scalar=False):
        """Subset distribution parameters to given rows and columns."""
        params = self._get_dist_params()

        if rowidx is None:
            return params

        if isinstance(rowidx, int):
            idxs = [rowidx]
        elif isinstance(rowidx, pd.Index):
            idxs = list(rowidx.values)
        elif isinstance(rowidx, np.ndarray):
            idxs = list(rowidx)
        else:
            idxs = list(rowidx)

        subset_param_dict = {}
        for param, val in params.items():
            subset_param_dict[param] = [val[i] for i in idxs]

        return subset_param_dict

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "intervals": [
                [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
                [(0.0, 1.5), (1.5, 3.0)],
            ],
            "pdf_values": [
                [0.2, 0.5, 0.3],
                [0.6, 0.4],
            ],
            "index": pd.RangeIndex(2),
            "columns": pd.Index([0]),
        }
        params2 = {
            "intervals": [
                [(0.0, 2.0), (2.0, 5.0)],
            ],
            "pdf_values": [
                [0.4, 0.2],
            ],
            "index": pd.RangeIndex(1),
            "columns": pd.Index([0]),
        }
        return [params1, params2]
