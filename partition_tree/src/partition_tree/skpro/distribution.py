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

    def measure(self) -> float:
        return float(self.high - self.low)

    def contains(self, x):
        lower_check = (x >= self.low) if self.lower_closed else (x > self.low)
        upper_check = (x <= self.high) if self.upper_closed else (x < self.high)
        return lower_check & upper_check


class IntervalDistribution(BaseDistribution):
    """
    Piecewise-uniform distributions over disjoint intervals.

    Parameters
    ----------
    intervals : list[Interval] | list[list[Interval]] | list[tuple]
        Either a shared interval list or a list per distribution instance. Tuples
        ``(low, high)`` are accepted and converted to `Interval` objects.
    pdf_values : array-like, optional
        Densities per interval, typically the direct output of
        ``pdf_with_intervals``. Accepts 1D (single instance) or 2D (rows per
        instance). If not provided, ``probability_measures`` must be given.
    probability_measures : array-like, optional
        Probability masses per interval. Used to derive densities when
        ``pdf_values`` is not supplied.
    index : pandas.Index, optional
        Index for the distribution instances. Defaults to a RangeIndex.
    columns : pandas.Index, optional
        Columns for pandas alignment. Defaults to a single column ``0``.
    normalize : bool, default True
        Whether to renormalize masses/densities to sum to 1 per instance.
    """

    _tags = {
        "authors": ["YourGitHubID"],
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "nonparametric",
        "broadcast_init": "off",
    }

    def __init__(self, intervals, pdf_values=None, index=None, columns=None):
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

        # Handle intervals with optional boundary flags
        # Each interval can be (low, high) or (low, high, lower_closed, upper_closed)
        def make_interval(x):
            if len(x) == 2:
                return Interval(x[0], x[1])
            elif len(x) == 4:
                return Interval(x[0], x[1], x[2], x[3])
            else:
                raise ValueError(f"Invalid interval tuple: {x}")

        self._intervals = list(
            map(lambda ints: list(map(make_interval, ints)), intervals)
        )
        # Density times volume = probability mass
        self._mass = []
        self._normalization_factor = []
        for i in range(len(self._intervals)):
            densities = self.pdf_values[i]
            intervals_i = self._intervals[i]
            masses_i = [
                densities[j] * intervals_i[j].measure() for j in range(len(intervals_i))
            ]
            self._mass.append(masses_i)
            # Normalization factor is the total mass (sum of densities × widths)
            self._normalization_factor.append(sum(masses_i))

    # ------------------------------------------------------------------
    # Core distribution methods
    # ------------------------------------------------------------------
    def _pdf(self, x):
        x = np.asarray(x)
        n_instances = len(self.index)
        if x.ndim == 0:
            x = np.full((n_instances, 1), x)
        elif x.ndim == 1:
            x = np.tile(x, (n_instances, 1))
        elif x.shape[0] != n_instances:
            raise ValueError("x must broadcast to (n_instances, n_points)")

        pdf_vals = np.zeros((x.shape[0]), dtype=float)
        for i in range(n_instances):
            densities = self.pdf_values[i]
            intervals = self._intervals[i]
            norm_factor = self._normalization_factor[i]
            for j, interval in enumerate(intervals):

                if interval.contains(x[i]):
                    # Normalize the density by the total mass
                    pdf_vals[i] = densities[j] / norm_factor

        return pd.DataFrame(pdf_vals + 1e-8, index=self.index, columns=self.columns)

    def _match_interval_idx(self, x: np.ndarray):

        idxs = np.empty_like(x, dtype=int)
        for intervals, val in zip(self._intervals, x):
            for idx, interval in enumerate(intervals):
                if interval.contains(val):
                    idxs[idx] = idx
                    break
        return idxs

    def plot(self, ax=None, colors=None, alpha=0.6, **kwargs):
        """Plot the piecewise constant PDF as histogram-like bars.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If ``None``, a new figure and axis are created.
        colors : sequence, optional
            Sequence of colors to cycle through for each instance.
            Defaults to Matplotlib's ``tab10`` cycle.
        alpha : float, optional
            Transparency for the bars. Defaults to ``0.6``.
        **kwargs
            Additional keyword arguments forwarded to ``ax.bar``.
        """
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

            lows = [interval.low for interval in intervals_i]
            widths = [interval.high - interval.low for interval in intervals_i]
            heights = densities_i

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

    def _cdf(self, x):
        x = np.asarray(x)
        n_instances = len(self.index)
        if x.ndim == 0:
            x = np.full((n_instances, 1), x)
        elif x.ndim == 1:
            x = np.tile(x, (n_instances, 1))
        elif x.shape[0] != n_instances:
            raise ValueError("x must broadcast to (n_instances, n_points)")

        cdf_vals = np.zeros((x.shape[0]), dtype=float)
        for i in range(n_instances):
            intervals = self._intervals[i]
            densities = self.pdf_values[i]
            val = x[i]

            cdf_val = 0.0
            for k, interval in enumerate(intervals):
                if val < interval.low:
                    break
                elif interval.contains(val):
                    cdf_val += densities[k] * (val - interval.low)
                    break
                else:
                    cdf_val += densities[k] * interval.measure()
            cdf_vals[i] = cdf_val

        return pd.DataFrame(cdf_vals, index=self.index, columns=self.columns)

    def _mean(self):
        n_instances = len(self.index)
        means = np.zeros((n_instances,), dtype=float)
        for i in range(len(self.index)):
            intervals_i = self._intervals[i]
            densities_i = self.pdf_values[i]
            total_mass = 0
            mean_i = 0.0
            for j, interval in enumerate(intervals_i):
                mean_i += densities_i[j] * (
                    0.5 * (interval.low + interval.high) * interval.measure()
                )
                total_mass += densities_i[j] * interval.measure()

            means[i] = mean_i / total_mass
        return pd.DataFrame(means, index=self.index, columns=self.columns)

    def _energy_x(self, x):
        """`\mathbb{E}[|X-x|]`, where :math:`X` is a copy of self,"""

        n_instances = len(self.index)
        energy = np.zeros((n_instances,), dtype=float)
        for i in range(n_instances):
            intervals = self._intervals[i]
            masses = np.asarray(self._mass[i], dtype=float)
            x_i = x[i]

            total_mass = masses.sum()
            if total_mass <= 0:
                energy[i] = np.nan
                continue

            energy_val = 0.0
            for mass, interval in zip(masses, intervals):
                midpoint = 0.5 * (interval.low + interval.high)
                energy_val += mass * abs(midpoint - x_i)
            energy[i] = energy_val / total_mass

        return energy.reshape(-1, 1)

    def _energy_self(self):
        n_instances = len(self.index)
        out = np.zeros((n_instances, 1), dtype=float)

        for i in range(n_instances):
            intervals = self._intervals[i]
            masses = np.asarray(self._mass[i], dtype=float)

            # sort by interval low (keeps "left-to-right" assumption true)
            order = np.argsort([iv.low for iv in intervals])
            intervals = [intervals[k] for k in order]
            masses = masses[order]

            widths = np.asarray([iv.measure() for iv in intervals], dtype=float)
            mids = np.asarray(
                [0.5 * (iv.low + iv.high) for iv in intervals], dtype=float
            )

            total_mass = masses.sum()
            if total_mass <= 0:
                out[i, 0] = np.nan
                continue

            # Between-interval part: 2 * sum_{i<j} m_i m_j (mid_j - mid_i)
            prefix_m = 0.0
            prefix_mm = 0.0  # sum m_k * mid_k over k<j
            between = 0.0
            for m, mid in zip(masses, mids):
                between += 2.0 * m * (mid * prefix_m - prefix_mm)
                prefix_m += m
                prefix_mm += m * mid

            # Within-interval part: sum m_j^2 * (width_j / 3)
            within = np.sum((masses**2) * (widths / 3.0))

            # Normalize if masses are not probabilities
            out[i, 0] = (between + within) / (total_mass**2)

        return out

    def _var(self):
        mean = self.mean().to_numpy().reshape(-1)
        n_instances = len(self.index)
        variances = np.zeros((n_instances,), dtype=float)
        for i in range(len(self.index)):
            intervals_i = self._intervals[i]
            masses_i = self._mass[i]
            total_mass = 0
            var_i = 0.0
            for j, interval in enumerate(intervals_i):
                midpoint = 0.5 * (interval.low + interval.high)
                var_i += masses_i[j] * (((midpoint - mean[i]) ** 2))
                total_mass += masses_i[j]

            variances[i] = var_i / total_mass

        return pd.DataFrame(variances, index=self.index, columns=self.columns)

    def _subset_params(self, rowidx, colidx, coerce_scalar=False):
        """Subset distribution parameters to given rows and columns.

        Parameters
        ----------
        rowidx : None, numpy index/slice coercible, or int
            Rows to subset to. If None, no subsetting is done.
        colidx : None, numpy index/slice coercible, or int
            Columns to subset to. If None, no subsetting is done.
        coerce_scalar : bool, optional, default=False
            If True, and the subsetted parameter is a scalar, coerce it to a scalar.

        Returns
        -------
        dict
            Dictionary with subsetted distribution parameters.
            Keys are parameter names of ``self``, values are the subsetted parameters.
        """
        params = self._get_dist_params()

        subset_param_dict = {}
        for param, val in params.items():
            if rowidx is None:
                subset_param_dict[param] = val
                continue
            if isinstance(rowidx, int):
                subset_param_dict[param] = val[rowidx]
                continue
            if isinstance(rowidx, pd.Index):
                idxs = rowidx.values
            if isinstance(rowidx, np.ndarray):
                idxs = rowidx
            subset_param_dict[param] = [val[i] for i in idxs]
        return subset_param_dict

    @staticmethod
    def from_mixture(
        distributions: List["IntervalDistribution"],
        weights: List[float] = None,
        index: pd.Index = None,
        columns: pd.Index = None,
    ) -> "IntervalDistribution":
        """Create a single IntervalDistribution by merging multiple distributions.

        This method takes a list of IntervalDistributions (potentially with
        overlapping intervals) and creates a new distribution with piecewise
        disjoint intervals. Where intervals overlap, the PDF value is the
        weighted average of the PDFs from the original distributions.

        Parameters
        ----------
        distributions : list of IntervalDistribution
            List of IntervalDistribution objects to merge. All distributions
            must have the same number of instances (same index length).
        weights : list of float, optional
            Weights for each distribution. If None, uniform weights are used.
            Will be normalized to sum to 1.
        index : pd.Index, optional
            Index for the resulting distribution. If None, uses the index
            from the first distribution.
        columns : pd.Index, optional
            Columns for the resulting distribution. If None, uses the columns
            from the first distribution.

        Returns
        -------
        IntervalDistribution
            A new IntervalDistribution with disjoint intervals and averaged PDFs.

        Examples
        --------
        >>> # Two overlapping interval distributions
        >>> dist1 = IntervalDistribution(
        ...     intervals=[[((0, 2),), ((3, 5),)]],
        ...     pdf_values=[[0.25, 0.25]]
        ... )
        >>> dist2 = IntervalDistribution(
        ...     intervals=[[((1, 4),)]],
        ...     pdf_values=[[0.333]]
        ... )
        >>> merged = IntervalDistribution.from_mixture([dist1, dist2])
        """
        if not distributions:
            raise ValueError("distributions list cannot be empty")

        n_dists = len(distributions)
        n_instances = len(distributions[0].index)

        # Validate all distributions have same number of instances
        for d in distributions:
            if len(d.index) != n_instances:
                raise ValueError(
                    "All distributions must have the same number of instances"
                )

        # Handle weights
        if weights is None:
            weights = np.ones(n_dists) / n_dists
        else:
            weights = np.asarray(weights, dtype=float)
            weights = weights / weights.sum()

        # Use index/columns from first distribution if not provided
        if index is None:
            index = distributions[0].index
        if columns is None:
            columns = distributions[0].columns

        merged_intervals = []
        merged_pdf_values = []

        for instance_idx in range(n_instances):
            # Collect all boundary points from all distributions for this instance
            boundaries = set()
            for d in distributions:
                intervals_i = d._intervals[instance_idx]
                for interval in intervals_i:
                    boundaries.add(interval.low)
                    boundaries.add(interval.high)

            # Sort boundaries to create disjoint intervals
            sorted_boundaries = sorted(boundaries)

            if len(sorted_boundaries) < 2:
                # Edge case: no valid intervals
                merged_intervals.append([])
                merged_pdf_values.append([])
                continue

            # Create disjoint intervals from consecutive boundary points
            instance_intervals = []
            instance_pdfs = []

            for j in range(len(sorted_boundaries) - 1):
                low = sorted_boundaries[j]
                high = sorted_boundaries[j + 1]

                if high <= low:
                    continue

                # Compute midpoint to test which original intervals contain this region
                midpoint = (low + high) / 2.0

                # Compute weighted average PDF at this point
                total_pdf = 0.0
                total_weight = 0.0

                for dist_idx, d in enumerate(distributions):
                    intervals_i = d._intervals[instance_idx]
                    densities_i = d.pdf_values[instance_idx]

                    for k, interval in enumerate(intervals_i):
                        if interval.low <= midpoint <= interval.high:
                            total_pdf += weights[dist_idx] * densities_i[k]
                            total_weight += weights[dist_idx]
                            break

                # Only add interval if it's covered by at least one distribution
                if total_weight > 0:
                    avg_pdf = total_pdf / total_weight
                    instance_intervals.append((low, high))
                    instance_pdfs.append(avg_pdf)

            merged_intervals.append(instance_intervals)
            merged_pdf_values.append(instance_pdfs)

        return IntervalDistribution(
            intervals=merged_intervals,
            pdf_values=merged_pdf_values,
            index=index,
            columns=columns,
        )
