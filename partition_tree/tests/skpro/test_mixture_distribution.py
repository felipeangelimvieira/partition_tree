"""Tests for MixtureIntervalDistribution.

Test plan
---------
1. **PDF integrates to ~1**  – numerical quadrature over the support.
2. **CDF is monotone, bounded in [0, 1]**  – evaluated at many x values.
3. **Mean = weighted sum of component means**  – mixture identity.
4. **Variance = mixture identity**  – Σ w_m(σ_m² + μ_m²) − μ_mix².
5. **PPF is the inverse of CDF**  – F(ppf(p)) ≈ p for many p values.
6. **energy_x = weighted sum of component energy_x values**.
7. **energy_self ≥ 0**  – energetic distance is non-negative.
8. **Sampling: empirical mean converges to the analytical mean**.
9. **Single component → same as original IntervalDistribution**.
10. **Correct PDF on fully disjoint components** – simple hand-computed check.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate as sci_integrate

from partition_tree.skpro.distribution import (
    IntervalDistribution,
    MixtureIntervalDistribution,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dist_a():
    """Component A: two intervals [0,1) and [2,3]."""
    return IntervalDistribution(
        intervals=[[(0.0, 1.0), (2.0, 3.0)]],
        pdf_values=[[0.5, 0.5]],
        index=pd.RangeIndex(1),
        columns=pd.Index([0]),
    )


@pytest.fixture()
def dist_b():
    """Component B: one wide interval [0,3]."""
    return IntervalDistribution(
        intervals=[[(0.0, 3.0)]],
        pdf_values=[[1.0 / 3.0]],
        index=pd.RangeIndex(1),
        columns=pd.Index([0]),
    )


@pytest.fixture()
def dist_c():
    """Component C: overlapping with A – [0.5, 2.5]."""
    return IntervalDistribution(
        intervals=[[(0.5, 2.5)]],
        pdf_values=[[0.5]],
        index=pd.RangeIndex(1),
        columns=pd.Index([0]),
    )


@pytest.fixture()
def mix_ab(dist_a, dist_b):
    return MixtureIntervalDistribution(
        distributions=[dist_a, dist_b],
        weights=[0.6, 0.4],
    )


@pytest.fixture()
def mix_abc(dist_a, dist_b, dist_c):
    return MixtureIntervalDistribution(
        distributions=[dist_a, dist_b, dist_c],
        weights=[0.5, 0.3, 0.2],
    )


# ---------------------------------------------------------------------------
# Multi-instance fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def multi_instance_dists():
    """Two-instance distributions."""
    d1 = IntervalDistribution(
        intervals=[
            [(0.0, 1.0), (1.0, 2.0)],
            [(0.0, 2.0), (2.0, 4.0)],
        ],
        pdf_values=[[0.6, 0.4], [0.3, 0.35]],
        index=pd.RangeIndex(2),
        columns=pd.Index([0]),
    )
    d2 = IntervalDistribution(
        intervals=[
            [(0.5, 1.5)],
            [(1.0, 3.0)],
        ],
        pdf_values=[[0.8], [0.4]],
        index=pd.RangeIndex(2),
        columns=pd.Index([0]),
    )
    return d1, d2


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _numerical_pdf(mixture, instance_idx, x_scalar):
    """Query mixture PDF at a scalar x for a given instance."""
    return mixture._mixture_pdf_at(instance_idx, x_scalar)


def _numerical_integral(mixture, instance_idx, lo, hi, n=5000):
    """Numerical integral of the mixture PDF over [lo, hi]."""
    xs = np.linspace(lo, hi, n)
    ys = [_numerical_pdf(mixture, instance_idx, float(xi)) for xi in xs]
    return float(np.trapezoid(ys, xs))


# ===========================================================================
# Test 1 – PDF integrates to 1
# ===========================================================================


class TestPdfIntegratesTo1:
    def test_two_component_mixture(self, mix_ab):
        total = _numerical_integral(mix_ab, 0, -0.5, 3.5)
        assert abs(total - 1.0) < 1e-2

    def test_three_component_mixture(self, mix_abc):
        total = _numerical_integral(mix_abc, 0, -0.5, 4.0)
        assert abs(total - 1.0) < 1e-2

    def test_multi_instance(self, multi_instance_dists):
        d1, d2 = multi_instance_dists
        mix = MixtureIntervalDistribution([d1, d2], weights=[0.5, 0.5])
        for i in range(2):
            total = _numerical_integral(mix, i, -1.0, 6.0)
            assert abs(total - 1.0) < 1e-2, f"Instance {i}: integral = {total}"


# ===========================================================================
# Test 2 – CDF is monotone and bounded in [0, 1]
# ===========================================================================


class TestCdfMonotoneBounded:
    def _xs(self):
        return np.linspace(-0.5, 3.5, 200)

    def _eval_cdf(self, mix, xs):
        """Evaluate the mixture CDF at an array of xs for instance 0."""
        return np.array([mix._mixture_cdf_at(0, float(xi)) for xi in xs], dtype=float)

    def test_bounded(self, mix_ab):
        vals = self._eval_cdf(mix_ab, self._xs())
        assert vals.min() >= -1e-9
        assert vals.max() <= 1.0 + 1e-9

    def test_monotone(self, mix_ab):
        xs = self._xs()
        vals = self._eval_cdf(mix_ab, xs)
        diffs = np.diff(vals)
        assert (diffs >= -1e-9).all(), "CDF is not monotone non-decreasing"

    def test_three_component(self, mix_abc):
        vals = self._eval_cdf(mix_abc, self._xs())
        diffs = np.diff(vals)
        assert (diffs >= -1e-9).all()
        assert vals.min() >= -1e-9
        assert vals.max() <= 1.0 + 1e-9


# ===========================================================================
# Test 3 – Mean = weighted sum of component means
# ===========================================================================


class TestMeanMixtureIdentity:
    def test_two_components(self, mix_ab, dist_a, dist_b):
        mu_mix = mix_ab._mean().to_numpy().ravel()[0]
        mu_a = dist_a._mean().to_numpy().ravel()[0]
        mu_b = dist_b._mean().to_numpy().ravel()[0]
        expected = 0.6 * mu_a + 0.4 * mu_b
        assert abs(mu_mix - expected) < 1e-10

    def test_three_components(self, mix_abc, dist_a, dist_b, dist_c):
        mu_mix = mix_abc._mean().to_numpy().ravel()[0]
        mu_a = dist_a._mean().to_numpy().ravel()[0]
        mu_b = dist_b._mean().to_numpy().ravel()[0]
        mu_c = dist_c._mean().to_numpy().ravel()[0]
        expected = 0.5 * mu_a + 0.3 * mu_b + 0.2 * mu_c
        assert abs(mu_mix - expected) < 1e-10

    def test_multi_instance(self, multi_instance_dists):
        d1, d2 = multi_instance_dists
        weights = [0.3, 0.7]
        mix = MixtureIntervalDistribution([d1, d2], weights=weights)
        mu_mix = mix._mean().to_numpy().ravel()
        mu1 = d1._mean().to_numpy().ravel()
        mu2 = d2._mean().to_numpy().ravel()
        expected = 0.3 * mu1 + 0.7 * mu2
        np.testing.assert_allclose(mu_mix, expected, atol=1e-10)


# ===========================================================================
# Test 4 – Variance = mixture identity
# ===========================================================================


class TestVarianceMixtureIdentity:
    def _mixture_var_formula(self, weights, dists):
        """Compute variance via the mixture identity directly."""
        ws = np.asarray(weights, dtype=float)
        ws = ws / ws.sum()
        n = len(dists[0].index)
        second_moment = np.zeros(n)
        for w, d in zip(ws, dists):
            var_m = d._var().to_numpy().ravel()
            mu_m = d._mean().to_numpy().ravel()
            second_moment += w * (var_m + mu_m**2)
        mu_mix = sum(w * d._mean().to_numpy().ravel() for w, d in zip(ws, dists))
        return second_moment - mu_mix**2

    def test_two_components(self, mix_ab, dist_a, dist_b):
        var_mix = mix_ab._var().to_numpy().ravel()
        expected = self._mixture_var_formula([0.6, 0.4], [dist_a, dist_b])
        np.testing.assert_allclose(var_mix, expected, atol=1e-10)

    def test_non_negative(self, mix_ab):
        var_mix = mix_ab._var().to_numpy().ravel()
        assert (var_mix >= 0).all()

    def test_three_components(self, mix_abc, dist_a, dist_b, dist_c):
        var_mix = mix_abc._var().to_numpy().ravel()
        expected = self._mixture_var_formula([0.5, 0.3, 0.2], [dist_a, dist_b, dist_c])
        np.testing.assert_allclose(var_mix, expected, atol=1e-10)


# ===========================================================================
# Test 5 – PPF is inverse of CDF: F(ppf(p)) ≈ p
# ===========================================================================


class TestPpfInverseOfCdf:
    @pytest.mark.parametrize("p", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    def test_roundtrip_two_components(self, mix_ab, p):
        p_arr = pd.DataFrame([[p]], index=pd.RangeIndex(1), columns=pd.Index([0]))
        x = mix_ab._ppf(np.array([[p]])).to_numpy().ravel()[0]
        # Evaluate CDF at the returned quantile
        cdf_val = mix_ab._mixture_cdf_at(0, x)
        assert abs(cdf_val - p) < 1e-4, f"CDF(ppf({p})) = {cdf_val}, expected {p}"

    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_roundtrip_three_components(self, mix_abc, p):
        x = mix_abc._ppf(np.array([[p]])).to_numpy().ravel()[0]
        cdf_val = mix_abc._mixture_cdf_at(0, x)
        assert abs(cdf_val - p) < 1e-4, f"CDF(ppf({p})) = {cdf_val}, expected {p}"


# ===========================================================================
# Test 6 – energy_x = weighted sum of component energy_x values
# ===========================================================================


class TestEnergyXMixtureIdentity:
    def test_two_components(self, mix_ab, dist_a, dist_b):
        x = np.array([[1.5]])  # (n_instances=1, 1)
        e_mix = mix_ab._energy_x(x).ravel()[0]
        e_a = dist_a._energy_x(x).ravel()[0]
        e_b = dist_b._energy_x(x).ravel()[0]
        expected = 0.6 * e_a + 0.4 * e_b
        assert abs(e_mix - expected) < 1e-10

    def test_various_points(self, mix_ab, dist_a, dist_b):
        for x_val in [0.0, 0.5, 1.5, 2.5, 3.5]:
            x = np.array([[x_val]])
            e_mix = mix_ab._energy_x(x).ravel()[0]
            e_a = dist_a._energy_x(x).ravel()[0]
            e_b = dist_b._energy_x(x).ravel()[0]
            expected = 0.6 * e_a + 0.4 * e_b
            assert abs(e_mix - expected) < 1e-10, f"x={x_val}: {e_mix} != {expected}"


# ===========================================================================
# Test 7 – energy_self ≥ 0
# ===========================================================================


class TestEnergySelfNonNegative:
    def test_two_components(self, mix_ab):
        es = mix_ab._energy_self()
        assert (es >= 0).all()

    def test_three_components(self, mix_abc):
        es = mix_abc._energy_self()
        assert (es >= 0).all()

    def test_multi_instance(self, multi_instance_dists):
        d1, d2 = multi_instance_dists
        mix = MixtureIntervalDistribution([d1, d2], weights=[0.5, 0.5])
        es = mix._energy_self()
        assert (es >= 0).all()

    def test_same_as_component_when_single(self, dist_a):
        """Single-component mixture energy_self == dist._energy_self()."""
        mix = MixtureIntervalDistribution([dist_a])
        es_mix = mix._energy_self().ravel()[0]
        es_raw = dist_a._energy_self().ravel()[0]
        assert abs(es_mix - es_raw) < 1e-8


# ===========================================================================
# Test 8 – Sampling: empirical mean converges to analytical mean
# ===========================================================================


class TestSamplingMean:
    def test_empirical_mean_close_to_analytical(self, mix_ab):
        """Sample from the mixture and check the empirical mean is close."""
        analytical_mean = mix_ab.mean().to_numpy().ravel()[0]
        samples = mix_ab.sample(n_samples=2000).to_numpy().ravel()
        empirical_mean = samples.mean()
        # Allow 3 std-deviations of MC error (generous tolerance)
        std_err = samples.std() / np.sqrt(len(samples))
        assert abs(empirical_mean - analytical_mean) < 5 * std_err + 0.05


# ===========================================================================
# Test 9 – Single component → results match original IntervalDistribution
# ===========================================================================


class TestSingleComponentEquivalence:
    """A mixture with a single component must give the same results as that component."""

    @pytest.mark.parametrize("x_val", [0.3, 1.0, 2.7])
    def test_pdf(self, dist_a, x_val):
        mix = MixtureIntervalDistribution([dist_a])
        x_df = pd.DataFrame([[x_val]], index=pd.RangeIndex(1), columns=pd.Index([0]))
        pdf_mix = mix._pdf(np.array([[x_val]])).to_numpy().ravel()[0]
        pdf_raw = dist_a._pdf(np.array([[x_val]])).to_numpy().ravel()[0]
        assert abs(pdf_mix - pdf_raw) < 1e-10

    @pytest.mark.parametrize("x_val", [0.3, 1.0, 2.7])
    def test_cdf(self, dist_a, x_val):
        mix = MixtureIntervalDistribution([dist_a])
        cdf_mix = mix._mixture_cdf_at(0, x_val)
        cdf_raw = dist_a._cdf(np.array([[x_val]])).to_numpy().ravel()[0]
        assert abs(cdf_mix - cdf_raw) < 1e-10

    def test_mean(self, dist_a):
        mix = MixtureIntervalDistribution([dist_a])
        assert (
            abs(
                mix._mean().to_numpy().ravel()[0] - dist_a._mean().to_numpy().ravel()[0]
            )
            < 1e-10
        )

    def test_var(self, dist_a):
        mix = MixtureIntervalDistribution([dist_a])
        assert (
            abs(mix._var().to_numpy().ravel()[0] - dist_a._var().to_numpy().ravel()[0])
            < 1e-10
        )


# ===========================================================================
# Test 10 – Correct PDF on fully disjoint components (hand-computed)
# ===========================================================================


class TestDisjointPdf:
    """When components are fully disjoint, the mixture PDF is simply w_m * f_m(x)."""

    def test_pdf_in_component_a_region(self):
        """dist_a covers [0,1] with density 0.5 (normalised to 1 total mass).
        dist_b covers [2,3] with density 0.5.
        With weights [0.5, 0.5]:
          At x=0.5, only component A is active: mix_pdf = 0.5 * f_a(0.5).
        """
        d_left = IntervalDistribution(
            intervals=[[(0.0, 1.0)]],
            pdf_values=[[1.0]],  # mass = 1*1 = 1 (normalised)
            index=pd.RangeIndex(1),
            columns=pd.Index([0]),
        )
        d_right = IntervalDistribution(
            intervals=[[(2.0, 3.0)]],
            pdf_values=[[1.0]],  # mass = 1*1 = 1 (normalised)
            index=pd.RangeIndex(1),
            columns=pd.Index([0]),
        )
        mix = MixtureIntervalDistribution([d_left, d_right], weights=[0.5, 0.5])

        # At x=0.5: only d_left is active, density = w_left * f_left / norm_left
        #          = 0.5 * 1.0 / 1.0 = 0.5
        pdf_at_05 = mix._mixture_pdf_at(0, 0.5)
        assert abs(pdf_at_05 - 0.5) < 1e-10

        # At x=2.5: only d_right is active
        pdf_at_25 = mix._mixture_pdf_at(0, 2.5)
        assert abs(pdf_at_25 - 0.5) < 1e-10

        # At x=1.5: neither active → 0
        pdf_at_15 = mix._mixture_pdf_at(0, 1.5)
        assert abs(pdf_at_15 - 0.0) < 1e-10

    def test_pdf_normalisation_disjoint(self):
        """The total integral of the disjoint mixture should be 1."""
        d_left = IntervalDistribution(
            intervals=[[(0.0, 1.0)]],
            pdf_values=[[1.0]],
            index=pd.RangeIndex(1),
            columns=pd.Index([0]),
        )
        d_right = IntervalDistribution(
            intervals=[[(2.0, 3.0)]],
            pdf_values=[[1.0]],
            index=pd.RangeIndex(1),
            columns=pd.Index([0]),
        )
        mix = MixtureIntervalDistribution([d_left, d_right], weights=[0.5, 0.5])
        total = _numerical_integral(mix, 0, -0.5, 3.5)
        assert abs(total - 1.0) < 1e-3


# ===========================================================================
# Test 11 – Weight normalisation
# ===========================================================================


class TestWeightNormalisation:
    """Weights should be normalised internally."""

    def test_unnormalised_weights_give_same_result(self, dist_a, dist_b):
        mix1 = MixtureIntervalDistribution([dist_a, dist_b], weights=[0.6, 0.4])
        mix2 = MixtureIntervalDistribution(
            [dist_a, dist_b], weights=[3.0, 2.0]
        )  # 3:2 ratio = same

        mu1 = mix1._mean().to_numpy().ravel()[0]
        mu2 = mix2._mean().to_numpy().ravel()[0]
        assert abs(mu1 - mu2) < 1e-10

        pdf1 = mix1._mixture_pdf_at(0, 0.5)
        pdf2 = mix2._mixture_pdf_at(0, 0.5)
        assert abs(pdf1 - pdf2) < 1e-10


# ===========================================================================
# Test 12 – uniform_cross_energy helper (unit tests)
# ===========================================================================


class TestUniformCrossEnergy:
    """Verify _uniform_cross_energy against numerical integration."""

    def test_same_interval(self):
        """E[|U-V|] for U,V ~ Uniform(0,1) = 1/3."""
        from partition_tree.skpro.distribution import _uniform_cross_energy

        result = _uniform_cross_energy(0.0, 1.0, 0.0, 1.0)
        assert abs(result - 1.0 / 3.0) < 1e-8

    def test_disjoint_intervals(self):
        """U ~ Uniform(0,1), V ~ Uniform(2,3): E[|U-V|] = 2 + 0 = 2 (midpoints 0.5, 2.5)."""
        from partition_tree.skpro.distribution import _uniform_cross_energy

        # Numerical ground truth
        def integrand(x, y):
            return abs(x - y)

        result_numerical, _ = sci_integrate.dblquad(integrand, 0, 1, 2, 3, epsabs=1e-8)
        result_numerical = result_numerical  # already the double integral over area = 1

        result = _uniform_cross_energy(0.0, 1.0, 2.0, 3.0)
        assert abs(result - result_numerical) < 1e-6

    def test_overlapping_intervals(self):
        """U ~ Uniform(0,2), V ~ Uniform(1,3): compared numerically."""
        from partition_tree.skpro.distribution import _uniform_cross_energy

        def integrand(x, y):
            return abs(x - y) / (2.0 * 2.0)

        result_numerical, _ = sci_integrate.dblquad(integrand, 0, 2, 1, 3, epsabs=1e-8)
        result = _uniform_cross_energy(0.0, 2.0, 1.0, 3.0)
        assert abs(result - result_numerical) < 1e-6
