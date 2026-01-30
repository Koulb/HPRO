"""
Unit tests for band-space Resolution of Identity (band-RI) convergence.

Tests use synthetic data only - no external DFT files required.
"""

import numpy as np
import pytest

from HPRO.mathutils import compute_local_h_band_ri, compute_band_ri_metric


class TestComputeLocalHBandRI:
    """Tests for compute_local_h_band_ri function."""

    def test_correctness_small(self):
        """Compare to explicit A† diag(eigs) A."""
        nband, norb = 10, 5
        rng = np.random.default_rng(42)
        eigs = rng.random(nband)
        A = rng.standard_normal((nband, norb)) + 1j * rng.standard_normal((nband, norb))

        H_ri = compute_local_h_band_ri(eigs, A)
        H_explicit = A.conj().T @ np.diag(eigs) @ A

        np.testing.assert_allclose(H_ri, H_explicit, rtol=1e-10)

    def test_hermiticity(self):
        """Result should be Hermitian."""
        nband, norb = 20, 8
        rng = np.random.default_rng(123)
        eigs = rng.random(nband)
        A = rng.standard_normal((nband, norb)) + 1j * rng.standard_normal((nband, norb))

        H_ri = compute_local_h_band_ri(eigs, A)
        np.testing.assert_allclose(H_ri, H_ri.conj().T, rtol=1e-12)

    def test_chunked_matches_direct(self):
        """Chunked computation matches direct."""
        nband, norb = 50, 12
        rng = np.random.default_rng(456)
        eigs = rng.random(nband)
        A = rng.standard_normal((nband, norb)) + 1j * rng.standard_normal((nband, norb))

        H_direct = compute_local_h_band_ri(eigs, A)
        H_chunked = compute_local_h_band_ri(eigs, A, chunk=7)

        np.testing.assert_allclose(H_chunked, H_direct, rtol=1e-10)

    def test_shape_mismatch_error(self):
        """Should raise ValueError on shape mismatch."""
        nband, norb = 10, 5
        rng = np.random.default_rng(789)
        eigs = rng.random(nband + 1)  # Wrong shape
        A = rng.standard_normal((nband, norb)) + 1j * rng.standard_normal((nband, norb))

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_local_h_band_ri(eigs, A)

    def test_real_eigenvalues_complex_A(self):
        """Real eigenvalues with complex A should work."""
        nband, norb = 15, 6
        rng = np.random.default_rng(101)
        eigs = rng.random(nband).astype(np.float64)
        A = rng.standard_normal((nband, norb)) + 1j * rng.standard_normal((nband, norb))

        H_ri = compute_local_h_band_ri(eigs, A)
        assert H_ri.dtype == np.complex128
        assert H_ri.shape == (norb, norb)

    def test_output_shape(self):
        """Output shape should be (norb, norb)."""
        nband, norb = 25, 10
        rng = np.random.default_rng(202)
        eigs = rng.random(nband)
        A = rng.standard_normal((nband, norb)) + 1j * rng.standard_normal((nband, norb))

        H_ri = compute_local_h_band_ri(eigs, A)
        assert H_ri.shape == (norb, norb)

    def test_single_band(self):
        """Edge case: single band."""
        norb = 5
        rng = np.random.default_rng(303)
        eigs = rng.random(1)
        A = rng.standard_normal((1, norb)) + 1j * rng.standard_normal((1, norb))

        H_ri = compute_local_h_band_ri(eigs, A)
        H_expected = eigs[0] * np.outer(A[0].conj(), A[0])

        np.testing.assert_allclose(H_ri, H_expected, rtol=1e-10)


class TestComputeBandRIMetric:
    """Tests for compute_band_ri_metric function."""

    def test_none_previous(self):
        """Should return inf when H_prev is None."""
        H_new = np.eye(5)
        err = compute_band_ri_metric(H_new, None)
        assert err == float('inf')

    def test_identical_matrices(self):
        """Identical matrices should give zero error."""
        H = np.random.randn(5, 5) + 1j * np.random.randn(5, 5)
        err = compute_band_ri_metric(H, H.copy())
        assert err < 1e-14

    def test_positive_error(self):
        """Different matrices should give positive error."""
        rng = np.random.default_rng(404)
        H1 = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
        H2 = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
        err = compute_band_ri_metric(H1, H2)
        assert err > 0

    def test_relative_scaling(self):
        """Error should be relative to norm of H_new."""
        rng = np.random.default_rng(505)
        H = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
        delta = 0.01 * (rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5)))

        err = compute_band_ri_metric(H + delta, H)
        expected = np.linalg.norm(delta, 'fro') / np.linalg.norm(H + delta, 'fro')
        np.testing.assert_allclose(err, expected, rtol=1e-10)


class TestConvergenceTrend:
    """Tests for convergence behavior with increasing band window."""

    def test_error_decreases_with_bands(self):
        """Error should decrease as band window grows."""
        # Build "full" reference with Nfull bands
        nband_full, norb = 100, 20
        rng = np.random.default_rng(606)
        eigs = np.sort(rng.random(nband_full))  # Sorted eigenvalues
        A = rng.standard_normal((nband_full, norb)) + 1j * rng.standard_normal((nband_full, norb))

        # Compute full reference
        H_full = compute_local_h_band_ri(eigs, A)

        # Test with increasing band windows
        nband_list = [10, 25, 50, 75, 100]
        errors = []

        for nbnd in nband_list:
            H_trunc = compute_local_h_band_ri(eigs[:nbnd], A[:nbnd, :])
            err = np.linalg.norm(H_trunc - H_full, 'fro') / np.linalg.norm(H_full, 'fro')
            errors.append(err)

        # Verify error decreases (or saturates) with more bands
        # Last error should be zero (full set)
        assert errors[-1] < 1e-12, "Full band set should match exactly"

        # Earlier truncations should have larger errors
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-10, (
                f"Error should decrease: errors[{i}]={errors[i]} vs errors[{i+1}]={errors[i+1]}"
            )

    def test_convergence_with_step_metric(self):
        """Test step-by-step convergence metric."""
        nband_full, norb = 80, 15
        rng = np.random.default_rng(707)
        eigs = np.sort(rng.random(nband_full))
        A = rng.standard_normal((nband_full, norb)) + 1j * rng.standard_normal((nband_full, norb))

        nband_list = [20, 40, 60, 80]
        step_errors = []
        H_prev = None

        for nbnd in nband_list:
            H_loc = compute_local_h_band_ri(eigs[:nbnd], A[:nbnd, :])
            err = compute_band_ri_metric(H_loc, H_prev)
            step_errors.append(err)
            H_prev = H_loc

        # First error should be inf (no previous)
        assert step_errors[0] == float('inf')

        # Subsequent errors should be finite
        for err in step_errors[1:]:
            assert np.isfinite(err)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
