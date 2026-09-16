"""Compare spekk FIR design with SciPy as an independent reference."""

import numpy as np
import pytest
from scipy import signal

from spekk import ops


@pytest.fixture(autouse=True)
def numpy_backend():
    with ops.backend.temporary_backend("numpy"):
        yield


@pytest.mark.parametrize("ripple", [8, 20, 21, 21.1, 40, 50, 50.1, 60, 100, -60])
@pytest.mark.parametrize("width", [0.001, 0.01, 0.1, 0.5, 1.0])
def test_kaiserord_matches_scipy(ripple, width):
    actual = ops.signal.kaiserord(ripple, width)
    expected = signal.kaiserord(ripple, width)
    assert isinstance(actual[0], int)
    assert actual[0] == expected[0]
    assert actual[1] == pytest.approx(expected[1], abs=1e-14)


@pytest.mark.parametrize("ripple", [0, 1, 7.99, -7])
def test_kaiserord_small_ripple_raises(ripple):
    with pytest.raises(ValueError):
        ops.signal.kaiserord(ripple, 0.1)
    with pytest.raises(ValueError):
        signal.kaiserord(ripple, 0.1)


@pytest.mark.parametrize("numtaps", [1, 3, 16, 31, 64, 101])
@pytest.mark.parametrize("window", [
    "boxcar", "hann", "hamming", "blackman", "blackmanharris", "nuttall",
    "flattop", "bartlett", "triang", "cosine", "bohman", "parzen", "barthann",
    "lanczos", "tukey", ("tukey", 0.2), ("tukey", 0), ("tukey", 1.5),
    ("kaiser", 0), ("kaiser", 5.65), ("kaiser", 14), ("kaiser", 80),
    6.0, ("gaussian", 4), ("general_gaussian", 1.5, 7),
    ("general_hamming", 0.6),
])
@pytest.mark.parametrize("scale", [True, False])
def test_firwin_windows_match_scipy(numtaps, window, scale):
    actual = ops.signal.firwin(numtaps, 0.3, window=window, scale=scale, dim="time")
    expected = signal.firwin(numtaps, 0.3, window=window, scale=scale)
    assert actual.dims == ["time"]
    np.testing.assert_allclose(ops.to_numpy(actual), expected, rtol=1e-10, atol=2e-14)


@pytest.mark.parametrize("numtaps", [1, 3, 16, 31, 64, 101])
@pytest.mark.parametrize("scale", [True, False])
def test_firwin_general_cosine_from_scipy_components(numtaps, scale):
    """SciPy's firwin currently rejects general_cosine through its xp dispatch."""
    weights = [0.3, 0.5, 0.2]
    expected = signal.firwin(numtaps, 0.3, window="boxcar", scale=False)
    expected *= signal.windows.general_cosine(numtaps, weights)
    if scale:
        expected /= expected.sum()
    actual = ops.signal.firwin(numtaps, 0.3, window=("general_cosine", weights), scale=scale)
    np.testing.assert_allclose(ops.to_numpy(actual), expected, rtol=1e-10, atol=2e-14)


@pytest.mark.parametrize("cutoff,pass_zero", [
    (0.2, True), (0.2, False), (0.2, "lowpass"), (0.2, "highpass"),
    ([0.2, 0.6], "bandpass"), ([0.2, 0.6], "bandstop"),
    ([0.1, 0.3, 0.5], True), ([0.1, 0.3, 0.5], False),
    ([0.1, 0.2, 0.5, 0.8], True), ([0.1, 0.2, 0.5, 0.8], False),
])
@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("fs", [None, 2, 30e6])
def test_firwin_bands_and_units(cutoff, pass_zero, scale, fs):
    cutoff = np.asarray(cutoff) * (1 if fs is None else fs / 2)
    actual = ops.signal.firwin(51, cutoff, pass_zero=pass_zero, scale=scale, fs=fs)
    expected = signal.firwin(51, cutoff, pass_zero=pass_zero, scale=scale, fs=fs)
    np.testing.assert_allclose(ops.to_numpy(actual), expected, rtol=1e-10, atol=2e-14)


@pytest.mark.parametrize("numtaps", [3, 16, 51, 201])
@pytest.mark.parametrize("width", [0.01, 0.1, 0.3])
def test_firwin_width_overrides_window(numtaps, width):
    actual = ops.signal.firwin(numtaps, 3, width=width * 10, fs=20, window="ignored")
    expected = signal.firwin(numtaps, 3, width=width * 10, fs=20, window="ignored")
    np.testing.assert_allclose(ops.to_numpy(actual), expected, rtol=1e-10, atol=2e-14)


@pytest.mark.parametrize("kwargs", [
    {"cutoff": []}, {"cutoff": 0}, {"cutoff": 1}, {"cutoff": -0.1},
    {"cutoff": [0.3, 0.2]}, {"cutoff": [0.3, 0.3]}, {"cutoff": [[0.2, 0.3]]},
    {"cutoff": 0.2, "pass_zero": "unknown"},
    {"cutoff": [0.2, 0.3], "pass_zero": "lowpass"},
    {"cutoff": [0.2, 0.3], "pass_zero": "highpass"},
    {"cutoff": 0.2, "pass_zero": "bandpass"},
    {"cutoff": 0.2, "pass_zero": "bandstop"},
    {"cutoff": 0.2, "numtaps": 10, "pass_zero": False},
    {"cutoff": [0.2, 0.3], "numtaps": 10, "pass_zero": True},
    {"cutoff": 0.2, "window": "invalid"}, {"cutoff": 0.2, "window": "kaiser"},
])
def test_firwin_invalid_inputs_match_scipy(kwargs):
    kwargs = {"numtaps": 31, **kwargs}
    with pytest.raises(ValueError):
        ops.signal.firwin(**kwargs)
    with pytest.raises(ValueError):
        signal.firwin(**kwargs)


@pytest.mark.parametrize("numtaps", [1, 2, 7, 32, 101])
@pytest.mark.parametrize("window", ["hamming_periodic", "hann_periodic", ("kaiser_periodic", 6), "hamming_symmetric"])
def test_firwin_window_symmetry_modes(numtaps, window):
    expected = signal.firwin(numtaps, 0.35, window=window)
    actual = ops.signal.firwin(numtaps, 0.35, window=window)
    np.testing.assert_allclose(ops.to_numpy(actual), expected, rtol=1e-10, atol=2e-14)


@pytest.mark.parametrize("length", [0, 1, 2, 3, 16, 51, 128])
@pytest.mark.parametrize("beta", [0, -5, 5, 14, 80])
def test_public_kaiser_matches_scipy(length, beta):
    actual = ops.signal.kaiser(length, beta, dim="time")
    assert actual.dims == ["time"]
    expected = signal.windows.kaiser(length, beta)
    np.testing.assert_allclose(ops.to_numpy(actual), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("length", [-1, 2.5])
def test_kaiser_invalid_length(length):
    with pytest.raises(ValueError):
        ops.signal.kaiser(length, 6)
    with pytest.raises(ValueError):
        signal.windows.kaiser(length, 6)


@pytest.mark.parametrize("backend_name", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("window", ["hamming", ("kaiser", 6), ("tukey", 0.25)])
def test_firwin_backends(backend_name, window):
    pytest.importorskip(backend_name)
    with ops.backend.temporary_backend(backend_name):
        result = ops.signal.firwin(63, [0.2, 0.55], pass_zero=False, window=window, dim="time")
        assert result.dims == ["time"]
        expected = signal.firwin(63, [0.2, 0.55], pass_zero=False, window=window)
        np.testing.assert_allclose(ops.to_numpy(result), expected, rtol=1e-5, atol=5e-8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_firwin_jax_jit_and_complex_convolution(dtype):
    jax = pytest.importorskip("jax")
    with jax.enable_x64(dtype == np.float64), ops.backend.temporary_backend("jax"):
        rng = np.random.default_rng(12)
        values = (rng.standard_normal((256, 3)) + 1j * rng.standard_normal((256, 3))).astype(
            np.complex64 if dtype == np.float32 else np.complex128
        )

        @ops.jit
        def filtered(data):
            coefficients = ops.signal.firwin(31, 0.25, window=("kaiser", 6), dim="time")
            return ops.image.convNd(data, coefficients, pad_mode="constant"), coefficients

        actual, coefficients = filtered(ops.array(values, dims=["time", "channel"]))
        expected_coefficients = signal.firwin(31, 0.25, window=("kaiser", 6))
        expected = signal.convolve(values, expected_coefficients[:, None], mode="same")
        tolerance = 2e-6 if dtype == np.float32 else 1e-12
        assert actual.dims == ["time", "channel"]
        np.testing.assert_allclose(ops.to_numpy(coefficients), expected_coefficients, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(ops.to_numpy(actual), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("cutoff,pass_zero,reference", [
    (0.3, True, 0), (0.3, False, 1), ([0.2, 0.5], False, 0.35),
    ([0.2, 0.5], True, 0),
])
def test_symmetry_and_unity_reference_gain(cutoff, pass_zero, reference):
    coefficients = ops.to_numpy(ops.signal.firwin(81, cutoff, pass_zero=pass_zero))
    np.testing.assert_allclose(coefficients, coefficients[::-1], atol=1e-15)
    positions = np.arange(81) - 40
    response = np.sum(coefficients * np.exp(-1j * np.pi * reference * positions))
    np.testing.assert_allclose(response, 1, atol=1e-14)


def test_kaiserord_firwin_response_and_centered_impulse():
    numtaps, beta = ops.signal.kaiserord(60, 0.1)
    numtaps += numtaps % 2 == 0
    coefficients = ops.signal.firwin(numtaps, 0.3, window=("kaiser", beta), dim="time")
    frequencies, response = signal.freqz(ops.to_numpy(coefficients), worN=16384, fs=2)
    assert np.max(np.abs(response[frequencies >= 0.35])) < 1.2e-3
    assert np.max(np.abs(np.abs(response[frequencies <= 0.25]) - 1)) < 1.2e-3
    impulse = np.zeros(4 * numtaps + 1)
    impulse[2 * numtaps] = 1
    result = ops.image.convNd(ops.array(impulse, dims=["time"]), coefficients, pad_mode="constant")
    expected = signal.convolve(impulse, ops.to_numpy(coefficients), mode="same")
    assert np.argmax(ops.to_numpy(result)) == 2 * numtaps
    np.testing.assert_allclose(ops.to_numpy(result), expected, atol=1e-14)


def test_spekk_cutoff_array():
    result = ops.signal.firwin(31, ops.array([0.2, 0.4], dims=["edges"]), pass_zero=False)
    expected = signal.firwin(31, [0.2, 0.4], pass_zero=False)
    np.testing.assert_allclose(ops.to_numpy(result), expected, atol=1e-14)


def test_no_scipy_imports_required(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise AssertionError("Spekk filter design must not import SciPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    numtaps, beta = ops.signal.kaiserord(60, 0.1)
    result = ops.signal.firwin(numtaps, 0.3, window=("kaiser", beta))
    assert result.shape == (numtaps,)