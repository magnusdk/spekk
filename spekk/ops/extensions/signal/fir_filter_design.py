"""Window-method FIR design using spekk arrays and scalar design parameters."""

import math
import operator

from spekk import ops
from spekk.ops._types import Dim, undefined_dim
from spekk.ops.extensions.signal.windows import kaiser


def _kaiser_beta(attenuation: float) -> float:
    if attenuation > 50:
        return 0.1102 * (attenuation - 8.7)
    if attenuation > 21:
        return 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21)
    return 0.0


def kaiserord(ripple: float, width: float) -> tuple[int, float]:
    """Estimate Kaiser FIR tap count and beta, as in scipy.signal.kaiserord.

    Parameters
    ----------
    ripple : float
        Magnitude of the required attenuation in dB (at least 8 dB).
    width : float
        Transition width normalized to Nyquist, not the sample rate.

    Returns
    -------
    tuple[int, float]
        Number of taps and Kaiser beta. The tap count is not forced odd.
    """
    attenuation = abs(float(ripple))
    if attenuation < 8:
        raise ValueError("Requested attenuation must be at least 8 dB")
    numtaps = math.ceil((attenuation - 7.95) / (2.285 * math.pi * width) + 1)
    return numtaps, _kaiser_beta(attenuation)


def _sinc(values):
    denominator = ops.where(values == 0, 1.0, math.pi * values)
    return ops.where(values == 0, 1.0, ops.sin(math.pi * values) / denominator)


def _fir_window(window, numtaps, dim):
    if isinstance(window, (int, float)):
        window = ("kaiser", float(window))
    if isinstance(window, str):
        name, parameters = window.lower(), ()
    elif isinstance(window, tuple) and window and isinstance(window[0], str):
        name, parameters = window[0].lower(), window[1:]
    else:
        raise ValueError("window must be a name, parameter tuple, or Kaiser beta")
    periodic = name.endswith("_periodic")
    name = name.removesuffix("_periodic").removesuffix("_symmetric")
    aliases = {
        "rect": "boxcar", "rectangular": "boxcar", "ones": "boxcar",
        "hanning": "hann", "hamm": "hamming", "black": "blackman",
        "blackharr": "blackmanharris", "tri": "triang",
    }
    name = aliases.get(name, name)
    cosine_coefficients = {
        "boxcar": (1.0,), "hann": (0.5, 0.5), "hamming": (0.54, 0.46),
        "blackman": (0.42, 0.5, 0.08),
        "blackmanharris": (0.35875, 0.48829, 0.14128, 0.01168),
        "nuttall": (0.3635819, 0.4891775, 0.1365995, 0.0106411),
        "flattop": (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368),
    }
    parameter_counts = {
        "kaiser": (1,), "gaussian": (1,), "general_gaussian": (2,),
        "general_cosine": (1,), "general_hamming": (1,), "tukey": (0, 1),
    }
    if name in parameter_counts:
        if len(parameters) not in parameter_counts[name]:
            raise ValueError(f"Invalid parameters for window {name!r}")
    elif name not in cosine_coefficients and name not in {"bartlett", "triang", "cosine", "bohman", "parzen", "barthann", "lanczos"}:
        raise ValueError(f"Unsupported FIR window: {name!r}")
    elif parameters:
        raise ValueError(f"Window {name!r} does not accept parameters")
    if numtaps <= 1:
        return ops.ones((numtaps,), dims=[dim])
    length = numtaps + int(periodic)
    samples = ops.arange(length, dim=dim) + 0.0
    fraction = samples / (length - 1)
    position = 2 * fraction - 1
    if name == "general_cosine":
        cosine_coefficients[name] = parameters[0]
    elif name == "general_hamming":
        cosine_coefficients[name] = (parameters[0], 1 - parameters[0])
    if name in cosine_coefficients:
        result = ops.zeros_like(samples)
        for order, coefficient in enumerate(cosine_coefficients[name]):
            result = result + coefficient * ops.cos(order * math.pi * position)
    elif name == "kaiser":
        result = kaiser(length, parameters[0], dim=dim)
    elif name == "gaussian":
        result = ops.exp(-0.5 * ((samples - (length - 1) / 2) / parameters[0]) ** 2)
    elif name == "general_gaussian":
        power, sigma = parameters
        result = ops.exp(-0.5 * ops.abs((samples - (length - 1) / 2) / sigma) ** (2 * power))
    elif name == "bartlett":
        result = 1 - ops.abs(position)
    elif name == "triang":
        result = 1 - ops.abs(samples - (length - 1) / 2) / ((length + length % 2) / 2)
    elif name == "cosine":
        result = ops.sin(math.pi * (samples + 0.5) / length)
    elif name == "bohman":
        distance = ops.abs(position)
        result = (1 - distance) * ops.cos(math.pi * distance) + ops.sin(math.pi * distance) / math.pi
        result = ops.where(distance >= 1, 0.0, result)
    elif name == "parzen":
        distance = ops.abs(samples - (length - 1) / 2) / (length / 2)
        result = ops.where(distance <= 0.5, 1 - 6 * distance**2 + 6 * distance**3, 2 * (1 - distance)**3)
    elif name == "barthann":
        result = 0.62 - 0.48 * ops.abs(fraction - 0.5) - 0.38 * ops.cos(2 * math.pi * fraction)
    elif name == "lanczos":
        result = _sinc(position)
    else:
        alpha = parameters[0] if parameters else 0.5
        if alpha <= 0:
            result = ops.ones_like(samples)
        elif alpha >= 1:
            result = 0.5 - 0.5 * ops.cos(2 * math.pi * fraction)
        else:
            edge = 0.5 * (1 + ops.cos(math.pi * (2 * ops.minimum(fraction, 1 - fraction) / alpha - 1)))
            result = ops.where(ops.abs(position) <= 1 - alpha, 1.0, edge)
    return result[:numtaps]


def firwin(
    numtaps: int,
    cutoff,
    *,
    width: float | None = None,
    window="hamming",
    pass_zero: bool | str = True,
    scale: bool = True,
    fs: float | None = None,
    dim: Dim = None,
) -> ops.array:
    """Design a linear-phase FIR using the window method.

    Matches scipy.signal.firwin for supported windows. ``cutoff`` contains
    half-amplitude (-6 dB) band edges, strictly between zero and ``fs / 2``.
    ``fs`` defaults to 2. ``width`` is a transition width in the same units;
    when supplied it overrides ``window`` with an estimated Kaiser window.
    ``pass_zero`` accepts a boolean or lowpass/highpass/bandpass/bandstop.
    Scaling gives unity gain at DC, Nyquist (single highpass), or the center
    of the first passband. Even tap counts cannot pass Nyquist.

    Supported windows: boxcar, hann, hamming, blackman, blackmanharris,
    nuttall, flattop, bartlett, triang, cosine, bohman, parzen, barthann,
    lanczos, tukey, kaiser, gaussian, general_gaussian, general_cosine and
    general_hamming. Parameterized windows use tuples as in SciPy; a numeric
    window denotes Kaiser beta. Windows are symmetric unless their name has
    a ``_periodic`` suffix. Other SciPy windows raise ValueError.

    Design parameters must be concrete (static under JIT). Coefficients use
    the active spekk backend's default floating-point precision and ``dim``.
    Odd taps with a symmetric window are suitable for centered convNd.
    """
    if dim is None:
        dim = undefined_dim
    numtaps = operator.index(numtaps)
    if numtaps < 1:
        raise ValueError("numtaps must be positive")
    nyquist = (2.0 if fs is None else float(fs)) / 2
    if not math.isfinite(nyquist) or nyquist <= 0:
        raise ValueError("fs must be finite and positive")
    if getattr(cutoff, "ndim", 0) > 1:
        raise ValueError("cutoff must be scalar or one-dimensional")
    try:
        edges = list(cutoff)
    except TypeError:
        edges = [cutoff]
    if not edges:
        raise ValueError("cutoff must contain at least one frequency")
    try:
        edges = [float(edge) / nyquist for edge in edges]
    except (TypeError, ValueError) as error:
        raise ValueError("cutoff must be scalar or one-dimensional") from error
    if any(not 0 < edge < 1 for edge in edges):
        raise ValueError("cutoff frequencies must lie strictly between zero and Nyquist")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("cutoff frequencies must be strictly increasing")
    if isinstance(pass_zero, str):
        if pass_zero not in {"lowpass", "highpass", "bandpass", "bandstop"}:
            raise ValueError("Invalid pass_zero filter type")
        if pass_zero in {"lowpass", "highpass"} and len(edges) != 1:
            raise ValueError("Lowpass and highpass require one cutoff")
        if pass_zero in {"bandpass", "bandstop"} and len(edges) < 2:
            raise ValueError("Bandpass and bandstop require at least two cutoffs")
        pass_zero = pass_zero in {"lowpass", "bandstop"}
    else:
        pass_zero = bool(operator.index(pass_zero))
    pass_nyquist = bool(len(edges) % 2) != pass_zero
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError("An even-length FIR cannot have a passband at Nyquist")
    if width is not None:
        attenuation = 2.285 * (numtaps - 1) * math.pi * float(width) / nyquist + 7.95
        window = ("kaiser", _kaiser_beta(attenuation))
    boundaries = ([0.0] if pass_zero else []) + edges + ([1.0] if pass_nyquist else [])
    bands = list(zip(boundaries[::2], boundaries[1::2]))
    positions = ops.arange(numtaps, dim=dim) - (numtaps - 1) / 2
    coefficients = ops.zeros_like(positions)
    for lower, upper in bands:
        coefficients = coefficients + upper * _sinc(upper * positions) - lower * _sinc(lower * positions)
    coefficients = coefficients * _fir_window(window, numtaps, dim)
    if scale:
        lower, upper = bands[0]
        reference = 0.0 if lower == 0 else (1.0 if upper == 1 else (lower + upper) / 2)
        gain = ops.sum(coefficients * ops.cos(math.pi * reference * positions))
        coefficients = coefficients / gain
    return coefficients