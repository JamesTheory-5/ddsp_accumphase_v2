# ddsp_accumphase_v2

MODULE NAME:
**ddsp_accumphase_v2**

DESCRIPTION:
Band-limited-capable, fully differentiable, pure-functional phase-accumulator oscillator in JAX.
It maintains a normalized phase in ([0, 1)), advances it by a frequency-dependent increment each sample, and produces one of several basic waveforms (sine, saw, square, triangle).
Parameters are passed as a tuple; state is a single JAX array and is updated via `lax.dynamic_update_slice`.
The design follows GDSP core style: functional DSP, no classes, fully JIT-/vmap-friendly.

---

### INPUTS

For `accumphase_v2_tick`:

* **x** : scalar JAX array (`jnp.ndarray`), interpreted as *frequency modulation* in Hz (additive offset to base frequency).
* **state** : JAX array, shape `(2,)`, oscillator state.
* **params** : tuple of JAX scalars,
  `(fs, base_freq_hz, radius, shape_id, smooth_coeff, bandlimited_flag)`

For `accumphase_v2_process`:

* **x** : 1D JAX array of shape `(T,)`, time series of frequency modulation in Hz, one value per sample.
* **state** : same as tick.
* **params** : same as tick.

---

### OUTPUTS

`accumphase_v2_tick(x, state, params)`:

* **y** : scalar JAX array, oscillator output sample.
* **new_state** : updated state array, same shape as `state`.

`accumphase_v2_process(x, state, params)`:

* **y** : 1D JAX array of length `T`, oscillator output.
* **final_state** : state after processing the entire block.

---

### STATE VARIABLES

State is a single JAX array:

```python
state = jnp.array([phase, freq_smooth], dtype=jnp.float32)
```

Tuple used externally is just this array; no dicts or classes.

* `state[0]` → `phase` (normalized, in `[0, 1)`)
* `state[1]` → `freq_smooth` (smoothed frequency in Hz)

All state updates use `lax.dynamic_update_slice`.

---

### EQUATIONS / MATH

Let:

* ( f_s ) = sampling rate (Hz)
* ( f_0 ) = base frequency (Hz)
* ( x[n] ) = frequency modulation (Hz offset)
* ( f_{\text{tgt}}[n] = f_0 + x[n] ) (target frequency)
* ( \alpha ) = `smooth_coeff` (0 → no smoothing, 1 → immediate tracking)
* ( f_{\text{smooth}}[n] ) = smoothed frequency
* ( \phi[n] ) = normalized phase in ([0, 1))

#### Frequency smoothing

One-pole smoothing:

[
f_{\text{smooth}}[n] = f_{\text{smooth}}[n-1] + \alpha \big(f_{\text{tgt}}[n] - f_{\text{smooth}}[n-1]\big)
]

If (\alpha = 0), frequency is effectively held constant (no update).
If (\alpha = 1), frequency immediately becomes ( f_{\text{tgt}}[n] ).

#### Phase increment and wrapping

Normalized phase increment:

[
\Delta \phi[n] = \frac{f_{\text{smooth}}[n]}{f_s}
]

Phase update with wrap:

[
\phi[n] = \operatorname{mod}\big(\phi[n-1] + \Delta \phi[n], 1.0\big)
]

(Using `jnp.mod`.)

Through-zero rule:
Phase is always wrapped to ([0, 1)) by `jnp.mod`. Negative or very large increments are allowed; wrap handles all cases.

#### Derived normalized bipolar phase

[
p[n] = 2\phi[n] - 1 \quad \in [-1, 1)
]

#### Waveforms

Let `shape_id` ∈ {0, 1, 2, 3}:

* **Sine** (ID = 0):

[
y_{\text{sin}}[n] = \sin\big(2\pi\phi[n]\big)
]

* **Saw** (ID = 1):

[
y_{\text{saw}}[n] = p[n]
]

* **Square** (ID = 2):

[
y_{\text{square}}[n] =
\begin{cases}
+1, & p[n] \ge 0 \
-1, & p[n] < 0
\end{cases}
]

implemented via `jnp.where`.

* **Triangle** (ID = 3):

[
y_{\text{tri}}[n] = 1 - 2|p[n]|
]

Waveform selection via one-hot mask (no Python branching, only `jnp.where`-derived masks):

[
\begin{aligned}
o_0 &= [\text{shape_id} = 0] \
o_1 &= [\text{shape_id} = 1] \
o_2 &= [\text{shape_id} = 2] \
o_3 &= [\text{shape_id} = 3] \
y_{\text{norm}}[n] &= o_0 y_{\text{sin}}[n] + o_1 y_{\text{saw}}[n]
+ o_2 y_{\text{square}}[n] + o_3 y_{\text{tri}}[n]
\end{aligned}
]

where `o_k` are 0/1 floats computed from equality comparisons.

#### Amplitude scaling

Let `radius` = output amplitude scalar:

[
y[n] = \text{radius} \cdot y_{\text{norm}}[n]
]

#### State update

Given old state vector ( s[n-1] = [\phi[n-1], f_{\text{smooth}}[n-1]] ):

[
s[n] = [\phi[n], f_{\text{smooth}}[n]]
]

implemented as:

```python
new_vals = jnp.stack([phi, f_smooth], axis=0)
state_next = lax.dynamic_update_slice(state_prev, new_vals, (0,))
```

#### Nonlinearities

Only basic nonlinear element is the square-wave sign via `jnp.where` (differentiable everywhere except at 0; JAX handles subgradients there).

#### Interpolation rules

No time or table interpolation in this version (direct analytic formulas). Parameter smoothing is via one-pole filter only.

---

### NOTES / CONSTRAINTS

* `smooth_coeff` ∈ ([0, 1]) recommended for stability and intuitive behavior.
* `base_freq_hz` and `freq_mod` should satisfy Nyquist: ( f_0 + x[n] < f_s / 2 ) to avoid aliasing.
* Saw/square/triangle are *not* band-limited in this base implementation (bandlimited_flag is presently a placeholder hook).
* All parameters and inputs inside JIT must be JAX scalars or arrays; do not pass Python numbers into jitted functions.
* All control flow in jitted code uses `jnp.where` or static tuple unpacking only (no Python `if`).

---

## FULL PYTHON FILE: `ddsp_accumphase_v2.py`

```python
"""
ddsp_accumphase_v2
==================

Pure functional, fully differentiable JAX phase-accumulator oscillator
in GDSP core style (no classes, no dicts, state as tuples/arrays only).

Features
--------
- Normalized phase accumulator in [0, 1)
- Frequency smoothing via one-pole filter
- Multiple shapes: sine, saw, square, triangle
- Functional DSP: tick(x, state, params) -> (y, new_state)
- Block processing via lax.scan in <name>_process()
- JIT-/vmap-friendly, no Python branching in jitted code
- State updates done via lax.dynamic_update_slice
- All shapes computed from analytic formulas (no lookup tables)

State layout
------------
state : jnp.ndarray, shape (2,)
    state[0] = phase       (normalized in [0, 1))
    state[1] = freq_smooth (smoothed frequency in Hz)

Params layout
-------------
params : tuple = (fs, base_freq_hz, radius, shape_id, smooth_coeff, bandlimited_flag)
    fs             : float32, sample rate (Hz)
    base_freq_hz   : float32, base oscillator frequency (Hz)
    radius         : float32, output amplitude
    shape_id       : int32, waveform selector (0=sin,1=saw,2=square,3=triangle)
    smooth_coeff   : float32, smoothing coefficient for frequency [0..1]
    bandlimited_flag: float32/int32, placeholder for future bandlimiting (0 or 1)

All parameters and inputs are JAX scalars/arrays when used under @jit.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax, jit


# ---------------------------------------------------------------------------
# Internal helpers (pure JAX, no side effects)
# ---------------------------------------------------------------------------

def _accumphase_v2_state_init(phase: jnp.ndarray,
                              freq_smooth: jnp.ndarray) -> jnp.ndarray:
    """Create initial state vector.

    Parameters
    ----------
    phase : jnp.ndarray scalar
        Initial phase in [0,1).
    freq_smooth : jnp.ndarray scalar
        Initial smoothed frequency (Hz).

    Returns
    -------
    state : jnp.ndarray shape (2,)
    """
    # Shape is static, no dynamic allocation based on runtime sizes.
    return jnp.stack(
        (jnp.asarray(phase, jnp.float32),
         jnp.asarray(freq_smooth, jnp.float32)),
        axis=0,
    )


def _accumphase_v2_unpack_state(state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Unpack state vector: (phase, freq_smooth)."""
    phase = state[0]
    freq_smooth = state[1]
    return phase, freq_smooth


def _accumphase_v2_pack_state(state: jnp.ndarray,
                              phase: jnp.ndarray,
                              freq_smooth: jnp.ndarray) -> jnp.ndarray:
    """Pack updated (phase, freq_smooth) into new state via dynamic_update_slice."""
    new_vals = jnp.stack(
        (jnp.asarray(phase, jnp.float32),
         jnp.asarray(freq_smooth, jnp.float32)),
        axis=0,
    )
    # Replace entire state (length=2) with new_vals at offset 0.
    return lax.dynamic_update_slice(state, new_vals, (0,))


def _accumphase_v2_wavetable(phase: jnp.ndarray,
                             phase_bipolar: jnp.ndarray,
                             shape_id: jnp.ndarray) -> jnp.ndarray:
    """Analytic waveform generator.

    Parameters
    ----------
    phase : jnp.ndarray scalar
        Normalized phase in [0,1).
    phase_bipolar : jnp.ndarray scalar
        Bipolar mapped phase: 2*phase - 1 in [-1,1).
    shape_id : jnp.ndarray scalar (int32)
        0=sine, 1=saw, 2=square, 3=triangle

    Returns
    -------
    y_norm : jnp.ndarray scalar
        Waveform sample in [-1,1], before amplitude scaling.
    """
    # --- basic shapes (analytic) ---
    # Sine in radians: sin(2*pi*phase)
    w_sin = jnp.sin(2.0 * jnp.pi * phase)

    # Saw: linear ramp in [-1,1)
    w_saw = phase_bipolar

    # Square: sign of bipolar phase
    w_square = jnp.where(phase_bipolar >= 0.0, 1.0, -1.0)

    # Triangle: 1 - 2*|p|
    w_tri = 1.0 - 2.0 * jnp.abs(phase_bipolar)

    # --- one-hot selection via comparisons (no dynamic indexing) ---
    sid = jnp.asarray(shape_id, jnp.int32)

    oh0 = (sid == jnp.int32(0)).astype(jnp.float32)
    oh1 = (sid == jnp.int32(1)).astype(jnp.float32)
    oh2 = (sid == jnp.int32(2)).astype(jnp.float32)
    oh3 = (sid == jnp.int32(3)).astype(jnp.float32)

    y_norm = (
        oh0 * w_sin +
        oh1 * w_saw +
        oh2 * w_square +
        oh3 * w_tri
    )

    return y_norm


# ---------------------------------------------------------------------------
# 1) accumphase_v2_init(...)
# ---------------------------------------------------------------------------

def accumphase_v2_init(
    fs: float,
    base_freq_hz: float,
    radius: float = 1.0,
    initial_phase: float = 0.0,
    shape_id: int = 0,
    smooth_coeff: float = 0.0,
    bandlimited_flag: float = 0.0,
) -> Tuple[jnp.ndarray, Tuple]:
    """Initialize accumphase_v2 oscillator state and params.

    Parameters
    ----------
    fs : float
        Sampling rate (Hz).
    base_freq_hz : float
        Base oscillator frequency (Hz).
    radius : float, optional
        Output amplitude, by default 1.0.
    initial_phase : float, optional
        Initial phase in [0,1), by default 0.0.
    shape_id : int, optional
        Waveform id: 0=sin, 1=saw, 2=square, 3=triangle; default 0.
    smooth_coeff : float, optional
        Frequency smoothing coefficient in [0,1], default 0.0 (no smoothing).
    bandlimited_flag : float, optional
        Placeholder for future bandlimiting control (0 or 1), default 0.0.

    Returns
    -------
    state : jnp.ndarray shape (2,)
        Initial state vector (phase, freq_smooth).
    params : tuple
        (fs, base_freq_hz, radius, shape_id, smooth_coeff, bandlimited_flag)
    """
    fs_j = jnp.asarray(fs, jnp.float32)
    base_freq_j = jnp.asarray(base_freq_hz, jnp.float32)
    radius_j = jnp.asarray(radius, jnp.float32)
    phase0 = jnp.mod(jnp.asarray(initial_phase, jnp.float32), jnp.float32(1.0))
    shape_j = jnp.asarray(shape_id, jnp.int32)
    smooth_j = jnp.asarray(smooth_coeff, jnp.float32)
    band_j = jnp.asarray(bandlimited_flag, jnp.float32)

    # Start smoothed frequency at base frequency
    state = _accumphase_v2_state_init(phase0, base_freq_j)
    params = (fs_j, base_freq_j, radius_j, shape_j, smooth_j, band_j)
    return state, params


# ---------------------------------------------------------------------------
# 2) accumphase_v2_update_state(...)
# ---------------------------------------------------------------------------

def accumphase_v2_update_state(
    state: jnp.ndarray,
    new_phase: jnp.ndarray,
    new_freq_smooth: jnp.ndarray,
) -> jnp.ndarray:
    """Update oscillator internal state (phase, freq_smooth).

    This function is purely functional, uses lax.dynamic_update_slice, and is
    jittable. It can be used to reset phase or change the smoothed frequency
    explicitly.

    Parameters
    ----------
    state : jnp.ndarray shape (2,)
        Current state.
    new_phase : jnp.ndarray scalar
        New phase (will be wrapped into [0,1) by the caller if desired).
    new_freq_smooth : jnp.ndarray scalar
        New smoothed frequency (Hz).

    Returns
    -------
    new_state : jnp.ndarray shape (2,)
        Updated state.
    """
    phase_wrapped = jnp.mod(new_phase, jnp.float32(1.0))
    return _accumphase_v2_pack_state(state, phase_wrapped, new_freq_smooth)


# ---------------------------------------------------------------------------
# 3) accumphase_v2_tick(x, state, params)
# ---------------------------------------------------------------------------

@jit
def accumphase_v2_tick(
    x: jnp.ndarray,
    state: jnp.ndarray,
    params: Tuple[jnp.ndarray, ...],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single-sample oscillator tick.

    Parameters
    ----------
    x : jnp.ndarray scalar
        Frequency modulation in Hz (additive offset to base frequency).
    state : jnp.ndarray shape (2,)
        Current oscillator state (phase, freq_smooth).
    params : tuple
        (fs, base_freq_hz, radius, shape_id, smooth_coeff, bandlimited_flag)

    Returns
    -------
    y : jnp.ndarray scalar
        Output sample.
    new_state : jnp.ndarray shape (2,)
        Updated state.
    """
    # Unpack params (static structure)
    fs, base_freq_hz, radius, shape_id, smooth_coeff, bandlimited_flag = params

    # Unpack state
    phase, freq_smooth_prev = _accumphase_v2_unpack_state(state)

    # Ensure x is float32
    x_j = jnp.asarray(x, jnp.float32)

    # Target frequency: base + modulation
    freq_target = base_freq_hz + x_j

    # One-pole smoothing: f_smooth = f_prev + a*(f_target - f_prev)
    # smooth_coeff in [0,1]
    freq_smooth = freq_smooth_prev + smooth_coeff * (freq_target - freq_smooth_prev)

    # Phase increment and wrapping
    inc = freq_smooth / fs
    phase_new = jnp.mod(phase + inc, jnp.float32(1.0))

    # Bipolar phase for non-sine shapes
    phase_bipolar = phase_new * jnp.float32(2.0) - jnp.float32(1.0)

    # Waveform (normalized in [-1,1])
    y_norm = _accumphase_v2_wavetable(phase_new, phase_bipolar, shape_id)

    # TODO: bandlimited_flag is a placeholder for future bandlimiting
    # For now, it is not used in the waveform computation to keep things simple.
    _ = bandlimited_flag  # avoid unused variable warning under jit

    # Apply amplitude
    y = y_norm * radius

    # Pack new state
    new_state = _accumphase_v2_pack_state(state, phase_new, freq_smooth)

    return y, new_state


# ---------------------------------------------------------------------------
# 4) accumphase_v2_process(x, state, params)
# ---------------------------------------------------------------------------

@jit
def accumphase_v2_process(
    x: jnp.ndarray,
    state: jnp.ndarray,
    params: Tuple[jnp.ndarray, ...],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Block processing wrapper using lax.scan.

    Parameters
    ----------
    x : jnp.ndarray shape (T,)
        Time-series of frequency modulation in Hz.
    state : jnp.ndarray shape (2,)
        Initial state.
    params : tuple
        (fs, base_freq_hz, radius, shape_id, smooth_coeff, bandlimited_flag)

    Returns
    -------
    y : jnp.ndarray shape (T,)
        Output waveform samples.
    final_state : jnp.ndarray shape (2,)
        State after processing the block.
    """
    def _step(carry_state, x_t):
        y_t, new_state = accumphase_v2_tick(x_t, carry_state, params)
        return new_state, y_t

    final_state, ys = lax.scan(_step, state, x)
    return ys, final_state


# ---------------------------------------------------------------------------
# 5) Smoke test, plotting, and listening example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Optional listening (sounddevice)
    try:
        import sounddevice as sd
        HAVE_SD = True
    except ImportError:
        HAVE_SD = False

    # -------------------------------------------------------
    # Smoke test
    # -------------------------------------------------------
    fs = 48_000.0
    f0 = 440.0
    radius = 0.5
    duration_sec = 0.02  # 20 ms preview
    N = int(fs * duration_sec)

    # Frequency modulation: zero for now (constant f0)
    # NOTE: shapes computed outside jit, using numpy then converted to jax
    fm_np = np.zeros(N, dtype=np.float32)
    fm = jnp.asarray(fm_np)

    # Initialize oscillator (saw wave for visualization)
    state, params = accumphase_v2_init(
        fs=fs,
        base_freq_hz=f0,
        radius=radius,
        initial_phase=0.0,
        shape_id=1,         # 0=sin,1=saw,2=square,3=triangle
        smooth_coeff=0.05,  # small smoothing
        bandlimited_flag=0.0,
    )

    # Run block
    y_block, final_state = accumphase_v2_process(fm, state, params)

    # Convert to numpy for plotting
    y_np = np.asarray(y_block)

    # -------------------------------------------------------
    # Plot preview (time domain)
    # -------------------------------------------------------
    t = np.arange(N) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t * 1000.0, y_np, label="Output")
    plt.title("ddsp_accumphase_v2 - Time-domain preview")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # Listen example (1 second, square wave)
    # -------------------------------------------------------
    duration_listen = 1.0
    N_listen = int(fs * duration_listen)

    fm_listen_np = np.zeros(N_listen, dtype=np.float32)
    fm_listen = jnp.asarray(fm_listen_np)

    state_sq, params_sq = accumphase_v2_init(
        fs=fs,
        base_freq_hz=110.0,   # lower pitch
        radius=0.3,
        initial_phase=0.0,
        shape_id=2,           # square
        smooth_coeff=0.01,
        bandlimited_flag=0.0,
    )

    y_listen_block, _ = accumphase_v2_process(fm_listen, state_sq, params_sq)
    y_listen = np.asarray(y_listen_block)

    if HAVE_SD:
        print("Playing 1 second of square wave at 110 Hz...")
        sd.play(y_listen, int(fs))
        sd.wait()
    else:
        print("sounddevice not installed; skipping audio playback.")
```

---

### Next things you could do

Some natural follow-ups for this module:

1. **Add bandlimiting (PolyBLEP)** using the `bandlimited_flag`:

   * Implement `_polyblep(t, dt)` and modify `_accumphase_v2_wavetable` for saw/square/triangle.
2. **Expose more modulation inputs**:

   * Make `x` a tuple `(freq_mod, amp_mod)` and adjust `tick` accordingly.
3. **Multi-voice support**:

   * Use `vmap` over `accumphase_v2_process` for polyphonic synthesis.
4. **Gradient-based control**:

   * Integrate into a larger DDSP graph and verify gradients flow correctly through frequency and phase.

If you want, I can do a **PolyBLEP-enabled v2.1** next, slotted into this exact API.
