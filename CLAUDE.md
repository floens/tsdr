# TSDR Architecture

## Overview

TSDR is a Textual TUI for Software Defined Radio with a command-line interface. It provides real-time spectrum visualization, waterfall display, and audio demodulation.

## Design

The design of the architecture, user interface, goals, non-goals, and vision is described in DESIGN.md.
Follow DESIGN.md strictly.

## Project Structure

```
src/tsdr/
├── core/
│   ├── events/       - EventBus pub-sub system for decoupled communication
│   ├── sdr/          - SDR runtime: engine, config, workers, pipeline, audio
│   └── workers.py    - Generic worker thread framework
├── devices/          - Hardware abstraction: SDR device drivers and factory
├── radio/            - Signal-processing content (no SDR-runtime deps)
│   ├── dsp/          - DSP primitives (filters, FFT, Costas, Mueller-Muller, FM disc.)
│   ├── demodulators/ - WFM, NFM, AM, SSB, CW
│   ├── decoders/     - Protocol decoders (RDS, FLEX, ADSB, DAB, DMR, TETRA, Morse)
│   └── vocoder/      - Speech codecs (AMBE, ACELP)
└── tui/
    ├── commands/     - Command framework, registry, and all commands
    ├── console/      - Console widget and terminal input
    └── widgets/      - Textual widgets
```

## Core Architecture

- **SDREngine**: Coordinator managing devices, pipelines, and audio output
- **SDRConfig**: Engine-global immutable config (FFT, display, audio). Lives on `SDREngine.config`.
- **DeviceConfig**: Per-device immutable config (frequency, gain, sample rate). Lives on `SDRDeviceContext.config`.
- **SDRDeviceContext**: Per-device state including config, pipelines, workers, and queues. Config access is thread-safe via a lock-protected property.

All inter-thread data is immutable (frozen dataclasses) for performance—no locks on the hot path.

## Worker Architecture

Each device runs two worker threads:

- **I/O Worker**: High-priority, minimal processing (read hardware + enqueue only)
- **Pipeline Worker**: Executes processing stages, publishes visualization events

Audio runs in a separate worker:

- **Audio Worker**: Separate thread per audio source, outputs via sounddevice

Queue-based communication between workers (queues live on `SDRDeviceContext`):
- `sample_queue`: I/O → Pipeline worker (raw IQ samples)
- `control_queue`: Main thread → I/O worker (`DeviceConfig` updates → hardware reconfig)
- `pipeline_control_queue`: Main thread → Pipeline worker (`DeviceConfig`/`SDRConfig` → stages get `on_config_change()`)
- `audio_queue`: Pipeline → Audio worker (`AudioBatch` payloads)

## Pipeline Pacing

Wall-clock pacers (where Python schedules "the next read") must advance
by absolute slots, not by actual return time:

```python
self._next_ts += period
if time.monotonic() < self._next_ts:
    time.sleep(self._next_ts - time.monotonic())
if time.monotonic() > self._next_ts + RESET_THRESHOLD:
    self._next_ts = time.monotonic()
```

`max(self._next_ts, time.monotonic()) + period` bakes `time.sleep` slop
into every next deadline and compounds it indefinitely, enough to starve
downstream consumers. Absolute scheduling self-corrects: a long sleep in
cycle N is offset by a shorter sleep in cycle N+1. The reset escape
handles cases where the pacer was paused much longer than one cycle
(e.g. rebuffer wait).

Min-interval throttles (`now - last >= interval`) are unrelated and don't
compound slop. Use them freely for event rate-limiting.

## UI and Event System

Textual app with custom widgets: SpectrumWidget, StatsWidget, TunerWidget, StatusBar.

**EventBus**: Type-based pub-sub system. Workers publish events directly.

**TextualEventAdapter**: Bridges EventBus to Textual messages.

Event flow: Worker → EventBus → TextualEventAdapter → Textual Message → Widget

## Command System

Commands extend the `Command` ABC from `tui/commands/base.py`. Required: `description` (property) and `run(args) -> str`. Optional: `configure(parser)` to add argparse arguments, and `complete(tokens, prefix, *, flag=None, subcommand=None) -> list[Completion]` for dynamic completions (positional values, flag values, or subcommand-scoped completions).

`CommandParser` wraps argparse and raises `CommandExit` instead of calling `sys.exit()`, keeping errors in-app.

`tui/commands/registry.py` exposes a module-level `COMMANDS: dict[str, Command]`, `register(name, command)`, and `execute(input_line)` which shlex-splits the line and dispatches to the matching command's `Command.execute(argv)`. Autocomplete uses fuzzy subsequence matching with priority for exact / prefix matches.

Commands are organized by domain:
- `tui/commands/builtin/` — echo, exit, help, paths, trace
- `tui/commands/sdr/` — add, bandplan, config, dab, demod, focus, frequency, list, memory, pipeline, record, remove, scan, squelch, start, stop
- `tui/commands/audio/` — audio

Adding a new command:
1. Create a class extending `Command` with at minimum `description` and `run()`. Add `configure()` if it takes arguments and `complete()` for dynamic completions.
2. Register it in `tui/commands/registry.py` with `register("name", MyCommand())`.

## Runtime Configuration (`config`)

The engine exposes two entrypoints, one per config layer:

- `SDREngine.update_global_config(**changes)` — `SDRConfig` (engine-wide: FFT, display, audio).
- `SDREngine.update_device_config(device_id, **changes)` — `DeviceConfig` (per-device: frequency, gain, sample rate, pipelines).

Device config flow (`update_device_config`):
1. Engine delegates to `SDRDeviceContext.update_config(**changes)`.
2. Context builds `new_config = old.with_changes(**changes)`, calls `validate()`, swaps the reference under a lock.
3. If `pipelines` changed, context re-materializes pipeline stages (reusing instances at matching positions to preserve state).
4. Context puts `new_config` on `control_queue` (I/O worker, hardware reconfig) and `pipeline_control_queue` (pipeline worker, stages get `on_config_change()`).
5. Engine publishes `ConfigChangedEvent` for UI subscribers.

Global config flow (`update_global_config`):
1. Engine builds `self.config = self.config.with_changes(**changes)` and calls `validate()`.
2. Engine notifies audio workers synchronously via `worker.on_config_change(self.config)`.
3. For each device: engine publishes `ConfigChangedEvent` and calls `context.notify_global_config_change(self.config)`, which puts the new `SDRConfig` on `pipeline_control_queue` only (no I/O worker, no hardware impact).

Adding a configurable parameter:
1. Add field to `SDRConfig` (global) or `DeviceConfig` (per-device).
2. Add field to `GlobalConfigChanges` or `DeviceConfigChanges` TypedDict (these drive `**changes` typing).
3. Implement `on_config_change()` in relevant stage(s) — check the config type with `isinstance()` since pipeline stages receive both `SDRConfig` and `DeviceConfig`.
4. Add CLI argument parsing in the relevant command.
5. Call `engine.update_global_config()` or `engine.update_device_config(device_id, ...)` from the command.

## Pipeline Architecture

Each device owns multiple peer pipelines via `SDRDeviceContext.pipelines: dict[str, ProcessingPipeline]`. Default composition (see `DEFAULT_PIPELINES` in `core/sdr/config.py` and `SDREngine.set_audio_demod`):

```
SDRDeviceContext.pipelines
    "visualization" → AGC → FFT → EventEmitter
    "audio"         → [FrequencyShift] → Demodulator → EventEmitter   (added on demod)
```

The audio pipeline is created on demand when an audio demodulator is selected; `FrequencyShift` is included only when an offset is set. Channel filtering and decimation happen *inside* the demodulator class (e.g. `WidebandFMDemodulator._setup_channel_filter`), not as separate stages.

All pipelines receive the same input data and execute independently. No parent/child nesting.

Use `SDREngine.add_pipeline()` / `remove_pipeline()` to manage pipelines dynamically.

## Pipeline Modification (`pipeline`)

Stages can be listed, added, or removed dynamically at runtime via the `pipeline` command.

Available stages: `agc`, `demodulator`, `event_emitter`, `fft`, `frequency_shift`, `record` (see `StageType` in `core/sdr/config.py`).

## Demodulators

- **WFM**: Broadcast FM (±75 kHz deviation, 50/75 µs de-emphasis, stereo via PLL-locked 19 kHz pilot, RDS decode)
- **NFM**: Narrowband FM (configurable deviation, 750 µs de-emphasis, squelch)
- **AM**: Envelope detection with DC blocker, AGC, audio LPF, squelch
- **USB/LSB**: Single sideband via shift-LPF-unshift (rejects opposite sideband)
- **CW**: Morse code with explicit BFO; Morse text decoder runs alongside audio output

All demodulators inherit from `radio/demodulators/__init__.py:Demodulator`. The base class owns the `_audio_batches` buffer and provides `_emit_audio(samples, sample_rate, timestamp, *, stereo=False)` and `get_audio()`. Subclass `__init__` must call `super().__init__()`; subclass `reset()` should call `super().reset()` to clear the buffer.

## Testing

Tests use real recorded IQ samples stored as `.cu8.zst` in `tests/samples/`. All tests load data via `load_iq()` from `core.sdr.io`, which handles decompression and format conversion to complex64 transparently. The `conftest.py` `run_pipeline` fixture runs IQ files through the full engine pipeline for end-to-end tests.

Tests cover DSP primitives (CostasLoop, MuellerMuller, FMDiscriminator), decoder correctness (RDS group decoding, FLEX sync/BCH/frame decode), demodulator output (WFM stereo detection), and signal processing diagnostics (chunk boundary artifacts, AGC behavior, resampling edge effects).

DSP building blocks in `radio.dsp` (CostasLoop, MuellerMuller, FMDiscriminator) are shared across decoders. New decoders should reuse these rather than reimplementing timing recovery, carrier tracking, or FM discrimination.

## Developing Decoders

Start by recording a sample of the target signal with the in-app `record` command — you need a known-good recording that contains the data you want to decode. Sample files live in `tests/samples/` (gitignored, `.cu8.zst`).

Write tests against this sample that validate each stage of the decoder pipeline (e.g. FM demod produces expected symbol distribution, sync words are detected, frames decode correctly). Tests are the primary verification tool since the TUI cannot be tested by the agent.

Decoder development is empirical and iterative. Build the pipeline in stages — decimation, demodulation, symbol recovery, sync detection, frame parsing — verifying each stage with tests before moving to the next. DSP parameters (filter cutoffs, timing gains, AGC constants) often need tuning against real signals, so expect to experiment and adjust based on test output.

A sample file is a hard prerequisite — do not write decoder code until it exists. Implement one stage at a time, testing against the real sample before moving to the next.

**I/Q and sign conventions**: PSK demodulators require careful attention to real/imaginary mapping and sign conventions. The mapping from complex symbols to soft bits (which component is I vs Q, and the sign of each) is coupled to downstream processing — constellation labeling, differential encoding direction, Viterbi polynomial convention (standard vs bit-reversed), and bit ordering all interact. A wrong convention can appear to work under high redundancy but fail when the code rate drops. Always cross-reference multiple working implementations and verify end-to-end at the lowest code rate the system must support.

## DSP Performance

### Numba kernels (`radio.dsp/_kernels.py`)

All hot-path DSP primitives live in `_kernels.py` as `@nb.njit(cache=True)` functions. Use `fastmath=True` only when the algorithm tolerates relaxed floating-point semantics (filters, discriminators -- not CRC/sync).

**When to use numba:**
- Inherently scalar algorithms with loop-carried state: PLL, FM discriminator, CRC, symbol timing recovery (Mueller-Muller), carrier tracking (Costas loop)
- Fused single-pass kernels that eliminate multiple numpy intermediate arrays on small-to-medium data (< ~4K elements): IQ conversion, frequency shifting, IQ metrics
- Decimating FIR filters where skipping outputs saves significant compute (decimation factor > 1)

**When to use numpy instead:**
- Bulk element-wise operations on large arrays (> ~4K elements): windowing, FFT power spectrum, `np.log10`, `np.mean`, `np.std`. Numpy uses AVX/SIMD intrinsics that process 4-8 elements per cycle; a numba scalar loop cannot compete.
- Operations with optimized BLAS/LAPACK implementations

**dtype conventions:**
- Hot-path audio/filter data: `float32`. Never promote to `float64` unless numerical stability requires it (high-order IIR filters).
- IQ data: `complex64`. Avoid Python `1j` literal in array expressions -- it is `complex128` and promotes the entire result. Use `np.complex64(1j)` or numba kernels that construct complex64 directly.
- Numba kernel outputs: always specify dtype explicitly (`np.empty(n, dtype=np.float32)`).

### Streaming filter classes

All have a `process(x) -> y` method and a `reset()` method, with state persisting across calls.

- **`StreamingFilter`**: Wraps `_lfilter_fir_f32` (FIR, Direct Form II Transposed) or `_lfilter_iir` / `_lfilter_iir_f32` (IIR). Best for non-decimating filters -- no buffer copy overhead, state updated in-place. Use `dtype=np.float32` for IIR filters that don't need float64 precision (de-emphasis, AGC).

- **`StreamingDecimFilter`**: Wraps `fir_decim_f32_into` / `fir_decim_c64_into` (FIR, Direct Form I with pre-allocated scratch buffers). Best for decimating filters (`m > 1`) -- computes only the decimated outputs. Pre-allocates padded scratch and output buffers to avoid per-call allocation. **Do not use with `m=1`** -- the padded buffer copy overhead makes it slower than `StreamingFilter` for the non-decimating case.

- **`StreamingPolyphaseResampler`**: Rational resampling (up/down) via polyphase decomposition.

- **`DCBlocker`**: Single-pole IIR high-pass at a configurable cutoff (typ. 16 Hz). Wraps `_dc_blocker_f32`. Used post-envelope in AM and post-real-projection in SSB to remove DC build-up.

Adjacent stateful kernels with the same `process()`/`reset()` shape but not strictly filters: `AGC` (envelope follower + gain), `SquelchGate` (power-gated mute envelope), `FMDiscriminator` (conjugate-product FM demod), `MuellerMuller` (symbol timing recovery), `CostasLoop` (carrier tracking).

### Allocation discipline on hot paths

- Pre-allocate output buffers and reuse across calls (see `StreamingDecimFilter._y_out`, `RDSDecoder._freq_shift_buf`)
- Use `np.clip(..., out=x)` for in-place operations
- Use `np.dot(x, x)` instead of `np.mean(x**2)` to avoid temporary squared arrays
- Return views from buffers when the caller consumes data before the next write (see `CircularBuffer._get_recent`)

## Error Handling

Let errors crash with full tracebacks during development. Catch specific exceptions only when there's a reason to handle them.

## Code style

Use pre-commit to verify code style.

Do not use broad exception clauses like `except Exception:`. Exception: when catching arbitrary errors raised by external libraries or hardware drivers (e.g. SoapySDR, rtlsdr, sounddevice), narrowing is impractical because the upstream code does not document its exception types — `except Exception:` with a `# noqa: BLE001` and a justifying comment is acceptable there.

Do not use imports on function level. Do not use `# noqa: PLC0415` to circumvent this requirement. Exception: importing optional third-party dependencies that may not be installed (e.g. `SoapySDR`, `rtlsdr`) — these belong inside the function that needs them so that the absence of the package raises an `ImportError` only when the user tries to use that backend.

## Comments

Use sparingly. Add comments when:
- Code is complex or non-obvious
- Code contains math/DSP algorithms
Avoid narrative comments.

## Verification

- Code: `uv run pre-commit run --all-files`
- UI: Manual testing by user (agent cannot test UI)
- Logs: `tsdr.log` in the platformdirs user config dir (see `config_dir()` in `core/storage.py`)
