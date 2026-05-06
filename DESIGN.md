# Design of TSDR

This document contains the architectural and visual design of the Terminal Software Defined Radio project.

By documenting this, we have a clear vision of what the project should become.

## Goals

- An all-in-one toolkit related to the exploration of the radio spectrum.
- Terminal based interface, but using features (images) if possible.
- A power tool, meant for power users.
- Prefer simplicity to flexibility.
- Implementing as many demodulators as possible, as long as we can implement them efficiently enough
- Use of Python for a low barrier to read and modify the code.
- "Agentic coding" friendly, with careful reviews.

## Non-Goals

- No plugin or module system. All demodulators are in this codebase.
- No programming languages other than Python. Cython is only allowed if performance or interop deems it necessary.

## Module Structure

- `core` - Non-UI logic
- `core.events` - Event bus.
- `core.sdr` - Core SDR framework, including engine and device context.
- `core.sdr.config` - SDRConfig and DeviceConfig data classes.
- `core.sdr.pipeline` - Pipeline framework.
- `core.sdr.pipeline.stages` - Pipeline implementations.
- `core.sdr.workers` - Worker framework and implementations
- `radio` - Signal-processing content, independent of the SDR runtime
- `radio.demodulators` - All demodulators. Registry in registry.py of all available demodulators.
- `radio.decoders` - Protocol decoders (RDS, FLEX, ADSB, DAB, DMR, TETRA)
- `radio.dsp` - DSP primitives to reuse in demodulators and decoders
- `radio.vocoder` - Speech codecs (AMBE, ACELP)
- `tui` - TUI core logic and entrypoint
- `tui.console` - Console-related
- `tui.commands` - Commands for console
- `tui.widgets` - Specific widgets
- `debug` - Debug related code, such as oob visualisation

The main split is between `core` and `tui`.

The core is the framework that does all business logic. Such as device communication, worker orchestration, sample
processing, and event publishing.

The `tui` uses `core` to show a visible UI to the user, in the terminal. `core` cannot import `tui`.

## Architecture

### System Overview

The core orchestration is handled by the engine class (SDREngine). It is also the entrypoint for modifications to the
engine configuration.

The engine API contains functions such as:

- add/remove/modify pipelines of devices
- modify configuration of devices
- add/remove device instances
- etc.

A combination of device, its pipelines, and its workers is managed by the device context class (SDRDeviceContext). The
device context is an internal unit: it encapsulates all state, configuration (with DeviceConfig), and resources for a
single device and its subsystems. External code, including the TUI, never accesses a device context directly. Instead,
the engine exposes methods that locate the relevant device context and delegate to it. The engine also coordinates
across device contexts, such as ensuring only one audio worker is actively playing.

The engine can manage multiple device contexts (and thus multiple devices). For ease of use, there is the concept of the
focused device. The focused device is used if the specific device is not specified.

### Data and memory

The preferred encapsulation of arrays of samples is the Numpy array. Numpy and related
libraries such as Scipy and Numba are preferred for compute-intensive operations over native Python constructs.

Data passing through subsystems is encapsulated by the thread-safe SamplesBatch dataclass. It carries data through the
subsystems. Its fields are populated by different subsystems. For example, the input samples field is populated by the
IO worker, and the
FFT data field by the FFT stage in the pipeline worker, and the then-active frequency and sample rate by the IO worker.

It is by design that the SamplesBatch class is reused by many subsystems. Therefore, the SamplesBatch class contains
many different fields, and consumers check which fields are populated. This prevents a sprawl of different data
containers and data-passing patterns.

### Configuration

The engine, device context, pipelines, workers, and many more systems are configurable. This can be difficult to manage,
so care is taken when working with configuration.

All configuration is typed, we never store config in untyped data structures.

Configuration can and is modified at runtime.

Configuration is never duplicated in different places.

The entrypoint to configuration modification is the engine. New configuration flows from the engine, to the device
context, to its subsystems.

The entrypoint to get all current active configuration in a thread-safe manner is from the engine.

These are the layers of configuration:

- The engine's SDRConfig. SDRConfig contains core global configuration.
- DeviceConfig that is part of a device context. It has configuration for hardware (gain control, frequency, sample
  rate, etc), processing (fft size, etc), etc.
- The stages of the pipelines of the devices.
- TUI-specific configuration related to the user interface.

Some configuration is pre-computed from other configuration, and therefore read-only. An example is the active
demodulator, that is derived from which stages are active in the pipeline.

Propagation of configuration works in a chain-of-responsibility. The engine tells the device context, which tells its
workers, pipelines and other subsystems.

Changes in configuration are notified to the TUI and other parts of the program by means of events. The receivers use
the thread-safe access method of the engine to get the active configuration, as described in "Thread Safety".

### Event-Driven Communication

The program contains an event bus.

The event bus is used to notify of, for example, updates, data or configuration changes.

### Thread Safety

The app is compatible with free-threaded Python.

Inter-thread communication is exclusively handled by Python's thread-safe Queue with immutable or otherwise thread-safe
data. This is used, for example, to propagate state and pass on sample data.

### Worker framework

Each worker owns a thread. The worker abstraction revolves around this. Workers belong to a device context.

#### I/O Worker

The IO worker gets samples from and sets config on the SDR hardware, and has exclusive access to it.

#### Pipeline Worker

The pipeline worker processes samples and works with the pipelines of the device context, and is generally the CPU
intensive part of the program.

#### Audio Worker

The audio worker manages the sound device for playing audio samples as produced by the processing parts.
Detail: when there are multiple active device contexts, only one audio worker can be actively playing.

#### Lifecycle & Health

The lifecycle of workers is managed by the worker framework. It controls the lifecycle (started, stopped, starting,
error, etc).

The worker framework manages the intricacies of managing the threads that back the workers.

## Devices

A device is a SDR-compatible device abstraction.

It can be a real device available through the operating system APIs, but also a device available only through the
network from some TCP or UDP address, or over the internet through some HTTP API.

It can also be a mock device, looping IQ samples from some pre-recorded sample file.

All available device implementations are registered in the device registry file.

### Device Abstraction

For all supported devices a class is implemented that extends an abstract class.

## Signal Processing

### Pipeline Model

The processing of signals happens in a pipeline.

The pipeline contains a linear set of stages.

Each stage processes the input and produces output. It populates SamplesBatch as described earlier.

A device context can have multiple pipelines. It is common to have two, one for the visualization pipeline, and another
for audio.

### Built-in Stages

Some common pipeline stages are:

- The FFT stage, that calculates the fast fourier transform.
- The frequency shift stage, if configuration demands a frequency shift.
- The stats stage, calculating statistics such as signal-to-noise ratio.
- The demodulator stage, that calls one of the demodulators.

### Demodulators

Demodulators turn signals into useful information.

For example, the WFM demodulator has as input samples. Its output is audio samples, and possibly RDS messages.

## User Interface

The user interface is a terminal based UI, also known as a TUI.

The TUI is interactive, it is clickable and scrollable.

### Layout

The general layout is, from top to bottom: tuner, spectrum, waterfall, console, status.
Then, there are sidebars for statistics, performance metrics. These are hidden by default, and can be toggled to
display.

### Events

Updates to the TUI come from the event-bus. A Textual adapter converts events from the core event bus system to
Textual's event system.

### Image mode

An optional but highly useful feature is the image mode. When enabled, the Kitty terminal graphics protocol is used to
efficiently show images and graphics to the user. The updating and rendering of these images is pushed to 60+fps when
optimized with shared memory transfers.

### Tuner & Stats

As the first widget it is prominent in view. It shows the active frequency, in big ascii letters.

It is also the thing that displays situational awareness. What's the current pipeline/demod? What's the signal-to-noise
ratio? What's the main configuration of the device such as gain?

### Spectrum and waterfall Display

The spectrum shows the active spectrum/fft bin view and updates frequently.

The waterfall shows the same data, but over time.

Both are adjusted with min/max decibel and zoom levels.

### Console & Commands

The very powerful feature is the console widget. This emulates a shell, including proper history, tab-to-complete,
readline editing, and aliases.

While the TUI is controlled via clicking, scrolling and shortcuts, the console commands are really the most powerful of
the app. By using commands much more configuration and features can be exposed, without needing to build a TUI interface
for it.

Commands use argparse to provide a native feel.

Commands generally use the engine API to perform their tasks.

### Keyboard Controls

Almost everything of the TUI interface is controlled by keyboard controls. Such as opening the console, adjusting gain
and the zoom, min and max parameters of the spectrum and waterfall.

## Persistence

Some persistence is available through the platformdirs library, mainly logs and preferences.

### Preferences

Some state of the configuration is stored in the preference. This is to allow restoring configuration on startup for
ease of use.
