# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unity library providing Fast Fourier Transform (FFT) with dual CPU and GPU implementations. Unity 6000.0.23f1.

## Architecture

**Namespace:** `Moderato.Mathematics`

**Assembly dependency chain:** `Moderato.Buffers` -> `Moderato.Mathematics` -> `Tests`

### CPU Path (Burst-compiled)
- `Assets/Moderato/Mathematics/FastFourierTransform/Core/FastFourierTransform.cs` — Internal `IJob` struct implementing Cooley-Tukey radix-2 FFT with bit-reversal. Burst-compiled, uses `System.Numerics.Complex` and `NativeArray`.
- `Assets/Moderato/Mathematics/FastFourierTransform/Processor/FFT.cs` — Public static API (`FFT.Transform`) with overloads for `float[]`, `double[]`, `Complex[]`. Schedules the job synchronously and returns managed arrays.
- `Assets/Moderato/Buffers/ArrayPool.cs` — `IDisposable` wrapper around `System.Buffers.ArrayPool<T>` used for temp allocations in FFT.

### GPU Path (Compute Shader)
- `Assets/Moderato/Mathematics/FastFourierTransform/Processor/FFT_GPU.cs` — Stateful driver class using Stockham auto-sort FFT. Ping-pong double-buffering, twiddle factor precomputation, LDS-optimized kernel selection. Three execution paths: `FFTLocal` (single-dispatch, N≤2048), `FFTStageLDS` (per-stage LDS), `FFTStage` (per-stage fallback). Supports `CommandBuffer` caching via `ForwardCached()`/`InverseCached()` and async readback via `GetResultAsync()`. Constants `THREADS=256`, `TILE=1024`, `LOCAL_N=2048` must match the compute shader.
- `Assets/Resources/StockhamFFT_Tiled.compute` — GPU compute shader with three kernels: `FFTLocal` (all stages in LDS, N≤2048, 32 KiB TGSM, inline sincos twiddle), `FFTStage` (per-stage fallback), `FFTStageLDS` (per-stage groupshared tiling). Loaded via `Resources.Load`.

### Shared
- `Assets/Moderato/Mathematics/Moderato.Mathematics.cs` — `Window` enum (Rectangular, Triangle, Hamming, Hanning, Blackman, BlackmanHarris).

## Running Tests

Tests use NUnit via Unity Test Framework with Unity Performance Testing Extension. Run from Unity Test Runner (Window > General > Test Runner) or CLI:

```
Unity.exe -runTests -batchmode -projectPath . -testResults TestResults/results.xml -testPlatform EditMode
```

Test files are in `Assets/Scripts/Tests/` and require the `UNITY_INCLUDE_TESTS` define (set in `Tests.asmdef`).

## Key Constraints

- FFT input is zero-padded to next power of 2 internally; GPU path requires power-of-2 input directly.
- GPU `THREADS` and `TILE` constants in `FFT_GPU.cs` must stay in sync with the compute shader defines.
- CPU FFT output is half-spectrum (Nyquist-pruned, length N/2).
