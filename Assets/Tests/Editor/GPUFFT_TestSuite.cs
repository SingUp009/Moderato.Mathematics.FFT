using System;
using System.Collections;
using System.Globalization;
using System.Linq;
using Moderato.Mathematics;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

public class GPUFFT_TestSuite
{
    const int THREADS = 256;
    const int TILE = 1024;

    ComputeShader _cs;

    int _kLDS, _kStage, _kLocal;

    [OneTimeSetUp]
    public void OneTimeSetup()
    {
        _cs = Resources.Load<ComputeShader>("StockhamFFT_Tiled");
        Assert.IsNotNull(_cs, "ComputeShader 'Resources/StockhamFFT_Tiled.compute' not found.");
        _kLDS = _cs.FindKernel("FFTStageLDS");
        _kStage = _cs.FindKernel("FFTStage");
        _kLocal = _cs.FindKernel("FFTLocal");
        Assert.That(_kLDS >= 0 && _kStage >= 0, "Kernels FFTStageLDS/FFTStage not found.");
        Assert.That(_kLocal >= 0, "Kernel FFTLocal not found.");
    }

    // ======= Small helpers =======

    static Vector2[] MakeSineReal(int N, int batches, int bin, float amp = 1f)
    {
        var x = new Vector2[N * batches];
        for (int n = 0; n < N; n++)
        {
            float v = amp * Mathf.Sin(2f * Mathf.PI * bin * n / N);
            x[n] = new Vector2(v, 0f); // batch 0 only -> enough to validate math
        }
        return x;
    }

    static Vector2[] MakeConstantReal(int N, int batches, float c)
    {
        var x = new Vector2[N * batches];
        for (int n = 0; n < N; n++) x[n] = new Vector2(c, 0f);
        return x;
    }

    static Vector2[] MakeNyquistReal(int N, int batches, float amp = 1f)
    {
        // cos(pi n) = (-1)^n -> single-sided Nyquist spike
        var x = new Vector2[N * batches];
        for (int n = 0; n < N; n++)
        {
            float v = amp * Mathf.Cos(Mathf.PI * n);
            x[n] = new Vector2(v, 0f);
        }
        return x;
    }

    static System.Random rng = new System.Random(1234);

    static Vector2[] MakeRandomReal(int N, int batches, float scale = 1f)
    {
        var x = new Vector2[N * batches];
        for (int n = 0; n < N; n++)
        {
            float v = (float)(scale * (rng.NextDouble() * 2 - 1));
            x[n] = new Vector2(v, 0f);
        }
        return x;
    }

    static Vector2[] MakeRandomComplex(int N, int batches, float scale = 1f)
    {
        var x = new Vector2[N * batches];
        for (int n = 0; n < N; n++)
        {
            float re = (float)(scale * (rng.NextDouble() * 2 - 1));
            float im = (float)(scale * (rng.NextDouble() * 2 - 1));
            x[n] = new Vector2(re, im);
        }
        return x;
    }

    static float[] ToSingleSidedAmplitude(Vector2[] spec, int N, int batches, int b = 0)
    {
        int off = b * N;
        var amp = new float[N / 2 + 1];
        amp[0] = spec[off + 0].magnitude / N;
        for (int k = 1; k < N / 2; ++k)
            amp[k] = 2f * spec[off + k].magnitude / N;
        amp[N / 2] = spec[off + N / 2].magnitude / N;
        return amp;
    }

    static float RMSE(float[] a, float[] b)
    {
        double s = 0; int n = a.Length;
        for (int i = 0; i < n; i++) { double d = a[i] - b[i]; s += d * d; }
        return (float)Math.Sqrt(s / n);
    }

    static float RMSE(Vector2[] a, Vector2[] b)
    {
        double s = 0; int n = a.Length;
        for (int i = 0; i < n; i++)
        {
            double dr = a[i].x - b[i].x;
            double di = a[i].y - b[i].y;
            s += dr * dr + di * di;
        }
        return (float)Math.Sqrt(s / n);
    }

    // ======= GPU runner (encapsulated driver with LDS toggle) =======

    enum ExecMode { FallbackOnly, LDSAuto }

    Vector2[] RunGPUForward(Vector2[] input, int N, int batches, ExecMode mode)
    {
        int stages = 0; for (int t = N; t > 1; t >>= 1) stages++;

        using var bufPing = new ComputeBuffer(N * batches, sizeof(float) * 2);
        using var bufPong = new ComputeBuffer(N * batches, sizeof(float) * 2);
        using var twiddle = new ComputeBuffer(N / 2, sizeof(float) * 2);

        // Upload input
        bufPing.SetData(input);

        // Precompute W_N^r = cos(2πr/N) - i sin(2πr/N)
        var tw = new Vector2[N / 2];
        double twoPiOverN = 2.0 * Math.PI / N;
        for (int r = 0; r < tw.Length; r++) { double ang = twoPiOverN * r; tw[r] = new Vector2((float)Math.Cos(ang), (float)-Math.Sin(ang)); }
        twiddle.SetData(tw);

        // Static params
        _cs.SetInt("N", N);
        _cs.SetInt("BatchCount", batches);
        _cs.SetInt("Stages", stages);
        _cs.SetFloat("InvN", 1.0f / N);
        _cs.SetInt("Direction", +1);

        _cs.SetBuffer(_kLDS, "Twiddle", twiddle);
        _cs.SetBuffer(_kStage, "Twiddle", twiddle);

        bool resultInPing = true;

        for (int s = 0; s < stages; s++)
        {
            var src = resultInPing ? bufPing : bufPong;
            var dst = resultInPing ? bufPong : bufPing;

            _cs.SetInt("StageDynamic", s);
            _cs.SetInt("ApplyScaleDynamic", 0);

            // LDS condition
            int m2 = 1 << (s + 1);
            bool canTile = (N % TILE) == 0 && (m2 <= TILE) && ((TILE % m2) == 0);

            if (mode == ExecMode.LDSAuto && canTile)
            {
                int tilesPerBatch = N / TILE;
                int groups = tilesPerBatch * batches;

                _cs.SetBuffer(_kLDS, "Ping", src);
                _cs.SetBuffer(_kLDS, "Pong", dst);
                _cs.Dispatch(_kLDS, groups, 1, 1);
            }
            else
            {
                int totalButterflies = batches * (N >> 1);
                int groups = Mathf.Max(1, (totalButterflies + THREADS - 1) / THREADS);
                int totalThreads = groups * THREADS;

                _cs.SetInt("ThreadCount", totalThreads);
                _cs.SetBuffer(_kStage, "Ping", src);
                _cs.SetBuffer(_kStage, "Pong", dst);
                _cs.Dispatch(_kStage, groups, 1, 1);
            }

            resultInPing = !resultInPing;
        }

        // Read final complex spectrum
        var outSpec = new Vector2[N * batches];
        if (resultInPing) bufPing.GetData(outSpec);
        else bufPong.GetData(outSpec);
        return outSpec;
    }

    Vector2[] RunGPUInverse(Vector2[] spec, int N, int batches, ExecMode mode)
    {
        int stages = 0; for (int t = N; t > 1; t >>= 1) stages++;

        using var bufPing = new ComputeBuffer(N * batches, sizeof(float) * 2);
        using var bufPong = new ComputeBuffer(N * batches, sizeof(float) * 2);
        using var twiddle = new ComputeBuffer(N / 2, sizeof(float) * 2);

        bufPing.SetData(spec);

        // Twiddle as forward (cos, -sin); inverse = conjugate by flipping sign in shader
        var tw = new Vector2[N / 2];
        double twoPiOverN = 2.0 * Math.PI / N;
        for (int r = 0; r < tw.Length; r++) { double ang = twoPiOverN * r; tw[r] = new Vector2((float)Math.Cos(ang), (float)-Math.Sin(ang)); }
        twiddle.SetData(tw);

        _cs.SetInt("N", N);
        _cs.SetInt("BatchCount", batches);
        _cs.SetInt("Stages", stages);
        _cs.SetFloat("InvN", 1.0f / N);
        _cs.SetInt("Direction", -1);

        _cs.SetBuffer(_kLDS, "Twiddle", twiddle);
        _cs.SetBuffer(_kStage, "Twiddle", twiddle);

        bool resultInPing = true;

        for (int s = 0; s < stages; s++)
        {
            var src = resultInPing ? bufPing : bufPong;
            var dst = resultInPing ? bufPong : bufPing;

            _cs.SetInt("StageDynamic", s);
            _cs.SetInt("ApplyScaleDynamic", (s == stages - 1) ? 1 : 0); // scale 1/N on the last stage

            int m2 = 1 << (s + 1);
            bool canTile = (N % TILE) == 0 && (m2 <= TILE) && ((TILE % m2) == 0);

            if (mode == ExecMode.LDSAuto && canTile)
            {
                int tilesPerBatch = N / TILE;
                int groups = tilesPerBatch * batches;

                _cs.SetBuffer(_kLDS, "Ping", src);
                _cs.SetBuffer(_kLDS, "Pong", dst);
                _cs.Dispatch(_kLDS, groups, 1, 1);
            }
            else
            {
                int totalButterflies = batches * (N >> 1);
                int groups = Mathf.Max(1, (totalButterflies + THREADS - 1) / THREADS);
                int totalThreads = groups * THREADS;

                _cs.SetInt("ThreadCount", totalThreads);
                _cs.SetBuffer(_kStage, "Ping", src);
                _cs.SetBuffer(_kStage, "Pong", dst);
                _cs.Dispatch(_kStage, groups, 1, 1);
            }

            resultInPing = !resultInPing;
        }

        var outTime = new Vector2[N * batches];
        if (resultInPing) bufPing.GetData(outTime);
        else bufPong.GetData(outTime);
        return outTime;
    }

    // ======= Tests =======

    [Test]
    public void PureTone_SinglePeak_FallbackAndLDS()
    {
        foreach (var N in new[] { 256, 1024, 4096 })
        {
            foreach (var bin in new[] { 1, 7, 64, (N / 2) - 1 })
            {
                var x = MakeSineReal(N, 1, bin, 1f);

                var X_fb = RunGPUForward(x, N, 1, ExecMode.FallbackOnly);
                var ampFB = ToSingleSidedAmplitude(X_fb, N, 1, 0);

                var X_lds = RunGPUForward(x, N, 1, ExecMode.LDSAuto);
                var ampLD = ToSingleSidedAmplitude(X_lds, N, 1, 0);

                // Expect a single spike at `bin` ≈ 1.0, others ≈ 0
                Assert.That(ampFB[bin], Is.InRange(0.999f, 1.001f), $"Fallback bin {bin} @ N={N}");
                Assert.That(ampLD[bin], Is.InRange(0.999f, 1.001f), $"LDS bin {bin} @ N={N}");

                // Leakage check
                float maxLeakFB = ampFB.Where((v, k) => k != 0 && k != N / 2 && k != bin).DefaultIfEmpty(0f).Max();
                float maxLeakLD = ampLD.Where((v, k) => k != 0 && k != N / 2 && k != bin).DefaultIfEmpty(0f).Max();
                Assert.Less(maxLeakFB, 1e-3f, $"Fallback leakage N={N}, bin={bin}");
                Assert.Less(maxLeakLD, 1e-3f, $"LDS leakage N={N}, bin={bin}");

                // DC & Nyquist ~ 0 for pure sine at non-0/non-Nyquist bins
                Assert.Less(Mathf.Abs(ampFB[0]), 1e-3f);
                Assert.Less(Mathf.Abs(ampFB[N / 2]), 1e-3f);
                Assert.Less(Mathf.Abs(ampLD[0]), 1e-3f);
                Assert.Less(Mathf.Abs(ampLD[N / 2]), 1e-3f);
            }
        }
    }

    [Test]
    public void DC_And_Nyquist_Spikes()
    {
        foreach (var N in new[] { 256, 1024, 4096 })
        {
            var dc = MakeConstantReal(N, 1, 1f);
            var X = RunGPUForward(dc, N, 1, ExecMode.LDSAuto);
            var A = ToSingleSidedAmplitude(X, N, 1, 0);
            Assert.That(A[0], Is.InRange(0.999f, 1.001f)); // DC
            Assert.Less(A.Skip(1).Take(N / 2 - 1).Max(), 1e-3f);
            Assert.Less(A[N / 2], 1e-3f);

            var nyq = MakeNyquistReal(N, 1, 1f);
            var X2 = RunGPUForward(nyq, N, 1, ExecMode.LDSAuto);
            var A2 = ToSingleSidedAmplitude(X2, N, 1, 0);
            Assert.That(A2[N / 2], Is.InRange(0.999f, 1.001f)); // Nyquist
            Assert.Less(A2.Take(N / 2).Where((_, k) => k != 0).Max(), 1e-3f);
        }
    }

    [Test]
    public void Hermitian_Symmetry_For_Real_Input()
    {
        int N = 1024;
        var x = MakeRandomReal(N, 1, 1f);
        var X = RunGPUForward(x, N, 1, ExecMode.LDSAuto);
        // X[k] == conj(X[N-k]) for real input (k=1..N/2-1)
        for (int k = 1; k < N / 2; k++)
        {
            var a = X[k];
            var b = X[N - k];
            Assert.Less(Mathf.Abs(a.x - b.x), 1e-4f);
            Assert.Less(Mathf.Abs(a.y + b.y), 1e-4f);
        }
    }

    [Test]
    public void Inverse_Reconstructs_Real_And_Complex()
    {
        foreach (var N in new[] { 256, 1024, 4096 })
        {
            foreach (var mode in new[] { ExecMode.FallbackOnly, ExecMode.LDSAuto })
            {
                // Real
                var xr = MakeRandomReal(N, 1, 0.5f);
                var Xr = RunGPUForward(xr, N, 1, mode);
                var yr = RunGPUInverse(Xr, N, 1, mode);
                Assert.Less(RMSE(xr, yr), 2e-5f, $"Real inverse RMSE N={N}, mode={mode}");

                // Complex
                var xc = MakeRandomComplex(N, 1, 0.5f);
                var Xc = RunGPUForward(xc, N, 1, mode);
                var yc = RunGPUInverse(Xc, N, 1, mode);
                Assert.Less(RMSE(xc, yc), 3e-5f, $"Complex inverse RMSE N={N}, mode={mode}");
            }
        }
    }

    [Test]
    public void LDS_vs_Fallback_Equality_On_Spectrum()
    {
        foreach (var N in new[] { 256, 1024, 4096 })
            foreach (var batches in new[] { 1, 4, 16 })
            {
                var x = MakeRandomComplex(N, batches, 0.25f);
                var Xfb = RunGPUForward(x, N, batches, ExecMode.FallbackOnly);
                var Xlds = RunGPUForward(x, N, batches, ExecMode.LDSAuto);

                // Compare complex spectra
                for (int i = 0; i < Xfb.Length; i++)
                {
                    Assert.Less(Mathf.Abs(Xfb[i].x - Xlds[i].x), 1e-4f, $"real mismatch @ {i} (N={N},B={batches})");
                    Assert.Less(Mathf.Abs(Xfb[i].y - Xlds[i].y), 1e-4f, $"imag mismatch @ {i} (N={N},B={batches})");
                }
            }
    }

    // ======= FFTLocal (single-dispatch) tests via FFT_GPU driver =======

    [Test]
    public void FFTLocal_vs_MultiDispatch_Parity()
    {
        // Compare FFTLocal (via FFT_GPU driver, N<=2048) against multi-dispatch fallback
        foreach (var N in new[] { 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 })
            foreach (var batches in new[] { 1, 4 })
            {
                var x = MakeRandomComplex(N, batches, 0.5f);

                // Multi-dispatch reference (fallback only)
                var Xref = RunGPUForward(x, N, batches, ExecMode.FallbackOnly);

                // FFTLocal via FFT_GPU driver (automatically selects FFTLocal for N<=1024)
                using var fft = new FFT_GPU(_cs, N, batches);
                fft.SetData(x);
                fft.Forward();
                var Xlocal = new Vector2[N * batches];
                fft.GetResult(Xlocal);

                float rmse = RMSE(Xref, Xlocal);
                Assert.Less(rmse, 1e-4f, $"FFTLocal vs Fallback RMSE={rmse} @ N={N},B={batches}");
            }
    }

    [Test]
    public void FFTLocal_Inverse_Reconstructs()
    {
        foreach (var N in new[] { 8, 64, 256, 512, 1024, 2048 })
        {
            var xr = MakeRandomReal(N, 1, 0.5f);
            using var fft = new FFT_GPU(_cs, N, 1);
            fft.SetData(xr);
            fft.Forward();

            // Read spectrum then feed back for inverse
            var spec = new Vector2[N];
            fft.GetResult(spec);

            fft.SetData(spec);
            fft.Inverse(true);
            var yr = new Vector2[N];
            fft.GetResult(yr);

            Assert.Less(RMSE(xr, yr), 3e-5f, $"FFTLocal inverse RMSE @ N={N}");
        }
    }

    [Test]
    public void FFTLocal_PureTone_SinglePeak()
    {
        foreach (var N in new[] { 64, 256, 512, 1024, 2048 })
        {
            int bin = 7;
            var x = MakeSineReal(N, 1, bin, 1f);
            using var fft = new FFT_GPU(_cs, N, 1);
            fft.SetData(x);
            fft.Forward();
            var X = new Vector2[N];
            fft.GetResult(X);
            var amp = ToSingleSidedAmplitude(X, N, 1, 0);

            Assert.That(amp[bin], Is.InRange(0.999f, 1.001f), $"FFTLocal peak @ bin={bin}, N={N}");
            float maxLeak = amp.Where((v, k) => k != 0 && k != N / 2 && k != bin).DefaultIfEmpty(0f).Max();
            Assert.Less(maxLeak, 1e-3f, $"FFTLocal leakage @ N={N}");
        }
    }

    [Test]
    public void FFTLocal_CachedCommandBuffer_Matches_Direct()
    {
        foreach (var N in new[] { 256, 1024, 2048 })
        {
            var x = MakeRandomComplex(N, 1, 0.5f);

            using var fft = new FFT_GPU(_cs, N, 1);

            // Direct path
            fft.SetData(x);
            fft.Forward();
            var Xdirect = new Vector2[N];
            fft.GetResult(Xdirect);

            // Cached path
            fft.SetData(x);
            fft.ForwardCached();
            var Xcached = new Vector2[N];
            fft.GetResult(Xcached);

            Assert.Less(RMSE(Xdirect, Xcached), 1e-6f, $"Cached vs Direct mismatch @ N={N}");
        }
    }

    [Test]
    public void CommandBuffer_Cached_LargeN_Matches_Direct()
    {
        // Test CommandBuffer caching for multi-dispatch path (N > 2048)
        int N = 4096;
        var x = MakeRandomComplex(N, 1, 0.5f);

        using var fft = new FFT_GPU(_cs, N, 1);

        fft.SetData(x);
        fft.Forward();
        var Xdirect = new Vector2[N];
        fft.GetResult(Xdirect);

        fft.SetData(x);
        fft.ForwardCached();
        var Xcached = new Vector2[N];
        fft.GetResult(Xcached);

        Assert.Less(RMSE(Xdirect, Xcached), 1e-6f, "Cached vs Direct mismatch @ N=4096");
    }
}
