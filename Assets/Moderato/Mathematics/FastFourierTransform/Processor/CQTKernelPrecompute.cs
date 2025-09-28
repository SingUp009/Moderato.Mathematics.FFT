using UnityEngine;
using System;
using System.Collections.Generic;

/// Precompute Constant-Q kernels per octave as frequency-domain spectra for fast projection.
/// Formulas based on Brown (1991) and efficient FFT-implementation by Schörkhuber & Klapuri.
/// We do analysis-only, so kernels are L2-normalized and pre-conjugated for inner products.
/// Comments in English as requested.
public static class CQTKernelPrecompute
{
    public struct OctaveKernels
    {
        public int octave;
        public int L;                // FFT length for this octave
        public int bins;             // bins in this octave (usually binsPerOctave)
        public ComputeBuffer kernels; // (L * bins) complex float2, pre-conjugated spectra
        public ComputeBuffer xSlice;  // temporary: FFT slice spectrum (L)
        public ComputeBuffer coefs;   // output coef buffer (bins)
    }

    public struct Plan
    {
        public int binsPerOctave;
        public float fmin;
        public int octaves;
        public float Q;             // 1/(2^(1/B)-1)
        public int sampleRate;
        public int[] L;             // FFT sizes per octave
        public int[] hop;           // hop sizes per octave
        public float[] fCenters;    // all center freqs
        public int totalBins;
        public OctaveKernels[] okt; // GPU buffers per octave
    }

    public static Plan Build(int sampleRate, float fmin = 55f, int binsPerOctave = 24, int octaves = 6, float hopPerKernel = 4f)
    {
        if (binsPerOctave <= 0 || octaves <= 0) throw new ArgumentException();
        var plan = new Plan
        {
            sampleRate = sampleRate,
            fmin = fmin,
            binsPerOctave = binsPerOctave,
            octaves = octaves
        };
        plan.Q = 1f / (Mathf.Pow(2f, 1f / binsPerOctave) - 1f); // Brown 1991

        // All center frequencies
        plan.totalBins = binsPerOctave * octaves;
        plan.fCenters = new float[plan.totalBins];
        for (int k = 0; k < plan.totalBins; k++)
            plan.fCenters[k] = fmin * Mathf.Pow(2f, k / (float)binsPerOctave);

        // Decide per-octave FFT sizes:
        // Group bins by octave; L_o >= max N_k in that octave, rounded to power of two.
        plan.L = new int[octaves];
        plan.hop = new int[octaves];
        var perOctBins = binsPerOctave;

        for (int o = 0; o < octaves; o++)
        {
            int k0 = o * binsPerOctave;
            int k1 = k0 + binsPerOctave - 1;
            float fminOct = plan.fCenters[k0];
            // Largest kernel occurs at lowest freq in the octave
            float Nk_f = plan.Q * sampleRate / fminOct;
            int Nk = Mathf.CeilToInt(Nk_f);
            int Lo = NextPow2(Nk);
            plan.L[o] = Mathf.Max(1024, Lo); // safety floor
            plan.hop[o] = Mathf.Max(64, plan.L[o] / (int)hopPerKernel);
        }

        return plan;
    }

    public static void AllocateKernels(ref Plan plan)
    {
        plan.okt = new OctaveKernels[plan.octaves];

        for (int o = 0; o < plan.octaves; o++)
        {
            int L = plan.L[o];
            int B = plan.binsPerOctave;
            var ok = new OctaveKernels
            {
                octave = o,
                L = L,
                bins = B,
                kernels = new ComputeBuffer(L * B, sizeof(float) * 2),
                xSlice = new ComputeBuffer(L, sizeof(float) * 2),
                coefs = new ComputeBuffer(B, sizeof(float) * 2),
            };

            // Build kernels (time -> freq -> conjugate)
            // For each bin in octave: time kernel length Nk, Hann windowed complex exponential,
            // zero-pad to L, FFT, then conjugate spectrum.
            var H = new UnityEngine.Vector2[L * B];

            for (int b = 0; b < B; b++)
            {
                int k = o * plan.binsPerOctave + b;
                float fk = plan.fCenters[k];
                int Nk = Mathf.CeilToInt(plan.Q * plan.sampleRate / fk);
                Nk = Mathf.Min(Nk, L); // clamp

                // time kernel with Hann
                var time = new System.Numerics.Complex[L];
                for (int n = 0; n < Nk; n++)
                {
                    float w = 0.5f * (1f - Mathf.Cos(2f * Mathf.PI * n / (Nk - 1))); // Hann
                    double phase = 2.0 * Math.PI * fk * n / plan.sampleRate;
                    var c = new System.Numerics.Complex(Math.Cos(phase), Math.Sin(phase));
                    time[n] = w * c / Nk; // normalize by Nk to keep amplitudes tame
                }
                for (int n = Nk; n < L; n++) time[n] = System.Numerics.Complex.Zero;

                // FFT (CPU-side) just once at startup; L is modest (<= 8192 typically).
                FFTInPlace(time); // Cooley–Tukey in C#, or call your CPU FFT if you have one

                // Store conjugated spectrum into H
                for (int f = 0; f < L; f++)
                {
                    var v = System.Numerics.Complex.Conjugate(time[f]);
                    H[f + L * b] = new UnityEngine.Vector2((float)v.Real, (float)v.Imaginary);
                }
            }

            ok.kernels.SetData(H);
            plan.okt[o] = ok;
        }
    }

    // ==== Utilities ====

    static int NextPow2(int x) { int p = 1; while (p < x) p <<= 1; return p; }

    // Minimal in-place FFT (decimation-in-time radix-2), for startup kernel build.
    // Not performance-critical; replace with your preferred CPU FFT if available.
    static void FFTInPlace(System.Numerics.Complex[] a, bool inverse = false)
    {
        int n = a.Length;
        int j = 0;
        for (int i = 1; i < n - 1; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j &= ~bit;
            j |= bit;
            if (i < j) { var t = a[i]; a[i] = a[j]; a[j] = t; }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = 2 * Math.PI / len * (inverse ? +1 : -1);
            var wlen = new System.Numerics.Complex(Math.Cos(ang), Math.Sin(ang));
            for (int i = 0; i < n; i += len)
            {
                var w = System.Numerics.Complex.One;
                for (int k = 0; k < len / 2; k++)
                {
                    var u = a[i + k];
                    var v = a[i + k + len / 2] * w;
                    a[i + k] = u + v;
                    a[i + k + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (inverse) { for (int i = 0; i < n; i++) a[i] /= n; }
    }
}
