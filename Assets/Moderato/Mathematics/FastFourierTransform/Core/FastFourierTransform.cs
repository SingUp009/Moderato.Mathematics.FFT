using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
using System.Numerics;
using Unity.Burst.CompilerServices;
using System;
using System.Runtime.InteropServices;

namespace Moderato.Mathematics
{
    [BurstCompile]
    [StructLayout(LayoutKind.Sequential)]
    internal struct FastFourierTransform : IJob
    {
        /// <summary>
        /// Type of window function.
        /// </summary>
        public Window Window;

        /// <summary>
        /// [ReadOnly]
        /// Input waveform. The length of the array must be a power of 2.
        /// </summary>
        [ReadOnly]
        public NativeArray<Complex> Waveform;

        /// <summary>
        /// Output spectrum. The length of the array must be the half of the input waveform.
        /// </summary>
        [WriteOnly]
        public NativeArray<Complex> Result;

        public bool EnableLowPassFilter { get; set; }

        /// <summary>
        /// Threshold for low-pass filter.
        /// Range: [0, 1]
        /// </summary>
        public double Threshold { get; set; }

        /// <summary>
        /// Constant values of ��.
        /// </summary>
        private const double TWO_PI = 2.0 * math.PI;
        private const double FOUR_PI = 4.0 * math.PI;
        private const double SIX_PI = 6.0 * math.PI;

        [SkipLocalsInit]
        public void Execute()
        {
            NativeArray<Complex> source = new(Waveform.Length, Allocator.Temp);
            Aliasing.ExpectNotAliased(in Waveform, in source);
            Waveform.AsReadOnlySpan().CopyTo(source);

            Windowing(Window, source);

            FFT(source, Result);

            source.Dispose();
        }

        #region Windowing

        private readonly void Windowing(Window WindowType, Span<Complex> source)
        {
            switch (WindowType)
            {
                case Window.Rectangular:
                    break;

                case Window.Triangle:
                    Triangle(source);
                    break;

                case Window.Hamming:
                    Hamming(source);
                    break;

                case Window.Hanning:
                    Hanning(source);
                    break;

                case Window.Blackman:
                    Blackman(source);
                    break;

                case Window.BlackmanHarris:
                    BlackmanHarris(source);
                    break;

                default:
                    break;
            }
        }

        private readonly void Triangle(Span<Complex> source)
        {
            int Length = source.Length;
            for (int i = 0; i < Length; i++)
            {
                source[i] *= 1.0 - math.abs(2.0 * i / Length);
            }
        }

        private readonly void Hamming(Span<Complex> source)
        {
            int Length = source.Length;
            for (int i = 0; i < Length; i++)
            {
                source[i] *= 0.54 - math.mul(0.46, math.cos(TWO_PI * i / Length));
            }
        }

        private readonly void Hanning(Span<Complex> source)
        {
            int Length = source.Length;
            for (int i = 0; i < Length; i++)
            {
                source[i] *= (1.0 - math.cos(TWO_PI * i / Length)) / 2;
            }
        }

        private readonly void Blackman(Span<Complex> source)
        {
            int Length = source.Length;
            for (int i = 0; i < Length; i++)
            {
                double ratio = i / (double)Length;
                source[i] *= 0.42
                    - math.cos(TWO_PI * ratio) / 2
                    + math.mul(0.08, math.cos(FOUR_PI * ratio));
            }
        }

        private readonly void BlackmanHarris(Span<Complex> source)
        {
            int Length = source.Length;
            for (int i = 0; i < Length; i++)
            {
                double ratio = i / (double)Length;
                source[i] *= 0.35875
                    - 0.48829 * math.cos(TWO_PI * ratio)
                    + 0.14128 * math.cos(FOUR_PI * ratio)
                    - 0.01168 * math.cos(SIX_PI * ratio);
            }
        }

        #endregion

        [SkipLocalsInit]
        private readonly void FFT(NativeArray<Complex> _source, NativeArray<Complex> _result)
        {
            ReadOnlySpan<Complex> source = _source.AsSpan();
            Span<Complex> result = _result.AsSpan();
            Aliasing.ExpectNotAliased(in _source, in _result);

            int Length = source.Length;
            Hint.Assume(Length > 0);

            int Log2Length = (int)math.log2(Length);
            Hint.Assume(Log2Length > 0);

            NativeArray<Complex> _rev = new(Length, Allocator.Temp);
            Span<Complex> rev = _rev.AsSpan();
            
            // Bit reversal
            for (int i = 0; i < Length; i++)
            {
                rev[ReverseBits(i, Log2Length)] = source[i];
            }

            NativeArray<Complex> _Spectrum = new(Length, Allocator.Temp);
            Span<Complex> Spectrum = _Spectrum.AsSpan();

            Aliasing.ExpectNotAliased(in _rev, in _Spectrum);

            // Cooley-Tukey FFT
            for (int s = 1; s <= Log2Length; s++)
            {
                int l = 1 << s;
                Complex wn = Complex.FromPolarCoordinates(1d, -TWO_PI / l);

                for (int k = 0; k < Length; k += l)
                {
                    Complex w = Complex.One;
                    int halfL = l / 2;

                    for (int j = 0; j < halfL; j++)
                    {
                        int KplusJ = k + j;
                        int KplusJplusHalfL = KplusJ + halfL;

                        Complex t = w * rev[KplusJplusHalfL];
                        Complex u = rev[KplusJ];

                        Spectrum[KplusJ] = u + t;
                        Spectrum[KplusJplusHalfL] = u - t;

                        w *= wn;
                    }
                }

                _Spectrum.CopyTo(_rev);
            }
            
            int nyquist = Length / 2;
            double normarization = 2.0 / Length;
            Span<Complex> _spectrum = Spectrum[..nyquist];
            
            // Normalize
            for (int i = 0; i < nyquist; i++)
            {
                result[i] = _spectrum[i] * normarization;
            }

            // Low-pass filter
            if (EnableLowPassFilter)
            {
                double max = default;
                for (int i = 0; i < nyquist; i++)
                {
                    max = math.max(max, result[i].Magnitude);
                }

                double threshold = max * Threshold;
                for (int i = 0; i < nyquist; i++)
                {
                    Complex value = result[i];

                    result[i] = value.Magnitude >= threshold ? value : Complex.Zero;
                }
            }

            _rev.Dispose();
            _Spectrum.Dispose();
        }

        [return: AssumeRange(0, int.MaxValue)]
        private readonly int ReverseBits(
            [AssumeRange(0, int.MaxValue)] int n,
            [AssumeRange(0, int.MaxValue)] int m)
        {
            int rev = 0;
            for (int i = 0; i < m; i++)
            {
                rev <<= 1;
                rev |= (n & 1);
                n >>= 1;
            }
            return rev;
        }
    }
}