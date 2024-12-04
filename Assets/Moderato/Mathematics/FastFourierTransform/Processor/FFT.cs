using System;
using System.Collections;
using System.Collections.Generic;
using Moderato.Buffers;
using System.Numerics;
using Unity.Collections;
using Unity.Jobs;

namespace Moderato.Mathematics
{
    public static class FFT
    {
        /// <summary>
        /// Fast Fourier Transform.
        /// </summary>
        /// <param name="source">
        /// Input waveform. The length of the array need not be a power of 2.
        /// </param>
        /// <param name="window">
        /// Window function. [Rectangular, Triangle, Hamming, Hanning, Blackman, BlackmanHarris]
        /// </param>
        /// <param name="enableLowPassFilter">
        /// Low-pass filter.
        /// </param>
        /// <param name="threshold">
        /// Low-pass filter threshold. [0, 1]
        /// </param>
        /// <returns>
        /// Output spectrum. The length of the array is half the next power of 2 of the input waveform.
        /// </returns>
        public static Complex[] Transform(ReadOnlySpan<float> source, Window window = Window.Hamming, bool enableLowPassFilter = false, double threshold = 0d)
        {
            #region Null or Empty Check
            if (source == null) return null;

            if (source.IsEmpty) return Array.Empty<Complex>();
            #endregion

            int length = source.Length;

            using ArrayPool<Complex> waveform = new(length);
            Span<Complex> _waveform = waveform.AsSpan();

            for (int i = 0; i < length; i++) _waveform[i] = new Complex(source[i], 0);

            return TransformInternal(waveform.Array, window, enableLowPassFilter, threshold);
        }

        /// <summary>
        /// Fast Fourier Transform.
        /// </summary>
        /// <param name="source">
        /// Input waveform. The length of the array need not be a power of 2.
        /// </param>
        /// <param name="window">
        /// Window function. [Rectangular, Triangle, Hamming, Hanning, Blackman, BlackmanHarris]
        /// </param>
        /// <param name="enableLowPassFilter">
        /// Low-pass filter.
        /// </param>
        /// <param name="threshold">
        /// Low-pass filter threshold. [0, 1]
        /// </param>
        /// <returns>
        /// Output spectrum. The length of the array is half the next power of 2 of the input waveform.
        /// </returns>
        public static Complex[] Transform(ReadOnlySpan<double> source, Window window = Window.Hamming, bool enableLowPassFilter = false, double threshold = 0d)
        {
            #region Null or Empty Check
            if (source == null) return null;

            if (source.IsEmpty) return Array.Empty<Complex>();
            #endregion

            int length = source.Length;

            using ArrayPool<Complex> waveform = new(length);
            Span<Complex> _waveform = waveform.AsSpan();

            for (int i = 0; i < length; i++) _waveform[i] = new Complex(source[i], 0);

            return TransformInternal(waveform.Array, window, enableLowPassFilter, threshold);
        }

        /// <summary>
        /// Fast Fourier Transform.
        /// </summary>
        /// <param name="source">
        /// Input waveform. The length of the array need not be a power of 2.
        /// </param>
        /// <param name="window">
        /// Window function. [Rectangular, Triangle, Hamming, Hanning, Blackman, BlackmanHarris]
        /// </param>
        /// <param name="enableLowPassFilter">
        /// Low-pass filter.
        /// </param>
        /// <param name="threshold">
        /// Low-pass filter threshold. [0, 1]
        /// </param>
        /// <returns>
        /// Output spectrum. The length of the array is half the next power of 2 of the input waveform.
        /// </returns>
        public static Complex[] Transform(ReadOnlySpan<Complex> source, Window window = Window.Hamming, bool enableLowPassFilter = false, double threshold = 0d)
        {
            #region Null or Empty Check
            if (source == null) return null;

            if (source.IsEmpty) return Array.Empty<Complex>();
            #endregion

            int length = source.Length;

            using ArrayPool<Complex> waveform = new(length);
            Span<Complex> _waveform = waveform.AsSpan();

            for (int i = 0; i < length; i++) _waveform[i] = source[i];

            return TransformInternal(waveform.Array, window, enableLowPassFilter, threshold);
        }

        private static Complex[] TransformInternal(Complex[] source, Window window, bool enableLowPassFilter, double threshold)
        {
            int length = source.Length;

            NativeArray<Complex> waveform = new(source, Allocator.TempJob);
            NativeArray<Complex> _result = new(length / 2, Allocator.TempJob);

            FastFourierTransform fft = new()
            {
                Window = window,
                Waveform = waveform,
                Result = _result,
                EnableLowPassFilter = enableLowPassFilter,
                Threshold = threshold,
            };

            fft.Schedule().Complete();

            Complex[] result = _result.ToArray();

            waveform.Dispose();
            _result.Dispose();

            return result;
        }
    }
}
