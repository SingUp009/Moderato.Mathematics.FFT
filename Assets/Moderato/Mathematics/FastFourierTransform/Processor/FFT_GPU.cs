using System;
using UnityEngine;

namespace Moderato.Mathematics
{
    /// GPU driver for StockhamFFT_Tiled.compute
    /// Comments are in English as requested.
    public sealed class FFT_GPU : IDisposable
    {
        const int THREADS = 256;   // must match HLSL
        const int TILE = 1024;  // must match HLSL

        readonly ComputeShader cs;
        readonly int kLDS, kStage;

        public int N { get; private set; }
        public int Batches { get; private set; }
        public int Stages { get; private set; }

        ComputeBuffer ping, pong, twiddle;
        bool resultInPing = true;

        public FFT_GPU(ComputeShader shader, int n, int batches)
        {
            if ((n & (n - 1)) != 0) throw new ArgumentException("N must be power of two.");
            if (batches <= 0) throw new ArgumentException("batches must be > 0.");
            cs = shader ?? throw new ArgumentNullException(nameof(shader));
            kLDS = cs.FindKernel("FFTStageLDS");
            kStage = cs.FindKernel("FFTStage");
            Init(n, batches);
        }

        public void Init(int n, int batches)
        {
            Release();
            N = n; Batches = batches; Stages = (int)Mathf.Log(N, 2);

            ping = new ComputeBuffer(N * Batches, sizeof(float) * 2);
            pong = new ComputeBuffer(N * Batches, sizeof(float) * 2);
            twiddle = new ComputeBuffer(N / 2, sizeof(float) * 2);
            UploadTwiddle();

            cs.SetInt("N", N);
            cs.SetInt("BatchCount", Batches);
            cs.SetInt("Stages", Stages);
            cs.SetFloat("InvN", 1.0f / N);
        }

        void UploadTwiddle()
        {
            var data = new Vector2[N / 2];
            double k = 2.0 * System.Math.PI / N;
            for (int r = 0; r < data.Length; r++)
            {
                double a = k * r;
                data[r] = new Vector2((float)System.Math.Cos(a), (float)-System.Math.Sin(a));
            }
            twiddle.SetData(data);
        }

        public void SetData(Vector2[] timeComplex)
        {
            if (timeComplex == null || timeComplex.Length != N * Batches)
                throw new ArgumentException("Length must be N*Batches.");
            ping.SetData(timeComplex);
            resultInPing = true;
        }

        public void Forward() => Execute(+1, normalizeInverse: false);
        public void Inverse(bool normalize = true) => Execute(-1, normalizeInverse: normalize);

        void Execute(int direction, bool normalizeInverse)
        {
            cs.SetInt("Direction", direction);
            cs.SetBuffer(kLDS, "Twiddle", twiddle);
            cs.SetBuffer(kStage, "Twiddle", twiddle);

            for (int s = 0; s < Stages; s++)
            {
                var src = resultInPing ? ping : pong;
                var dst = resultInPing ? pong : ping;

                cs.SetInt("StageDynamic", s);
                cs.SetInt("ApplyScaleDynamic", (normalizeInverse && direction < 0 && s == Stages - 1) ? 1 : 0);

                int m2 = 1 << (s + 1);
                bool useLDS = (N % TILE) == 0 && (m2 <= TILE) && ((TILE % m2) == 0);

                if (useLDS)
                {
                    int tilesPerBatch = (N >> 1) / (TILE);   // halfN / TILE
                    if (tilesPerBatch == 0) tilesPerBatch = 1;
                    int groups = tilesPerBatch * Batches;

                    cs.SetBuffer(kLDS, "Ping", src);
                    cs.SetBuffer(kLDS, "Pong", dst);
                    cs.Dispatch(kLDS, groups, 1, 1);
                }
                else
                {
                    int totalButterflies = Batches * (N >> 1);
                    int groups = Mathf.Max(1, (totalButterflies + THREADS - 1) / THREADS);
                    cs.SetInt("ThreadCount", groups * THREADS);

                    cs.SetBuffer(kStage, "Ping", src);
                    cs.SetBuffer(kStage, "Pong", dst);
                    cs.Dispatch(kStage, groups, 1, 1);
                }
                resultInPing = !resultInPing;
            }
        }

        public void GetResult(Vector2[] dst)
        {
            if (dst == null || dst.Length != N * Batches) throw new ArgumentException();
            if (resultInPing) ping.GetData(dst);
            else pong.GetData(dst);
        }
        public ComputeBuffer ResultBuffer => resultInPing ? ping : pong;

        public void Dispose() => Release();
        void Release() { ping?.Dispose(); pong?.Dispose(); twiddle?.Dispose(); ping = pong = twiddle = null; }
    }

}