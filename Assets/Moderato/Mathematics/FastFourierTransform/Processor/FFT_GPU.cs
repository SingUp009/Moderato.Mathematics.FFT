using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Moderato.Mathematics
{
    /// GPU driver for StockhamFFT_Tiled.compute
    public sealed class FFT_GPU : IDisposable
    {
        const int THREADS = 256;    // must match HLSL
        const int TILE = 1024;   // must match HLSL
        const int LOCAL_N = 2048; // must match HLSL LOCAL_N

        readonly ComputeShader cs;
        readonly int kLDS, kStage, kLocal;

        public int N { get; private set; }
        public int Batches { get; private set; }
        public int Stages { get; private set; }

        ComputeBuffer ping, pong, twiddle;
        bool resultInPing = true;

        struct StageParams
        {
            public int kernel;
            public int groups;
            public int threadCount; // only used by fallback kernel
        }
        StageParams[] stageParams;
        bool useLocalKernel;
        bool cachedResultInPing; // resultInPing state after a full forward/inverse pass

        CommandBuffer cmdForward, cmdInverse, cmdInverseNoNorm;

        public FFT_GPU(ComputeShader shader, int n, int batches)
        {
            if ((n & (n - 1)) != 0) throw new ArgumentException("N must be power of two.");
            if (batches <= 0) throw new ArgumentException("batches must be > 0.");
            cs = shader ?? throw new ArgumentNullException(nameof(shader));
            kLDS = cs.FindKernel("FFTStageLDS");
            kStage = cs.FindKernel("FFTStage");
            kLocal = cs.FindKernel("FFTLocal");
            Init(n, batches);
        }

        public void Init(int n, int batches)
        {
            Release();
            N = n; Batches = batches;
            int s = 0; for (int t = N; t > 1; t >>= 1) s++;
            Stages = s;

            ping = new ComputeBuffer(N * Batches, sizeof(float) * 2);
            pong = new ComputeBuffer(N * Batches, sizeof(float) * 2);

            useLocalKernel = (N <= LOCAL_N);

            cs.SetInt("N", N);
            cs.SetInt("BatchCount", Batches);
            cs.SetInt("Stages", Stages);
            cs.SetFloat("InvN", 1.0f / N);

            if (!useLocalKernel)
            {
                twiddle = new ComputeBuffer(N / 2, sizeof(float) * 2);
                UploadTwiddle();

                // Bind twiddle once — it never changes after Init
                cs.SetBuffer(kLDS, "Twiddle", twiddle);
                cs.SetBuffer(kStage, "Twiddle", twiddle);

                // Pre-compute per-stage dispatch parameters
                stageParams = new StageParams[Stages];
                for (int i = 0; i < Stages; i++)
                {
                    int m2 = 1 << (i + 1);
                    bool useLDS = (N % TILE) == 0 && (m2 <= TILE) && ((TILE % m2) == 0);

                    if (useLDS)
                    {
                        int tilesPerBatch = (N >> 1) / TILE;
                        if (tilesPerBatch == 0) tilesPerBatch = 1;
                        stageParams[i] = new StageParams { kernel = kLDS, groups = tilesPerBatch * Batches };
                    }
                    else
                    {
                        int totalButterflies = Batches * (N >> 1);
                        int groups = Mathf.Max(1, (totalButterflies + THREADS - 1) / THREADS);
                        stageParams[i] = new StageParams { kernel = kStage, groups = groups, threadCount = groups * THREADS };
                    }
                }
            }

            InvalidateCommandBuffers();
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

        public void ForwardCached()
        {
            cmdForward ??= BuildCommandBuffer(+1, false);
            Graphics.ExecuteCommandBuffer(cmdForward);
            resultInPing = cachedResultInPing;
        }

        public void InverseCached(bool normalize = true)
        {
            if (normalize)
            {
                cmdInverse ??= BuildCommandBuffer(-1, true);
                Graphics.ExecuteCommandBuffer(cmdInverse);
            }
            else
            {
                cmdInverseNoNorm ??= BuildCommandBuffer(-1, false);
                Graphics.ExecuteCommandBuffer(cmdInverseNoNorm);
            }
            resultInPing = cachedResultInPing;
        }

        CommandBuffer BuildCommandBuffer(int direction, bool normalizeInverse)
        {
            var cmd = new CommandBuffer();
            cmd.name = direction > 0 ? "FFT_Forward" : "FFT_Inverse";

            cmd.SetComputeIntParam(cs, "Direction", direction);

            if (useLocalKernel)
            {
                cmd.SetComputeIntParam(cs, "ApplyScaleDynamic",
                    (normalizeInverse && direction < 0) ? 1 : 0);
                // Always start from ping (SetData writes to ping)
                cmd.SetComputeBufferParam(cs, kLocal, "Ping", ping);
                cmd.SetComputeBufferParam(cs, kLocal, "Pong", pong);
                cmd.DispatchCompute(cs, kLocal, Batches, 1, 1);
                cachedResultInPing = false; // result in pong after single dispatch
            }
            else
            {
                bool inPing = true; // always start from ping
                for (int s = 0; s < Stages; s++)
                {
                    var sp = stageParams[s];
                    var src = inPing ? ping : pong;
                    var dst = inPing ? pong : ping;

                    cmd.SetComputeIntParam(cs, "StageDynamic", s);
                    cmd.SetComputeIntParam(cs, "ApplyScaleDynamic",
                        (normalizeInverse && direction < 0 && s == Stages - 1) ? 1 : 0);

                    if (sp.kernel == kStage)
                        cmd.SetComputeIntParam(cs, "ThreadCount", sp.threadCount);

                    cmd.SetComputeBufferParam(cs, sp.kernel, "Ping", src);
                    cmd.SetComputeBufferParam(cs, sp.kernel, "Pong", dst);
                    cmd.DispatchCompute(cs, sp.kernel, sp.groups, 1, 1);
                    inPing = !inPing;
                }
                cachedResultInPing = inPing;
            }

            return cmd;
        }

        void InvalidateCommandBuffers()
        {
            cmdForward?.Dispose(); cmdForward = null;
            cmdInverse?.Dispose(); cmdInverse = null;
            cmdInverseNoNorm?.Dispose(); cmdInverseNoNorm = null;
        }

        void Execute(int direction, bool normalizeInverse)
        {
            cs.SetInt("Direction", direction);

            if (useLocalKernel)
            {
                cs.SetInt("ApplyScaleDynamic",
                    (normalizeInverse && direction < 0) ? 1 : 0);

                var src = resultInPing ? ping : pong;
                var dst = resultInPing ? pong : ping;
                cs.SetBuffer(kLocal, "Ping", src);
                cs.SetBuffer(kLocal, "Pong", dst);
                cs.Dispatch(kLocal, Batches, 1, 1);
                resultInPing = !resultInPing;
                return;
            }

            for (int s = 0; s < Stages; s++)
            {
                var sp = stageParams[s];
                var src = resultInPing ? ping : pong;
                var dst = resultInPing ? pong : ping;

                cs.SetInt("StageDynamic", s);
                cs.SetInt("ApplyScaleDynamic",
                    (normalizeInverse && direction < 0 && s == Stages - 1) ? 1 : 0);

                if (sp.kernel == kStage)
                    cs.SetInt("ThreadCount", sp.threadCount);

                cs.SetBuffer(sp.kernel, "Ping", src);
                cs.SetBuffer(sp.kernel, "Pong", dst);
                cs.Dispatch(sp.kernel, sp.groups, 1, 1);
                resultInPing = !resultInPing;
            }
        }

        public void GetResult(Vector2[] dst)
        {
            if (dst == null || dst.Length != N * Batches) throw new ArgumentException();
            if (resultInPing) ping.GetData(dst);
            else pong.GetData(dst);
        }

        public void GetResultAsync(Action<AsyncGPUReadbackRequest> callback)
        {
            AsyncGPUReadback.Request(resultInPing ? ping : pong, callback);
        }

        public ComputeBuffer ResultBuffer => resultInPing ? ping : pong;

        public void Dispose() => Release();

        void Release()
        {
            InvalidateCommandBuffers();
            ping?.Dispose(); pong?.Dispose(); twiddle?.Dispose();
            ping = pong = twiddle = null;
        }
    }
}
