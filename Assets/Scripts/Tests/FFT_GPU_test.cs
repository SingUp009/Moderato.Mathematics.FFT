using Moderato.Mathematics;
using NUnit.Framework;
using Unity.PerformanceTesting;
using UnityEngine;

public class FFT_GPU_test
{
    private const int BATCH = 1;

    private ComputeShader shader;

    [OneTimeSetUp]
    public void SetUp()
    {
        shader = Resources.Load<ComputeShader>("StockhamFFT_Tiled");
    }

    static Vector2[] MakeSine(int N, int batches)
    {
        var data = new Vector2[N * batches];
        for (int i = 0; i < N; i++)
            data[i] = new Vector2(Mathf.Sin(440f * Mathf.PI * i / N), 0f);
        return data;
    }

    // Forward + GetData (end-to-end including readback)
    [Test, Performance]
    public void Forward_WithReadback([NUnit.Framework.Range(1, 24)] int length)
    {
        int N = 1 << length;
        var data = MakeSine(N, BATCH);
        var result = new Vector2[N];

        using var fft = new FFT_GPU(shader, N, BATCH);
        fft.SetData(data);

        Measure.Method(() =>
            {
                fft.Forward();
                fft.ResultBuffer.GetData(result);
            })
            .WarmupCount(10)
            .MeasurementCount(1000)
            .Run();
    }

    // Forward only (GPU dispatch without CPU readback stall)
    [Test, Performance]
    public void Forward_DispatchOnly([NUnit.Framework.Range(1, 24)] int length)
    {
        int N = 1 << length;
        var data = MakeSine(N, BATCH);

        using var fft = new FFT_GPU(shader, N, BATCH);
        fft.SetData(data);

        Measure.Method(() =>
            {
                fft.Forward();
            })
            .WarmupCount(10)
            .MeasurementCount(1000)
            .Run();
    }

    // ForwardCached (CommandBuffer path)
    [Test, Performance]
    public void ForwardCached_WithReadback([NUnit.Framework.Range(1, 24)] int length)
    {
        int N = 1 << length;
        var data = MakeSine(N, BATCH);
        var result = new Vector2[N];

        using var fft = new FFT_GPU(shader, N, BATCH);
        fft.SetData(data);

        Measure.Method(() =>
            {
                fft.ForwardCached();
                fft.ResultBuffer.GetData(result);
            })
            .WarmupCount(10)
            .MeasurementCount(1000)
            .Run();
    }
}
