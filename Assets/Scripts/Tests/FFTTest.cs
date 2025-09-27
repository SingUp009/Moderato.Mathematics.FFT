using Moderato.Mathematics;
using NUnit.Framework;
using System.Collections;
using Unity.PerformanceTesting;
using UnityEngine;

public class FFTTest
{
    private float[] data;

    private const int LENGTH = 1 << 16;

    [OneTimeSetUp]
    public void SetUp()
    {
        data = new float[LENGTH];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = Mathf.Sin(440 * Mathf.PI * i / (LENGTH));
        }
    }

    [Test, Performance]
    public void CpuTest()
    {
        Measure.Method(() => FFT.Transform(data))
            .WarmupCount(10)
            .MeasurementCount(100)
            .Run();
    }

    [Test, Performance]
    public void GpuTest()
    {

    }
}
