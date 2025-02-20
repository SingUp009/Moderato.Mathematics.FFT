using System.Collections;
using NUnit.Framework;
using Unity.PerformanceTesting;
using UnityEngine;
using Moderato.Mathematics;

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
    public void Test()
    {
        Measure.Method(() => FFT.Transform(data))
            .WarmupCount(10)
            .MeasurementCount(100)
            .Run();
    }
}
