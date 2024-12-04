Moderato.Mathematics.FFT
---
Fast Fourier Transform (FFT) in Unity using Burst Compiler &amp; C# Job System.

## Table of Contents

- [Getting started](#getting-started)
- [Arguments](#arguments)
- [UPM Package](#upm-package)
- [License](#license)

Getting started
---
Install via UPM package with git reference

```csharp
using Moderato.Mathematics;
using System.Numerics;

void Hoge()
{
  float[] array_float = new float[100];
  double[] array_double = new double[100];
  Complex[] array_Complex = new Complex[100];

  Complex[] result = FFT.Transform(array_float);
  Complex[] result = FFT.Transform(array_double);
  Complex[] result = FFT.Transform(array_Complex);

  Complex[] result = FFT.Transform(source: array_float, window: Window.Rectangular, enableLowPassFilter: true, threshold: 0.5);
}
```

Arguments
---
### source : **\[Required]**
`float[]` or `double[]` or `System.Numerics.Complex[]`<br>
Input waveform.
> [!NOTE]
> The length of the array need not be a power of 2.
---
### window : **\[Optional]**
`Moderato.Mathematics.Window`<br>
Window function.
  * Rectangular: $W\[n] = 1.0$.
  * Triangle: $W\[n] = 1.0 - |(2.0 * n / N)|$.
  * Hamming: $W\[n] = 0.54 - 0.46 * \cos (2.0 * \pi * n / N)$.
  * Hanning: $W\[n] = (1.0 - \cos (2.0 * \pi * n / N)) / 2.0$.
  * Blackman: $W\[n] = 0.42 - ( \cos (2.0 * \pi * n / N) / 2.0) + (0.08 * \cos (4.0 * \pi * n / N))$.
  * BlackmanHarris: $W\[n] = 0.35875 - (0.48829 * \cos (2.0 * \pi * n / N)) + (0.14128 * \cos (4.0 * \pi * n / N)) - (0.01168 * \cos (6.0 * \pi * n / N))$.

> [!NOTE]
> Default is `Hamming`.
---
### enableLowPassFilter : **\[Optional]**
`bool`<br>
Low-pass Filter.

> [!NOTE]
> Default is `false`.
---
### threshold : **Optional**
`double`<br>
Low-pass filter threshold.

> [!NOTE]
> [0, 1]
> Default is `0d`.
> Also, this value is never read if enableLowPassFilter is `false`.
---
### return
`System.Numerics.Complex[]`<br>
Output Spectrum.
> [!IMPORTANT]
> The length of the array is half the next power of 2 of the `source`.

UPM Package
---
## Install via git URL
You can add `https://github.com/SingUp009/Moderato.Mathematics.FFT.git?path=Assets/Moderato` to Package Manager.
> Package Manager: Window -> Package Manager -> Add package from git URL...

License
---
This library is under the MIT License.
