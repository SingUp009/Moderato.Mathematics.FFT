using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

namespace Moderato.Mathematics
{
    public static class MathExtensions
    {
        public static float Remap(this float value, float fromMin, float fromMax, float toMin, float toMax, bool clamp = true)
        {
            var val = (value - fromMin) / (fromMax - fromMin) * (toMax - toMin) + toMin;
            return clamp ? Math.Clamp(val, toMin, toMax) : val;
        }
    }
}
