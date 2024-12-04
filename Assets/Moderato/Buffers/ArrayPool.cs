using System;

namespace Moderato.Buffers
{
    public readonly struct ArrayPool<T> : IDisposable
    {
        public readonly T[] Array { get; }

        public readonly int Length => Array.Length;

        public T this[int index] => Array[index];

        public Span<T> AsSpan()
            => Array.AsSpan();

        public Span<T> AsSpan(int start)
            => Array.AsSpan(start);

        public Span<T> AsSpan(int start, int length)
            => Array.AsSpan(start, length);

        public ArrayPool(int length, bool skipLocalInit = false)
        {
            Array = System.Buffers.ArrayPool<T>.Shared.Rent(length);

            if (!skipLocalInit) Array.AsSpan().Clear();
        }

        public void Dispose()
            => System.Buffers.ArrayPool<T>.Shared.Return(Array);
    }
}
