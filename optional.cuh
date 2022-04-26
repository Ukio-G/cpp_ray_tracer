#ifndef CUDA_OPTIONAL
#define CUDA_OPTIONAL

#include <cuda_runtime.h>

namespace cu {
    struct nullopt_t { };

    static const nullopt_t nullopt = nullopt_t();

    template<class T>
    class optional {
    public:
        __device__ __host__ optional(const T& value) : _has_value(true), _value(value) {}
        __device__ __host__ optional(const optional<T>& other) : _has_value(other._has_value), _value(other._value) {}
        __device__ __host__ optional(nullopt_t) : _has_value(false) {}
        __device__ __host__ optional() : _has_value(false) {}

        __device__ __host__ T& operator*() {
            if (!_has_value)
                *((int*)0xdead) = 0xdead;
            return _value;
        }

        __device__ __host__ operator bool() {
            return _has_value;
        }

        __device__ __host__ optional<T>& operator=(const T& other)
        {
            _value = other;
            _has_value = true;
            return *this;
        }

        __device__ __host__ optional<T>& operator=(const optional<T>& other)
        {
            if (&other == this)
                return *this;

            _value = other._value;
            _has_value = other._has_value;
            return *this;
        }

        __device__ __host__ T& value() {
            if (!_has_value)
                *((int*)0xdead) = 0xdead;
            return _value;
        }

        __device__ __host__ T& value_or(const T& val) {
            if (!_has_value)
                return val;
            return _value;
        }

        __device__ __host__ bool has_value() {
            return _has_value;
        }

        __device__ __host__ void reset() {
            _has_value = false;
            _value.T::~T();
        }

    private:
        bool _has_value;
        T _value;
    };
}
#endif