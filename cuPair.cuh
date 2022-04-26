#ifndef CUDA_PAIR
#define CUDA_PAIR

template<class T, class TT>
struct cuPair {
public:
	__device__ __host__ cuPair() : first(T()), second(TT()) { }
	__device__ __host__ cuPair(const cuPair& other) : first(other.first), second(other.second) { }
	__device__ __host__ cuPair(const T& first_, const TT& second_) : first(first_), second(second_) { }
	__device__ __host__ cuPair& operator=(const cuPair& other) {
		if (&other == this)
			return *this;
		first = other.first;
		second = other.second;
		return *this;
	}
	T first;
	TT second;
};

#endif