#pragma once

#include <cuda_runtime.h>

template <class T> class CudaList {
public:

	__device__ CudaList() {}

	__device__ CudaList(int _num_obj) {
		num_obj = _num_obj;
		list = new T[num_obj];
		index = -1;
	}
	__device__ int size(void) const {
		return num_obj;
	}
	__device__ void add(T t) {
		index += 1;
		if (index < num_obj)
			list[index] = t;
	}
	__device__ T& operator [](int i) const { return list[i]; }

private:
	int num_obj;
	int index;
	T* list;
};
