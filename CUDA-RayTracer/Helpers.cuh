#pragma once

#include "cuda_runtime.h"

template<typename T> __device__ int binary_search(T* list, int size, const float& val);
template<typename T> __device__ int upper_bound(T* list, int size, const float& val);
template<typename T> __device__ int lower_bound(T* list, int size, const float& val);

__device__ int binary_search(double* list, int size, const float& val);
__device__ int upper_bound(double* list, int size, const float& val);
__device__ int lower_bound(double* list, int size, const float& val);