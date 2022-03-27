#include "Helpers.cuh"

__device__ int binary_search(double* list, int size, const float& val)
{
	int middle, left = 0, right = size;
	while (right >= 1) {
		middle = (right - left) / 2 + left;

		if (val == middle) return middle;
		else if (val < middle) right = middle - 1;
		else left = middle + 1;
	}
	return -1;
}

__device__ int upper_bound(double* list, int size, const float& val)
{
	int middle, left = 0, right = size;
	while (left < right) {
		middle = (right - left) / 2 + left;

		if (val >= list[middle]) {
			left = middle + 1;
		}
		else {
			right = middle;
		}
	}
	if (left < size && list[left <= val]) left++;
	return left;
}

__device__ int lower_bound(double* list, int size, const float& val)
{
	int middle, left = 0, right = size;
	while (left < right) {
		middle = (right - left) / 2 + left;

		if (val <= list[middle]) {
			right = middle;
		}
		else {
			left = middle + 1;
		}
	}
	if (left < size && list[left < val]) left++;
	return left;
}