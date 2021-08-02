#pragma once

#include <bitset>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <texture_indirect_functions.h>

#include <vector>
#include <algorithm>

#include "Bounds3f.cuh"
#include "Triangle.cuh"
#include "MemoryArena.h"
#include "BVH.cuh"
#include "CudaHelpers.cuh"
#include "kernel.cuh"

__constant__ const int MAX_LEAF = 100;

__host__ __device__ Bounds3f intersect(const Bounds3f& b1, const Bounds3f& b2) {
	return Bounds3f(
		make_float3(
			fmaxf(b1.pMin.x, b2.pMin.x),
			fmaxf(b1.pMin.y, b2.pMin.y),
			fmaxf(b1.pMin.z, b2.pMin.z)),
		make_float3(
			fminf(b1.pMax.x, b2.pMax.x),
			fminf(b1.pMax.y, b2.pMax.y),
			fminf(b1.pMax.z, b2.pMax.z)));
};

__host__ __device__ bool overlaps(const Bounds3f& b1, const Bounds3f& b2) {
	bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
	bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
	bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
	return (x && y && z);
};

__host__ __device__ bool inside(float3& p, const Bounds3f& b) {
	return (p.x >= b.pMin.x && p.x <= b.pMax.x &&
		p.y >= b.pMin.y && p.y <= b.pMax.y &&
		p.z >= b.pMin.z && p.z <= b.pMax.z);
};

__host__ __device__ bool inside_exclusive(const float3& p, const Bounds3f& b) {
	return (p.x >= b.pMin.x && p.x < b.pMax.x&&
		p.y >= b.pMin.y && p.y < b.pMax.y&&
		p.z >= b.pMin.z && p.z < b.pMax.z);
};

__host__ __device__ Bounds3f Union(const Bounds3f& b, const float3& p) {
	return Bounds3f(
		make_float3(
			fminf(b.pMin.x, p.x),
			fminf(b.pMin.y, p.y),
			fminf(b.pMin.z, p.z)),
		make_float3(
			fmaxf(b.pMax.x, p.x),
			fmaxf(b.pMax.y, p.y),
			fmaxf(b.pMax.z, p.z)));
};

__host__ __device__ Bounds3f Union(const Bounds3f& b1, const Bounds3f& b2) {
	return Bounds3f(
		make_float3(
			fminf(b1.pMin.x, b2.pMin.x),
			fminf(b1.pMin.y, b2.pMin.y),
			fminf(b1.pMin.z, b2.pMin.z)),
		make_float3(
			fmaxf(b1.pMax.x, b2.pMax.x),
			fmaxf(b1.pMax.y, b2.pMax.y),
			fmaxf(b1.pMax.z, b2.pMax.z)));
};

enum class SplitMethod { SAH, HLBVH, Middle, EqualCounts };

struct BVHPrimitiveInfo
{
	__host__ __device__ BVHPrimitiveInfo() {}

	__host__ __device__ BVHPrimitiveInfo(size_t primitiveNumber, const Bounds3f& bounds) :
		primitiveNumber(primitiveNumber), bounds(bounds),
		centroid(0.5f * bounds.pMin + 0.5f * bounds.pMax)
	{}
	size_t primitiveNumber;
	Bounds3f bounds;
	float3 centroid;
};

struct BVHBuildNode
{
	Bounds3f bounds;
	BVHBuildNode* children[2];
	size_t primitiveNumber;
	int splitAxis, firstPrimOffset, nPrimitives, obj_id;

	__host__ __device__ void init_leaf(int first, int n, const Bounds3f& b)
	{
		firstPrimOffset = first;
		nPrimitives = n;
		bounds = b;
		children[0] = children[1] = nullptr;
	}

	__host__ __device__ void init_interior(int axis, BVHBuildNode* c0, BVHBuildNode* c1)
	{
		children[0] = c0;
		children[1] = c1;
		splitAxis = axis;
		bounds = Union(c0->bounds, c1->bounds);
		nPrimitives = 0;
	}
};

struct LinearBVHNode {
	Bounds3f bounds;
	union {
		int primitivesOffset;    // leaf
		int secondChildOffset;   // interior
	};
	uint16_t nPrimitives;  // 0 -> interior node
	uint8_t axis;          // interior node: xyz
	uint8_t pad[1];        // ensure 32 byte total size
};

__global__ void reorder_primitives(Triangle triangles[], int ordered_prims[], int size) {
	for (int i = 0; i < size; i++) {
		int index = ordered_prims[i];
		g_triangles[i] = triangles[index];
	}
}

__device__ inline bool bounds_hit(float4 pMin, float4 pMax, const Ray& __restrict__ ray, const float3& __restrict__ invDir, const int dirIsNeg[3])
{
	float tmin, tmax, tymin, tymax, tzmin, tzmax;

	tmin = (((dirIsNeg[0] == 0) ? pMin : pMax).x - ray.o.x) * invDir.x;
	tmin = ((((1 - dirIsNeg[0]) == 0) ? pMin : pMax).x - ray.o.x) * invDir.x;

	tymin = (((dirIsNeg[1] == 0) ? pMin : pMax).y - ray.o.y) * invDir.y;
	tymax = ((((1 - dirIsNeg[1]) == 0) ? pMin : pMax).y - ray.o.y) * invDir.y;

	tzmin = (((dirIsNeg[2] == 0) ? pMin : pMax).z - ray.o.z) * invDir.z;
	tzmax = ((((1 - dirIsNeg[2]) == 0) ? pMin : pMax).z - ray.o.z) * invDir.z;

	float tminbox = min_max(tmin, tmax, min_max(tymin, tymax, min_max(tzmin, tzmax, 0)));
	float tmaxbox = max_min(tmin, tmax, max_min(tymin, tymax, max_min(tzmin, tzmax, K_HUGE)));

	return (tminbox <= tmaxbox);
}

/*
__device__ void Intersect(const Ray& __restrict__ ray, ShadeRec& sr, const LinearBVHNode* __restrict__ nodes)
{
	float		t;
	float3		normal;
	float3		local_hit_point;
	float		tmin = K_HUGE;

	float3 invDir = make_float3(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
	int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

	// Follow ray through BVH nodes to find primitive intersections //
	int toVisitOffset = 0, currentNodeIndex = 0;
	int nodesToVisit[64];
	while (true) {

		const float4 pMin = tex1Dfetch(t_BVHbounds, 2 * currentNodeIndex + 0);
		const float4 pMax = tex1Dfetch(t_BVHbounds, 2 * currentNodeIndex + 1);
		
		//const LinearBVHNode node = nodes[currentNodeIndex];

		// Check ray against BVH node //
		if (bounds_hit(pMin, pMax, ray, invDir, dirIsNeg)) {

			const int4 node = tex1Dfetch(t_BVHnodes, currentNodeIndex);

			const int primitivesOffset = node.x;
			const int secondChildOffset = node.y;
			const int nPrimitives = node.z;
			const int axis = node.w;

			if (nPrimitives > 0) {
				// Intersect ray with primitives in leaf BVH node //
				for (int i = 0; i < nPrimitives; ++i) {
					if (g_triangles[primitivesOffset + i].hit(ray, t, sr) && (t < tmin)) {
						sr.hit_an_obj = true;
						sr.material_ptr = g_triangles[primitivesOffset + i].material_ptr;
						tmin = t;
						normal = sr.normal;
						local_hit_point = sr.local_hit_point;
					}
				}
				if (toVisitOffset == 0) break;
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
			else {
				// Put far BVH node on nodesToVisit stack, advance to near node //
				if (dirIsNeg[axis]) {
					nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = secondChildOffset;
				}
				else {
					nodesToVisit[toVisitOffset++] = secondChildOffset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			}
		}
		else {
			if (toVisitOffset == 0) break;
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}
	}

	if (sr.hit_an_obj) {
		sr.t = tmin;
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
		sr.ray = ray;
	}
	
}
*/

/*
__device__ void Intersect(const Ray& __restrict__ ray, ShadeRec& sr, const LinearBVHNode* __restrict__ nodes)
{
	float		t;
	float3		normal;
	float3		local_hit_point;
	float		tmin = K_HUGE;

	float3 invDir = make_float3(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
	int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

	// Follow ray through BVH nodes to find primitive intersections //
	int toVisitOffset = 0, currentNodeIndex = 0;
	int nodesToVisit[64];
	while (true) {

		const LinearBVHNode node = nodes[currentNodeIndex];

		// Check ray against BVH node //
		if (node.bounds.hit(ray, invDir, dirIsNeg)) {
			if (node.nPrimitives > 0) {
				// Intersect ray with primitives in leaf BVH node //

				for (int i = 0; i < node.nPrimitives; ++i) {
					if (g_triangles[node.primitivesOffset + i].hit(ray, t, sr) && (t < tmin)) {
						sr.hit_an_obj = true;
						sr.material_ptr = g_triangles[node.primitivesOffset + i].material_ptr;
						tmin = t;
						normal = sr.normal;
						local_hit_point = sr.local_hit_point;
					}
				}
				if (toVisitOffset == 0) break;
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
			else {
				// Put far BVH node on nodesToVisit stack, advance to near node //
				if (dirIsNeg[node.axis]) {
					nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = node.secondChildOffset;
				}
				else {
					nodesToVisit[toVisitOffset++] = node.secondChildOffset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			}
		}
		else {
			if (toVisitOffset == 0) break;
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}
	}

	if (sr.hit_an_obj) {
		sr.t = tmin;
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
		sr.ray = ray;
	}
}
*/

__device__ void Intersect(const Ray& __restrict ray, ShadeRec& sr, const LinearBVHNode* __restrict nodes)
{
	float		t;
	float3		normal;
	float3		local_hit_point;
	float		tmin = K_HUGE;

	float3 invDir = make_float3(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
	int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

	// Follow ray through BVH nodes to find primitive intersections //
	int toVisitOffset = 0, currentNodeIndex = 0;
	int nodesToVisit[64];

	bool searching = true;

	while (true) {

		const LinearBVHNode node = nodes[currentNodeIndex];
		searching = true;
		// Check ray against BVH node //
		if (node.bounds.hit(ray, invDir, dirIsNeg)) {
			if (node.nPrimitives > 0) {
				searching = false;
				if (!__any(searching)) {
					// Intersect ray with primitives in leaf BVH node //
					for (int i = 0; i < node.nPrimitives; ++i) {
						if (g_triangles[node.primitivesOffset + i].hit(ray, t, sr) && (t < tmin)) {
							sr.hit_an_obj = true;
							sr.material_ptr = g_triangles[node.primitivesOffset + i].material_ptr;
							tmin = t;
							normal = sr.normal;
							local_hit_point = sr.local_hit_point;
						}
					}
				}
				if (toVisitOffset == 0) break;
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
			else {
				// Put far BVH node on nodesToVisit stack, advance to near node //
				if (dirIsNeg[node.axis]) {
					nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = node.secondChildOffset;
				}
				else {
					nodesToVisit[toVisitOffset++] = node.secondChildOffset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			}
		}
		else {
			if (toVisitOffset == 0) break;
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}
	}

	if (sr.hit_an_obj) {
		sr.t = tmin;
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
		sr.ray = ray;
		//sr.id = ray.id;
	}
}

__device__ void Intersect(const Ray& __restrict ray, Isect& isect, const LinearBVHNode* __restrict nodes)
{
	float		t;
	float3		normal;
	float3		local_hit_point;
	float		tmin = K_HUGE;

	float3 invDir = make_float3(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
	int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

	// Follow ray through BVH nodes to find primitive intersections //
	int toVisitOffset = 0, currentNodeIndex = 0;
	int nodesToVisit[64];

	bool searching = true;

	while (true) {
		const LinearBVHNode node = nodes[currentNodeIndex];
		searching = true;
		// Check ray against BVH node //
		if (node.bounds.hit(ray, invDir, dirIsNeg)) {
			if (node.nPrimitives > 0) {
				searching = false;
				if (!__any(searching)) {
					// Intersect ray with primitives in leaf BVH node //
					for (int i = 0; i < node.nPrimitives; ++i) {
						if (g_triangles[node.primitivesOffset + i].hit(ray, t, isect) && (t < tmin)) {
							isect.wasFound = true;
							isect.materialIndex = 0;
							tmin = t;
							normal = isect.normal;
							local_hit_point = isect.position;
						}
					}
				}
				if (toVisitOffset == 0) break;
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
			else {
				// Put far BVH node on nodesToVisit stack, advance to near node //
				if (dirIsNeg[node.axis]) {
					nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = node.secondChildOffset;
				}
				else {
					nodesToVisit[toVisitOffset++] = node.secondChildOffset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			}
		}
		else {
			if (toVisitOffset == 0) break;
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}
	}

	if (isect.wasFound) {
		isect.distance = tmin;
		isect.normal = normal;
		isect.position = local_hit_point;
	}
}

__device__ bool shadow_hit(const Ray& __restrict__ ray, float& tmin, const LinearBVHNode* __restrict__ nodes)
{
	float		t;

	float3 invDir = make_float3(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
	int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

	// Follow ray through BVH nodes to find primitive intersections //
	int toVisitOffset = 0, currentNodeIndex = 0;
	int nodesToVisit[64];
	while (true) {
		const LinearBVHNode* node = &nodes[currentNodeIndex];
		// Check ray against BVH node //
		if (node->bounds.hit(ray, invDir, dirIsNeg)) {
			if (node->nPrimitives > 0) {
				// Intersect ray with primitives in leaf BVH node //
				for (int i = 0; i < node->nPrimitives; ++i) {
					if (g_triangles[node->primitivesOffset + i].shadow_hit(ray, t) && (t < tmin)) {
						return (true);
					}
				}
				if (toVisitOffset == 0) break;
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
			else {
				// Put far BVH node on nodesToVisit stack, advance to near node //
				if (dirIsNeg[node->axis]) {
					nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
					currentNodeIndex = node->secondChildOffset;
				}
				else {
					nodesToVisit[toVisitOffset++] = node->secondChildOffset;
					currentNodeIndex = currentNodeIndex + 1;
				}
			}
		}
		else {
			if (toVisitOffset == 0) break;
			currentNodeIndex = nodesToVisit[--toVisitOffset];
		}
	}

	return (false);
}

class BVHAccel {
public:
	__host__ BVHAccel(std::vector<BVHPrimitiveInfo> primitiveInfo, Triangle* d_triangles, SplitMethod splitMethod, int maxPrimsInNode) :
		maxPrimsInNode(maxPrimsInNode), splitMethod(splitMethod)
	{
		int totalNodes = 0;
		std::vector<int> orderedPrims;

		MemoryArena arena(1024 * 1024);

		root = recursive_build(arena, primitiveInfo, 0, primitiveInfo.size(), &totalNodes, orderedPrims);

		nodes = AllocAligned<LinearBVHNode>(totalNodes);
		int offset = 0;
		flattenBVHTree(root, &offset);
		prepareBVHForGPU();

		size_t node_size = sizeof(LinearBVHNode) * totalNodes;
		checkCudaErrors(cudaMalloc((void**)&d_nodes, node_size));
		checkCudaErrors(cudaMemcpy(d_nodes, nodes, node_size, cudaMemcpyHostToDevice));

		size_t orderedPrims_size = sizeof(int) * orderedPrims.size();
		checkCudaErrors(cudaMalloc((void**)&d_orderedPrims, orderedPrims_size));
		checkCudaErrors(cudaMemcpy(d_orderedPrims, orderedPrims.data(), orderedPrims_size, cudaMemcpyHostToDevice));

		Triangle* d_triangle_ptr;
		checkCudaErrors(cudaMalloc((void**)&d_triangle_ptr, sizeof(Triangle)*orderedPrims.size()));
		checkCudaErrors(cudaMemcpyToSymbol(g_triangles, &d_triangle_ptr, sizeof(Triangle*)));

		reorder_primitives <<< 1, 1 >>> (d_triangles, d_orderedPrims, orderedPrims.size());
	};

	__host__ BVHBuildNode* recursive_build(MemoryArena& arena, std::vector<BVHPrimitiveInfo>& primitiveInfo, int start, int end, int* totalNodes, std::vector<int>& orderedPrims)
	{
		BVHBuildNode* node = arena.Alloc<BVHBuildNode>();
		(*totalNodes)++;

		Bounds3f bounds;
		for (int i = start; i < end; ++i)
			bounds = Union(bounds, primitiveInfo[i].bounds);

		int nPrimitives = end - start;
		if (nPrimitives == 1) {
			// Create leaf BVHBuildNode //
			int firstPrimOffset = orderedPrims.size();
			for (int i = start; i < end; ++i) {
				int primNum = primitiveInfo[i].primitiveNumber;
				orderedPrims.push_back(primNum);
			}
			node->init_leaf(firstPrimOffset, nPrimitives, bounds);
			return node;
		}
		else {
			// Compute bound of primitive centroids, choose split dimension dim //
			Bounds3f centroidBounds;
			for (int i = start; i < end; ++i)
				centroidBounds = Union(centroidBounds, primitiveInfo[i].centroid);
			int dim = centroidBounds.maximum_extent();

			// Partition primitives into two sets and build children //
			int mid = (start + end) / 2;

			float con;
			switch (dim) {
			case 0:
				con = centroidBounds.pMax.x == centroidBounds.pMin.x;
				break;
			case 1:
				con = centroidBounds.pMax.y == centroidBounds.pMin.y;
				break;
			default:
				con = centroidBounds.pMax.z == centroidBounds.pMin.z;
			}
			if (con) {
				//if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
					// Create leaf BVHBuildNode //
				int firstPrimOffset = orderedPrims.size();
				for (int i = start; i < end; ++i) {
					int primNum = primitiveInfo[i].primitiveNumber;
					orderedPrims.push_back(primNum);
				}
				node->init_leaf(firstPrimOffset, nPrimitives, bounds);
				return node;
			}
			else {
				// Partition primitives based on splitMethod //
				switch (splitMethod) {
				case SplitMethod::Middle: {
					// Partition primitives through node’s midpoint /
					float pmid;
					switch (dim) {
					case 0:
						pmid = (centroidBounds.pMin.x + centroidBounds.pMax.x) / 2;
						break;
					case 1:
						pmid = (centroidBounds.pMin.y + centroidBounds.pMax.y) / 2;
						break;
					default:
						pmid = (centroidBounds.pMin.z + centroidBounds.pMax.z) / 2;
					}
					//float pmid = (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2;
					BVHPrimitiveInfo* midPtr = std::partition(
						&primitiveInfo[start], &primitiveInfo[end - 1] + 1,
						[dim, pmid](const BVHPrimitiveInfo& pi) {
							switch (dim) {
							case 0:
								return pi.centroid.x < pmid;
							case 1:
								return pi.centroid.y < pmid;
							default:
								return pi.centroid.z < pmid;
							}
							//return pi.centroid[dim] < pmid;
						});
					mid = midPtr - &primitiveInfo[0];
					if (mid != start && mid != end) break;
				}
				case SplitMethod::EqualCounts: {
					// Partition primitives into equally sized subsets //
					mid = (start + end) / 2;
					std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
						&primitiveInfo[end - 1] + 1,
						[dim](const BVHPrimitiveInfo& a,
							const BVHPrimitiveInfo& b) {
								switch (dim) {
								case 0:
									return a.centroid.x < b.centroid.x;
								case 1:
									return a.centroid.y < b.centroid.y;
								default:
									return a.centroid.z < b.centroid.z;
								}
								//return a.centroid[dim] < b.centroid[dim];
						});

					break;
				}
				case SplitMethod::SAH:
				default: {
					// Partition primitives using approximate SAH //
					if (nPrimitives <= 2) {
						// Partition primitives into equally sized subsets //
						mid = (start + end) / 2;
						std::nth_element(
							&primitiveInfo[start],
							&primitiveInfo[mid],
							&primitiveInfo[end - 1] + 1,
							[dim](const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b) {
								switch (dim) {
								case 0:
									return a.centroid.x < b.centroid.x;
								case 1:
									return a.centroid.y < b.centroid.y;
								default:
									return a.centroid.z < b.centroid.z;
								}
								//return a.centroid[dim] < b.centroid[dim];
							});
					}
					else {

						// Allocate BucketInfo for SAH partition buckets //
						constexpr int nBuckets = 12;
						struct BucketInfo {
							int count = 0;
							Bounds3f bounds;
						};
						BucketInfo buckets[nBuckets];

						// Initialize BucketInfo for SAH partition buckets //
						for (int i = start; i < end; ++i) {
							//int b = nBuckets * centroidBounds.offset(primitiveInfo[i].centroid)[dim];
							int b;
							switch (dim) {
							case 0:
								b = nBuckets * centroidBounds.offset(primitiveInfo[i].centroid).x;
								break;
							case 1:
								b = nBuckets * centroidBounds.offset(primitiveInfo[i].centroid).y;
								break;
							default:
								b = nBuckets * centroidBounds.offset(primitiveInfo[i].centroid).z;
							}
							if (b == nBuckets) b = nBuckets - 1;
							buckets[b].count++;
							buckets[b].bounds = Union(buckets[b].bounds, primitiveInfo[i].bounds);
						}

						// Compute costs for splitting after each bucket //
						float cost[nBuckets - 1];
						for (int i = 0; i < nBuckets - 1; ++i) {
							Bounds3f b0, b1;
							int count0 = 0, count1 = 0;
							for (int j = 0; j <= i; ++j) {
								b0 = Union(b0, buckets[j].bounds);
								count0 += buckets[j].count;
							}
							for (int j = i + 1; j < nBuckets; ++j) {
								b1 = Union(b1, buckets[j].bounds);
								count1 += buckets[j].count;
							}
							cost[i] = .125f + (count0 * b0.surface_area() + count1 * b1.surface_area()) / bounds.surface_area();
						}

						// Find bucket to split at that minimizes SAH metric //
						float minCost = cost[0];
						int minCostSplitBucket = 0;
						for (int i = 1; i < nBuckets - 1; ++i) {
							if (cost[i] < minCost) {
								minCost = cost[i];
								minCostSplitBucket = i;
							}
						}

						// Either create leaf or split primitives at selected SAH bucket //
						float leafCost = nPrimitives;
						if (nPrimitives > maxPrimsInNode || minCost < leafCost) {
							BVHPrimitiveInfo* pmid = std::partition(
								&primitiveInfo[start],
								&primitiveInfo[end - 1] + 1,
								[=](const BVHPrimitiveInfo& pi) {
									//int b = nBuckets * centroidBounds.offset(pi.centroid)[dim];
									int b;
									switch (dim) {
									case 0:
										b = nBuckets * centroidBounds.offset(pi.centroid).x;
										break;
									case 1:
										b = nBuckets * centroidBounds.offset(pi.centroid).y;
										break;
									default:
										b = nBuckets * centroidBounds.offset(pi.centroid).z;
									}
									if (b == nBuckets) b = nBuckets - 1;
									return b <= minCostSplitBucket;
								});
							mid = pmid - &primitiveInfo[0];
						}
						else {
							// Create leaf BVHBuildNode //
							int firstPrimOffset = orderedPrims.size();
							for (int i = start; i < end; ++i) {
								int primNum = primitiveInfo[i].primitiveNumber;
								orderedPrims.push_back(primNum);
							}
							node->init_leaf(firstPrimOffset, nPrimitives, bounds);
							return node;
						}
					}
					break;
				}
				}
				auto child0 = recursive_build(arena, primitiveInfo, start, mid, totalNodes, orderedPrims);
				auto child1 = recursive_build(arena, primitiveInfo, mid, end, totalNodes, orderedPrims);
				node->init_interior(dim, child0, child1);
			}
		}
		return node;
	}

	void prepareBVHForGPU() {
		// Loading BVH into textures.
		// textures are better for incoherent cache access

		// Bounding boxes
		// float3 pMin	12 bytes
		// float3 pMax	12 bytes

		// 1. Init data on host
		float* d_BVHbounds = NULL;
		size_t BVHbounds_size = size_t(num_nodes * 8 * sizeof(float));
		float* h_BVHbounds = (float*)malloc(BVHbounds_size);

		for (unsigned i = 0; i < num_nodes; i++) {
			h_BVHbounds[8 * i + 0] = nodes[i].bounds.pMin.x;
			h_BVHbounds[8 * i + 1] = nodes[i].bounds.pMin.y;
			h_BVHbounds[8 * i + 2] = nodes[i].bounds.pMin.z;
			h_BVHbounds[8 * i + 3] = 0.f;

			h_BVHbounds[8 * i + 4] = nodes[i].bounds.pMax.x;
			h_BVHbounds[8 * i + 5] = nodes[i].bounds.pMax.y;
			h_BVHbounds[8 * i + 6] = nodes[i].bounds.pMax.z;
			h_BVHbounds[8 * i + 7] = 0.f;
		}

		// 2. Move data from host to device
		checkCudaErrors(cudaMalloc((void**)&d_BVHbounds, BVHbounds_size));
		checkCudaErrors(cudaMemcpy(d_BVHbounds, h_BVHbounds, BVHbounds_size, cudaMemcpyHostToDevice));

		// 3. Bind to texture
		cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTexture(size_t(0), &t_BVHbounds, d_BVHbounds, &channel1desc, BVHbounds_size));

		// BVH Nodes
		// int primitivesOffset;    4 bytes
		// int secondChildOffset;   4 bytes
		// uint16_t nPrimitives;    2 bytes
		// uint8_t axis;			1 byte

		// 1. Init data on host
		int* d_BVHnodes = NULL;
		size_t BVHnodes_size = size_t(num_nodes * 4 * sizeof(int));
		int* h_BVHnodes = (int*)malloc(BVHnodes_size);

		for (unsigned i = 0; i < num_nodes; i++) {
			h_BVHnodes[4 * i + 0] = nodes[i].primitivesOffset;
			h_BVHnodes[4 * i + 1] = nodes[i].secondChildOffset;
			h_BVHnodes[4 * i + 2] = nodes[i].nPrimitives;
			h_BVHnodes[4 * i + 3] = nodes[i].axis;
		}

		// 2. Move data from host to device
		checkCudaErrors(cudaMalloc((void**)&d_BVHnodes, BVHnodes_size));
		checkCudaErrors(cudaMemcpy(d_BVHnodes, h_BVHnodes, BVHnodes_size, cudaMemcpyHostToDevice));

		// 3. Bind to texture
		cudaChannelFormatDesc channel2desc = cudaCreateChannelDesc<int4>();
		checkCudaErrors(cudaBindTexture(size_t(0), &t_BVHnodes, d_BVHnodes, &channel2desc, BVHnodes_size));
	}

	__host__ int flattenBVHTree(BVHBuildNode* node, int* offset) {
		LinearBVHNode* linearNode = &nodes[*offset];
		num_nodes++;
		linearNode->bounds = node->bounds;
		int myOffset = (*offset)++;
		if (node->nPrimitives > 0)
		{
			linearNode->primitivesOffset = node->firstPrimOffset;
			linearNode->nPrimitives = node->nPrimitives;
		}
		else
		{
			// Create interior flattened BVH node //
			linearNode->axis = node->splitAxis;
			linearNode->nPrimitives = 0;
			flattenBVHTree(node->children[0], offset);
			linearNode->secondChildOffset = flattenBVHTree(node->children[1], offset);
		}
		return myOffset;
	}

	const int maxPrimsInNode;
	const SplitMethod splitMethod;
	BVHBuildNode* root;
	LinearBVHNode* nodes = nullptr;
	LinearBVHNode* d_nodes;
	int* d_orderedPrims;
	int num_nodes = 0;
};