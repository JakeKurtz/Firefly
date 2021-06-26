#pragma once

#include <bitset>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <glm/glm.hpp>

#include <vector>
#include <algorithm>

#include "../GeometricObjects/GeometricObj.h"
#include "../GeometricObjects/Compound/Mesh.h"
#include "../Acceleration/Bounds3.h"
#include "../Utilities/CudaList.h"
#include "../Utilities/MemoryArena.h"

__constant__ const int MAX_LEAF = 100;

enum class SplitMethod { SAH, HLBVH, Middle, EqualCounts };

int counter_0 = 0;
int counter_1 = 0;

struct BVHPrimitiveInfo
{
	__host__ __device__ BVHPrimitiveInfo() {}

	__host__ __device__ BVHPrimitiveInfo(size_t primitiveNumber, const Bounds3f& bounds) :
		primitiveNumber(primitiveNumber), bounds(bounds),
		centroid(0.5f * bounds.pMin + 0.5f * bounds.pMax)
	{}
	size_t primitiveNumber;
	Bounds3f bounds;
	glm::vec3 centroid;
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

__device__ void Intersect(const Ray& ray, ShadeRec& sr, const LinearBVHNode* nodes, const CudaList<GeometricObj*> primitives) 
{
	double		t;
	glm::dvec3	normal;
	glm::vec3	local_hit_point;
	double		tmin = K_HUGE;

	vec3 invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
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
					if (primitives[node->primitivesOffset + i]->hit(ray, t, sr) && (t < tmin)) {
						sr.hit_an_obj = true;
						tmin = t;
						sr.material_ptr = primitives[node->primitivesOffset + i]->get_material();
						normal = sr.normal;
						local_hit_point = sr.local_hit_point;
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

	// BRUTE FORCE
	/*
	for (int i = 0; i < primitives.size(); ++i) {
		if (primitives[i]->hit(ray, t, sr) && (t < tmin)) {
			sr.hit_an_obj = true;
			tmin = t;
			sr.material_ptr = primitives[i]->get_material();
			normal = sr.normal;
			local_hit_point = sr.local_hit_point;
		}
	}*/

	if (sr.hit_an_obj) {
		sr.t = tmin;
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
	}
}

__device__ bool shadow_hit(const Ray& ray, double& tmin, const LinearBVHNode* nodes, const CudaList<GeometricObj*> primitives)
{
	double		t;
	
	vec3 invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
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
					if (primitives[node->primitivesOffset + i]->shadow_hit(ray, t) && (t < tmin)) {
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
	
	// BRUTE FORCE
	/*
	for (int i = 0; i < primitives.size(); ++i) {
		if (primitives[i]->shadow_hit(ray, t) && (t < tmin)) {
			return true;
		}
	}*/

	return (false);
}

class BVHAccel {
public:
	__host__ BVHAccel(std::vector<BVHPrimitiveInfo> primitiveInfo, SplitMethod splitMethod, int maxPrimsInNode) :
		maxPrimsInNode(maxPrimsInNode), splitMethod(splitMethod)
	{
		int totalNodes = 0;
		std::vector<int> orderedPrims;

		MemoryArena arena(1024 * 1024);

		root = recursive_build(arena, primitiveInfo, 0, primitiveInfo.size(), &totalNodes, orderedPrims);

		nodes = AllocAligned<LinearBVHNode>(totalNodes);
		int offset = 0;
		flattenBVHTree(root, &offset);

		size_t node_size = sizeof(LinearBVHNode) * totalNodes;
		checkCudaErrors(cudaMalloc((void**)&d_nodes, node_size));
		checkCudaErrors(cudaMemcpy(d_nodes, nodes, node_size, cudaMemcpyHostToDevice));

		size_t orderedPrims_size = sizeof(int) * orderedPrims.size();
		checkCudaErrors(cudaMalloc((void**)&d_orderedPrims, orderedPrims_size));
		checkCudaErrors(cudaMemcpy(d_orderedPrims, orderedPrims.data(), orderedPrims_size, cudaMemcpyHostToDevice));
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
			if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
				// Create leaf BVHBuildNode //
				int firstPrimOffset = orderedPrims.size();
				for (int i = start; i < end; ++i) {
					int primNum = primitiveInfo[i].primitiveNumber;
					orderedPrims.push_back(primNum);
				}
				node->init_leaf(firstPrimOffset, nPrimitives, bounds);
				return node;
			} else {
				// Partition primitives based on splitMethod //
				switch (splitMethod) {
				case SplitMethod::Middle: {
					// Partition primitives through node’s midpoint /
					float pmid = 
						(centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2;
					BVHPrimitiveInfo* midPtr = std::partition(
						&primitiveInfo[start], &primitiveInfo[end - 1] + 1,
						[dim, pmid](const BVHPrimitiveInfo& pi) {
							return pi.centroid[dim] < pmid;
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
										return a.centroid[dim] < b.centroid[dim];
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
							return a.centroid[dim] < b.centroid[dim];
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
								int b = nBuckets * centroidBounds.offset(primitiveInfo[i].centroid)[dim];
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
							BVHPrimitiveInfo* pmid = std::partition(&primitiveInfo[start],
								&primitiveInfo[end - 1] + 1,
								[=](const BVHPrimitiveInfo& pi) {
									int b = nBuckets * centroidBounds.offset(pi.centroid)[dim];
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

	__host__ int flattenBVHTree(BVHBuildNode* node, int* offset) {
		LinearBVHNode* linearNode = &nodes[*offset];
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
};