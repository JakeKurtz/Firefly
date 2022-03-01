#include "dTriangle.cuh"
#include "dMath.cuh"
#include "BVH.h"

__device__ dTriangle::dTriangle(void)
{
	v0.position = make_float3(0, 0, 0);
	v1.position = make_float3(0, 0, 1);
	v2.position = make_float3(1, 0, 0);

	init();
};

__device__ dTriangle::dTriangle(const dVertex v0, const dVertex v1, const dVertex v2) :
	v0(v0), v1(v1), v2(v2)
{
	init();
};

__device__ void dTriangle::init() {
	float3 v0v1 = v1.position - v0.position;
	float3 v0v2 = v2.position - v0.position;

	float3 ortho = cross(v0v1, v0v2);

	double area = length(ortho) * 0.5;
	inv_area = 1.0 / area;

	face_normal = normalize(ortho);
}

__device__ bool dTriangle::intersect(const dRay& ray, float& u, float& v, float& t) const {
	float3 e1 = v1.position - v0.position;
	float3 e2 = v2.position - v0.position;

	float3 pvec = cross(ray.d, e2);
	double det = dot(e1, pvec);

#ifdef TEST_CULL
	if (det < K_EPSILON)
		return false;

	float3 tvec = ray.o - v0.Position;
	u = dot(tvec, pvec);
	if (u < 0.0 || u > det)
		return false;

	float3 qvec = cross(tvec, e1);
	v = dot(ray.d, qvec);
	if (v < 0.0 || u + v > det)
		return false;

	t = dot(e2, qvec);

	double inv_det = 1.0 / det;

	t *= inv_det;
	u *= inv_det;
	v *= inv_det;
#else
	if (det > -K_EPSILON && det < K_EPSILON)
		return false;

	double inv_det = 1.0 / det;

	float3 tvec = ray.o - v0.position;
	u = dot(tvec, pvec) * inv_det;
	if (u < 0.0 || u > 1.0)
		return false;

	float3 qvec = cross(tvec, e1);
	v = dot(ray.d, qvec) * inv_det;
	if (v < 0.0 || u + v > 1.0)
		return false;

	t = dot(e2, qvec) * inv_det;
#endif
	return true;
}

__device__ bool dTriangle::hit(const dRay& ray) const
{
	float u, v, t;
	return intersect(ray, u, v, t);
}

__device__ bool dTriangle::hit(const dRay& ray, float& tmin, Isect& isect) const
{
	float u, v;
	bool hit = intersect(ray, u, v, tmin);

	float3 normal = normalize(u * v1.normal + v * v2.normal + (1 - u - v) * v0.normal);
	float2 texcoord = u * v1.texcoords + v * v2.texcoords + (1 - u - v) * v0.texcoords;

	if (tmin < 0) return false;

	isect.normal = normal;
	isect.texcoord = texcoord;
	isect.position = ray.o + (tmin * ray.d);

	return hit;
};

__device__ bool dTriangle::shadow_hit(const dRay& ray, float& tmin) const
{
	float3 e1 = v1.position - v0.position;
	float3 e2 = v2.position - v0.position;

	float3 pvec = cross(ray.d, e2);
	double det = dot(e1, pvec);

	if (det > -K_EPSILON && det < K_EPSILON)
		return false;

	double inv_det = 1.0 / det;

	float3 tvec = ray.o - v0.position;
	double u = dot(tvec, pvec) * inv_det;
	if (u < 0.0 || u > 1.0)
		return false;

	float3 qvec = cross(tvec, e1);
	double v = dot(ray.d, qvec) * inv_det;
	if (v < 0.0 || u + v > 1.0)
		return false;

	tmin = dot(e2, qvec) * inv_det;

	if (tmin < 0) return false;

	return true;
};

__device__ void intersect(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& __restrict ray, Isect& isect)
{
	float		t;
	int			triangle_id;
	float3		normal;
	float2		texcoord;
	float3		local_hit_point;
	float		tmin = K_HUGE;

	float3 invDir = make_float3(1.f / (float)ray.d.x, 1.f / (float)ray.d.y, 1.f / (float)ray.d.z);
	int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

	// Follow ray through BVH nodes to find primitive intersections //
	int toVisitOffset = 0, currentNodeIndex = 0;
	int nodesToVisit[64];

	bool searching = true;

	while (true) {
		const LinearBVHNode* node = &nodes[currentNodeIndex];
		searching = true;
		// Check ray against BVH node //
		if (node->bounds.hit(ray, invDir, dirIsNeg)) {
			if (node->nPrimitives > 0) {
				searching = false;
				if (!__any(searching)) {
					// Intersect ray with primitives in leaf BVH node //
					for (int i = 0; i < node->nPrimitives; ++i) {
						int triangle_id = node->primitivesOffset + i;
						if (triangles[triangle_id].hit(ray, t, isect) && (t < tmin) && (t > 0.00001)) {
							isect.wasFound = true;
							triangle_id = node->primitivesOffset + i;
							isect.material = triangles[triangle_id].material;
							tmin = t;
							normal = isect.normal;
							texcoord = isect.texcoord;
							local_hit_point = isect.position;
						}
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

	if (isect.wasFound) {
		isect.distance = tmin;
		isect.triangle_id = triangle_id;
		isect.normal = normal;
		isect.texcoord = texcoord;
		isect.position = local_hit_point;
	}
}