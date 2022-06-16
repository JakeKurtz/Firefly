#include "dTriangle.cuh"
#include "dMath.cuh"
#include "BVH.h"

//#define TEST_CULL

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

__device__ bool dTriangle::intersect(const dRay& ray, float& u, float& v, float& t) const 
{
	float3 e1 = v1.position - v0.position;
	float3 e2 = v2.position - v0.position;

	float3 pvec = cross(ray.d, e2);
	double det = dot(e1, pvec);

#ifdef TEST_CULL
	if (det < K_EPSILON)
		return false;

	float3 tvec = ray.o - v0.position;
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
	// apply inverse transformation matrix to ray
	dRay ray_p = transform->inv_matrix * ray;

	float u, v, t;
	return intersect(ray_p, u, v, t);
}

__device__ bool dTriangle::hit(const dRay& ray, float& tmin, Isect& isect) const
{
	// apply inverse transformation matrix to ray
	dRay ray_p = transform->inv_matrix * ray;

	//transform->inv_matrix.print();

	//printf("o:\t(%f,%f,%f)\no':\t(%f,%f,%f)\n\n", ray.o.x, ray.o.y, ray.o.z, ray_p.o.x, ray_p.o.y, ray_p.o.z);

	float u, v;
	bool hit = intersect(ray_p, u, v, tmin);

	float3 normal = normalize(u * v1.normal + v * v2.normal + (1 - u - v) * v0.normal);
	float3 tangent = normalize(u * v1.tangent + v * v2.tangent + (1 - u - v) * v0.tangent);
	float3 bitangent = normalize(u * v1.bitangent + v * v2.bitangent + (1 - u - v) * v0.bitangent);
	float2 texcoord = u * v1.texcoords + v * v2.texcoords + (1 - u - v) * v0.texcoords;

	if (tmin < 0) return false;

	// apply transformation to normal, tangent, bitangent, position.

	isect.normal = normalize(float3_cast(transform->matrix * make_float4(normal, 0.f)));
	isect.tangent = normalize(float3_cast(transform->matrix * make_float4(tangent, 0.f)));
	isect.bitangent = normalize(float3_cast(transform->matrix * make_float4(bitangent, 0.f)));
	isect.texcoord = texcoord;
	isect.position = transform->matrix * (ray_p.o + (tmin * ray_p.d));

	return hit;
};

__device__ bool dTriangle::shadow_hit(const dRay& ray, float& tmin) const
{
	dRay ray_p = transform->inv_matrix * ray;

	float3 e1 = v1.position - v0.position;
	float3 e2 = v2.position - v0.position;

	float3 pvec = cross(ray_p.d, e2);
	double det = dot(e1, pvec);

	if (det > -K_EPSILON && det < K_EPSILON)
		return false;

	double inv_det = 1.0 / det;

	float3 tvec = ray_p.o - v0.position;
	double u = dot(tvec, pvec) * inv_det;
	if (u < 0.0 || u > 1.0)
		return false;

	float3 qvec = cross(tvec, e1);
	double v = dot(ray_p.d, qvec) * inv_det;
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
	float3		tangent;
	float3		bitangent;
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
						if (triangles[node->primitivesOffset + i].hit(ray, t, isect) && (t < tmin)) {
							isect.wasFound = true;
							triangle_id = node->primitivesOffset + i;
							isect.material = triangles[node->primitivesOffset + i].material;
							tmin = t;
							normal = isect.normal;
							tangent = isect.tangent;
							bitangent = isect.bitangent;
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
		isect.tangent = tangent;
		isect.bitangent = bitangent;
		isect.texcoord = texcoord;
		isect.position = local_hit_point;
	}
}

__device__ bool intersect_shadows(const LinearBVHNode* nodes, const dTriangle* triangles, const dRay& __restrict ray, float& tmin)
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
					int triangle_id = node->primitivesOffset + i;
					if (triangles[node->primitivesOffset + i].shadow_hit(ray, t) && (t < tmin)) {
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