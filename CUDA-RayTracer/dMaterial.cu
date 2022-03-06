#include "dMaterial.cuh"
#include "Isect.cuh"
#include "dRandom.cuh"
#include "dMath.cuh"
#include "dLight.cuh"
#include "dRay.cuh"

__device__ float3 get_albedo(const Isect& isect) {
	int id = isect.material->baseColorTexture;

	float3 albedo;
	if (id == -1) {
		albedo = isect.material->baseColorFactor;
	}
	else {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		float4 tex = tex2DLod<float4>(id, u, v, 0);
		albedo = make_float3(tex.x, tex.y, tex.z);
	}
	return albedo;
}
__device__ float get_roughness(const Isect& isect) {
	int id = isect.material->roughnessTexture;

	float r;
	if (id == -1) {
		r = isect.material->roughnessFactor;
	}
	else {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		float4 tex = tex2DLod<float4>(id, u, v, 0);
		r = tex.x;
	}
	return r;
}
__device__ float get_metallic(const Isect& isect) {
	int id = isect.material->metallicTexture;

	float m;
	if (id == -1) {
		m = isect.material->metallicFactor;
	}
	else {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		float4 tex = tex2DLod<float4>(id, u, v, 0);
		m = tex.x;
	}
	return 0.f;
}
__device__ float3 get_normal(const Isect& isect) {
	int id = isect.material->normalTexture;

	float3 normal;
	if (id == -1) {
		normal = isect.normal;
	}
	else {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		float4 tex = tex2DLod<float4>(id, u, v, 0);
		normal = make_float3(tex.x, tex.y, tex.z);
	}
	return normal;
}

__device__ float3 refract(const float3& I, const float3& N, const float& ior)
{
	float cosi = clamp(-1.0, 1.0, dot(I, N));
	float etai = 1, etat = ior;
	float3 n = N;
	if (cosi < 0) { cosi = -cosi; }
	else {
		//std::swap(etai, etat);
		float tmp = etat;
		etat = etai;
		etai = tmp;
		n = -N;
	}
	float eta = etai / etat;
	float k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0 ? make_float3(0) : eta * I + (eta * cosi - sqrtf(k)) * n;
}
__device__ double power_heuristic(int nf, double fPdf, int ng, double gPdf)
{
	double f = nf * fPdf;
	double g = ng * gPdf;
	return (f * f) / (f * f + g * g);
}
__device__ double ggxtr_ndf(float3 n, float3 h, float r)
{
	double a2 = pow(r * r, 2);
	double NH2 = pow(fmaxf(0.0, dot(n, h)), 2);
	return a2 / (M_PI * (pow(NH2 * (a2 - 1.f) + 1.f, 2)));
};
__device__ double geo_atten(float3 wi, float3 wo, float3 n, float r)
{
	double k = pow(r + 1.f, 2.f) / 8.f;

	double NL = fmaxf(dot(n, wi), 0.0);
	double NV = fmaxf(dot(n, wo), 0.0);

	double G1 = NL / (NL * (1.f - k) + k);
	double G2 = NV / (NV * (1.f - k) + k);

	return G1 * G2;
};

__device__ float3 diff_sample(const Isect& isect)
{
	float e0 = random();
	float e1 = random();

	float sinTheta = sqrtf(1.f - e0 * e0);
	float phi = 2.f * M_PI * e1;
	float x = sinTheta * cos(phi);
	float z = sinTheta * sin(phi);
	float3 sp = make_float3(x, e0, z);

	float3 N = isect.normal;//get_normal(isect);
	float3 T = normalize(cross(N, get_orthogonal_vec(N)));
	float3 B = normalize(cross(N, T));

	//createCoordinateSystem(N, T, B);

	float3 wi = normalize(T * sp.x + N * sp.y + B * sp.z);

	return (wi);
};
__device__ double diff_get_pdf()
{
	return 1.f / (2.f * M_PI);
};
__device__ float3 diff_f(const Isect& isect, const float3& wi, const float3& wo)
{
	//float n_dot_wi = fabsf(dot(isect.normal, wi));
	//float n_dot_wo = fabsf(dot(isect.normal, wo));
	//return (28.f / (23.f * M_PI)) * isect.material->baseColor * (make_float3(1.f) - isect.material->fresnel) * (1 - powf(1 - .5f * n_dot_wi, 5)) * (1 - powf(1 - .5f * n_dot_wo, 5));
	//return (isect.material->baseColor * M_1_PI);

	/*float m = get_metallic(isect);
	float3 a = get_albedo(isect);
	float3 wh = normalize(wo + wi);
	float3 f0 = lerp(a, isect.material->fresnel, m);
	float3 F = fresnel(f0, wh, wi);
	float3 kD = make_float3(1.f) - F;
	kD *= 1.0 - m;
	*/

	//return (get_albedo(isect) * M_1_PI * isect.material->kd);
	//return (make_float3(1,0,0) * M_1_PI * 1.f);
	return (get_albedo(isect) * M_1_PI * 1.f);
	//return (isect.position);// *M_1_PI * 1.f);
};
__device__ float3 diff_L(dLight** lights, const Isect& isect, const float3& wi, const float3& wo, int light_id, const float3& sample_point)
{
	float3 L = make_float3(0, 0, 0);

	//for (int i = 0; i < 5; i++) {
	float n_dot_wi = fmaxf(dot(normalize(isect.normal), wi), 0.f);
	//float n_dot_wi = fmaxf(dot(get_normal(isect), wi), 0.f);

	//	float n_dot_wi = fabsf(dot(isect.normal, wi));
	//	float n_dot_wo = fabsf(dot(isect.normal, wo));

	if (n_dot_wi > 0.f) {

		float3 f = diff_f(isect, wi, wo) * n_dot_wi;
		float3 Li = lights[light_id]->L(isect, wi, sample_point);

		//L = (28.f / (23.f * M_PI)) * isect.material->baseColor * (make_float3(1.f) - isect.material->fresnel) * (1 - powf(1 - .5f * n_dot_wi, 5)) * (1 - powf(1 - .5f * n_dot_wo, 5)) * Li;
		//L /= n_dot_wi * M_1_PI;
		L += f * Li * n_dot_wi / diff_get_pdf();
	}
	//}

	return L;
}

__device__ float3 ct_sample(const Isect& isect, const float3& wo)
{
	float r = get_roughness(isect);

	float e0 = random();
	float e1 = random();

	double theta = atan(r * r * sqrtf(e0 / (1.f - e0)));
	double phi = 2 * M_PI * e1;

	float3 h = make_float3(
		sin(theta) * cos(phi),
		cos(theta),
		sin(theta) * sin(phi)
	);

	float3 N = isect.normal;//get_normal(isect);
	float3 T = normalize(cross(N, get_orthogonal_vec(N)));
	float3 B = normalize(cross(N, T));

	float3 sample = T * h.x + N * h.y + B * h.z;

	float3 wi = -reflect(wo, normalize(sample));

	return (wi);
};
__device__ float ct_get_pdf(float3 n, float3 wi, float3 wo, float r)
{
	float3 wh = normalize(wo + wi);

	double wh_dot_n = abs(dot(wh, n));
	double wo_dot_wh = abs(dot(wo, wh));

	double D = ggxtr_ndf(n, wh, r);

	return (D * wh_dot_n) / ((4.f * wo_dot_wh));
};
__device__ float3 ct_f(const Isect& isect, const float3& wi, const float3& wo)
{
	float r = get_roughness(isect);
	//float m = get_metallic(isect);
	//float3 a = get_albedo(isect);

	//float3 f0 = lerp(a, isect.material->fresnel, m);
	//float r = isect.material->roughness;

	float3 L = make_float3(0, 0, 0);

	float3 n = isect.normal;//get_normal(isect);
	float3 wh = normalize(wo + wi);

	double n_dot_wi = abs(dot(n, wi));
	double n_dot_wo = abs(dot(n, wo));

	double D = ggxtr_ndf(n, wh, r);
	double G = geo_atten(wi, wo, n, r);
	float3 F = make_float3(1.f);//fresnel(isect.material->fresnel, wh, wi);
	//float3 F = fresnel(f0, wh, wi);

	L = (D * G * F) / (4.f * n_dot_wo * n_dot_wi);
	//L = isect.material->ks * (D * G * F) / (4.f * n_dot_wo * n_dot_wi);

	return (L);
};
__device__ float3 ct_sample_f(const Isect& isect, const float3& wo, float3& wi, float& pdf)
{
	float r = get_roughness(isect);

	float e0 = random();
	float e1 = random();

	double theta = atan(r * r * sqrtf(e0 / (1.f - e0)));
	double phi = 2.f * M_PI * e1;

	float3 h = make_float3(
		sin(theta) * cos(phi),
		cos(theta),
		sin(theta) * sin(phi)
	);

	float3 N = isect.normal;//get_normal(isect);
	float3 T = normalize(cross(N, get_orthogonal_vec(N)));
	float3 B = normalize(cross(N, T));

	wi = -reflect(wo, normalize(T * h.x + N * h.y + B * h.z));
	pdf = ct_get_pdf(N, wi, wo, r);

	return ct_f(isect, wi, wo);
};
__device__ float3 ct_L(dLight** lights, const Isect& isect, const float3& wi, const float3& wo, int light_id, const float3& sample_point, float r)
{
	float3 L = make_float3(0, 0, 0);
	float3 f = make_float3(0, 0, 0);
	float3 Li = make_float3(0, 0, 0);

	double brdf_pdf, light_pdf, weight;

	//for (int i = 0; i < 5; i++) {
	//int i = rand_int(0,4);
	float n_dot_wi = dot(isect.normal, wi);
	//float n_dot_wi = dot(get_normal(isect), wi);
	if (n_dot_wi > 0.f) {

		brdf_pdf = ct_get_pdf(isect.normal, wi, wo, r);
		//brdf_pdf = ct_get_pdf(get_normal(isect), wi, wo, r);
		light_pdf = lights[light_id]->get_pdf(isect);
		weight = power_heuristic(1, light_pdf, 1, brdf_pdf);

		f = ct_f(isect, wi, wo) * weight * n_dot_wi;
		Li = lights[light_id]->L(isect, wi, sample_point);

		if (f != make_float3(0.f) && light_pdf != 0.f && Li != make_float3(0.f)) {
			L += fmaxf(make_float3(0.f), f * Li / light_pdf);
		}
	}

	// Sample BRDF
	float3 wi_brdf = ct_sample(isect, wo);
	dRay visibility_ray(isect.position, wi_brdf);
	Isect isect_2;
	float tmin;
	//n_dot_wi = dot(get_normal(isect), wi_brdf);
	n_dot_wi = dot(isect.normal, wi_brdf);

	if (n_dot_wi > 0.f && lights[light_id]->visible(visibility_ray, tmin, isect_2)) {

		//brdf_pdf = ct_get_pdf(get_normal(isect), wi_brdf, wo, r);
		brdf_pdf = ct_get_pdf(isect.normal, wi_brdf, wo, r);
		light_pdf = lights[light_id]->get_pdf(isect_2, visibility_ray);
		weight = power_heuristic(1, brdf_pdf, 1, light_pdf);

		f = ct_f(isect, wi_brdf, wo) * weight * n_dot_wi;
		Li = lights[light_id]->L(isect, wi_brdf, sample_point);

		if (f != make_float3(0.f) && brdf_pdf != 0.f && Li != make_float3(0, 0, 0)) {
			L += fmaxf(make_float3(0.f), f * Li / brdf_pdf);
		}
	}
	//}

	return L;
}

__device__ float3 emissive_L(const Isect& isect, const float3& ray_dir)
{
	//if (dot(-isect.normal, ray_dir) > 0.0)
	//	return (isect.material->radiance * isect.material->emissiveColor);
	if (dot(-get_normal(isect), ray_dir) > 0.0)
		return (isect.material->radiance * isect.material->emissiveColorFactor);
	else
		return (make_float3(0, 0, 0));
};
__device__ float3 emissive_L(const Isect& isect)
{
	return (isect.material->radiance * isect.material->emissiveColorFactor);
};
__device__ float3 emissive_L(const dMaterial* material)
{
	return (material->radiance * material->emissiveColorFactor);
};