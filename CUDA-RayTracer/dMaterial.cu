#include "dMaterial.cuh"
#include "Isect.cuh"
#include "dRandom.cuh"
#include "dMath.cuh"
#include "dLight.cuh"
#include "dRay.cuh"
#include "dMatrix.cuh"

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
	float r;
	if (isect.material->metallicRoughnessTexture != -1) {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		float4 tex = tex2DLod<float4>(isect.material->metallicRoughnessTexture, u, v, 0);
		r = fmaxf(tex.y, 0.01f);
	}
	else if (isect.material->roughnessTexture != -1 ){
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		float4 tex = tex2DLod<float4>(isect.material->roughnessTexture, u, v, 0);
		r = fmaxf(tex.x, 0.01f);
	}
	else {
		r = isect.material->roughnessFactor;
	}
	return r;
}
__device__ float get_metallic(const Isect& isect) {
	if (isect.material->metallicRoughnessTexture != -1) {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		float4 tex = tex2DLod<float4>(isect.material->metallicRoughnessTexture, u, v, 0);
		return tex.z;
	}
	else if (isect.material->metallicTexture != -1) {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		float4 tex = tex2DLod<float4>(isect.material->metallicTexture, u, v, 0);
		return tex.x;
	}
	else {
		return isect.material->metallicFactor;
	}
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
		
		float3 T = isect.tangent;
		float3 B = isect.bitangent;
		float3 N = isect.normal;

		Matrix4x4 TBN = Matrix4x4(
			T.x, T.y, T.z, 0.f,
			B.x, B.y, B.z, 0.f,
			N.x, N.y, N.z, 0.f,
			0.f, 0.f, 0.f, 1.f
		);
		
		float4 n = tex2DLod<float4>(id, u, v, 0);
		n = n * 2.0 - 1.0;
		normal = normalize(TBN * make_float3(n.x, n.y, n.z));
	}
	return normal;
}

__device__ float3 fresnel(float3 f0, float3 h, float3 wo) {
	return f0 + (1 - f0) * pow(1 - fmaxf(0.f, dot(h, wo)), 5);
}
__device__ float3 fresnel_roughness(float3 f0, float3 n, float3 wo, float r)
{
	return f0 + (fmaxf(make_float3(1.f - r), f0) - f0) * pow(clamp(1.f - fmaxf(0.f, dot(n, wo)), 0.f, 1.f), 5.f);
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
	double a2 = r * r * r * r;
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

	float3 N = get_normal(isect);
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
	float m = get_metallic(isect);
	float3 a = get_albedo(isect);

	float3 f0 = lerp(isect.material->fresnel, a, m);

	float3 wh = normalize(wo + wi);
	float3 F = fresnel(f0, wh, wi);

	float3 kD = make_float3(1.f) - F;
	kD *= 1.0 - m;
	
	return (kD * a * M_1_PI);
};
__device__ float3 diff_L(dLight** lights, const Isect& isect, const float3& wi, const float3& wo, int light_id, const float3& sample_point)
{
	float3 L = make_float3(0.f);
	float3 f = make_float3(0.f);
	float3 Li = make_float3(0.f);

	double brdf_pdf, light_pdf, weight;

	Li = lights[light_id]->L(isect, wi, sample_point);

	float n_dot_wi = dot(get_normal(isect), wi);
	if (n_dot_wi > 0.f) {

		brdf_pdf = diff_get_pdf();
		light_pdf = lights[light_id]->get_pdf(isect);
		weight = power_heuristic(1, light_pdf, 1, brdf_pdf);

		f = diff_f(isect, wi, wo) * n_dot_wi;

		if (f != make_float3(0.f) && light_pdf > 0.f && Li != make_float3(0)) {
			L += f * Li * weight / light_pdf;
		}
	}

	// Sample BRDF
	float3 wi_brdf = diff_sample(isect);
	dRay visibility_ray(isect.position, wi_brdf);
	Isect isect_2;
	float tmin;

	n_dot_wi = dot(get_normal(isect), wi_brdf);

	if (n_dot_wi > 0.f && lights[light_id]->visible(visibility_ray, tmin, isect_2)) {

		brdf_pdf = diff_get_pdf();
		light_pdf = lights[light_id]->get_pdf(isect);
		weight = power_heuristic(1, brdf_pdf, 1, light_pdf);

		f = diff_f(isect, wi_brdf, wo) * n_dot_wi;

		if (f != make_float3(0.f) && brdf_pdf > 0.f && Li != make_float3(0)) {
			L += f * Li * weight / brdf_pdf;
		}
	}
	return (L/2.f);
}

__device__ float3 spec_sample(const Isect& isect, const float3& wo)
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

	float3 N = get_normal(isect);
	float3 T = normalize(cross(N, get_orthogonal_vec(N)));
	float3 B = normalize(cross(N, T));

	float3 sample = T * h.x + N * h.y + B * h.z;

	float3 wi = -reflect(wo, normalize(sample));

	return (wi);
};
__device__ float spec_get_pdf(float3 n, float3 wi, float3 wo, float r)
{
	float3 wh = normalize(wo + wi);

	double wh_dot_n = fmaxf(dot(wh, n), 0.f);
	double wo_dot_wh = fmaxf(dot(wo, wh), 0.f);

	double D = ggxtr_ndf(n, wh, r);

	return (D * wh_dot_n) / fmaxf((4.f * wo_dot_wh), 0.001f);
};
__device__ float3 spec_f(const Isect& isect, const float3& wi, const float3& wo)
{
	float r = get_roughness(isect);
	float m = get_metallic(isect);
	float3 a = get_albedo(isect);
	float3 f0 = lerp(isect.material->fresnel, a, m);

	float3 L = make_float3(0, 0, 0);

	float3 n = get_normal(isect);
	float3 wh = normalize(wo + wi);

	double n_dot_wi = fmaxf(dot(n, wi), 0.f);
	double n_dot_wo = fmaxf(dot(n, wo), 0.f);

	double D = ggxtr_ndf(n, wh, r);
	double G = geo_atten(wi, wo, n, r);
	float3 F = fresnel(f0, wh, wi);

	L = (D * G * F) / fmaxf((4.f * n_dot_wo * n_dot_wi), 0.001);

	return (L);
};
__device__ float3 spec_sample_f(const Isect& isect, const float3& wo, float3& wi, float& pdf)
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

	float3 N = get_normal(isect);
	float3 T = normalize(cross(N, get_orthogonal_vec(N)));
	float3 B = normalize(cross(N, T));

	wi = -reflect(wo, normalize(T * h.x + N * h.y + B * h.z));
	pdf = spec_get_pdf(N, wi, wo, r);

	return spec_f(isect, wi, wo);
};
__device__ float3 spec_L(dLight** lights, const Isect& isect, const float3& wi, const float3& wo, int light_id, const float3& sample_point, float r)
{
	float3 L = make_float3(0, 0, 0);
	float3 f = make_float3(0, 0, 0);
	float3 Li = make_float3(0.f);
	
	double brdf_pdf, light_pdf, weight;

	//for (int i = 0; i < 5; i++) {
	//int i = rand_int(0,4);

	Li = lights[light_id]->L(isect, wi, sample_point);

	float n_dot_wi = dot(get_normal(isect), wi);
	if (n_dot_wi > 0.f) {

		brdf_pdf = spec_get_pdf(get_normal(isect), wi, wo, r);
		light_pdf = lights[light_id]->get_pdf(isect);
		weight = power_heuristic(1, light_pdf, 1, brdf_pdf);

		f = spec_f(isect, wi, wo) * n_dot_wi;

		if (f != make_float3(0.f) && light_pdf > 0.f && Li != make_float3(0)) {
			L += f * Li * weight / light_pdf;
		}
	}
	
	// Sample BRDF
	float3 wi_brdf = spec_sample(isect, wo);
	dRay visibility_ray(isect.position, wi_brdf);
	Isect isect_2;
	float tmin;
	n_dot_wi = dot(get_normal(isect), wi_brdf);

	if (n_dot_wi > 0.f && lights[light_id]->visible(visibility_ray, tmin, isect_2)) {
		brdf_pdf = spec_get_pdf(get_normal(isect), wi_brdf, wo, r);
		light_pdf = lights[light_id]->get_pdf(isect);
		weight = power_heuristic(1, brdf_pdf, 1, light_pdf);

		f = spec_f(isect, wi_brdf, wo) * n_dot_wi;

		if (f != make_float3(0.f) && brdf_pdf > 0.f && Li != make_float3(0)) {
			L += f * Li * weight / brdf_pdf;
		}
	}
	//}

	return (L/2.f);
}

__device__ float3 BRDF_L(dLight** lights, const Isect& isect, const float3& wi, const float3& wo, int light_id, const float3& sample_point, float3& sample_dir)
{
	float3 L = make_float3(0.f);
	float3 f = make_float3(0.f);
	float3 Li = make_float3(0.f);

	double diff_pdf, diff_weight;
	double spec_pdf, spec_weight;
	double light_pdf, weight;

	float r = get_roughness(isect);
	float3 n = get_normal(isect);

	//int i = rand_int(0,4);

	Li = lights[light_id]->L(isect, wi, sample_point);

	float n_dot_wi = dot(n, wi);

	if (n_dot_wi > 0.f) {	
		diff_pdf = diff_get_pdf();
		spec_pdf = spec_get_pdf(n, wi, wo, r);
		light_pdf = lights[light_id]->get_pdf(isect);

		diff_weight = power_heuristic(1, light_pdf, 1, diff_pdf);
		spec_weight = power_heuristic(1, light_pdf, 1, spec_pdf);

		if (light_pdf > 0.f && Li != make_float3(0)) {

			if (lights[light_id]->is_delta()) {
				spec_weight = 1.f;
				diff_weight = 1.f;
			}

			f = spec_f(isect, wi, wo) * n_dot_wi;
			if (f != make_float3(0.f)) L += f * spec_weight * Li / light_pdf;

			f = diff_f(isect, wi, wo) * n_dot_wi;
			if (f != make_float3(0.f)) L += f * diff_weight * Li / light_pdf;
		}
	}

	// Sample BRDF
	dRay visibility_ray;
	Isect it;
	float tmin;

	float3 wi_spec = spec_sample(isect, wo);
	float3 wi_diff = diff_sample(isect);

	if (!lights[light_id]->is_delta()) {
		visibility_ray = dRay(isect.position, wi_spec);
		n_dot_wi = dot(get_normal(isect), wi_spec);

		if (n_dot_wi > 0.f && lights[light_id]->visible(visibility_ray, tmin, it)) {
			spec_pdf = spec_get_pdf(get_normal(isect), wi_spec, wo, r);
			light_pdf = lights[light_id]->get_pdf(isect);
			spec_weight = power_heuristic(1, spec_pdf, 1, light_pdf);

			f = spec_f(isect, wi_spec, wo) * n_dot_wi;

			if (f != make_float3(0.f) && spec_pdf > 0.f && Li != make_float3(0)) {
				L += f * Li * spec_weight / spec_pdf;
			}
		}

		visibility_ray = dRay(isect.position, wi_diff);
		n_dot_wi = dot(get_normal(isect), wi_diff);

		if (n_dot_wi > 0.f && lights[light_id]->visible(visibility_ray, tmin, it)) {
			diff_pdf = diff_get_pdf();
			light_pdf = lights[light_id]->get_pdf(isect);
			diff_weight = power_heuristic(1, diff_pdf, 1, light_pdf);

			f = diff_f(isect, wi_diff, wo) * n_dot_wi;

			if (f != make_float3(0.f) && diff_pdf > 0.f && Li != make_float3(0)) {
				L += f * Li * diff_weight / diff_pdf;
			}
		}
	}

	if (random() < 0.5f) {
		sample_dir = wi_diff;
	}
	else {
		sample_dir = wi_spec;
	}

	return (L);
}
__device__ float3 BRDF_f(const Isect& isect, const float3& wi, const float3& wo)
{
	return spec_f(isect, wi, wo) + diff_f(isect, wi, wo);
}
__device__ float BRDF_pdf(const Isect& isect, const float3 wi, const float3 wo)
{
	float spec_pdf = spec_get_pdf(get_normal(isect), wi, wo, get_roughness(isect));
	float diff_pdf = diff_get_pdf();

	return 0.5 * (spec_pdf + diff_pdf);
}

__device__ float3 emissive_L(const Isect& isect, const float3& ray_dir)
{
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