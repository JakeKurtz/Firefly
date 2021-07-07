#pragma once
#include "../GeometricObj.h"
#include "../../Utilities/CudaList.h"

class Compound : public GeometricObj {
public:

	__device__ Compound(int num_objects) : GeometricObj() 
	{
		objects = CudaList<GeometricObj*>(num_objects);
	};

	__device__ Compound* clone(void) const
	{
		return (new Compound(*this));
	}

	__device__ virtual void set_material(Material* material_ptr)
	{
		int num_objects = objects.size();

		for (int j = 0; j < num_objects; j++)
			objects[j]->set_material(material_ptr);
	};

	__device__ virtual void add_object(GeometricObj* object_ptr)
	{
		objects.add(object_ptr);
	};

	__device__ int get_num_objects(void)
	{
		return objects.size();
	};

	__device__ virtual bool hit(const Ray& ray, float& tmin, ShadeRec& sr) const
	{
		float			t;
		float3			normal;
		float3			local_hit_point;
		bool			hit = false;
		tmin =			K_HUGE;
		int 			num_objects = objects.size();

		for (int j = 0; j < num_objects; j++)
			if (objects[j]->hit(ray, t, sr) && (t < tmin)) {
				hit = true;
				tmin = t;
				material_ptr = objects[j]->get_material();	// lhs is GeometricObject::material_ptr
				normal = sr.normal;
				local_hit_point = sr.local_hit_point;
					
			}

		if (hit) {
			sr.t = tmin;
			sr.normal = normal;
			sr.local_hit_point = local_hit_point;
		}

		return (hit);
	};

	__device__ virtual bool hit(const Ray& ray) const
	{
		float			t;
		float3			normal;
		float3			local_hit_point;
		bool			hit = false;
		int 			num_objects = objects.size();

		for (int j = 0; j < num_objects; j++)
			if (objects[j]->hit(ray) && (t < K_HUGE)) {
				hit = true;
			}
		return (hit);
	};

	__device__ virtual bool shadow_hit(const Ray& ray, float& tmin) const
	{
		float			t;
		float3			normal;
		float3			local_hit_point;
		bool			hit = false;
		tmin =			K_HUGE;
		int 			num_objects = objects.size();

		for (int j = 0; j < num_objects; j++)
			if (objects[j]->shadow_hit(ray, t) && (t < tmin)) {
				hit = true;
				tmin = t;
				//material_ptr = objects[j]->get_material();	// lhs is GeometricObject::material_ptr
			}
		return (hit);
	};

protected:

	CudaList<GeometricObj*> objects;

private:

	__device__ void delete_objects(void)
	{
		int num_objects = objects.size();

		for (int j = 0; j < num_objects; j++) {
			delete objects[j];
			objects[j] = NULL;
		}
		//objects.erase(objects.begin(), objects.end());
	};

	__device__ void copy_objects(const CudaList<GeometricObj*>& rhs_objects)
	{
		delete_objects();
		int num_objects = rhs_objects.size();

		for (int j = 0; j < num_objects; j++)
			objects.add(rhs_objects[j]->clone());
	};

};