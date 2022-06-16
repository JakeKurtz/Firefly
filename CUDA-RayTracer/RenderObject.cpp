#include "RenderObject.h"

#include "Curve.h"
#include "Model.h"

int RenderObject::get_id()
{
    return id;
}

void RenderObject::set_name(std::string name)
{
    (this)->name = name;
}

std::string RenderObject::get_name()
{
    return name;
}

void RenderObject::set_transform(Transform* transform)
{
    (this)->transform = transform;
}

Transform* RenderObject::get_transform()
{
    return transform;
}

int RenderObject::type()
{
    if (dynamic_cast<Curve*>(this) != nullptr)
    {
        return TYPE_CURVE;
    }
    else if (dynamic_cast<Model*>(this) != nullptr)
    {
        return TYPE_TRIANGLE_MESH;
    }
    else {
        //std::cout << "ERROR: invalid object type." << std::endl;
        return -1;
    }
}
