#include "Bezier.h"

Bezier::Bezier()
{
    id = gen_id();

    this->name = "NULL";

    this->control_points.push_back(glm::vec3(0, 1, 0));
    this->control_points.push_back(glm::vec3(0, 0, 1));

    transform = new Transform();

    update();
}

Bezier::Bezier(std::vector<glm::vec3> control_points)
{
    id = gen_id();

    this->name = "NULL";
    this->control_points = control_points;

    transform = new Transform();

    update();
}

void Bezier::update()
{
    this->vertices = de_casteljau(control_points, depth);
    setup_buffers();
}

std::vector<glm::vec3> Bezier::de_casteljau(std::vector<glm::vec3> const& points, int depth)
{
    std::vector<glm::vec3> out, tmp, cp;
    out.push_back(points[0]);

    float t = 1.f / depth;

    for (int i = 1; i <= depth; i++) {
        cp = points;
        while (cp.size() != 1) {
            for (int j = 1; j < cp.size(); j++) {
                tmp.push_back(glm::mix(cp[j - 1], cp[j], t * i));
            }
            cp.swap(tmp);
            tmp.clear();
        }
        out.push_back(cp[0]);
    }

    return out;
}
