#include "ClosedCurve.h"

ClosedCurve::ClosedCurve(std::vector<glm::vec3> control_points)
{
    id = gen_id();

    this->name = "NULL";
    this->control_points = control_points;

    transform = new Transform();

    update();
}

void ClosedCurve::update()
{
    this->vertices = subdiv_closed(control_points, degree, depth);
    setup_buffers();
}

std::vector<glm::vec3> ClosedCurve::subdiv_closed(std::vector<glm::vec3> const& points, int degree, int depth) {
    std::vector<glm::vec3> out;

    // add midpoints
    for (int i = 0; i < points.size(); i++) {
        glm::vec3 a = points[i];
        glm::vec3 b = points[(i + 1) % points.size()];

        glm::vec3 midPoint = a + 0.5f * (b - a);
        out.push_back(a);
        out.push_back(midPoint);
    }
    // chase
    for (int i = 0; i < degree - 1; i++) {
        std::vector<glm::vec3> temp = out;
        for (int j = 0; j < temp.size(); j++) {
            glm::vec3 a = temp[j];
            glm::vec3 b = temp[(j + 1) % temp.size()];

            glm::vec3 midPoint = a + 0.5f * (b - a);
            out[j] = midPoint;
        }
    }
    if (depth > 1) out = subdiv_closed(out, degree, depth -= 1);
    out.push_back(out[0]);
    return out;
}
