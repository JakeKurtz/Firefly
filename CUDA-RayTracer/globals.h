#pragma once

extern bool GLFW_INIT;

extern int id_counter;

static int gen_id() {
	return id_counter++;
}