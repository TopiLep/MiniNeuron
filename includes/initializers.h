#pragma once

#include <cstddef>

enum class InitializerType {
	HeInit,
	Xavier
};


float HeInit(size_t fan_in);

float Xavier(size_t fan_in, size_t fan_out);