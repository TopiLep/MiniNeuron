#include "initializers.h"
#include <cmath>
#include <random>

float HeInit(size_t fan_in) {
	float stddev = std::sqrt(2.0f / static_cast<float>(fan_in));

	static std::default_random_engine gen(std::random_device{}());

	std::normal_distribution<float> dis(0.0, stddev);

	return dis(gen);
}

float Xavier(size_t fan_in, size_t fan_out) {
	float stddev = std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));

	static std::default_random_engine gen(std::random_device{}());

	std::normal_distribution<float> dis(0.0, stddev);

	return dis(gen);
}