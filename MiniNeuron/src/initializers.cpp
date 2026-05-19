#include "initializers.h"
#include <cmath>
#include <random>
#include <mutex>

static std::mutex rng_mutex;
static std::default_random_engine gen(std::random_device{}());

float HeInit(size_t fan_in) {
	if (fan_in == 0) {
		throw std::invalid_argument("fan_in must be greater than 0");
	}
	float stddev = std::sqrt(2.0f / static_cast<float>(fan_in));

	std::lock_guard<std::mutex> lock(rng_mutex);
	std::normal_distribution<float> dis(0.0, stddev);

	return dis(gen);
}

float Xavier(size_t fan_in, size_t fan_out) {
	if (fan_in == 0 || fan_out == 0) {
		throw std::invalid_argument("fan_in and fan_out must be greater than 0");
	}
	float stddev = std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));

	std::lock_guard<std::mutex> lock(rng_mutex);
	std::normal_distribution<float> dis(0.0, stddev);

	return dis(gen);
}