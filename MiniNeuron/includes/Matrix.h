#pragma once

#include <vector>
#include <stdexcept>


namespace MiniNeuron {
	//row major matrix that stores the data in a flat vector for better cache.
	struct Matrix {

		int rows, cols;
		std::vector<float> data;

		Matrix(int x, int y) : rows(x), cols(y), data(x* y, 0.0f) {}
		Matrix() : rows(0), cols(0) {}

		//non const acces
		inline float& operator()(int x, int y) {
		if (x < 0 || x >= rows || y < 0 || y >= cols) {
			throw std::out_of_range("Matrix index out of bounds");
		}
		return data[x * cols + y];
	}
	//const acces
	inline float operator()(int r, int c) const {
		if (r < 0 || r >= rows || c < 0 || c >= cols) {
			throw std::out_of_range("Matrix index out of bounds");
		}
		return data[r * cols + c];
	}

	//return rows address for faster lookup
	inline float* row(int i) {
		if (i < 0 || i >= rows) {
			throw std::out_of_range("Matrix row index out of bounds");
		}
		return &data[i * cols];
	}
	//const version
	inline const float* row(int i) const {
		if (i < 0 || i >= rows) {
			throw std::out_of_range("Matrix row index out of bounds");
		}

		//zero all values in the matrix
		void clear() {
			std::fill(data.begin(), data.end(), 0.0f);
		}

	};
}