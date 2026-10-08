#pragma once

#include <vector>


namespace MiniNeuron {
	//row major matrix that stores the data in a flat vector for better cache.
	struct Matrix {

		int rows, cols;
		std::vector<float> data;

		Matrix(int x, int y) : rows(x), cols(y), data(x* y, 0.0f) {}
		Matrix() : rows(0), cols(0) {}

		//non const acces
		inline float& operator()(int x, int y) {
			return data[x * cols + y];
		}
		//const acces
		inline float operator()(int r, int c) const {
			return data[r * cols + c];
		}

		//return rows address for faster lookup
		inline float* row(int i) {
			return &data[i * cols];
		}
		//const version
		inline const float* row(int i) const {
			return &data[i * cols];
		}

		//zero all values in the matrix
		void clear() {
			std::fill(data.begin(), data.end(), 0.0f);
		}

	};
}