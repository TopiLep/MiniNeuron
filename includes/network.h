#pragma once

#include "layer.h"
#include "loss_types.h"
#include "optimizer_types.h"
#include "matrix.h"
#include <vector>
#include <string>
#include <chrono>

namespace MiniNeuron {

	class Network {
	public:

		//core functions
		void add(Layer&& x);
		void initLayers ();
		std::vector<float> forward(const std::vector<float>& inputs);
		void backpropagate(const std::vector<float>& targets, const std::vector<float>& inputs);
		void updateNetwork(int batchSize, float learningRate);
		float epoch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int batchSize, float learningRate, LossTypes losstype);
		void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets,int batchSize, int epochs, float learningRate, LossTypes losstype = LossTypes::MSE);

		//loss
		float loss(const std::vector<float>& p, const std::vector<float>& y, LossTypes losstype);
		float lossDerivative(float p, float y, LossTypes losstype);

		//set optimizer
		void setOptimizer(OptimizerType = OptimizerType::SGD);

		//save load
		void saveModel(const std::string& filePath);
		void loadModel(const std::string& filePath);

		//debug
		void printNetwork();

	private:
		std::vector<Layer> layers;

		std::chrono::high_resolution_clock::time_point start;

	};

}