#include "network.h"
#include "Layer.h"
#include "Matrix.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>

#include <chrono>

namespace MiniNeuron {

	//add layer to the back of the network. also prints a debug text.
	void Network::add(Layer&& x) {
		layers.push_back(std::move(x));
		//std::cout << "added layer to network" << std::endl;
	}

	//initialize layers
	void Network::initLayers() {
		for (size_t i = 0; i < layers.size(); i++) {
			layers[i].initLayer();
		}
	}

	void Network::setOptimizer(OptimizerType optimizer) {
		for (size_t i = 0; i < layers.size(); i++) {
			layers[i].initOptimizer(optimizer);
		}
	}

	//prints all layers and shows thier neuron count. used for debug, to be changed.
	void Network::printNetwork() {
		for (size_t i = 0; i < layers.size(); i++) {
			std::cout << "Layer(" << i << ")" << " neurons(" << layers[i].getNeuronCount() << ")" << std::endl;
		}
	}

	//used to predict/use the neural network
	std::vector<float> Network::forward(const std::vector<float>& inputs) {
		if (layers.empty()) {
			throw std::invalid_argument("Network has no layers");
		}
		if (inputs.size() != layers[0].getInputCount()) {
			throw std::invalid_argument("Input size does not match first layer input count");
		}
		std::vector<float> x = inputs;
		for (size_t i = 0; i < layers.size(); i++) {
			x = layers[i].Forward(x);
		}
		return x;
	}

	//score the overall network perfomance, used to tell user how good the network is.
	float Network::loss(const std::vector<float>& p, const std::vector<float>& y, LossTypes lossType) {
		float loss = 0.0f;
		for (size_t i = 0; i < p.size(); i++) {

			switch (lossType) {
				case LossTypes::crossEntropy:
				{
					loss += -y[i] * std::log(p[i] + 1e-8f);
					break;
				}
				case LossTypes::MSE:
				{
					float diff = y[i] - p[i];
					loss += 0.5f * (diff * diff);
					break;
				}
			}
		}
		if (lossType == LossTypes::MSE) {
			return loss / p.size();
		}
		return loss;
	}

	//score the neural network
	float Network::lossDerivative(float p, float y, LossTypes lossType) {
		switch (lossType) {
			case LossTypes::MSE:
			{
				return p - y;
			}
			case LossTypes::crossEntropy:
			{
				return -y / (p + 1e-8f);
			}
			default:
				return 0.0f;
		}
	}

	void Network::backpropagate(const std::vector<float>& targets, const std::vector<float>& inputs) {
		for (int i = (int)layers.size() - 1; i >= 0; i--) {
			bool isOutput = (i == layers.size() - 1);

			if (isOutput) {
				Matrix empty(0, 0);
				layers[i].backpropagation(targets, empty, layers[i - 1].getResult(), true);
			} else if (i == 0) {
				layers[i].backpropagation(layers[i + 1].getDelta(), layers[i + 1].getWeights(), inputs, false);
			} else {
				layers[i].backpropagation(layers[i + 1].getDelta(), layers[i + 1].getWeights(), layers[i - 1].getResult(), false);
			}
		}
	}

	void Network::updateNetwork(int batchSize, float learningRate) {
		for (size_t i = 0; i < layers.size(); i++) {
			layers[i].updateWeights(batchSize, learningRate);
		}
	}

	float Network::epoch(const std::vector<std::vector<float>>& inputs, const const std::vector<std::vector<float>>& targets, int batchSize, float learningRate, LossTypes losstype) {
		float total_loss = 0.0f;
		assert(inputs.size() == targets.size());
		for (size_t i = 0; i < inputs.size(); i++)
		{
			const std::vector<float>& x = inputs[i];
			const std::vector<float>& y = targets[i];

			std::vector<float> prediction = forward(x);
			total_loss += loss(prediction, y, LossTypes::crossEntropy);
			backpropagate(y, x);
			if ((i + 1) % batchSize == 0) {
				updateNetwork(batchSize, learningRate);
			}
		}
		return total_loss / inputs.size();
	}

	void Network::train(const const std::vector<std::vector<float>>& inputs, const const std::vector<std::vector<float>>& targets, int batchSize, int epochs, float learningRate, LossTypes losstype) {
		std::cout << "Starting Training..." << std::endl;
		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < epochs; i++) {
			float avgLoss = epoch(inputs, targets, batchSize, learningRate, losstype);
			if (i % 1 == 0) {
				std::cout.precision(8);
				std::cout << std::fixed;
				std::cout << "Epoch: " << i
					<< " | Loss: " << avgLoss << std::endl;
				std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
				std::chrono::seconds duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
				std::cout << "Program ran for " << duration.count() << " seconds\n\n";
			}
			std::cout.unsetf(std::ios::fixed);
		}
	}

	void Network::saveModel(const std::string& filePath) {
		std::fstream fout;

		fout.open(filePath, std::ios::out | std::ios::binary);

		unsigned short numLayers = (unsigned short)layers.size();

		if (fout) {
			fout.write(reinterpret_cast<char*>(&numLayers), sizeof(unsigned short));
			
			for (size_t l= 0; l< layers.size(); l++) {
				unsigned short neuronCount = (unsigned short)layers[l].getNeuronCount();
				unsigned short inputCount = (unsigned short)layers[l].getInputCount();
				fout.write(reinterpret_cast<char*>(&neuronCount), sizeof(unsigned short));
				fout.write(reinterpret_cast<char*>(&inputCount), sizeof(unsigned short));
				int act = (int)layers[l].getActivation();
				int init = (int)layers[l].getInitializer();
				fout.write(reinterpret_cast<char*>(&act), sizeof(int));
				fout.write(reinterpret_cast<char*>(&init), sizeof(int));
				for (size_t i = 0; i < layers[l].getNeuronCount(); i++) {
					fout.write(reinterpret_cast<const char*>(layers[l].getWeights().data.data()),
						layers[l].getWeights().data.size() * sizeof(float));
				}
				fout.write(reinterpret_cast<const char*>(layers[l].getBiases().data()), layers[l].getNeuronCount() * sizeof(float));
			}
			fout.close();
			std::cout << "Model saved" << std::endl;
		}
		else {
			std::cout << "Failed to open file to save model" << std::endl;
		}

	}

	void Network::loadModel(const std::string& filePath) {
		std::fstream fin;

		fin.open(filePath, std::ios::in | std::ios::binary);

		if (fin) {
			unsigned short numLayers;
			fin.read(reinterpret_cast<char*>(&numLayers), sizeof(unsigned short));

			for (int i = 0; i < numLayers; i++) {
				unsigned short neuronCount;
				unsigned short inputCount;

				fin.read(reinterpret_cast<char*>(&neuronCount), sizeof(unsigned short));
				fin.read(reinterpret_cast<char*>(&inputCount), sizeof(unsigned short));

				int actInt;
				int initInt;

				fin.read(reinterpret_cast<char*>(&actInt), sizeof(int));
				fin.read(reinterpret_cast<char*>(&initInt), sizeof(int));

				ActivationType act = (ActivationType)actInt;
				InitializerType init = (InitializerType)initInt;

				Layer tempLayer(neuronCount, inputCount, act, init);

				Matrix weights(inputCount, neuronCount );
				for (size_t i = 0; i < neuronCount; i++) {
					fin.read(reinterpret_cast<char*>(weights.data.data()), weights.data.size() * sizeof(float));
				}

				tempLayer.initSizes();

				tempLayer.setWeights(weights);

				std::vector<float> biases(neuronCount);
				fin.read(reinterpret_cast<char*>(biases.data()), neuronCount * sizeof(float));
				tempLayer.setBiases(biases);

				add(std::move(tempLayer));
			}

			fin.close();

			std::cout << "Model loaded" << std::endl;
		}
		else {

		}
	}

}