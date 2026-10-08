#include "layer.h"
#include "network.h"
#include "activation_types.h"
#include "loss_types.h"
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
	//make the neural network object
	MiniNeuron::Network net;

	//structure of layers 2 -> 4 -> 1
	net.add(MiniNeuron::Layer(4, 2, ActivationType::Sigmoid, InitializerType::Xavier));
	net.add(MiniNeuron::Layer(1, 4, ActivationType::Sigmoid, InitializerType::Xavier));

	//init all layer weights, bias and other stuff
	net.initLayers();

	//define learning data for xor
    std::vector<std::vector<float>> inputs = { {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f} };
    std::vector<std::vector<float>> targets = { {0.0f}, {1.0f}, {1.0f}, {0.0f} };

	float learningRate = 5.0f;

	//train function, 5000 epochs.
	net.train(inputs, targets, 1, 500, learningRate, LossTypes::crossEntropy);

	//print prediction for all inputs to check final perfomance
	std::cout << std::fixed << std::setprecision(2);
	for (int i = 0; i < (int)inputs.size(); i++) {
		std::vector<float> pred = net.forward(inputs[i]);
		std::cout << "{ " << inputs[i][0] << ", " << inputs[i][1] << " }" << " = " << pred[0] << std::endl;
	}

	net.saveModel("xor-sigmoid-v1.1.0.mn");

	return 0;
}