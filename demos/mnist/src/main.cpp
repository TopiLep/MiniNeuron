#include "Layer.h"
#include "network.h"
#include "activation_types.h"
#include "loss_types.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

//struct to store mnsit data
struct MNISTData {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
};

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<int> loadLabels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open: " << path << std::endl;
        return {};
    }
    int magicNumber = 0;
    file.read((char*)&magicNumber, 4);
    if (!file) {
        std::cout << "Failed to read magic number" << std::endl;
        return {};
    }
    magicNumber = reverseInt(magicNumber);

    int numLabels = 0;
    file.read((char*)&numLabels, 4);
    if (!file) {
        std::cout << "Failed to read numLabels" << std::endl;
        return {};
    }
    numLabels = reverseInt(numLabels);

    // Validate numLabels to prevent memory exhaustion
    const int MAX_LABELS = 1000000; // 1 million max
    if (numLabels <= 0 || numLabels > MAX_LABELS) {
        std::cout << "Invalid numLabels: " << numLabels << std::endl;
        return {};
    }

    std::vector<int> labels(numLabels);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        file.read((char*)&label, 1);
        if (!file) {
            std::cout << "Failed to read label at index " << i << std::endl;
            return {};
        }
        labels[i] = (int)label;
    }

    return labels;
}

std::vector<std::vector<float>> loadImages(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open: " << path << std::endl;
        return {};
    }
    int magicNumber = 0;
    file.read((char*)&magicNumber, 4);
    if (!file) {
        std::cout << "Failed to read magic number" << std::endl;
        return {};
    }
    magicNumber = reverseInt(magicNumber);

    int numImgs = 0;
    file.read((char*)&numImgs, 4);
    if (!file) {
        std::cout << "Failed to read numImgs" << std::endl;
        return {};
    }
    numImgs = reverseInt(numImgs);

    int rows = 0, cols = 0;
    file.read((char*)&rows, 4);
    file.read((char*)&cols, 4);
    if (!file) {
        std::cout << "Failed to read image dimensions" << std::endl;
        return {};
    }
    rows = reverseInt(rows);
    cols = reverseInt(cols);

    // Validate dimensions and image count to prevent memory exhaustion
    const int MAX_IMAGES = 1000000; // 1 million max
    const int MAX_DIMENSION = 10000; // 10k x 10k max
    if (numImgs <= 0 || numImgs > MAX_IMAGES) {
        std::cout << "Invalid numImgs: " << numImgs << std::endl;
        return {};
    }
    if (rows <= 0 || rows > MAX_DIMENSION || cols <= 0 || cols > MAX_DIMENSION) {
        std::cout << "Invalid image dimensions: " << rows << "x" << cols << std::endl;
        return {};
    }
    int pixelsPerImage = rows * cols;
    if (pixelsPerImage != 784) {
        std::cout << "Warning: expected 784 pixels per image, got " << pixelsPerImage << std::endl;
    }

    std::vector<std::vector<float>> imgs(numImgs);
    for (int i = 0; i < numImgs; i++) {
        std::vector<float> img;
        img.reserve(pixelsPerImage);
        for (int j = 0; j < pixelsPerImage; j++) {
            unsigned char pixel;
            file.read((char*)&pixel, 1);
            if (!file) {
                std::cout << "Failed to read pixel at image " << i << ", pixel " << j << std::endl;
                return {};
            }
            img.push_back(pixel / 255.0f);
        }
        imgs[i] = img;
    }
    return imgs;
}

MNISTData loadMnist(const std::string& imgPath, const std::string& labelPath) {
    MNISTData data;
    data.inputs = loadImages(imgPath);
    std::vector<int> labels = loadLabels(labelPath);
    for (int i = 0; i < labels.size(); i++) {
        std::vector<float> target(10, 0.0f);
        target[labels[i]] = 1.0f;
        data.targets.push_back(target);
    }
    return data;
}


int main() {
    //make the neural network object
    MiniNeuron::Network net;

    
    //structure of layers 784 -> 256 -> 128 -> 10
    net.add(MiniNeuron::Layer(256, 784, ActivationType::ReLU, InitializerType::HeInit));
    net.add(MiniNeuron::Layer(128, 256, ActivationType::ReLU, InitializerType::HeInit));
    net.add(MiniNeuron::Layer(10, 128, ActivationType::Softmax, InitializerType::Xavier));

    //structure of layers 784 -> 128 -> 10
    //net.add(MiniNeuron::Layer(128, 784, ActivationType::Sigmoid, InitializerType::Xavier));
    //net.add(MiniNeuron::Layer(10, 128, ActivationType::Softmax, InitializerType::Xavier));

    //init all layer weights, bias and other stuff
    net.initLayers();

    
    //net.loadModel("mnist-relu-v1.1.0.mn");

    //define learning data for mnist
    MNISTData training = loadMnist("Mnist/train-images-idx3-ubyte", "Mnist/train-labels-idx1-ubyte");
    MNISTData testing = loadMnist("Mnist/t10k-images-idx3-ubyte", "Mnist/t10k-labels-idx1-ubyte");
    std::cout << "Training samples: " << training.inputs.size() << std::endl;
    std::cout << "Testing samples: " << testing.inputs.size() << std::endl;

    float learningRate = 0.001f;

    net.setOptimizer(OptimizerType::Adam);

    //train function, 1 epoch.
    net.train(training.inputs, training.targets, 32, 3, learningRate, LossTypes::crossEntropy);

    
    //Global Accuracy Calculation
    std::cout << "\nCalculating Global Accuracy..." << std::endl;

    int correctPredictions = 0;
    size_t totalTestSamples = testing.inputs.size();

    for (size_t i = 0; i < totalTestSamples; i++) {
        std::vector<float> res = net.forward(testing.inputs[i]);

        int prediction = 0;
        float maxProb = res[0];
        for (int j = 1; j < 10; j++) {
            if (res[j] > maxProb) {
                maxProb = res[j];
                prediction = j;
            }
        }

        int actual = 0;
        for (int j = 0; j < 10; j++) {
            if (testing.targets[i][j] == 1.0f) {
                actual = j;
                break;
            }
        }

        if (prediction == actual) {
            correctPredictions++;
        }

        if (i % 500 == 0) {
            std::cout << "\rEvaluating: " << i << "/" << totalTestSamples << std::flush;
        }
    }

    float accuracy = (static_cast<float>(correctPredictions) / totalTestSamples) * 100.0f;

    std::cout << "\rEvaluation Complete!                          " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " Final Test Accuracy: " << accuracy << "%" << std::endl;
    std::cout << " Total Correct: " << correctPredictions << " / " << totalTestSamples << std::endl;
    std::cout << "========================================" << std::endl;

    net.saveModel("mnist-sigmoid-v1.1.0.mn");

    return 0;
}