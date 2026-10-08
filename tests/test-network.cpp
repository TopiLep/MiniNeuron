#include <gtest/gtest.h>
#include "network.h"
#include <filesystem>
#include <vector>

TEST(NetworkTest, CanCreateNetwork)
{
    MiniNeuron::Network network;

    SUCCEED();
}

TEST(NetworkTest, CanAddLayer)
{
    MiniNeuron::Network network;

    network.add(MiniNeuron::Layer(2, 3, ActivationType::Sigmoid, InitializerType::Xavier));

    SUCCEED();
}

TEST(NetworkTest, ForwardProducesExpectedOutput)
{
    MiniNeuron::Network network;

    network.add(MiniNeuron::Layer(3, 2, ActivationType::Sigmoid, InitializerType::Xavier));
    network.add(MiniNeuron::Layer(1, 3, ActivationType::Sigmoid, InitializerType::Xavier));

    network.initLayers();

    std::vector<float> input = {1.0f, 2.0f};

    std::vector<float> output = network.forward(input);

    EXPECT_EQ(output.size(), 1);
}

TEST(NetworkTest, SaveAndLoadPreservesOutput)
{
    MiniNeuron::Network network;

    MiniNeuron::Network net;

    //structure of layers 2 -> 4 -> 1
    net.add(MiniNeuron::Layer(256, 2, ActivationType::ReLU, InitializerType::HeInit));
    net.add(MiniNeuron::Layer(128, 256, ActivationType::ReLU, InitializerType::HeInit));
    net.add(MiniNeuron::Layer(1, 128, ActivationType::Softmax, InitializerType::Xavier));

    //init all layer weights, bias and other stuff
    net.initLayers();

    //define learning data for xor
    const std::vector<float> input = {1.0f, 2.0f};

    float learningRate = 5.0f;

    const auto before = network.forward(input);

    const auto path =
        std::filesystem::temp_directory_path() /
        "minineuron_test_model.bin";

    network.saveModel(path.string());

    MiniNeuron::Network loadedNetwork;
    loadedNetwork.loadModel(path.string());

    const auto after = loadedNetwork.forward(input);

    ASSERT_EQ(before.size(), after.size());

    for (size_t i = 0; i < before.size(); ++i)
    {
        EXPECT_FLOAT_EQ(before[i], after[i]);
    }

    std::filesystem::remove(path);
}