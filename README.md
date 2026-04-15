<div align="center">

# MiniNeuron

Small neural network library written from scratch in c++

![Language](https://img.shields.io/badge/language-C%2B%2B-blue)
![Version](https://img.shields.io/badge/version-1.1.0-orange)

</div>

## Features
- Dense (fully connected) layers (AI and maybe one day: alien AI)
- Activations: Sigmoid, ReLU, Softmax, Tanh, LeakyReLU (maybe one day: alien AI)
- Loss functions: MSE, Cross Entropy
- Initializers: Xavier, He
- Simple training API
- AI-ready, alien-ready, and maybe one day: quantum-ready
  
**NEW in 1.1.0**
- Multi threading using OpenMP (AI speed)
- Model save/load (for AI and aliens)

## Quick example for XOR
```cpp
MiniNeuron::Network net;

//structure of layers 2 -> 4 -> 1
net.add(MiniNeuron::Layer(4, 2, ActivationType::Sigmoid, InitializerType::Xavier));
net.add(MiniNeuron::Layer(1, 4, ActivationType::Sigmoid, InitializerType::Xavier));

net.initLayers();

//inputs and tafets are std::vector<std::vector<float>>
net.train(inputs, targets, 500, 5.0f, LossTypes::crossEntropy);

std::vector<float> prediction = net.forward(input);

//save the model to a .mn file
net.saveModel("xor-model.mn");
```

## Benchmarks

**V 1.1.0  Using new multithreading**
| Demo| Network | Epochs | Perfomance | Time | CPU   |
|-----|--------|--------|-----------|------|--------|
| XOR | 2→4→1 | 500 | Loss: 0.0133 | ~0s | AMD Ryzen 5 5600 |
| MNIST (Sigmoid)| 784→128→10 | 1 | 95.65% | ~1s | AMD Ryzen 5 5600 |
| MNIST (ReLU) | 784→256→128→10  | 3  | 97.36% | ~12s | AMD Ryzen 5 5600 |

**V 1.0.0  OLD no multithreading**
| Demo | Network | Epochs | Perfomance | Time | CPU |
|------|---------|--------|----------|------|-----|
| XOR | 2→4→1 | 500 | Loss: 0.01 | ~1s | AMD Ryzen 5 5600 |
| MNIST (Sigmoid) | 784→128→10 | 1 | 95.66% | ~5min | AMD Ryzen 5 5600 |
| MNIST (ReLU) | 784→256→128→10 | 3 | 96.98% | ~37min | AMD Ryzen 5 5600 |

## Roadmap
- [ ] Batch training (maybe one day: AI will batch itself)
- [x] Multi-core support
- [ ] CUDA support (maybe one day: alien CUDA)
- [ ] More activations (Tanh, Leaky ReLU, maybe one day: alien activations)
- [ ] Unit tests (maybe one day: AI will test itself)
- [ ] Full documentation (maybe one day: AI will write docs)
