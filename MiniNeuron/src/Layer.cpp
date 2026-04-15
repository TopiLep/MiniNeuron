#include "Layer.h"
#include "initializers.h"
#include "activation_types.h"
#include "Matrix.h"
#include <iostream>
#include <cmath>

namespace MiniNeuron {

	Layer::Layer(size_t neuronCount, size_t inputCount, ActivationType activation, InitializerType initializer)
		: m_weights(neuronCount, inputCount),
		m_weightGradients(neuronCount, inputCount)
	{
		this->m_neuronCount = neuronCount;
		this->m_inputCount = inputCount;
		this->activation = activation;
		this->initializer = initializer;	

		this->m_timeStep = 1;
	}

	void Layer::initSizes() {
		m_biases.resize(m_neuronCount, 0.0f);
		m_biasGradients.resize(m_neuronCount, 0.0f);
		m_delta.resize(m_neuronCount, 0.0f);
		m_zResult.resize(m_neuronCount, 0.0f);
		m_result.resize(m_neuronCount, 0.0f);
	}

	void Layer::initLayer() {

		initSizes();
		
		switch (initializer) {
		case InitializerType::HeInit:
		{
			#pragma omp parallel for
			for (int i = 0; i < m_weights.cols * m_weights.rows; i++) {
				m_weights.data[i] = HeInit(m_inputCount);
			}
			break;
		}
		case InitializerType::Xavier:
		{
			#pragma omp parallel for
			for (int i = 0; i < m_weights.cols * m_weights.rows; i++) {
					m_weights.data[i] = Xavier(m_inputCount, m_neuronCount);
			}
			break;
		}
		}
		std::cout << "weights initialized" << std::endl; // debug print
	}

	void Layer::initOptimizer(OptimizerType op) {
		switch (op) {
		case OptimizerType::SGD:
			optimizer = OptimizerType::SGD;
			break;
		case OptimizerType::Adam:
			optimizer = OptimizerType::Adam;
			m_firstMoment = Matrix(m_neuronCount, m_inputCount);
			m_secondMoment = Matrix(m_neuronCount, m_inputCount);
			m_firstMomentBias.resize(m_neuronCount, 0.0f);
			m_secondMomentBias.resize(m_neuronCount, 0.0f);
			break;
		}
	}

	const std::vector<float>& Layer::Forward(const std::vector<float>& input) {
		const float* __restrict inPtr = input.data();

		if (m_weights.cols != input.size()) {
			std::cout << "Invalid input in matrix multiplication " << std::endl;
			throw std::invalid_argument("...");
		}

		//matrix multiplication
		#pragma omp parallel for
		for (int i = 0; i < m_neuronCount; i++) {
			float* __restrict rptr = m_weights.row(i);
			float sum = m_biases[i];
			#pragma omp simd
			for (int j = 0; j < m_inputCount; j++) {
				sum += inPtr[j] * rptr[j];
			}
			m_zResult[i] = sum;
		}

		//switch case to choose the activasion method
		switch (activation) {
		case ActivationType::ReLU:
		{
			for (int i = 0; i < m_neuronCount; i++) {
				m_result[i] = std::max(0.0f, m_zResult[i]);
			}
			break;
		}
		case ActivationType::Softmax:
		{
			float max_val = m_zResult[0];
			for (int i = 1; i < m_neuronCount; i++) {
				if (m_zResult[i] > max_val) max_val = m_zResult[i];
			}
			float sum = 0.0f;
			for (int i = 0; i < m_neuronCount; i++) {
				m_result[i] = std::expf(m_zResult[i] - max_val);
				sum += m_result[i];
			}
			if (sum == 0.0f) {
				throw std::invalid_argument("Softmax sum is zero");
			}
			for (int i = 0; i < m_neuronCount; i++) {
				m_result[i] /= sum;
			}
			break;
		}
		case ActivationType::Sigmoid:
		{
			for (int i = 0; i < m_neuronCount; i++) {
				m_result[i] = 1.0f / (1.0f + std::expf(-m_zResult[i]));
			}
			break;
		}
		       case ActivationType::Tanh:
		       {
			       for (int i = 0; i < m_neuronCount; i++) {
				       m_result[i] = std::tanh(m_zResult[i]);
			       }
			       break;
		       }
		       case ActivationType::LeakyReLU:
		       {
			       for (int i = 0; i < m_neuronCount; i++) {
				       m_result[i] = (m_zResult[i] > 0.0f) ? m_zResult[i] : 0.01f * m_zResult[i];
			       }
			       break;
		       }
		       default:
			       //default to linear (maybe one day: alien AI will improve this)
			       break;
		}


		return(m_result);
	}

	inline float Layer::calculateActivationDerivative(float val) {
		       switch (activation) {
		       case ActivationType::ReLU:
			       return (val > 0.0f) ? 1.0f : 0.0f;
		       case ActivationType::Softmax:
			       return 1.0f;
		       case ActivationType::Sigmoid:
			       {
				       float s = 1.0f / (1.0f + std::expf(-val));
				       return s * (1.0f - s);
			       }
		       case ActivationType::Tanh:
			       {
				       float t = std::tanh(val);
				       return 1.0f - t * t;
			       }
		       case ActivationType::LeakyReLU:
			       return (val > 0.0f) ? 1.0f : 0.01f;
		       default:
			       return 1.0f; // linear (maybe one day: alien AI)
		       }
	}

	void Layer::backpropagation(const std::vector<float>& prev_delta, const Matrix& previous_weights, const std::vector<float>& inputs, bool isOutput) {
		const int prevSize = (int)prev_delta.size();
		const float* __restrict inPtr = inputs.data();
		const int inputSize = (int)inputs.size();

		if (isOutput) {
#pragma omp parallel for
			for (int i = 0; i < m_neuronCount; i++) {
				float error = (m_result[i] - prev_delta[i]);

				m_zResult[i] = calculateActivationDerivative(m_zResult[i]);
				this->m_delta[i] = error * m_zResult[i];

				//store gradients
				float scaled_delta = m_delta[i];
				float* weightGradptr = m_weightGradients.row(i);
				#pragma omp simd
				for (int j = 0; j < inputSize; j++) {
					weightGradptr[j] += scaled_delta * inPtr[j];
				}
				m_biasGradients[i] += scaled_delta;
			}
		}
		else {
#pragma omp parallel for
			for (int i = 0; i < m_neuronCount; i++) {
				float error = 0.0f;
				for (int j = 0; j < prevSize; j++) {
					error += prev_delta[j] * previous_weights(j, i);
				}

				m_zResult[i] = calculateActivationDerivative(m_zResult[i]);
				this->m_delta[i] = error * m_zResult[i];

				//store gradients
				float scaled_delta = m_delta[i];
				float* weightGradptr = m_weightGradients.row(i);
				#pragma omp simd
				for (int j = 0; j < inputSize; j++) {
					weightGradptr[j] += scaled_delta * inPtr[j];
				}
				m_biasGradients[i] += scaled_delta;
			}
		}
	}

	void Layer::updateWeights(int batchSize, float learningRate) {

		switch (optimizer) {
		case OptimizerType::SGD:

		{
			float scale = learningRate / batchSize;
			#pragma omp parallel for
			for (int i = 0; i < m_neuronCount; i++) {
				float* weightptr = m_weights.row(i);
				float* weightGradptr = m_weightGradients.row(i);
				#pragma omp simd
				for (int j = 0; j < m_inputCount; j++) {
					weightptr[j] -= scale * weightGradptr[j];
				}
				m_biases[i] -= scale * m_biasGradients[i];;
			}
			m_weightGradients.clear();
			m_biasGradients.assign(m_neuronCount, 0.0f);
			break;
		}

		case OptimizerType::Adam:

		{

			const float beta1 = 0.9f;
			const float beta2 = 0.999f;
			const float epsilon = 1e-8f;

			float beta1Correction = 1.0f - pow(beta1, m_timeStep);
			float beta2Correction = 1.0f - pow(beta2, m_timeStep);

			const float oneMbeta1 = 1.0f - beta1;
			const float oneMbeta2 = 1.0f - beta2;

			#pragma omp parallel for
			for (int i = 0; i < m_neuronCount; i++) {
				

				float* weightptr = m_weights.row(i);
				float* weightGradptr = m_weightGradients.row(i);

				float* fMptr = m_firstMoment.row(i);
				float* sMptr = m_secondMoment.row(i);

				#pragma omp simd
				for (int j = 0; j < m_inputCount; j++) {
					;
					fMptr[j] = beta1 * fMptr[j] + oneMbeta1 * weightGradptr[j];
					sMptr[j] = beta2 * sMptr[j] + oneMbeta2 * weightGradptr[j] * weightGradptr[j];

					float mHat = fMptr[j] / beta1Correction;
					float vHat = sMptr[j] / beta2Correction;

					weightptr[j] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
				}
				m_firstMomentBias[i] = beta1 * m_firstMomentBias[i] + oneMbeta1 * m_biasGradients[i];
				m_secondMomentBias[i] = beta2 * m_secondMomentBias[i] + oneMbeta2 * m_biasGradients[i] * m_biasGradients[i];

				float mHat = m_firstMomentBias[i] / beta1Correction;
				float vHat = m_secondMomentBias[i] / beta2Correction;

				m_biases[i] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
			}
			m_weightGradients.clear();
			m_biasGradients.assign(m_neuronCount, 0.0f);
			m_timeStep++;
			break;
		}

		}

	}

}