//
// Created by SCPZ24 on 2025/11/14.
//

#include "../../includes/layers/Linear.h"

#include <random>
#include <cmath>
#include <iostream>

Linear::Linear(const uint inputSize, const uint outputSize):
input_size(inputSize),
output_size(outputSize) {
    const size_t weight_size = outputSize * inputSize;
    weights.resize(weight_size);
    bias.resize(outputSize);

    std::random_device rd;
    std::mt19937 gen(rd());

    const auto limit = static_cast<DataType>(std::sqrt(6.0 / (inputSize + outputSize)));
    std::uniform_real_distribution<DataType> weight_dist(-limit, limit);

    for (DataType & weight : weights) {
        weight = weight_dist(gen);
    }

    std::uniform_real_distribution<DataType> bias_dist(0.0, 0.0);
    for (DataType & bia : bias) {
        bia = bias_dist(gen);
    }
}

vector<vector<DataType>> Linear::forward(const vector<vector<DataType>> &input) {
    /**
     *W : output_size * input_size
     *b : output_size * 1
     *input x : input_size * 1
     *return : W @ x + b
     */

    inputs = vector<vector<DataType>>(input);
    vector<vector<DataType>> output(input.size());
    for (uint m = 0; m < output.size(); ++m) {
        output[m].resize(output_size,0.0);
        uint j = 0;
        for (uint i = 0; i < output_size; ++i) {
            DataType sum = 0.0;
            for (uint k = 0; k < input_size; ++k, ++j) {
                sum += weights[j] * input[m][k];
            }
            output[m][i] = sum + bias[i];
        }
    }
    return output;
}

vector<vector<DataType>> Linear::backward(const vector<vector<DataType>> &gradOutput, const DataType learningRate) {
    /**
     *W^T : input_size * output_size
     *gradOutput : output * 1
     *dj/dx : ret = W^T @ gradOutput (input_size * 1)
     *
     *dj/dw : grad_W = gradOutput @ x^T
     *x^T : 1 * input_size
     *
     *dj/dy : grad_b = gradOutput
     */

    const uint batch_size = gradOutput.size();
    vector<vector<DataType>> ret(batch_size);
    for (uint m = 0; m < ret.size(); ++m) {
        ret[m].resize(input_size,0.0);
        uint j = 0;
        for (uint i = 0; i < output_size; ++i) {
            const DataType current = gradOutput[m][i];
            for (uint k = 0; k < input_size; ++k, ++j) {
                ret[m][k] += weights[j] * current;
            }
        }
    }

    const uint weight_size = output_size * input_size;
    vector<vector<DataType>> grad_W(batch_size);
    for (uint m = 0; m < batch_size; ++m) {
        grad_W[m].resize(weight_size,0.0);
        uint j = 0;
        for (uint i = 0; i < output_size; ++i) {
            const DataType current = gradOutput[m][i];
            for (uint k = 0; k < input_size; ++k, ++j) {
                grad_W[m][j] = current * inputs[m][k];
            }
        }
    }
    vector<DataType> grad_weight(weight_size);
    for (uint i = 0 ; i < batch_size; ++i) {
        for (uint j = 0; j < weight_size; ++j) {
            grad_weight[j] += grad_W[i][j];
        }
    }
    for (uint i = 0 ; i < weight_size; ++i) {
        grad_weight[i] /= static_cast<DataType>(batch_size);
        weights[i] -= learningRate * grad_weight[i];
    }

    vector<DataType> grad_b(output_size);
    for (uint i = 0 ; i < batch_size; ++i) {
        for (uint j = 0; j < output_size; ++j) {
            grad_b[j] += gradOutput[i][j];
        }
    }
    for (uint i = 0 ; i < output_size; ++i) {
        grad_b[i] /= static_cast<DataType>(batch_size);
        bias[i] -= learningRate * grad_b[i];
    }

    return ret;
}

void Linear::save(std::ostream& os) const {
    uint is = input_size;
    uint osz = output_size;
    os.write(reinterpret_cast<const char*>(&is), sizeof(is));
    os.write(reinterpret_cast<const char*>(&osz), sizeof(osz));
    uint wlen = static_cast<uint>(weights.size());
    os.write(reinterpret_cast<const char*>(&wlen), sizeof(wlen));
    for (const DataType& w : weights) {
        os.write(reinterpret_cast<const char*>(&w), sizeof(w));
    }
    uint blen = static_cast<uint>(bias.size());
    os.write(reinterpret_cast<const char*>(&blen), sizeof(blen));
    for (const DataType& b : bias) {
        os.write(reinterpret_cast<const char*>(&b), sizeof(b));
    }
}

void Linear::load(std::istream& is_) {
    uint isz = 0, osz = 0; is_.read(reinterpret_cast<char*>(&isz), sizeof(isz)); is_.read(reinterpret_cast<char*>(&osz), sizeof(osz));
    uint wlen = 0; is_.read(reinterpret_cast<char*>(&wlen), sizeof(wlen));
    weights.resize(wlen);
    for (uint i = 0; i < wlen; ++i) {
        is_.read(reinterpret_cast<char*>(&weights[i]), sizeof(weights[i]));
    }
    uint blen = 0; is_.read(reinterpret_cast<char*>(&blen), sizeof(blen));
    bias.resize(blen);
    for (uint i = 0; i < blen; ++i) {
        is_.read(reinterpret_cast<char*>(&bias[i]), sizeof(bias[i]));
    }
    input_size = isz;
    output_size = osz;
}

string Linear::type() const { return "Linear"; }