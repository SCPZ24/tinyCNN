//
// Created by SCPZ24 on 2025/11/14.
//

#include "../../includes/layers/Softmax.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <ostream>
#include <istream>

Softmax::Softmax(const uint _size):size(_size){}

vector<vector<DataType>> Softmax::forward(const vector<vector<DataType>>& input) {
    const uint batch_size = input.size();
    outputs.resize(batch_size);

    for (uint b = 0; b < batch_size; ++b) {
        DataType max_val = -std::numeric_limits<DataType>::infinity();
        for (uint i = 0; i < size; ++i) {
            const DataType v = input[b][i];
            if (std::isfinite(v) && v > max_val) max_val = v;
        }
        if (!std::isfinite(max_val)) max_val = 0.0;

        std::vector<DataType> exps(size);
        DataType sum_exp = 0.0;
        for (uint i = 0; i < size; ++i) {
            const DataType v = input[b][i];
            exps[i] = std::isfinite(v) ? std::exp(v - max_val) : 0.0;
            sum_exp += exps[i];
        }

        outputs[b].resize(size);
        const DataType eps = 1e-12;
        const DataType denom = sum_exp + eps;
        for (uint i = 0; i < size; ++i) {
            outputs[b][i] = exps[i] / denom;
        }
    }

    return outputs;
}

std::vector<std::vector<DataType>> Softmax::backward(const std::vector<std::vector<DataType>>& gradOutputs, const DataType /*learningRate*/) {
    const uint batch_size = gradOutputs.size();
    std::vector<std::vector<DataType>> gradInputs(batch_size);

    for (uint b = 0; b < batch_size; ++b) {
        gradInputs[b].resize(size, 0.0);

        DataType dot_product = 0.0;
        for (uint j = 0; j < size; ++j) {
            const DataType go = gradOutputs[b][j];
            const DataType yj = outputs[b][j];
            if (!std::isfinite(go) || !std::isfinite(yj)) continue;
            dot_product += yj * go;
        }

        for (uint i = 0; i < size; ++i) {
            gradInputs[b][i] = outputs[b][i] * (gradOutputs[b][i] - dot_product);
        }
    }

    return gradInputs;
}

void Softmax::save(std::ostream& os) const {
    uint sz = size;
    os.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
}

void Softmax::load(std::istream& is_) {
    uint sz = 0; is_.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    size = sz;
}

string Softmax::type() const { return "Softmax"; }

// vector<vector<DataType>> Softmax::forward(const vector<vector<DataType>> &input) {
//     const uint batch_size = input.size();
//     vector<vector<DataType>> output(batch_size);
//     for (uint b = 0 ; b < batch_size; ++b) {
//         output[b].resize(size,0.0);
//         DataType exps[size];
//         for (uint i = 0 ; i < size ; ++i) {
//             exps[i] = exp(input[b][i]);
//         }
//         DataType sum = 0;
//         for (uint i = 0 ; i < size ; ++i) {
//             sum += exps[i];
//         }
//         for (uint i = 0 ; i < size ; ++i) {
//             output[b][i] = exps[i] / sum;
//         }
//     }
//     outputs = vector(output);
//     return std::move(output);
// }
//
// vector<vector<DataType>> Softmax::backward(const vector<vector<DataType>> &gradOutputs, const DataType learningRate) {
//     const uint batch_size = gradOutputs.size();
//     vector<vector<DataType>> ret(batch_size);
//
//     for (uint b = 0; b < batch_size; ++b) {
//         ret[b].resize(size,0.0);
//         DataType dot_product = 0.0;
//         for (uint j = 0; j < size; ++j) {
//             dot_product += outputs[b][j] * gradOutputs[b][j];
//         }
//         for (uint i = 0; i < size; ++i) {
//             ret[b][i] = outputs[b][i] * (gradOutputs[b][i] - dot_product);
//         }
//     }
//
//     return ret;
// }


