//
// Created by SCPZ24 on 2025/11/16.
//

#include "../../includes/losses/CrossEntropy.h"
#include <cmath>

CrossEntropy::CrossEntropy(const uint _size): size(_size) {}

vector<DataType> CrossEntropy::forward(const vector<vector<DataType>> &input, const vector<vector<DataType>> &label) {
    const uint batch_size = input.size();
    vector<DataType> output(batch_size);
    grad_loss = vector<vector<DataType>>(batch_size);
    for (uint b = 0; b < batch_size; ++b) {
        grad_loss[b].resize(size,0.0);
        for (uint i = 0; i < size; ++i) {
            if (label[b][i] > 1e-5) {
                const DataType value = std::max<DataType>(input[b][i], 1e-10); //truncate the small value to prevent gradient explosion
                output[b] = -log(value);
                grad_loss[b][i] = -1.0 / value;
            }
        }
    }
    return output;
}

vector<vector<DataType>> CrossEntropy::backward() {
    return std::move(grad_loss);
}
