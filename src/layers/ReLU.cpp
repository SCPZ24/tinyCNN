//
// Created by SCPZ24 on 2025/11/14.
//

#include "../../includes/layers/ReLU.h"

ReLU::ReLU(const uint size): size(size) {}

vector<vector<DataType>> ReLU::forward(const vector<vector<DataType>> &input) {
    vector<vector<DataType>> output(input.size());
    hasGradient = vector<vector<bool>>(input.size());
    for (uint i = 0; i < output.size(); ++i) {
        output[i].resize(size,0.0);
        hasGradient[i].resize(size);
        for (uint j = 0; j < size; ++j) {
            const bool has_gradient = input[i][j] > 0.0;
            output[i][j] = has_gradient ? input[i][j] : 0;
            hasGradient[i][j] = has_gradient;
        }
    }
    return std::move(output);
}

vector<vector<DataType>> ReLU::backward(const vector<vector<DataType>> &gradOutput, DataType learningRate) {
    /**
     *gradOutput : size * 1
     *dy/dx : size * size
     *
     *return : dj/dx (size * 1)
     */

    vector<vector<DataType>> ret(gradOutput.size());
    for (uint i = 0; i < ret.size(); ++i) {
        ret[i].resize(size,0.0);
        for (uint j = 0; j < size; ++j) {
            ret[i][j] = hasGradient[i][j] ? gradOutput[i][j] : 0;
        }
    }
    return ret;
}

void ReLU::save(std::ostream& os) const {
    uint sz = size;
    os.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
}

void ReLU::load(std::istream& is_) {
    uint sz = 0; is_.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    size = sz;
}

string ReLU::type() const { return "ReLU"; }
