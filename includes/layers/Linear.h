//
// Created by SCPZ24 on 2025/11/14.
//

#ifndef CPPMNIST_LINEARLAYER_H
#define CPPMNIST_LINEARLAYER_H

#include "Layer.h"

class Linear final : public Layer{
public:
    Linear(uint inputSize, uint outputSize);
    ~Linear() override = default;

    vector<vector<DataType>> forward(const vector<vector<DataType>> &input) override;
    vector<vector<DataType>> backward(const vector<vector<DataType>> &gradOutput, DataType learningRate) override;
    void save(std::ostream&) const override;
    void load(std::istream&) override;
    [[nodiscard]] string type() const override;
private:
    vector<DataType> weights;
    vector<DataType> bias;
    uint input_size;
    uint output_size;

    vector<vector<DataType>> inputs;
};


#endif //CPPMNIST_LINEARLAYER_H