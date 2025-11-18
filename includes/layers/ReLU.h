//
// Created by SCPZ24 on 2025/11/14.
//

#ifndef CPPMNIST_RELU_H
#define CPPMNIST_RELU_H
#include "Layer.h"


class ReLU final : public Layer{
public:
    explicit ReLU(uint size);
    ~ReLU() override = default;

    vector<vector<DataType>> forward(const vector<vector<DataType>> &input) override;
    vector<vector<DataType>> backward(const vector<vector<DataType>> &gradOutput, DataType learningRate) override;
    void save(std::ostream&) const override;
    void load(std::istream&) override;
    [[nodiscard]] string type() const override;

private:
    uint size;

    vector<vector<bool>> hasGradient;
};


#endif //CPPMNIST_RELU_H