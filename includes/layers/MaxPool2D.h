//
// Created by SCPZ24 on 2025/11/14.
//

#ifndef CPPMNIST_MAXPOOL2D_H
#define CPPMNIST_MAXPOOL2D_H
#include "Layer.h"


class MaxPool2D final : public Layer {
public:
    MaxPool2D(uint kernel_size, uint _stride, uint _channels, uint _padding, uint input_height, uint input_width);
    ~MaxPool2D() override = default;

    vector<vector<DataType>> forward(const vector<vector<DataType>> &input)override;
    vector<vector<DataType>> backward(const vector<vector<DataType>> &, DataType learningRate) override;
    void save(std::ostream&) const override;
    void load(std::istream&) override;
    [[nodiscard]] string type() const override;
private:
    uint kernelSize;
    uint stride;
    uint channels;
    uint padding;

    uint inputHeight;
    uint inputWidth;
    uint inputParameters;

    uint outputHeight;
    uint outputWidth;
    uint outputParameters;

    vector<vector<uint>> maxPosition; // batch_size * channel * outputHeight * outputWeight.
};


#endif //CPPMNIST_MAXPOOL2D_H