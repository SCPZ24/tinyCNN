//
// Created by SCPZ24 on 2025/11/15.
//

#ifndef CPPMNIST_CONVOLUTION2D_H
#define CPPMNIST_CONVOLUTION2D_H
#include "Layer.h"

class Convolution2D final : public Layer{
public:
    Convolution2D(uint input_channels, uint output_channels, uint kernel_size, uint _stride, uint _padding, uint input_height, uint input_width);
    ~Convolution2D() override = default;

    vector<vector<DataType>> forward(const vector<vector<DataType>> &) override;
    vector<vector<DataType>> backward(const vector<vector<DataType>> &, DataType learningRate) override;
    void save(std::ostream&) const override;
    void load(std::istream&) override;
    [[nodiscard]] string type() const override;
private:
    uint inputChannels;
    uint kernelSize;
    uint stride;
    uint padding;

    uint inputHeight;
    uint inputWidth;

    uint inputParameters;

    uint outputHeight;
    uint outputWidth;

    uint outputChannels;
    uint outputParameters;

    vector<vector<vector<vector<DataType>>>> weights;
    //outputChannels * inputChannels * height(kernelSize) * width(kernelSize)
    vector<DataType> bias;
    //outputChannels

    vector<vector<vector<vector<DataType>>>> inputTensor;
};


#endif //CPPMNIST_CONVOLUTION2D_H