//
// Created by SCPZ24 on 2025/11/14.
//

#include "../../includes/layers/MaxPool2D.h"
#include "../../includes/utils/VectorEncapsulator.h"
#include <limits>
#include <ostream>
#include <istream>

MaxPool2D::MaxPool2D(const uint kernel_size, const uint _stride, const uint _channels, const uint _padding, const uint input_height, const uint input_width):
kernelSize(kernel_size), stride(_stride), channels(_channels), padding(_padding), inputHeight(input_height), inputWidth(input_width){
    outputHeight = (input_height + 2 * padding - kernel_size) / stride + 1;
    outputWidth = (input_width + 2 * padding - kernel_size) / stride + 1;
    inputParameters = channels * inputHeight * inputWidth;
    outputParameters = channels * outputHeight * outputWidth;
}

vector<vector<DataType>> MaxPool2D::forward(const vector<vector<DataType>> &input) {
    const uint batch_size = input.size();
    maxPosition = vector<vector<uint>>(batch_size);
    vector<vector<DataType>> output(batch_size);
    for (uint b = 0; b < batch_size; ++b) {
        output[b].resize(outputParameters);
        maxPosition[b].resize(outputParameters);
        vector<vector<vector<DataType>>> inputTensor, outputTensor;
        vector<vector<vector<uint>>> maxPositionTensor;
        tensor_allocate(inputTensor, channels, inputHeight+2*padding, inputWidth+2*padding);
        vector_to_tensor_pad<DataType>(inputTensor, input[b], channels, inputHeight+2*padding, inputWidth+2*padding, padding, -std::numeric_limits<DataType>::infinity());
        tensor_allocate(outputTensor, channels, outputHeight, outputWidth);
        tensor_allocate(maxPositionTensor, channels, outputHeight, outputWidth);
        for (uint c = 0; c < channels; ++c) {
            for (uint h = 0, o_h = 0 ; o_h < outputHeight ; h+=stride, ++o_h) {
                for (uint w = 0, o_w = 0 ; o_w < outputWidth ; w+=stride, ++o_w) {
                    DataType maxValue = -std::numeric_limits<DataType>::infinity();
                    uint maxIndex = 0;
                    for (uint i = 0; i < kernelSize; ++i) {
                        for (uint j = 0; j < kernelSize; ++j) {
                            if (maxValue < inputTensor[c][h+i][w+j]) {
                                maxValue = inputTensor[c][h+i][w+j];
                                maxIndex = i * kernelSize + j;
                            }
                        }
                    }
                    maxPositionTensor[c][o_h][o_w] = maxIndex;
                    outputTensor[c][o_h][o_w] = maxValue;
                }
            }
        }
        tensor_to_vector<DataType>(output[b], outputTensor, channels, outputHeight, outputWidth);
        tensor_to_vector<uint>(maxPosition[b], maxPositionTensor, channels, outputHeight, outputWidth);
    }
    return std::move(output);
}

vector<vector<DataType>> MaxPool2D::backward(const vector<vector<DataType>> &gradOutput, const DataType learningRate) {
    const uint batch_size = gradOutput.size();
    vector<vector<DataType>> ret(batch_size);
    for (uint b = 0; b < batch_size; ++b) {
        ret[b].resize(inputParameters);
        vector<vector<vector<DataType>>> gradTensor, gradInputTensor;
        vector<vector<vector<uint>>> maxPositionTensor;
        tensor_allocate(gradTensor, channels, outputHeight, outputWidth);
        vector_to_tensor<DataType>(gradTensor, gradOutput[b], channels, outputHeight, outputWidth);
        tensor_allocate(maxPositionTensor, channels, outputHeight, outputWidth);
        vector_to_tensor<uint>(maxPositionTensor, maxPosition[b], channels, outputHeight, outputWidth);
        tensor_allocate<DataType>(gradInputTensor, channels, inputHeight+2*padding, inputWidth+2*padding, 0.0);
        for (uint c = 0; c < channels; ++c) {
            for (uint h_o = 0 ; h_o < outputHeight ; ++h_o) {
                for (uint w_o = 0 ; w_o < outputWidth ; ++w_o) {
                    const uint maxIndex = maxPositionTensor[c][h_o][w_o];
                    const uint di = maxIndex / kernelSize, dj = maxIndex % kernelSize;
                    gradInputTensor[c][h_o * stride + di][w_o * stride + dj] += gradTensor[c][h_o][w_o];
                }
            }
        }
        tensor_to_vector_pad(ret[b], gradInputTensor, channels, inputHeight+2*padding, inputWidth+2*padding, padding);
    }
    return ret;
}

void MaxPool2D::save(std::ostream& os) const {
    uint ks = kernelSize, st = stride, ch = channels, pd = padding, ih = inputHeight, iw = inputWidth;
    os.write(reinterpret_cast<const char*>(&ks), sizeof(ks));
    os.write(reinterpret_cast<const char*>(&st), sizeof(st));
    os.write(reinterpret_cast<const char*>(&ch), sizeof(ch));
    os.write(reinterpret_cast<const char*>(&pd), sizeof(pd));
    os.write(reinterpret_cast<const char*>(&ih), sizeof(ih));
    os.write(reinterpret_cast<const char*>(&iw), sizeof(iw));
}

void MaxPool2D::load(std::istream& is_) {
    uint ks=0, st=0, ch=0, pd=0, ih=0, iw=0;
    is_.read(reinterpret_cast<char*>(&ks), sizeof(ks));
    is_.read(reinterpret_cast<char*>(&st), sizeof(st));
    is_.read(reinterpret_cast<char*>(&ch), sizeof(ch));
    is_.read(reinterpret_cast<char*>(&pd), sizeof(pd));
    is_.read(reinterpret_cast<char*>(&ih), sizeof(ih));
    is_.read(reinterpret_cast<char*>(&iw), sizeof(iw));
    kernelSize = ks; stride = st; channels = ch; padding = pd; inputHeight = ih; inputWidth = iw;
    outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
    outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;
    inputParameters = channels * inputHeight * inputWidth;
    outputParameters = channels * outputHeight * outputWidth;
}

string MaxPool2D::type() const { return "MaxPool2D"; }
