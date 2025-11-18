//
// Created by SCPZ24 on 2025/11/15.
//

#include "../../includes/layers/Convolution2D.h"
#include "../../includes/utils/VectorEncapsulator.h"

#include <random>
#include <cmath>
#include <utility>
#include <ostream>
#include <istream>

Convolution2D::Convolution2D(const uint input_channels, const uint output_channels, const uint kernel_size,
                             const uint _stride, const uint _padding, const uint input_height, const uint input_width) :
    inputChannels(input_channels), kernelSize(kernel_size), stride(_stride),
    padding(_padding), inputHeight(input_height), inputWidth(input_width), outputChannels(output_channels) {
    outputHeight = (input_height - kernelSize + 2 * padding) / stride + 1;
    outputWidth = (input_width - kernelSize + 2 * padding) / stride + 1;
    inputParameters = inputChannels * inputHeight * inputWidth;
    outputParameters = outputChannels * outputHeight * outputWidth;

    std::random_device rd;
    std::mt19937 gen(rd());
    const DataType fan_in = static_cast<DataType>(inputChannels * kernelSize * kernelSize);
    const DataType he_std = std::sqrt(static_cast<DataType>(2.0) / fan_in);
    std::normal_distribution<DataType> dist(0.0, he_std);

    weights.resize(outputChannels);
    for (uint o = 0; o < outputChannels; ++o) {
        weights[o].resize(inputChannels);
        for (uint i = 0; i < inputChannels; ++i) {
            weights[o][i].resize(kernelSize);
            for (uint h = 0; h < kernelSize; ++h) {
                weights[o][i][h].resize(kernelSize);
                for (uint w = 0; w < kernelSize; ++w) {
                    weights[o][i][h][w] = dist(gen);
                }
            }
        }
    }
    bias.resize(outputChannels, 0.0);
}

vector<vector<DataType>> Convolution2D::forward(const vector<vector<DataType>> &input) {
    /**
     *input : batch_size * input_channel * input_height * input_width
     *
     *weights : input_channel * input_height * input_width FOR output_channel
     *bias : output_channel * 1
     */

    const uint batch_size = input.size();
    vector<vector<DataType>> output(batch_size);
    inputTensor = vector<vector<vector<vector<DataType>>>>(batch_size);
    for (uint b = 0; b < batch_size; ++b) {
        output[b].resize(outputParameters);
        vector<vector<vector<DataType>>> padded_input_tensor;
        tensor_allocate(padded_input_tensor, inputChannels, inputHeight+2*padding, inputWidth+2*padding);
        vector_to_tensor_pad(padded_input_tensor, input[b], inputChannels, inputHeight+2*padding, inputWidth+2*padding, padding,DataType());
        vector<vector<vector<DataType>>> output_tensor;
        tensor_allocate<DataType>(output_tensor, outputChannels, outputHeight, outputWidth);
        for (uint o_c = 0; o_c < outputChannels; ++o_c) {
            for (uint o_h = 0; o_h < outputHeight; ++o_h) {
                for (uint o_w = 0; o_w < outputWidth; ++o_w) {
                    DataType sum = 0.0;
                    for (uint i_c = 0; i_c < inputChannels; ++i_c) {
                        for (uint i_h = 0; i_h < kernelSize; ++i_h) {
                            for (uint i_w = 0; i_w < kernelSize; ++i_w) {
                                sum += padded_input_tensor[i_c][o_h*stride+i_h][o_w*stride+i_w] * weights[o_c][i_c][i_h][i_w];
                            }
                        }
                    }
                    output_tensor[o_c][o_h][o_w] = sum + bias[o_c];
                }
            }
        }
        tensor_to_vector(output[b], output_tensor, outputChannels, outputHeight, outputWidth);
        inputTensor[b] = std::move(padded_input_tensor);
    }
    return output;
}

vector<vector<DataType>> Convolution2D::backward(const vector<vector<DataType>> &gradOutput, const DataType learningRate) {
    const uint batch_size = gradOutput.size();
    vector<vector<vector<vector<DataType>>>> grad_W(outputChannels, vector<vector<vector<DataType>>>(inputChannels, vector<vector<DataType>>(kernelSize, vector<DataType>(kernelSize, 0.0))));
    vector<DataType> grad_b(outputChannels, 0.0);
    vector<vector<vector<vector<DataType>>>> grad_x(batch_size);
    vector<vector<vector<vector<DataType>>>> gradOutput_tensor(batch_size);
    const uint input_height_features = inputHeight + 2 * padding;
    const uint input_width_features = inputWidth + 2 * padding;

    for (uint b = 0; b < batch_size; ++b) {
        tensor_allocate(grad_x[b], inputChannels, inputHeight+2*padding, inputWidth+2*padding);
        tensor_allocate(gradOutput_tensor[b], outputChannels, outputHeight, outputWidth);
        vector_to_tensor(gradOutput_tensor[b], gradOutput[b], outputChannels, outputHeight, outputWidth);

        for (uint o_c = 0; o_c < outputChannels; ++o_c) {
            //calculate the grad(bias) on this bias and this channel
            for (uint o_h = 0; o_h < outputHeight; ++o_h) {
                for (uint o_w = 0; o_w < outputWidth; ++o_w) {
                    grad_b[o_c] += gradOutput_tensor[b][o_c][o_h][o_w];
                }
            }

            //calculate the grad(weights) on this bias and this channel
            for (uint i_c = 0; i_c < inputChannels; ++i_c) {
                for (uint i_h = 0; i_h < kernelSize; ++i_h) {
                    for (uint i_w = 0; i_w < kernelSize; ++i_w) {
                        DataType sum = 0.0;
                        for (uint o_h = 0; o_h < outputHeight; ++o_h) {
                            for (uint o_w = 0; o_w < outputWidth; ++o_w) {
                                sum += inputTensor[b][i_c][o_h*stride+i_h][o_w*stride+i_w] * gradOutput_tensor[b][o_c][o_h][o_w];
                            }
                        }
                        grad_W[o_c][i_c][i_h][i_w] += sum;
                    }
                }
            }

            //calculate the grad(input) on this bias and this channel
            const vector<vector<vector<DataType>>> flippedKernel = flipKernel<DataType>(weights[o_c], kernelSize);
            for (uint i_c = 0; i_c < inputChannels; ++i_c) {
                for (uint i_h = 0; i_h < input_height_features; ++i_h) {  //input_height_features is inputHeight+2*padding
                    for (uint i_w = 0; i_w < input_width_features; ++i_w) {
                        DataType sum = 0.0;
                        for (uint k_h = 0; k_h < kernelSize; ++k_h) {
                            for (uint k_w = 0; k_w < kernelSize; ++k_w) {
                                if (i_h < k_h) continue;
                                if (i_w < k_w) continue;
                                const uint o_h_candidate = i_h - k_h;
                                const uint o_w_candidate = i_w - k_w;
                                if (o_h_candidate % stride != 0 || o_w_candidate % stride != 0) continue;
                                const uint o_h = o_h_candidate / stride;
                                const uint o_w = o_w_candidate / stride;
                                if (o_h < outputHeight && o_w < outputWidth) {
                                    sum += flippedKernel[i_c][k_h][k_w] * gradOutput_tensor[b][o_c][o_h][o_w];
                                }
                            }
                        }
                        grad_x[b][i_c][i_h][i_w] += sum;
                    }
                }
            }
        }
    }

    //gd for weights
    for (uint o_c = 0; o_c < outputChannels; ++o_c) {
        for (uint i_c = 0; i_c < inputChannels; ++i_c) {
            for (uint h = 0; h < kernelSize; ++h) {
                for (uint w = 0; w < kernelSize; ++w) {
                    weights[o_c][i_c][h][w] -= learningRate * grad_W[o_c][i_c][h][w] / static_cast<DataType>(batch_size);
                }
            }
        }
    }

    //gd for bias
    for (uint o_c = 0; o_c < outputChannels; ++o_c) {
        bias[o_c] -= learningRate * (grad_b[o_c] / static_cast<DataType>(batch_size));
    }

    //cut the padding and view grad(input)
    vector<vector<DataType>> ret(batch_size);
    for (uint b = 0; b < batch_size; ++b) {
        ret[b].resize(inputParameters);
        tensor_to_vector_pad(ret[b], grad_x[b], inputChannels, inputHeight+2*padding, inputWidth+2*padding, padding);
    }

    return ret;
}

void Convolution2D::save(std::ostream& os) const {
    uint ic = inputChannels, oc = outputChannels, ks = kernelSize, st = stride, pd = padding, ih = inputHeight, iw = inputWidth, oh = outputHeight, ow = outputWidth;
    os.write(reinterpret_cast<const char*>(&ic), sizeof(ic));
    os.write(reinterpret_cast<const char*>(&oc), sizeof(oc));
    os.write(reinterpret_cast<const char*>(&ks), sizeof(ks));
    os.write(reinterpret_cast<const char*>(&st), sizeof(st));
    os.write(reinterpret_cast<const char*>(&pd), sizeof(pd));
    os.write(reinterpret_cast<const char*>(&ih), sizeof(ih));
    os.write(reinterpret_cast<const char*>(&iw), sizeof(iw));
    os.write(reinterpret_cast<const char*>(&oh), sizeof(oh));
    os.write(reinterpret_cast<const char*>(&ow), sizeof(ow));
    for (uint o = 0; o < oc; ++o) {
        for (uint i = 0; i < ic; ++i) {
            for (uint h = 0; h < ks; ++h) {
                for (uint w = 0; w < ks; ++w) {
                    const DataType v = weights[o][i][h][w];
                    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
                }
            }
        }
    }
    for (uint o = 0; o < oc; ++o) {
        const DataType b = bias[o];
        os.write(reinterpret_cast<const char*>(&b), sizeof(b));
    }
}

void Convolution2D::load(std::istream& is_) {
    uint ic=0, oc=0, ks=0, st=0, pd=0, ih=0, iw=0, oh=0, ow=0;
    is_.read(reinterpret_cast<char*>(&ic), sizeof(ic));
    is_.read(reinterpret_cast<char*>(&oc), sizeof(oc));
    is_.read(reinterpret_cast<char*>(&ks), sizeof(ks));
    is_.read(reinterpret_cast<char*>(&st), sizeof(st));
    is_.read(reinterpret_cast<char*>(&pd), sizeof(pd));
    is_.read(reinterpret_cast<char*>(&ih), sizeof(ih));
    is_.read(reinterpret_cast<char*>(&iw), sizeof(iw));
    is_.read(reinterpret_cast<char*>(&oh), sizeof(oh));
    is_.read(reinterpret_cast<char*>(&ow), sizeof(ow));
    inputChannels = ic; outputChannels = oc; kernelSize = ks; stride = st; padding = pd; inputHeight = ih; inputWidth = iw; outputHeight = oh; outputWidth = ow;
    inputParameters = inputChannels * inputHeight * inputWidth;
    outputParameters = outputChannels * outputHeight * outputWidth;
    weights.resize(oc);
    for (uint o = 0; o < oc; ++o) {
        weights[o].resize(ic);
        for (uint i = 0; i < ic; ++i) {
            weights[o][i].resize(ks);
            for (uint h = 0; h < ks; ++h) {
                weights[o][i][h].resize(ks);
                for (uint w = 0; w < ks; ++w) {
                    is_.read(reinterpret_cast<char*>(&weights[o][i][h][w]), sizeof(weights[o][i][h][w]));
                }
            }
        }
    }
    bias.resize(oc);
    for (uint o = 0; o < oc; ++o) {
        is_.read(reinterpret_cast<char*>(&bias[o]), sizeof(bias[o]));
    }
}

string Convolution2D::type() const { return "Convolution2D"; }