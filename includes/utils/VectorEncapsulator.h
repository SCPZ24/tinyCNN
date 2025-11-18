//
// Created by SCPZ24 on 2025/11/15.
//

#ifndef CPPMNIST_VECTORENCAPSULATOR_H
#define CPPMNIST_VECTORENCAPSULATOR_H
#include <iostream>
#include <vector>

template<typename T>
void DataTypeintShape(std::vector<std::vector<std::vector<T>>>& tensor) {
    cout<<"Shape:"<<tensor.size()<<' '<<tensor[0].size()<<' '<<tensor[0][0].size()<<endl;
}

template<typename T>
void vector_to_tensor(std::vector<std::vector<std::vector<T>>>& dst, const std::vector<T>& src, const uint channels, const uint height, const uint width) {
    uint j = 0;
    for (uint c = 0; c < channels; ++c) {
        for (uint h = 0; h < height; ++h) {
            for (uint w = 0; w < width; ++w, ++j) {
                dst[c][h][w] = src[j];
            }
        }
    }
}

template<typename T>
void vector_to_tensor_pad(std::vector<std::vector<std::vector<T>>>& dst,const std::vector<T>& src,
    const uint channels, const uint height, const uint width, const uint padding, const T pad_content) {
    /**
     *height/width: pass in the height/weight after padding.
     */

    uint j = 0;
    for (uint c = 0; c < channels; ++c) {
        for (uint h = 0; h < padding; ++h) {
            for (uint w = 0; w < width; ++w) {
                dst[c][h][w] = pad_content;
            }
        }
        for (uint h = padding; h < height-padding; ++h) {
            for (uint w = 0; w < padding; ++w) {
                dst[c][h][w] = pad_content;
            }
            for (uint w = padding; w < width-padding; ++w, ++j) {
                dst[c][h][w] = src[j];
            }
            for (uint w = width-padding; w < width; ++w) {
                dst[c][h][w] = pad_content;
            }
        }
        for (uint h = height-padding; h < height; ++h) {
            for (uint w = 0; w < width; ++w) {
                dst[c][h][w] = pad_content;
            }
        }
    }
}

template<typename T>
void tensor_to_vector(std::vector<T>& dst, const std::vector<std::vector<std::vector<T>>>& src, const uint channels, const uint height, const uint width) {
    uint j = 0;
    for (uint c = 0; c < channels; ++c) {
        for (uint h = 0; h < height; ++h) {
            for (uint w = 0; w < width; ++w, ++j) {
                dst[j] = src[c][h][w];
            }
        }
    }
}

template<typename T>
void tensor_to_vector_pad(std::vector<T>& dst, const std::vector<std::vector<std::vector<T>>>& src,
    const uint channels, const uint height, const uint width, const uint padding) {
    /**
     *height, width passed in are values after padding.
     */
    uint j = 0;
    for (uint c = 0; c < channels; ++c) {
        for (uint h = padding; h < height - padding; ++h) {
            for (uint w = padding; w < width - padding; ++w, ++j) {
                dst[j] = src[c][h][w];
            }
        }
    }
}

template<typename T>
void tensor_allocate(std::vector<std::vector<std::vector<T>>>& tensor, const uint channels, const uint height, const uint width) {
    tensor.resize(channels);
    for (uint c = 0 ; c < channels ; ++c) {
        tensor[c].resize(height);
        for (uint h = 0 ; h < height ; ++h) {
            tensor[c][h].resize(width);
        }
    }
}

template<typename T>
void tensor_allocate(std::vector<std::vector<std::vector<T>>>& tensor, const uint channels, const uint height, const uint width, const T fill_value) {
    tensor.resize(channels);
    for (uint c = 0 ; c < channels ; ++c) {
        tensor[c].resize(height);
        for (uint h = 0 ; h < height ; ++h) {
            tensor[c][h].resize(width);
            for (uint w = 0 ; w < width ; ++w) {
                tensor[c][h][w] = fill_value;
            }
        }
    }
}

template<typename T>
vector<vector<vector<T>>> flipKernel(const vector<vector<vector<T>>>& kernel, const uint kernel_size) {
    const uint channels = kernel.size();
    vector flipped(channels, vector(kernel_size, vector(kernel_size, T())));
    for (uint c = 0; c < channels; ++c) {
        for (uint h = 0; h < kernel_size; ++h) {
            for (uint w = 0; w < kernel_size; ++w) {
                flipped[c][h][w] = kernel[c][kernel_size - 1 - h][kernel_size - 1 - w];
            }
        }
    }
    return flipped;
}

#endif //CPPMNIST_VECTORENCAPSULATOR_H