//
// Created by SCPZ24 on 2025/11/13.
//

#ifndef CPPMNIST_LAYER_H
#define CPPMNIST_LAYER_H

#include <vector>
#include "../public.h"
#include <ostream>
#include <istream>

class Layer {
public:
    Layer() = default;
    virtual ~Layer() = default;
    virtual vector<vector<DataType>> forward(const vector<vector<DataType>> &) = 0;
    virtual vector<vector<DataType>> backward(const vector<vector<DataType>> &, DataType learningRate) = 0;
    virtual void save(std::ostream&) const = 0;
    virtual void load(std::istream&) = 0;
    [[nodiscard]] virtual string type() const = 0;
};


#endif //CPPMNIST_LAYER_H