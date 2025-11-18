//
// Created by SCPZ24 on 2025/11/16.
//

#ifndef CPPMNIST_MODEL_H
#define CPPMNIST_MODEL_H
#include "../layers/Layer.h"
#include "../losses/Loss.h"
#include <memory>
#include <string>

class Model {
public:
    explicit Model(vector<unique_ptr<Layer>>&& _layers);

    [[nodiscard]] vector<vector<DataType>> forwardProp(const vector<vector<DataType>>& input) const;
    void backprop(const vector<vector<DataType>>& outputGrad, DataType learningRate) const;
    void save(const string& path) const;
    void load(const string& path) const;

private:
    vector<unique_ptr<Layer>> layers;
};


#endif //CPPMNIST_MODEL_H