//
// Created by SCPZ24 on 2025/11/14.
//

#ifndef CPPMNIST_SOFTMAX_H
#define CPPMNIST_SOFTMAX_H
#include "Layer.h"


class Softmax final : public Layer{
public:
    explicit Softmax(uint _size);
    ~Softmax() override = default;

    vector<vector<DataType>> forward(const vector<vector<DataType>> &) override;
    vector<vector<DataType>> backward(const vector<vector<DataType>> &, DataType learningRate) override;
    void save(std::ostream&) const override;
    void load(std::istream&) override;
    [[nodiscard]] string type() const override;
private:
    uint size;

    vector<vector<DataType>> outputs;
};


#endif //CPPMNIST_SOFTMAX_H