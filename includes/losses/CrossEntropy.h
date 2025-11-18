//
// Created by SCPZ24 on 2025/11/16.
//

#ifndef CPPMNIST_CROSSENTROPY_H
#define CPPMNIST_CROSSENTROPY_H
#include "Loss.h"


class CrossEntropy final : public Loss{
public:
    explicit CrossEntropy(uint _size);
    ~CrossEntropy() override = default;

    vector<DataType> forward(const vector<vector<DataType>> &input, const vector<vector<DataType>> &label) override;
    vector<vector<DataType>> backward() override;
private:
    uint size;

    vector<vector<DataType>> grad_loss;
};


#endif //CPPMNIST_CROSSENTROPY_H