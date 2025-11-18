//
// Created by SCPZ24 on 2025/11/16.
//

#ifndef CPPMNIST_LOSS_H
#define CPPMNIST_LOSS_H
#include <vector>
#include "../public.h"

class Loss {
public:
    virtual ~Loss() = default;

    virtual vector<DataType> forward(const vector<vector<DataType>>& input, const vector<vector<DataType>>& label) = 0; //consider batches
    virtual vector<vector<DataType>> backward() = 0;
};


#endif //CPPMNIST_LOSS_H