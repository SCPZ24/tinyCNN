//
// Created by SCPZ24 on 2025/11/17.
//

#ifndef CPPMNIST_TEST_H
#define CPPMNIST_TEST_H
#include <iostream>

#include "Model.h"
#include "../loader/Loader.h"
#include "../public.h"

inline void test(const Model& model, const unique_ptr<Loader> &loader, const uint batches_per_epoch) {
    uint correct = 0;
    const uint total = loader->getTotalSize();
    for (uint i = 0 ; i < batches_per_epoch; ++i) {
        vector<vector<DataType>> data = loader->loadData();
        vector<vector<DataType>> label = loader->loadLabel();
        vector<vector<DataType>> pred = model.forwardProp(data);

        for (uint b = 0 ; b < data.size(); ++b) {
            auto max_it = ranges::max_element(pred[b]);
            const uint max_index_pred = std::distance(pred[b].begin(), max_it);
            max_it = ranges::max_element(label[b]);
            const uint max_index_label = std::distance(label[b].begin(), max_it);
            if (max_index_label == max_index_pred) {
                ++correct;
            }
        }
    }
    const double accuracy = static_cast<double>(correct) / total;
    std::cout<<"Accuracy : "<<accuracy * 100<<"%"<<std::endl;
}

#endif //CPPMNIST_TEST_H