//
// Created by SCPZ24 on 2025/11/16.
//

#ifndef CPPMNIST_TRAINER_H
#define CPPMNIST_TRAINER_H
#include "Model.h"
#include "../loader/Loader.h"


class Trainer {
public:
    Trainer(Model& _model, unique_ptr<Loss> _loss, unique_ptr<Loader> _loader, DataType learning_rate, uint batches_per_epoch, uint _epoch);

    void train() const;
private:
    Model& model;
    unique_ptr<Loss> loss;
    unique_ptr<Loader> loader;

    DataType learningRate;

    uint batchesPerEpoch;
    uint epoch;
};


#endif //CPPMNIST_TRAINER_H