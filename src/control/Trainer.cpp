//
// Created by SCPZ24 on 2025/11/16.
//

#include <iostream>
#include "../../includes/control/Trainer.h"
#include <filesystem>
#include <algorithm>

Trainer::Trainer(Model& _model, unique_ptr<Loss> _loss, unique_ptr<Loader> _loader, const DataType learning_rate,
                 const uint batches_per_epoch, const uint _epoch):
model(_model), loss(std::move(_loss)), loader(std::move(_loader)), learningRate(learning_rate),
batchesPerEpoch(batches_per_epoch), epoch(_epoch){}

void Trainer::train() const {
    const uint totalSize = loader->getTotalSize();
    const std::filesystem::path ckpt_dir = std::filesystem::path("checkpoints");
    const std::filesystem::path last = ckpt_dir / "last.bin";
    if (std::filesystem::exists(last)) model.load(last.string());
    for (uint e = 0; e < epoch; ++e) {
        cout<<"Epoch "<<e+1<<endl;
        DataType sum_loss = 0.0;
        for (uint bpe = 0; bpe < batchesPerEpoch; ++bpe) {
            vector<vector<DataType>> data = loader->loadData();
            vector<vector<DataType>> label = loader->loadLabel();

            vector<vector<DataType>> pred = model.forwardProp(data);
            vector<DataType> losses = loss->forward(pred, label);
            vector<vector<DataType>> loss_grad = loss->backward();

            for (const DataType l : losses) {
                sum_loss += l;
            }

            model.backprop(loss_grad, learningRate);
        }
        cout<<"Average loss : "<<sum_loss / static_cast<float>(totalSize)<<endl<<endl;
        std::filesystem::create_directories(ckpt_dir);
        model.save((ckpt_dir / ("epoch_" + std::to_string(e) + ".bin")).string());
        model.save(last.string());
    }
    cout<<"Training finished!!!"<<endl;
}
