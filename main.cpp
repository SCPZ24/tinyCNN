#include <iostream>
#include <vector>

#include "includes/control/Model.h"
#include "includes/control/Trainer.h"
#include "includes/layers/Convolution2D.h"
#include "includes/layers/Linear.h"
#include "includes/layers/MaxPool2D.h"
#include "includes/layers/ReLU.h"
#include "includes/layers/Softmax.h"
#include "includes/loader/Loader.h"
#include "includes/losses/Loss.h"
#include "includes/losses/CrossEntropy.h"
#include "includes/loader/MnistLoader.h"
#include "includes/control/Test.h"

#define train_test

int main() {
    constexpr uint batch = 100;
    vector<unique_ptr<Layer>> layers;
    layers.push_back(make_unique<Convolution2D>(1,4,3,1,1,28,28));
    layers.push_back(make_unique<ReLU>(4*28*28));
    layers.push_back(make_unique<MaxPool2D>(2,2,4,0,28,28));
    layers.push_back(make_unique<Convolution2D>(4,8,3,1,1,14,14));
    layers.push_back(make_unique<MaxPool2D>(2,2,8,0,14,14));
    layers.push_back(make_unique<Linear>(7*7*8,16));
    layers.push_back(make_unique<Linear>(16,10));
    layers.push_back(make_unique<Softmax>(10));
    Model model(std::move(layers));

#ifdef first_train_trial
    unique_ptr<Loader> loader(
        new MnistLoader("data/train-images-idx3-ubyte","data/train-labels.idx1-ubyte",batch)
        );
    unique_ptr<Loss> loss(new CrossEntropy(10));

    const uint bpe = loader->getTotalSize() / batch;
    const Trainer trainer(model,std::move(loss),std::move(loader),5e-3,bpe,10);
    trainer.train();
#endif

#ifdef extra_train_trial
    unique_ptr<Loader> loader(
        new MnistLoader("data/train-images-idx3-ubyte","data/train-labels.idx1-ubyte",batch)
        );
    unique_ptr<Loss> loss(new CrossEntropy(10));

    const uint bpe = loader->getTotalSize() / batch;
    model.load("checkpoints/last.bin");
    const Trainer trainer(model,std::move(loss),std::move(loader),5e-3,bpe,5);
    trainer.train();
#endif

#ifdef train_test
    const unique_ptr<Loader> loader(
        new MnistLoader("data/t10k-images.idx3-ubyte","data/t10k-labels.idx1-ubyte",batch)
        );
    model.load("checkpoints/last.bin");
    cout<<"Model loaded, testing..."<<endl;
    const uint bpe = loader->getTotalSize() / batch;
    test(model,loader,bpe);
#endif

    return 0;
}