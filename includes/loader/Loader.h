//
// Created by SCPZ24 on 2025/11/16.
//

#ifndef CPPMNIST_LOADER_H
#define CPPMNIST_LOADER_H

#include <string>
#include <vector>
#include "../public.h"

class Loader {
public:
    virtual ~Loader() = default;

    Loader(std::string data_path, std::string label_path, uint batch_size);

    virtual std::vector<std::vector<DataType>> loadData() = 0;
    virtual std::vector<std::vector<DataType>> loadLabel() = 0;

    virtual uint getTotalSize() = 0;
protected:
    std::string dataPath;
    std::string labelPath;

    uint batchSize;
};


#endif //CPPMNIST_LOADER_H