//
// Created by SCPZ24 on 2025/11/16.
//

#ifndef CPPMNIST_MNISTLOADER_H
#define CPPMNIST_MNISTLOADER_H

#include <vector>
#include <string>
#include <fstream>

#include "Loader.h"

class MnistLoader final : public Loader {
public:
        MnistLoader(const std::string& data_path, const std::string& label_path, uint batch_size);

        std::vector<std::vector<DataType>> loadData() override;
        std::vector<std::vector<DataType>> loadLabel() override;

        ~MnistLoader() override;

        uint getTotalSize() override;
private:
        std::ifstream dataFile;
        std::ifstream labelFile;

        uint totalImages;
        uint imageHeight;
        uint imageWidth;

        uint currentPosData;
        uint currentPosLabel;

        void resetData();
        void resetLabel();

        static uint32_t swapEndian(uint32_t value);
};

#endif //CPPMNIST_MNISTLOADER_H