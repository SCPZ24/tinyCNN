//
// Created by DouBao on 2025/11/16.
//

#include "../../includes/loader/MnistLoader.h"
#include <fstream>

MnistLoader::MnistLoader(const std::string &data_path, const std::string &label_path, const uint batch_size) :
Loader(data_path, label_path, batch_size) {
    dataFile = std::ifstream(dataPath, std::ios::binary);
    labelFile = std::ifstream(labelPath, std::ios::binary);

    if (!dataFile.is_open()) {
        throw std::runtime_error("Failed to load data:" + dataPath);
    }
    if (!labelFile.is_open()) {
        throw std::runtime_error("Failed to load label:" + labelPath);
    }

    uint32_t magicNumImages, magicNumLabels;
    uint32_t totalLabels;
    dataFile.read(reinterpret_cast<char*>(&magicNumImages), 4);
    dataFile.read(reinterpret_cast<char*>(&totalImages), 4);
    dataFile.read(reinterpret_cast<char*>(&imageHeight), 4);
    dataFile.read(reinterpret_cast<char*>(&imageWidth), 4);
    labelFile.read(reinterpret_cast<char*>(&magicNumLabels), 4);
    labelFile.read(reinterpret_cast<char*>(&totalLabels), 4);

    magicNumImages = swapEndian(magicNumImages);
    totalImages = swapEndian(totalImages);
    imageHeight = swapEndian(imageHeight);
    imageWidth = swapEndian(imageWidth);
    magicNumLabels = swapEndian(magicNumLabels);
    totalLabels = swapEndian(totalLabels);

    if (magicNumImages != 0x00000803 || magicNumLabels != 0x00000801) {
        throw std::runtime_error("Not MNIST Dataset!");
    }
    if (totalImages != totalLabels) {
        throw std::runtime_error("Unmatched Data and Label Pairs!");
    }

    currentPosData = 0;
    currentPosLabel = 0;
}

std::vector<std::vector<DataType>> MnistLoader::loadData() {
    uint actualSize = std::min(batchSize, totalImages - currentPosData);
    if (actualSize == 0) {
        resetData();
        actualSize = std::min(batchSize, totalImages);
    }

    std::vector<std::vector<DataType>> ret(actualSize);

    const uint imageMemory = imageHeight * imageWidth;
    for (uint b = 0; b < actualSize; ++b) {
        std::vector<DataType> image(imageMemory);
        unsigned char pixel;
        for (uint i = 0; i < imageMemory; ++i) {
            dataFile.read(reinterpret_cast<char*>(&pixel), 1);
            image[i] = static_cast<DataType>(pixel) / 255.0; // normalize to [0,1]
        }
        ret[b] = std::move(image);
    }
    currentPosData += actualSize;

    return ret;
}

std::vector<std::vector<DataType>> MnistLoader::loadLabel() {
    uint actualSize = std::min(batchSize, totalImages - currentPosLabel);
    if (actualSize == 0) {
        resetLabel();
        actualSize = std::min(batchSize, totalImages);
    }

    std::vector<std::vector<DataType>> ret(actualSize);

    for (size_t i = 0; i < actualSize; ++i) {
        ret[i].resize(10,0.0);
        unsigned char label;
        labelFile.read(reinterpret_cast<char*>(&label), 1);
        ret[i][label] = 1.0;
    }
    currentPosLabel += actualSize;

    return ret;
}

MnistLoader::~MnistLoader() {
    dataFile.close();
    labelFile.close();
}

uint MnistLoader::getTotalSize(){
    return totalImages;
}

void MnistLoader::resetData() {
    currentPosData = 0;
    // offset: 4 * 32
    dataFile.seekg(4 * sizeof(uint32_t), std::ios::beg);
}

void MnistLoader::resetLabel() {
    currentPosLabel = 0;
    // offset: 2 * 32
    labelFile.seekg(2 * sizeof(uint32_t), std::ios::beg);
}

uint32_t MnistLoader::swapEndian(uint32_t value) {
    const auto* bytes = reinterpret_cast<uint8_t*>(&value);
    return static_cast<int32_t>(bytes[0]) << 24 | static_cast<int32_t>(bytes[1]) << 16 | static_cast<int32_t>(bytes[2]) << 8 | static_cast<int32_t>(bytes[3]);
}
