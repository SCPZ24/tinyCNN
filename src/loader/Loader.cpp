//
// Created by SCPZ24 on 2025/11/16.
//

#include <utility>

#include "../../includes/loader/Loader.h"

Loader::Loader(std::string data_path, std::string label_path, const uint batch_size)
    :dataPath(std::move(data_path)), labelPath(std::move(label_path)), batchSize(batch_size){}