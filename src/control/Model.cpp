//
// Created by SCPZ24 on 2025/11/16.
//

#include "../../includes/control/Model.h"
#include <fstream>
#include <filesystem>
#include <ranges>

void clipGradients(vector<vector<DataType>>& grads, const DataType maxNorm) {
    if (grads.empty()) return;
    DataType normSquared = 0.0;
    for (auto& sampleGrad : grads) {
        for (auto& g : sampleGrad) {
            if (!std::isfinite(g)) g = 0.0;
            normSquared += g * g;
        }
    }
    const DataType norm = std::sqrt(normSquared);
    if (norm > maxNorm && norm > 0.0) {
        const DataType scale = maxNorm / norm;
        for (auto& sampleGrad : grads) {
            for (auto& g : sampleGrad) {
                g *= scale;
            }
        }
    }
}

Model::Model(vector<unique_ptr<Layer>>&& _layers) : layers(std::move(_layers)) {}

vector<vector<DataType>> Model::forwardProp(const vector<vector<DataType>> &input) const {
    vector<vector<DataType>> output = input;
    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void Model::backprop(const vector<vector<DataType>> &outputGrad, const DataType learningRate) const {
    vector<vector<DataType>> inputGrad = outputGrad;
    for (const auto & layer : std::ranges::reverse_view(layers)) {
        inputGrad = layer->backward(inputGrad, learningRate);
    }
}

void Model::save(const string& path) const {
    std::filesystem::path p(path);
    std::filesystem::create_directories(p.parent_path());
    std::ofstream ofs(path, std::ios::binary);
    constexpr uint32_t ver = 1;
    const auto n = static_cast<uint32_t>(layers.size());
    constexpr char magic[9] = { 'C','P','P','M','N','I','S','T','\0' };
    ofs.write(magic, 9);
    ofs.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto& lyr : layers) {
        const string t = lyr->type();
        const auto len = static_cast<uint32_t>(t.size());
        ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
        ofs.write(t.data(), len);
        lyr->save(ofs);
    }
}

void Model::load(const string& path) const {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.good()) return;
    char magic[9]; ifs.read(magic, 9);
    uint32_t ver = 0; ifs.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    uint32_t n = 0; ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    const auto cur = static_cast<uint32_t>(layers.size());
    const uint32_t cnt = std::min(cur, n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t len = 0; ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
        string t(len, '\0'); ifs.read(t.data(), len);
        if (i < cnt) {
            if (layers[i]->type() == t) {
                layers[i]->load(ifs);
            } else {
                return;
            }
        }
    }
}
