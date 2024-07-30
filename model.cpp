#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "model.hpp"
#include "utils.hpp"


using namespace std;
using namespace nvinfer1;

class Logger : public ILogger {
public:
    virtual void log(Severity severity, const char* msg) noexcept override {
        string str;
        switch (severity) {
        case Severity::kINTERNAL_ERROR: str = RED    "[fatal]: " CLEAR;
        case Severity::kERROR:          str = RED    "[error]: " CLEAR;
        case Severity::kWARNING:        str = BLUE   "[warn]: "  CLEAR;
        case Severity::kINFO:           str = YELLOW "[info]: "  CLEAR;
        case Severity::kVERBOSE:        str = PURPLE "[verb]: "  CLEAR;
        }
        if (severity <= Severity::kINFO)
            cout << str << string(msg) << endl;
    }
};

bool Model::initEngine(string enginePath, int input_w, int output_w) {
    if (!fileExists(enginePath)) {
        LOGE("ERROR: %s not found", enginePath.c_str());
        return false;
    }
    vector<unsigned char> modelData;
    modelData = loadFile(enginePath);

    Logger logger;
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size(), nullptr);
    context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims3(81, input_w, input_w));

    input_dim = input_w;
    output_dim = output_w;
    input_size = 81 * input_dim * input_dim * sizeof(float);
    output_size = 61 * output_dim * output_dim * sizeof(float);

    return true;
}

bool Model::infer(float* input_host, float* output_host) {
    /* 1. host->device */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* buffers[2];
    cudaMalloc(&buffers[0], input_size);
    cudaMalloc(&buffers[1], output_size);
    cudaMemcpyAsync(buffers[0], input_host, input_size, cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

    /* 2. infer */
    context->enqueueV2(buffers, stream, nullptr);

    /* 3. device->host */
    cudaMemcpyAsync(output_host, buffers[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    /* 4. free cuda */
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    LOG("finished inference");
    return true;
}

void Model::doInference(const std::vector<cv::Mat>& inputImgVec, std::vector<cv::Mat>& outputImgVec) {

    float* input_host = new float[input_size];
    float* output_host = new float[output_size];

    for (int i = 0; i < inputImgVec.size(); ++i) {
        cv::Mat img32;
        inputImgVec[i].convertTo(img32, CV_32FC1);
        memcpy(input_host + i * input_dim * input_dim, img32.data, sizeof(float) * input_dim * input_dim);
    }
    infer(input_host, output_host);
    for (int i = 0; i < outputImgVec.size(); ++i) {
        outputImgVec[i].create(output_dim, output_dim, CV_32FC1);
        memcpy(outputImgVec[i].data, output_host + i * output_dim*output_dim, sizeof(float) * output_dim*output_dim);
        outputImgVec[i].convertTo(outputImgVec[i], CV_16UC1);
    }

    delete[] input_host;
    delete[] output_host;



}
