#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <string>
#include <opencv2/opencv.hpp>

class Model {
	public:
		bool initEngine(std::string enginePath, int input_w, int output_w);
		void doInference(const std::vector<cv::Mat>& inputImgVec, std::vector<cv::Mat>& outputImgVec);

	private:
		nvinfer1::IRuntime* runtime;
		nvinfer1::ICudaEngine* engine;
		nvinfer1::IExecutionContext* context;
		bool infer(float* input_host, float* output_host);
		int input_dim;
		int output_dim;
		int input_size;
		int output_size;
};		

#endif
