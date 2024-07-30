#ifndef _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#endif
#include <experimental/filesystem>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <tiffio.h>
#include "NvInfer.h"

using namespace std;

bool fileExists(const string filename) {
	if (!experimental::filesystem::exists(
		experimental::filesystem::path(filename))) {
		return false;
	}
	else {
		return true;
	}
}
/**
 * @brief 获取engine的大小size，并将engine的信息载入到data中，
 *
 * @param path engine的路径
 * @param data 存储engine数据的vector
 * @param size engine的大小
 * @return true 文件读取成功
 * @return false 文件读取失败
 */
bool fileRead(const string &path, vector<unsigned char> &data, size_t &size) {
	stringstream trtModelStream;
	ifstream cache(path);
	if (!cache.is_open()) {
		cerr << "Unable to open file: " << path << endl;
		return false;
	}

	/* 将engine的内容写入trtModelStream中*/
	trtModelStream.seekg(0, trtModelStream.beg);
	trtModelStream << cache.rdbuf();
	cache.close();

	/* 计算model的大小*/
	trtModelStream.seekg(0, ios::end);
	size = trtModelStream.tellg();
	data.resize(size);

	trtModelStream.seekg(0, ios::beg);
	// read方法将从trtModelStream读取的数据写入以data[0]为起始地址的内存位置
	trtModelStream.read(reinterpret_cast<char*>(data.data()), size);
	return true;
}

vector<unsigned char> loadFile(const string &file) {
	ifstream in(file, ios::in | ios::binary);
	if (!in.is_open())
		return {};
	in.seekg(0, ios::end);
	size_t length = in.tellg();
	vector<unsigned char> data;
	if (length > 0){
		in.seekg(0, ios::beg);
		data.resize(length);
		in.read(reinterpret_cast<char*>(data.data()), length);
	}
	in.close();
	return data;
}

string printDims(const nvinfer1::Dims dims) {
	int n = 0;
	char buff[100];
	string result;

	n += snprintf(buff + n, sizeof(buff) - n, "[ ");
	for (int i = 0; i < dims.nbDims; i++) {
		n += snprintf(buff + n, sizeof(buff) - n, "%d", dims.d[i]);
		if (i != dims.nbDims - 1) {
			n += snprintf(buff + n, sizeof(buff) - n, ", ");
		}
	}
	n += snprintf(buff + n, sizeof(buff) - n, " ]");
	result = buff;
	return result;
}

string printTensor(float* tensor, int size) {
	int n = 0;
	char buff[100];
	string result;
	n += snprintf(buff + n, sizeof(buff) - n, "[ ");
	for (int i = 0; i < size; i++) {
		n += snprintf(buff + n, sizeof(buff) - n, "%8.4lf", tensor[i]);
		if (i != size - 1) {
			n += snprintf(buff + n, sizeof(buff) - n, ", ");
		}
	}
	n += snprintf(buff + n, sizeof(buff) - n, " ]");
	result = buff;
	return result;
}

// models/onnx/sample.onnx
string getEnginePath(string onnxPath) {
	int name_l = onnxPath.rfind("/");
	int name_r = onnxPath.rfind(".");

	int dir_r = onnxPath.find("/");

	string enginePath;
	enginePath = onnxPath.substr(0, dir_r);
	enginePath += "/engine";
	enginePath += onnxPath.substr(name_l, name_r - name_l);
	enginePath += ".engine";
	return enginePath;
}

vector<cv::Mat> readMultiframeTiff(const string& filename) {
	vector<cv::Mat> mats;

	TIFF* tif = TIFFOpen(filename.c_str(), "r");
	if (!tif) {
		throw runtime_error("Could not open file: " + filename);
	}

	uint32 width, height;
	if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) || !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height)) {
		TIFFClose(tif);
		throw runtime_error("Could not read image dimensions from file: " + filename);
	}

	for (int frame = 0; frame < TIFFNumberOfDirectories(tif); ++frame) {
		TIFFSetDirectory(tif, frame);

		cv::Mat img(height, width, CV_16UC1);
		unsigned char* buf = img.ptr();
		for (int strip = 0; strip < TIFFNumberOfStrips(tif); ++strip) {
			if (!TIFFReadRawStrip(tif, strip, buf + strip * TIFFStripSize(tif), TIFFStripSize(tif))) {
				TIFFClose(tif);
				throw runtime_error("Could not read strip from file: " + filename);
			}
		}

		mats.push_back(img);
	}

	TIFFClose(tif);

	return mats;
}

void saveMatVectorToTiff(std::vector<cv::Mat>& images, const std::string& filename) {
	TIFF* tif = TIFFOpen(filename.c_str(), "w");
	if (!tif) {
		throw std::runtime_error("Could not open output TIFF file");
	}

	for (size_t i = 0; i < images.size(); ++i) {
		cv::Mat& img = images[i];

		TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img.cols);
		TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img.rows);
		TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, img.channels());
		TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 16);
		TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
		TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, img.channels() == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB);

		// Write the image data to the TIFF file
		for (int row = 0; row < img.rows; ++row) {
			TIFFWriteScanline(tif, img.ptr(row), row, 0);
		}

		// Write the current directory (image) and create a new one for the next image
		if (i < images.size() - 1) {
			TIFFWriteDirectory(tif);
		}
	}

	TIFFClose(tif);
}
