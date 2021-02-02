#ifndef WEBDATACONVERTER_H
#define WEBDATACONVERTER_H
#pragma once

//#include <Base64Encoder.h>
//#include <rapidjson\document.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>


#include <WebAPIDataConverter.h>
//#include "happyhttp.h"
#include <string>

class WebAPIDataConverter {
public:
	static std::string ConvertImageToString(cv::Mat img, int id);
	static std::string ConvertNumberToString(int id);
	static std::string ConvertNumberToString(int id1, int id2);
	static void ConvertStringToPoints(const char* data, std::vector<cv::Point2f>& vPTs, cv::Mat& desc);
	static void ConvertStringToLabels(const char* data, std::vector<int>& vMatches);
	static void ConvertStringToDepthImage(const char* data, cv::Mat& res);
};
#endif