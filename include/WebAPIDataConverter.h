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
	//static void ConvertStringToPoints(const char* data, std::vector<cv::Point2f>& vPTs, cv::Mat& desc); //삭제 예정
	static void ConvertStringToNumber(const char* data, int &n);
	static void ConvertStringToPoints(const char* data, int n, std::vector<cv::Point2f>& vPTs);
	static void ConvertStringToDesc(const char* data, int n, cv::Mat& desc);
	static void ConvertStringToMatches(const char* data, int n, std::vector<int>& vMatches);
	static void ConvertStringToDepthImage(const char* data, cv::Mat& res);
};
#endif