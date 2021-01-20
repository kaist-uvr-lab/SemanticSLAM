

#ifndef JSON_CONVERTER_H
#define JSON_CONVERTER_H
#pragma once

#include <rapidjson\document.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <Base64Encoder.h>
#include "happyhttp.h"
#include <cstdio>
#include <cstring>


//##include <prettywriter.h>

class JSONConverter {
public:
	//static void Init();
	static std::string ConvertImageToJSONStr(int nFrameID, cv::Mat img);
	static cv::Mat ConvertStringToImage(const char* data, int N);
	static cv::Mat ConvertStringToLabel(const char* data, int N);
	static bool RequestPOST(std::string ip, int port, cv::Mat img, cv::Mat& dst, int mnFrameID, int& stat);
public:
	
private:
};


#endif