#ifndef FEATUREMATCHINGWEBAPI_H
#define FEATUREMATCHINGWEBAPI_H
#pragma once

#include <rapidjson\document.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <Base64Encoder.h>
#include "happyhttp.h"
#include <string>
//#include <cstdio>
//#include <cstring>

class FeatureMatchingWebAPI{
public:
	//static std::string ConvertImageToString(cv::Mat img, int type);
	static bool Reset(std::string ip, int port);
	static bool RequestDetect(std::string ip, int port, cv::Mat src, int id, std::vector<cv::Point2f>& vPTs);
	static bool RequestMatch(std::string ip, int port, int id1, int id2, std::vector<int>& vMatches);
private:

};
#endif