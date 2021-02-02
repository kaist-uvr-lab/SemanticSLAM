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
	static bool SendImage(std::string ip, int port, cv::Mat src, int id);
	static bool RequestDetect(std::string ip, int port, int id, std::vector<cv::Point2f>& vPTs, cv::Mat& desc);
	static bool RequestMatch(std::string ip, int port, int id1, int id2, std::vector<int>& vMatches);
	static bool RequestDepthEstimate(std::string ip, int port, int id, cv::Mat& dst);

	static bool RequestDepthEstimate(std::string ip, int port, cv::Mat src, int id, cv::Mat& dst);
	static bool RequestDetect(std::string ip, int port, cv::Mat src, int id, std::vector<cv::Point2f>& vPTs);

	static void FeatueOnBeginAA(const happyhttp::Response* r, void* userdata)
	{
		//ssWebData.str("");
		//feaeture_count = 0;
	}
private:

};
#endif