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
	static void ConvertBytesToDesc(const char* data, int n, cv::Mat& desc);

	static std::string ConvertImageToString(cv::Mat img, int id);
	static std::string ConvertNumberToString(int id);
	static std::string ConvertNumberToString(int id1, int id2);
	//static void ConvertStringToPoints(const char* data, std::vector<cv::Point2f>& vPTs, cv::Mat& desc); //삭제 예정
	static void ConvertStringToImage(const char* data, cv::Mat& img);
	static void ConvertStringToNumber(const char* data, int &id1, int& id2);
	static void ConvertStringToNumber(const char* data, int &n);
	static void ConvertStringToPoints(const char* data, std::vector<cv::Point2f>& vPTs);
	static void ConvertStringToDesc(const char* data, int n, cv::Mat& desc);
	static void ConvertStringToMatches(const char* data, int n, std::vector<int>& vMatches);
	static void ConvertStringToMatches(const char* data, int n, cv::Mat& mMatches);
	static void ConvertStringToDepthImage(const char* data, cv::Mat& res);

	////단말과 매핑서버의 통신 관련
	//매핑 서버 결과를 json string으로 변환
	static std::string ConvertMapDataToJson(cv::Mat mpIDs, cv::Mat x3Ds, cv::Mat kfids, cv::Mat poses, cv::Mat idxs);
	static void ConvertMapName(const char* data, std::string& map);
	static void ConvertInitConnectToServer(const char* data, float& _fx, float& _fy, float& _cx, float& _cy, int& _w, int & _h, bool& _b);
	static void ConvertDeviceToServer(const char* data, int& id, bool& init);
	static void ConvertDeviceFrameIDToServer(const char* data, std::string& map, int& id);
	static std::string ConvertInitializationToJsonString(int id, bool bInit, cv::Mat R, cv::Mat t, cv::Mat keypoints, cv::Mat mappoints);
};
#endif