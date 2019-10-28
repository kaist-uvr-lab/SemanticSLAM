

#ifndef JSON_CONVERTER_H
#define JSON_CONVERTER_H
#pragma once

#include <rapidjson\document.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "happyhttp.h"
#include <cstdio>
#include <cstring>


//##include <prettywriter.h>

class JSONConverter {
public:
	static void Init();
	static std::string ConvertImageToJSONStr(int nFrameID,cv::Mat img) {
		/*
		auto reqJsonData = R"(
          {
            "UserSeq": 1,
            "UserID": "jacking75",
            "UserPW": "123qwe"
          }
        )";
		const char json[] = " { \"hello\" : \"world\", \"t\" : true , \"f\" : false, \"n\": null, \"i\":123, \"pi\": 3.1416, \"a\":[1, 2, 3, 4] } ";
		*/
		std::string json;
		
		int r = img.rows;
		int c = img.cols;

		std::stringstream ss;
		ss << "{\"image\":"<<"[";
		for (int y = 0; y < r; y++) {
			ss<<"[";
			for (int x = 0; x < c; x++) {
				cv::Vec3b colorVec = img.at<cv::Vec3b>(y, x);
				ss << "[" << (int)colorVec.val[0] << ", " << (int)colorVec.val[1] << ", " << (int)colorVec.val[2] << "]";
				if (x < c - 1)
					ss<<",";
			}
			ss << "]";
			if( y < r-1)
				ss<<",";
		}
		ss << "]}";
		return ss.str();
	}
	static cv::Mat ConvertStringToImage(const char* data, int N);
	static bool RequestPOST(cv::Mat img, cv::Mat& dst, int mnFrameID);
public:
	const static char* headers[];
};


#endif