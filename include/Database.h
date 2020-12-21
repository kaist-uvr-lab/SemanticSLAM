#ifndef UVR_SLAM_DATABASE_H
#define UVR_SLAM_DATABASE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <mutex>

namespace UVR_SLAM {

	class Database {
	public:
		Database();
		virtual ~Database();
		void AddData(unsigned long long id, int data);
		int  GetData(unsigned long long it);
	private:
		int mnObjColorSize;
		std::mutex mMutexLBP;
		std::map<unsigned long long, cv::Mat> mmObjLBP;
	};
}
#endif
