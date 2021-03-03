
#ifndef UVR_SLAM_USER_H
#define UVR_SLAM_USER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class MapPoint;
	class Frame;
	class MapGrid;
	class User {
	public:
		User();
		User(std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, bool _b);
		virtual ~User();
		std::string mapName;
		float fx, fy, cx, cy;
		int mnWidth, mnHeight;
		bool mbMapping;
		cv::Mat K, InvK;
		
		std::vector<MapPoint*> mvpLocalMPs;
		std::set<MapGrid*> mspLocalGrids;
		Frame* mpLastFrame;
		int mnLastMatch;

		void SetPose(cv::Mat _R, cv::Mat _t);
		void GetPose(cv::Mat&_R, cv::Mat& _t);
		cv::Mat GetPose();
		void GetInversePose(cv::Mat& _Rinv, cv::Mat& _Tinv);
		cv::Mat GetInversePose();
		cv::Mat GetCameraCenter();
	private:
		std::mutex mMutexUserPose;
		cv::Mat Pose, InversePose;
		cv::Mat R, t;
		cv::Mat Rinv, Tinv;
		cv::Mat Center;
	};
}
#endif