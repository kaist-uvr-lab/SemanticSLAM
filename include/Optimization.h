

#ifndef UVR_SLAM_OPTIMIZATION_H
#define UVR_SLAM_OPTIMIZATION_H
#pragma once
#include <Frame.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Vertex.h>
#include <InitialData.h>

namespace UVR_SLAM {
	//class Edge;
	//class Vertex;
	//class Optimizer;
	//class PlaneEdgeOnlyPoseNMap;
	class FrameWindow;
	class Optimization {
	public:
		static void LocalBundleAdjustment(UVR_SLAM::FrameWindow* pWindow, int trial1 = 2, int trial2 = 5, bool bShowStatus = false);
		static int InitOptimization(InitialData* data, std::vector<cv::DMatch> Matches, UVR_SLAM::Frame* pInitFrame1, UVR_SLAM::Frame* pInitFrame2, cv::Mat K, bool& bInit, int trial1 = 2, int trial2 = 5);
		static int PoseOptimization(UVR_SLAM::FrameWindow* pWindow, UVR_SLAM::Frame* pF, bool bStatus = false, int trial1 = 2, int trial2 = 5);
		/*double CalibrationGridPlaneTest(UVR_SLAM::Pose* pPose, std::vector<cv::Point2f> pts, std::vector<cv::Point3f> wPts, cv::Mat K, int mnWidth, int mnHeight);
		double PlaneMapDepthEstimationTest(UVR_SLAM::Pose* pPose,
			std::vector<cv::Point3f> wPts1, std::vector<cv::Point2f> iPts1, std::vector<float>& depths1,
			std::vector<cv::Point3f> wPts2, std::vector<cv::Point2f> iPts2, std::vector<float>& depths2,
			cv::Mat K, cv::Mat P1, cv::Mat P2, int mnWidth, int mnHeight);*/
	};
}

#endif //ANDROIDOPENCVPLUGINPROJECT_OPTIMIZER_H