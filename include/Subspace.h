#ifndef UVR_SLAM_SUBSPACE_H
#define UVR_SLAM_SUBSPACE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class Frame;
	class MapPoint;
	class Subspace {
	public:
		Subspace();
		virtual ~Subspace();
	public:
		bool CheckNeedNewSubspace(std::vector<MapPoint*> vpTempMPs);
		cv::Mat GetData();
	public:
		Frame *mpStartFrame, *mpEndFrame;
		std::vector<Frame*> mvpSubspaceFrames;
		std::vector<cv::Mat> mvParams;
		std::set<MapPoint*> mspSubspaceMapPoints;
		std::map<int, std::set<MapPoint*>> mapData;
		cv::Mat avgParam; //avg normal
		int mnID;
	private:

	public:
		static int nSubspaceID;
	};
}

#endif