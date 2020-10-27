#ifndef UVR_SLAM_PLANAR_OPTIMIZATION_H
#define UVR_SLAM_PLANAR_OPTIMIZATION_H
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace UVR_SLAM {
	class Frame;
	class MapPoint;
	class MapOptimizer;
	class Map;

	class PlaneProcessInformation;
	class PlanarOptimization {
	public:
		static void OpticalLocalBundleAdjustmentWithPlane(UVR_SLAM::MapOptimizer* pMapOptimizer, PlaneProcessInformation* pPlaneInfo, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::Frame*> vpKFs, std::vector<UVR_SLAM::Frame*> vpFixedKFs);
	};
}

#endif //ANDROIDOPENCVPLUGINPROJECT_OPTIMIZER_H