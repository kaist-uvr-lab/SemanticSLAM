#pragma once
#ifndef UVR_SLAM_SEGMENTATION_DATA_H
#define UVR_SLAM_SEGMENTATION_DATA_H

#include <opencv2/opencv.hpp>

namespace UVR_SLAM {
	const enum ObjectType {
		OBJECT_FLOOR,
		OBJECT_WALL,
		OBJECT_CEILING,
		OBJECT_NONE
	};
	const cv::Vec3b COLOR_NONE = cv::Vec3b(0, 0, 0);
	const cv::Vec3b COLOR_WALL = cv::Vec3b(128, 0, 0);
	const cv::Vec3b COLOR_FLOOR = cv::Vec3b(0, 0, 128);
	const cv::Vec3b COLOR_CEILING = cv::Vec3b(0, 128, 128);

	class ObjectColors {
	public:
		static std::vector<cv::Vec3b> mvObjectLabelColors;
		static void Init() {
			mvObjectLabelColors.push_back(COLOR_FLOOR);
			mvObjectLabelColors.push_back(COLOR_WALL);
			mvObjectLabelColors.push_back(COLOR_CEILING);
			mvObjectLabelColors.push_back(COLOR_NONE);
		}
	};

	
}

#endif
