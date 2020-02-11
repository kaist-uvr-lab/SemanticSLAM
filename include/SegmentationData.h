#pragma once
#ifndef UVR_SLAM_SEGMENTATION_DATA_H
#define UVR_SLAM_SEGMENTATION_DATA_H

#include <opencv2/opencv.hpp>

namespace UVR_SLAM {
	const enum ObjectType {
		OBJECT_FLOOR = 4,
		OBJECT_WALL = 1,
		OBJECT_CEILING = 6,
		OBJECT_NONE = 0
	};
	const cv::Vec3b COLOR_NONE = cv::Vec3b(0, 0, 0);
	const cv::Vec3b COLOR_WALL = cv::Vec3b(128, 0, 0);
	const cv::Vec3b COLOR_FLOOR = cv::Vec3b(0, 0, 128);
	const cv::Vec3b COLOR_CEILING = cv::Vec3b(0, 128, 128);

	const cv::Vec3b COLOR_OUTDOOR_NONE = cv::Vec3b(255, 255, 255);
	const cv::Vec3b COLOR_OUTDOOR_ROAD = cv::Vec3b(0, 0, 0);
	const cv::Vec3b COLOR_OUTDOOR_BUILDING = cv::Vec3b(0, 128, 0);
	const cv::Vec3b COLOR_OUTDOOR_SKY = cv::Vec3b(64, 128, 0);
	
	class ObjectColors {
	public:
		static std::vector<cv::Vec3b> mvObjectLabelColors;
		static void Init() {
			cv::Mat colormap = cv::Mat::zeros(256, 3, CV_8UC1);
			cv::Mat ind = cv::Mat::zeros(256, 1, CV_8UC1);
			for (int i = 1; i < ind.rows; i++) {
				ind.at<uchar>(i) = i;
			}

			for (int i = 7; i >= 0; i--) {
				for (int j = 0; j < 3; j++) {
					cv::Mat tempCol = colormap.col(j);
					int a = pow(2, j);
					int b = pow(2, i);
					cv::Mat temp = ((ind / a) & 1) * b;
					tempCol |= temp;
					tempCol.copyTo(colormap.col(j));
				}
				ind /= 8;
			}
			std::cout << colormap << std::endl;
			for (int i = 0; i < colormap.rows; i++) {
				cv::Vec3b color = cv::Vec3b(colormap.at<uchar>(i, 0), colormap.at<uchar>(i, 1), colormap.at<uchar>(i, 2));
				mvObjectLabelColors.push_back(color);
			}

			//Indoor
			/*mvObjectLabelColors.push_back(COLOR_FLOOR);
			mvObjectLabelColors.push_back(COLOR_WALL);
			mvObjectLabelColors.push_back(COLOR_CEILING);
			mvObjectLabelColors.push_back(COLOR_NONE);*/
			//Outdoor
			/*mvObjectLabelColors.push_back(COLOR_OUTDOOR_ROAD);
			mvObjectLabelColors.push_back(COLOR_OUTDOOR_BUILDING);
			mvObjectLabelColors.push_back(COLOR_OUTDOOR_SKY);
			mvObjectLabelColors.push_back(COLOR_OUTDOOR_NONE);*/

		}
	};

	
}

#endif
