#ifndef POINT_LESS_H
#define POINT_LESS_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace UVR_SLAM {
	struct Point2fLess
	{
		bool operator()(cv::Point2f const&lhs, cv::Point2f const& rhs) const
		{
			return lhs.x == rhs.x ? lhs.y < rhs.y : lhs.x < rhs.x;
		}
	};
	struct Point3fLess
	{
		bool operator()(cv::Point3f const&lhs, cv::Point3f const& rhs) const
		{
			return lhs.x == rhs.x ? lhs.y == rhs.y ? lhs.z < rhs.z : lhs.y < rhs.y : lhs.x < rhs.x;
		}
	};
}
#endif
