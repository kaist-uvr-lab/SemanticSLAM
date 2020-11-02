#ifndef UVR_SLAM_DEPTH_FILTER_H
#define UVR_SLAM_FRAME_GRID_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace UVR_SLAM {
	class Frame;
	class CandidatePoint;
	class Seed {
	public:
		Seed(float depth_mean, float depth_min);
		float a;
		float b;
		float mu;
		float z_range;
		float sigma2;
	private:
	};
	class DepthFilter {
	public:
		DepthFilter();
		virtual ~DepthFilter();
	public:
		void Update(Frame* pF);
	private:
	};

}
#endif