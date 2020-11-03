#ifndef UVR_SLAM_DEPTH_FILTER_H
#define UVR_SLAM_FRAME_GRID_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace UVR_SLAM {
	class System;
	class Frame;
	class CandidatePoint;
	class Seed {
	public:
		Seed(cv::Mat ray, float _px_err_angle,float depth_mean, float depth_min);
		float a;
		float b;
		float mu;
		float z_range;
		float sigma2;
		int count;
		float px_err_angle;
		cv::Mat ray;

		float ComputeTau(cv::Mat t, float z); //호출시 t는 상대좌표이며, Xcam depth는 레퍼런스의 것. 이거 시드로 옮겨도 됨.

	private:
	};
	class DepthFilter {
	public:
		DepthFilter();
		DepthFilter(System* pSys);
		virtual ~DepthFilter();
	public:
		void Init();
		void Update(Frame* pF, Frame* pPrev);
		void UpdateSeed(Seed* pSeed, float invz, float tau2);
	private:
		cv::Mat K, invK;
		System* mpSystem;
	};

}
#endif