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
		Seed(cv::Mat ray, float depth_mean, float depth_min);
		float a;
		float b;
		float mu;
		float z_range;
		float sigma2;
		int count;
		static float px_err_angle;
		cv::Mat ray;
		void UpdateDepth(float invz, float tau2);
		float ComputeDepth(cv::Point2f& est, cv::Point2f src, cv::Mat R, cv::Mat t, cv::Mat K);
		float ComputeTau(cv::Mat t, float z); //ȣ��� t�� �����ǥ�̸�, Xcam depth�� ���۷����� ��. �̰� �õ�� �Űܵ� ��.
		std::vector<float> mvfDepths;
		cv::Mat matDepths;
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