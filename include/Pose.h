
#ifndef UVR_SLAM_POSE_H
#define UVR_SLAM_POSE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace UVR_SLAM {
	class Pose {
	public:
		Eigen::Matrix3d R, Rold;
		Eigen::Vector3d t, Told;
		//초기 포즈 만들 때는 double형으로 형변환
		Pose();
		Pose(cv::Mat _R, cv::Mat _t);
		virtual ~Pose();

		void LeftUpdate(Eigen::Matrix3d DeltaR, Eigen::Vector3d DeltaT) {
			//P = DeltaP*P;
			R = DeltaR*R;
			t = DeltaR*t + DeltaT;
		}
		void RightUpdate(Eigen::Matrix3d DeltaR, Eigen::Vector3d DeltaT) {
			//P = P*DeltaP;
			t = R*DeltaT + t;
			R = R*DeltaR;
		}
		void Accept() {
			Rold = R;
			Told = t;
		}
		void Reject() {
			R = Rold;
			t = Told;
		}
		void Get(cv::Mat& _R, cv::Mat& _t);
		void Set(cv::Mat _R, cv::Mat _t);
		void GetInv(Eigen::Matrix3d& _Rinv, Eigen::Vector3d& _Tinv) {
			_Rinv = R.transpose();
			_Tinv = -_Rinv*t;
		}
	private:
	};
}

#endif