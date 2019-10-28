//
// Created by UVR-KAIST on 2019-01-14.
//

#ifndef UVR_SLAM_MATRIXOPERATOR_H
#define UVR_SLAM_MATRIXOPERATOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace cv;

namespace UVR_SLAM {
	class MatrixOperator {
	public:
		MatrixOperator() {

		}
		virtual ~MatrixOperator() {
		}
	public:
		cv::Mat static EXP(double wx, double wy, double wz);
		cv::Mat static EXPD(cv::Mat mat);
		cv::Mat static EXP6(double vx, double vy, double vz, double wx, double wy, double wz);
		cv::Mat static EXP6X44(double vx, double vy, double vz, double wx, double wy, double wz);
		cv::Mat static LOG(cv::Mat rmat);
		cv::Mat static LOGD(cv::Mat mat);
		cv::Mat static GetSkewSymetricMatrix(cv::Mat t);
		cv::Mat static GetSkewSymetricMatrix(double x, double y, double z);

		cv::Mat static Jacobian(cv::Mat mat);
		cv::Mat static InvJacobian(cv::Mat mat);

		double static CalcDistance(cv::Mat mat) {
			double res = sqrt(mat.dot(mat));
			return res;
		}

		int static DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

	public:
		Eigen::Matrix3d static Skew_Eigen(Eigen::Vector3d v);
		Eigen::Matrix3d static EXP_Eigen(Eigen::Vector3d d);
		Eigen::Matrix4d static EXP_Eigen(Eigen::VectorXd d);
		void Mat2Eigen(cv::Mat src, Eigen::VectorXd& dst);
		void Mat2Eigen(cv::Mat src, Eigen::MatrixXd& dst);

	public:
		void static BuildSparseProblem(std::vector<Eigen::Triplet<double>>& coeffs, cv::Mat m, int xidx, int yidx) {
			for (int x = 0; x < m.cols; x++) {
				for (int y = 0; y < m.rows; y++) {
					double val = m.at<double>(y, x);
					if (val != 0.0)
						coeffs.push_back(Eigen::Triplet<double>(y + yidx, x + xidx, val));
				}
			}
		}
		void static BuildSparseProblem(Eigen::VectorXd& _b, cv::Mat m, int idx) {
			for (int y = 0; y < m.rows; y++) {
				double val = m.at<double>(y);
				_b[y + idx] = val;
			}
		}
		void static BuildSparseProblem(std::vector<Eigen::Triplet<float>>& coeffs, cv::Mat m, int xidx, int yidx) {
			for (int x = 0; x < m.cols; x++) {
				for (int y = 0; y < m.rows; y++) {
					float val = m.at<float>(y, x);
					if (val != 0.0)
						coeffs.push_back(Eigen::Triplet<float>(y + yidx, x + xidx, val));
				}
			}
		}
		void static BuildSparseProblem(Eigen::VectorXf& _b, cv::Mat m, int idx) {
			for (int y = 0; y < m.rows; y++) {
				float val = m.at<float>(y);
				_b[y + idx] = val;
			}
		}
	public:
		void static CholeskyDecomposition(const Mat A, Mat& U);
		cv::Mat static Lsolve(cv::Mat L, cv::Mat b);
		cv::Mat static Usolve(cv::Mat U, cv::Mat b);
		cv::Mat static Diag(cv::Mat src);
	public:
		const static float deg2rad;
		const static float rad2deg;
	};

}

#endif //ANDROIDOPENCVPLUGINPROJECT_MATRIXOPERATOR_H
