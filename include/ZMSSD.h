#ifndef UVR_SLAM_ZMSSSD_H
#define UVR_SLAM_ZMSSSD_H
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace UVR_SLAM {
	class ZMSSD {
	public:
		static const int patch_size_ = 2 * 4;
		static const int patch_area_ = patch_size_*patch_size_;
		static const int threshold_ = 2000 * patch_area_;
		ZMSSD(cv::Mat src):ref(src)
		{
			int sumA_uint = 0, sumAA_uint = 0;

			for (int y = 0; y < src.rows; y++) {
				for (int x = 0; x < src.cols; x++) {
					int val = src.at<uchar>(y, x);
					sumA_uint += val;
					sumAA_uint += val*val;
				}
			}
			sumA_ = sumA_uint;
			sumAA_ = sumAA_uint;
		}
		int computeScore(cv::Mat patch) {
			int sumB_uint = 0, sumBB_uint = 0, sumAB_uint = 0;
			for (int y = 0; y < patch.rows; y++) {
				for (int x = 0; x < patch.cols; x++) {
					int val = patch.at<uchar>(y, x);
					sumB_uint  += val;
					sumBB_uint += val*val;
					sumAB_uint += val * ref.at<uchar>(y, x);
				}
			}
			int sumB = sumB_uint;
			int sumBB = sumBB_uint;
			int sumAB = sumAB_uint;
			return sumAA_ - 2 * sumAB + sumBB - (sumA_*sumA_ - 2 * sumA_*sumB + sumB*sumB) / patch_area_;
		}
		int sumA_, sumAA_;
		cv::Mat ref;

	private:
	};
}
#endif