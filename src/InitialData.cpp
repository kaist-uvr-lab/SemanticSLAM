#include <InitialData.h>

UVR_SLAM::InitialData::InitialData() {
	vbTriangulated = std::vector<bool>(0);
	parallax = 0.0f;
	nGood = 0;
	R0 = cv::Mat::eye(3, 3, CV_32FC1);
	t0 = cv::Mat::zeros(3, 1, CV_32FC1);
}
UVR_SLAM::InitialData::InitialData(int minGood) :nMinGood(minGood) {
	vbTriangulated = std::vector<bool>(0);
	parallax = 0.0f;
	nGood = 0;
	R0 = cv::Mat::eye(3, 3, CV_32FC1);
	t0 = cv::Mat::zeros(3, 1, CV_32FC1);
}
UVR_SLAM::InitialData::~InitialData() {
}

void UVR_SLAM::InitialData::SetRt(cv::Mat _R, cv::Mat _t) {
	R = _R.clone();
	t = _t.clone();
}