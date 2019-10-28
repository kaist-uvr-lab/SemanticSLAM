#include <Pose.h>
#include <opencv2/core/eigen.hpp>

UVR_SLAM::Pose::Pose(){}
UVR_SLAM::Pose::Pose(cv::Mat _R, cv::Mat _t) {
	cv::Mat tempR, tempT;
	//_R.convertTo(tempR, CV_64FC1);
	//_t.convertTo(tempT, CV_64FC1);
	R = Eigen::Matrix3d();
	t = Eigen::Vector3d();
	Set(_R, _t);
	Accept();
}
UVR_SLAM::Pose::~Pose() {

}

void UVR_SLAM::Pose::Get(cv::Mat& _R, cv::Mat& _t) {
	//_R, _t 타입 체크 필요
	cv::eigen2cv(R, _R);
	cv::eigen2cv(t, _t);
}
void UVR_SLAM::Pose::Set(cv::Mat _R, cv::Mat _t) {
	//UVR::MatrixOperator::Mat2Eigen(_R, R);
	cv::cv2eigen(_R, R);
	cv::cv2eigen(_t, t);
}