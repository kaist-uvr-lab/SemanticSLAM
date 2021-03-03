#include <User.h>
#include <MapPoint.h>
#include <MapGrid.h>
#include <Frame.h>

namespace UVR_SLAM {
	User::User() {

	}
	User::User(std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, bool _b):mapName(_map),mnWidth(_w), mnHeight(_h), fx(_fx), fy(_fy), cx(_cx), cy(_cy), mbMapping(_b), mnLastMatch(0){
		K = cv::Mat::eye(3, 3, CV_32FC1);
		K.at<float>(0, 0) = fx;
		K.at<float>(1, 1) = fy;
		K.at<float>(0, 2) = cx;
		K.at<float>(1, 2) = cy;
		InvK = K.inv();
		mpLastFrame = nullptr;
		SetPose(cv::Mat::eye(3, 3, CV_32FC1), cv::Mat::zeros(3, 1, CV_32FC1));
	}
	User::~User() {

	}

	void User::SetPose(cv::Mat _R, cv::Mat _t) {
		std::unique_lock<std::mutex>(mMutexUserPose);
		R = _R.clone();
		t = _t.clone();
		cv::hconcat(R, t, Pose);
	}
	void User::GetPose(cv::Mat&_R, cv::Mat& _t){
		std::unique_lock<std::mutex>(mMutexUserPose);
		_R = R.clone();
		_t = t.clone();
	}
	cv::Mat User::GetPose(){
		std::unique_lock<std::mutex>(mMutexUserPose);
		return Pose.clone();
	}
	void User::GetInversePose(cv::Mat& _Rinv, cv::Mat& _Tinv){
		std::unique_lock<std::mutex>(mMutexUserPose);
		_Rinv = R.t();
		_Tinv = -_Rinv*t;
	}
	cv::Mat User::GetInversePose(){
		std::unique_lock<std::mutex>(mMutexUserPose);
		Rinv = R.t();
		Tinv = -Rinv*t;
		cv::hconcat(Rinv, Tinv, InversePose);
		return InversePose.clone();
	}
	cv::Mat User::GetCameraCenter(){
		std::unique_lock<std::mutex>(mMutexUserPose);
		return -R.t()*t;
	}
}