#ifndef UVR_SLAM_MATCHER_H
#define UVR_SLAM_MATCHER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>


namespace UVR_SLAM {
	
	class Frame;
	class FrameWindow;
	class Matcher {
	public:
		Matcher();
		Matcher(cv::Ptr < cv::DescriptorMatcher> _matcher, int w, int h);
		virtual ~Matcher();

	public:
		//Matching 관련
		int FeatureMatchingWithSemanticFrames(Frame* pSemantic, Frame* pFrame);
		int MatchingProcessForInitialization(Frame* init, Frame* curr, cv::Mat& F, std::vector<cv::DMatch>& resMatches);
		int FeatureMatchingForInitialPoseTracking(FrameWindow* pWindow, Frame* pF);
		int FeatureMatchingForInitialPoseTracking(Frame* pPrev, Frame* pCurr, UVR_SLAM::FrameWindow* pWindow);
		int FeatureMatchingForPoseTrackingByProjection(FrameWindow* pWindow, Frame* pF, float rr);
	public:
		//Epipolar geometry to create new map points
		bool CheckEpiConstraints(cv::Mat F12, cv::Point2f pt1, cv::Point2f pt2, float sigma, float& res);
		cv::Mat CalcFundamentalMatrix(cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K);
		bool FeatureMatchingWithEpipolarConstraints(int& matchIDX, UVR_SLAM::Frame* pTargetKF, cv::Mat F12, cv::KeyPoint kp, cv::Mat desc, float sigma, int thresh);
	private:
		//Fundamental Matrix관련초기화
		void FindFundamental(Frame* pInit, Frame* pCurr, std::vector<cv::DMatch> vMatches, std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21);
		void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
		cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
		float CheckFundamental(Frame* pInit, Frame* pCurr, const cv::Mat &F21, std::vector<cv::DMatch> vMatches, std::vector<bool> &vbMatchesInliers, float sigma);

	private:
		cv::Mat K, D;
		int mWidth, mHeight;
		cv::Ptr<cv::DescriptorMatcher> matcher;
		float mfNNratio; //projection maching에서 이용
		int TH_HIGH;     //projection maching에서 이용
	};
}

#endif