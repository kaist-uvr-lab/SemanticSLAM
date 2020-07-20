 #ifndef UVR_SLAM_MATCHER_H
#define UVR_SLAM_MATCHER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>


namespace UVR_SLAM {
	
	class MapPoint;
	class Frame;
	class FrameWindow;
	class PlaneInformation;
	class Map;
	class Matcher {
	public:
		Matcher();
		Matcher(cv::Ptr < cv::DescriptorMatcher> _matcher, int w, int h);
		virtual ~Matcher();

///////////////////////////////////////////////////////////
////200410 Optical flow 적용 버전
	public:
		int OpticalMatchingForInitialization(Frame* init, Frame* curr, std::vector<std::pair<cv::Point2f, cv::Point2f>>& resMatches);
		int OpticalMatchingForInitialization(Frame* prev, Frame* curr, std::vector<cv::Point2f>& vpPts2, std::vector<bool>& vbInliers, std::vector<int>& vnIDXs, cv::Mat& debug);
		int OpticalMatchingForTracking(Frame* prev, Frame* curr, std::vector<UVR_SLAM::MapPoint*>& vpMPs, std::vector<cv::Point2f>& vpPts, std::vector<cv::Point2f>& vpPts1, std::vector<cv::Point3f>& vpPts2, std::vector<bool>& vbInliers, std::vector<int>& vnIDXs, std::vector<int>& vnMPIDXs, cv::Mat& overlap, cv::Mat& debug);
		int OpticalMatchingForTracking(Frame* pKF, Frame* pF, std::vector<UVR_SLAM::MapPoint*>& vLocalMPs, std::vector<cv::Point2f>& vLocalKPs, std::vector<bool>& vLocalInliers, std::vector<int>& vLocalIndexes, std::vector<cv::Point2f>& vNewKPs, std::vector<int>& vNewIndexes, cv::Mat& overlap, cv::Mat& debug);
		int OpticalMatchingForTracking2(Frame* prev, Frame* curr, cv::Mat& debug);
		int OpticalKeyframeAndFrameMatchingForTracking(Frame* prev, Frame* curr, cv::Mat& debug);
		int OpticalMatchingForMapping(Frame* init, Frame* curr, std::vector<std::pair<cv::Point2f, cv::Point2f>>& resMatches, cv::Mat& debugging);
		int OpticalMatchingForFuseWithEpipolarGeometry(Frame* prev, Frame* curr, cv::Mat& debug);
		int Fuse(Frame* pKF1, Frame* pKF2, Frame* pKF3, cv::Mat& debug);
		/////////////F를 이용한 매칭
		void FindFundamental(Frame* pInit, Frame* pCurr, std::vector<std::pair<cv::Point2f, cv::Point2f>> vMatches, std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21);
		float CheckFundamental(Frame* pInit, Frame* pCurr, const cv::Mat &F21, std::vector<std::pair<cv::Point2f, cv::Point2f>> vMatches, std::vector<bool> &vbMatchesInliers, float sigma);
		void Normalize(const std::vector<cv::Point2f> &vPts, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
		cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
		/////////////F를 이용한 매칭
////200410
///////////////////////////////////////////////////////////
	public:

		void FindFundamental(Frame* pInit, Frame* pCurr, std::vector<cv::DMatch> vMatches, std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21);

	public:
		//Epipolar geometry to create new map points
		bool CheckEpiConstraints(cv::Mat F12, cv::Point2f pt1, cv::Point2f pt2, float sigma, float& res);
		cv::Mat CalcFundamentalMatrix(cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K);
		bool FeatureMatchingWithEpipolarConstraints(int& matchIDX, UVR_SLAM::Frame* pTargetKF, cv::Mat F12, cv::KeyPoint kp, cv::Mat desc, float sigma, int thresh);
	private:
		 
		int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
		void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

		//Fundamental Matrix관련초기화
		
		void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
		float CheckFundamental(Frame* pInit, Frame* pCurr, const cv::Mat &F21, std::vector<cv::DMatch> vMatches, std::vector<bool> &vbMatchesInliers, float sigma);

	private:
		cv::Mat K, D;
		int mWidth, mHeight;
		cv::Ptr<cv::DescriptorMatcher> matcher;
		bool mbCheckOrientation;
		float mfNNratio; //projection maching에서 이용
		int TH_HIGH;     //projection maching에서 이용
		int TH_LOW;
		int HISTO_LENGTH;
	};
}

#endif