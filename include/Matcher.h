 #ifndef UVR_SLAM_MATCHER_H
#define UVR_SLAM_MATCHER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>


namespace UVR_SLAM {
	class CandidatePoint;
	class MapPoint;
	class Frame;
	class FrameGrid;
	class Visualizer;
	class PlaneInformation;
	class Map;
	class System;
	class Matcher {
	
	public:
		Matcher();
		Matcher(System* pSys, cv::Ptr < cv::DescriptorMatcher> _matcher, int w, int h);
		virtual ~Matcher();
		void Init();

///////////////////////////////////////////////////////////
////201114 Epipolar constraints
	public:
		cv::Mat ComputeLineEquation(cv::Point2f pt1, cv::Point2f pt2);
		bool CheckLineDistance(cv::Mat line, cv::Point2f pt, float sigma);
		void ComputeEpiLinePoint(cv::Point2f& sPt, cv::Point2f& ePt,cv::Mat ray, float minDepth, float maxDepth, cv::Mat Rrel, cv::Mat Trel, cv::Mat K);
		cv::Point2f CalcLinePoint(float val, cv::Mat mLine, bool opt);
////201114 Epipolar constraints
///////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////
////210128 ��Ƽ�� �÷ο� ��� ��Ī
		////�ʱ�ȭ
		int OpticalFlowMatching(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> vecPoints, std::vector<cv::Point2f>& vecMatchPoints1, std::vector<cv::Point2f>& vecMatchPoints2, std::vector<int>& vecIndexes);
		////Ʈ��ŷ
		int OpticalFlowMatching(int nFrameID, cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> vecPoints, std::vector<MapPoint*> vecMPs, std::vector<cv::Point2f>& vecMatchPoints1, std::vector<cv::Point2f>& vecMatchPoints2, std::vector<MapPoint*>& vecMatchMPs, std::vector<bool>& vecInliers, cv::Mat& overlap);
//////////////////////////////////////////////////////////

		int BagOfWordsMatching(Frame* pF1, Frame* pF2, std::vector<MapPoint*>& vpMatches12);
		float SuperPointDescriptorDistance(const cv::Mat &a, const cv::Mat &b);

///////////////////////////////////////////////////////////
////200410 Optical flow ���� ����
	public:
		int OpticalMatchingForTracking(Frame* prev, Frame* curr, std::vector<UVR_SLAM::MapPoint*>& vpMPs, std::vector<cv::Point2f>& vCurrPtsMP, std::vector<bool>& vbInliers, 
			std::vector<UVR_SLAM::CandidatePoint*>& vpCPs, std::vector<cv::Point2f>& vPrevPts, std::vector<cv::Point2f>& vCurrPtsCP, std::vector<int>& vnIDXs, cv::Mat& overlap);

		
		
		int OpticalMatchingForTracking(Frame* prev, Frame* curr, std::vector<UVR_SLAM::CandidatePoint*>& vpCPs, std::vector<cv::Point2f>& vPrevPts, std::vector<cv::Point2f>& vCurrPts, 
			std::vector<bool>& vbInliers, std::vector<int>& vnIDXs);


		
		bool OpticalGridMatching(FrameGrid* grid1, cv::Mat src1, cv::Mat src2, std::vector<cv::Point2f>& vPrevPTs, std::vector<cv::Point2f>& vCurrPTs);
		int OpticalGridsMatching(Frame* pFrame1, Frame* pFrame2, std::vector<cv::Point2f>& vpPTs);
		int OpticalMatchingForInitialization(Frame* init, Frame* curr, std::vector<std::pair<cv::Point2f, cv::Point2f>>& resMatches);
		int OpticalMatchingForInitialization(Frame* prev, Frame* curr, std::vector<cv::Point2f>& vpPts1, std::vector<cv::Point2f>& vpPts2, std::vector<bool>& vbInliers, std::vector<int>& vnIDXs, cv::Mat& debug);
		int OpticalMatchingForTracking(Frame* prev, Frame* curr, std::vector<UVR_SLAM::CandidatePoint*>& vpCPs, std::vector<UVR_SLAM::MapPoint*>& vpMPs, std::vector<cv::Point2f>& vpPts, std::vector<bool>& vbInliers, std::vector<int>& vnIDXs, cv::Mat& overlap);
		int OpticalMatchingForMapping(Map* pMap, Frame* pCurrKF, Frame* pPrevKF, std::vector<cv::Point2f>& vMatchedPrevPts, std::vector<cv::Point2f>& vMatchedCurrPts, std::vector<CandidatePoint*>& vMatchedCPs, cv::Mat K, cv::Mat InvK, double& dtime, cv::Mat& debugging);
		int OpticalMatchingForMapping(Map* pMap, Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, std::vector<cv::Point2f>& vMatchedPPrevPts, std::vector<cv::Point2f>& vMatchedPrevPts, std::vector<cv::Point2f>& vMatchedCurrPts, std::vector<CandidatePoint*>& vMatchedCPs, cv::Mat K, cv::Mat InvK, double& dtime, cv::Mat& debugging);
		int OpticalMatchingForMapping2(Map* pMap, Frame* pCurrKF, Frame* pPrevKF, std::vector<cv::Point2f>& vMatchedPrevPts, std::vector<cv::Point2f>& vMatchedCurrPts, std::vector<CandidatePoint*>& vMatchedCPs, cv::Mat K, cv::Mat InvK, double& dtime, cv::Mat& debugging);
		/////////////////

		/////////////F�� �̿��� ��Ī
		void FindFundamental(Frame* pInit, Frame* pCurr, std::vector<std::pair<cv::Point2f, cv::Point2f>> vMatches, std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21);
		float CheckFundamental(Frame* pInit, Frame* pCurr, const cv::Mat &F21, std::vector<std::pair<cv::Point2f, cv::Point2f>> vMatches, std::vector<bool> &vbMatchesInliers, float sigma);
		void Normalize(const std::vector<cv::Point2f> &vPts, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
		cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
		/////////////F�� �̿��� ��Ī
////200410
///////////////////////////////////////////////////////////
	public:
		void FindFundamental(Frame* pInit, Frame* pCurr, std::vector<cv::DMatch> vMatches, std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21);
	public:
		//Epipolar geometry to create new map points
		bool CheckEpiConstraints(cv::Mat F12, cv::Point2f pt1, cv::Point2f pt2, float sigma, cv::Mat & epiLine, float& res, bool& bLine);
		cv::Mat CalcFundamentalMatrix(cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K);
		bool FeatureMatchingWithEpipolarConstraints(int& matchIDX, UVR_SLAM::Frame* pTargetKF, cv::Mat F12, cv::KeyPoint kp, cv::Mat desc, float sigma, int thresh);
	private:
		int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
		void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
		//Fundamental Matrix�����ʱ�ȭ
		void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
		float CheckFundamental(Frame* pInit, Frame* pCurr, const cv::Mat &F21, std::vector<cv::DMatch> vMatches, std::vector<bool> &vbMatchesInliers, float sigma);
	private:
		System* mpSystem;
		cv::Mat K, D;
		int mWidth, mHeight;
		Visualizer* mpVisualizer;
		cv::Ptr<cv::DescriptorMatcher> matcher;
		bool mbCheckOrientation;
		float mfNNratio; //projection maching���� �̿�
		const int TH_HIGH;     //projection maching���� �̿�
		const int TH_LOW;
		const int HISTO_LENGTH;
	};
}
#endif