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
	class Matcher {
	public:
		Matcher();
		Matcher(cv::Ptr < cv::DescriptorMatcher> _matcher, int w, int h);
		virtual ~Matcher();

	public:

		int MatchingWithPrevFrame(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr,  std::vector<std::pair<int,bool>>& mvMatches, int nLocalMapID);
		int MatchingWithLocalMap(Frame* pF, std::vector<MapPoint*> mvpLocalMPs, cv::Mat mLocalMapDesc, std::vector<bool>& mvbLocalMapInliers, std::vector<std::pair<int, bool>>& mvMatches, float rr);

		void FindFundamental(Frame* pInit, Frame* pCurr, std::vector<cv::DMatch> vMatches, std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21);

		//�ʱ�ȭ ��Ī 20.01.03
		int SearchForInitialization(Frame* F1, Frame* F2, std::vector<cv::DMatch>& resMatches, int windowSize);

		//fuse �������� ����.
		int MatchingForFuse(const std::vector<MapPoint*> &vpMapPoints, Frame *pKF, float th = 3.0f);

		//Pose Tracking ���� �̿��ϴ� ��
		int FeatureMatchingForInitialPoseTracking(Frame* pPrev, Frame* pCurr, std::vector<MapPoint*> mvpLocalMPs, cv::Mat mLocalMapDesc, std::vector<bool>& mvbLocalMapInliers, std::vector<cv::DMatch>& mvMatches, int nLocalMapID);
		int FeatureMatchingForInitialPoseTracking(FrameWindow* pWindow, Frame* pF, std::vector<MapPoint*> mvpLocalMPs, cv::Mat mLocalMapDesc, std::vector<bool>& mvbLocalMapInliers);
		int FeatureMatchingForPoseTrackingByProjection(Frame* pF, std::vector<MapPoint*> mvpLocalMPs, cv::Mat mLocalMapDesc, std::vector<bool>& mvbLocalMapInliers, std::vector<cv::DMatch>& mvMatches, float rr);

		int KeyFrameFuseFeatureMatching(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr);
		int KeyFrameFeatureMatching(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<cv::DMatch>& vMatches);

		//�ʱ�ȭ�� ���� �̿��ϴ� ��
		int MatchingProcessForInitialization(Frame* init, Frame* curr, cv::Mat& F, std::vector<cv::DMatch>& resMatches);

		//��׵��� Ȯ���� �ʿ���.
		int FeatureMatchingWithSemanticFrames(Frame* pSemantic, Frame* pFrame);
		
		
	public:
		//Epipolar geometry to create new map points
		bool CheckEpiConstraints(cv::Mat F12, cv::Point2f pt1, cv::Point2f pt2, float sigma, float& res);
		cv::Mat CalcFundamentalMatrix(cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K);
		bool FeatureMatchingWithEpipolarConstraints(int& matchIDX, UVR_SLAM::Frame* pTargetKF, cv::Mat F12, cv::KeyPoint kp, cv::Mat desc, float sigma, int thresh);
	private:

		int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
		void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

		//Fundamental Matrix�����ʱ�ȭ
		
		void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
		cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
		float CheckFundamental(Frame* pInit, Frame* pCurr, const cv::Mat &F21, std::vector<cv::DMatch> vMatches, std::vector<bool> &vbMatchesInliers, float sigma);

	private:
		cv::Mat K, D;
		int mWidth, mHeight;
		cv::Ptr<cv::DescriptorMatcher> matcher;
		bool mbCheckOrientation;
		float mfNNratio; //projection maching���� �̿�
		int TH_HIGH;     //projection maching���� �̿�
		int TH_LOW;
		int HISTO_LENGTH;
	};
}

#endif