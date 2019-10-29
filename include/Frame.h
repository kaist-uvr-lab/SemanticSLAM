

//
// Created by UVR-KAIST on 2019-01-08.
//

#ifndef UVR_SLAM_FRAME_H
#define UVR_SLAM_FRAME_H
#pragma once
#include <fbow.h> //include windows header.
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <SegmentationData.h>
#include <MapPoint.h>
//#include "KeyFrame.h"

namespace UVR_SLAM {
#define FRAME_GRID_ROWS 24
#define FRAME_GRID_COLS 32

	
	const unsigned char FLAG_KEY_FRAME = 0x1;
	const unsigned char FLAG_SEGMENTED_FRAME = 0x2;
	const unsigned char FLAG_INIT_FRAME = 0x4;

	//class MapPoint;
	class ORBextractor;
	class Frame {
	public:
		Frame(cv::Mat _src, int w, int h);
		Frame(void* ptr, int id, int w, int h);
		Frame(void* ptr, int id, int w, int h, cv::Mat _R, cv::Mat _t);
		virtual ~Frame();
		void close();
	public:
		void process(cv::Ptr<cv::Feature2D> detector);
		fbow::fBow GetBowVec();
		void SetBowVec(fbow::Vocabulary* pfvoc);
		double Score(UVR_SLAM::Frame* pF);
	public:
		cv::Mat GetFrame();
		cv::Mat GetOriginalImage();
		void SetFrameType(int n);
		int GetFrameType();
		void SetPose(cv::Mat _R, cv::Mat _t);
		void GetPose(cv::Mat&_R, cv::Mat& _t);
		cv::Mat GetRotation();
		cv::Mat GetTranslation();
		void AddMP(UVR_SLAM::MapPoint* pMP, int idx);
		void RemoveMP(int idx);
		void UpdateMPs();
		UVR_SLAM::MapPoint* GetMapPoint(int idx);
		void SetMapPoint(UVR_SLAM::MapPoint* pMP, int idx);
		bool GetBoolInlier(int idx);
		void SetBoolInlier(bool flag, int idx);
		void SetObjectType(UVR_SLAM::ObjectType type, int idx);
		ObjectType GetObjectType(int idx);
		int GetNumInliers();
		unsigned char GetType();
		void TurnOnFlag(unsigned char opt);
		void TurnOffFlag(unsigned char opt);
		bool CheckType(unsigned char opt);
		int GetFrameID();
		bool CheckBaseLine(Frame* pF1, Frame* pF2);
		bool ComputeSceneMedianDepth(float& fMedianDepth);
		cv::Mat GetCameraCenter();
	public:
		
		std::vector<bool> mvbMPInliers;
		std::vector<cv::KeyPoint> mvKeyPoints, mvKeyPointsUn, mvkInliers, mvTempKPs;
		cv::Mat matDescriptor;
		cv::Mat undistorted;
		fbow::fBow mBowVec;
	
	private:
		void Increase();
		void Decrease();
		void SetFrameID();
	private:
		int mnFrameID;
		std::mutex mMutexID;

		std::vector<ObjectType> mvObjectTypes; //모든 키포인트에 대해서 미리 정의된 레이블인지 재할당
		std::mutex mMutexObjectTypes;

		std::vector<UVR_SLAM::MapPoint*> mvpMPs;
		cv::Mat matFrame, matOri;

		cv::Mat R, t;
		int mnInliers;
		unsigned char mnType;

		std::mutex mMutexMPs, mMutexBoolInliers, mMutexNumInliers, mMutexPose, mMutexType;
		std::mutex mMutexImage;
	public:
		//from ORB_SLAM2
		cv::Mat mDistCoef, mK;
		ORBextractor* mpORBextractor;
		static float fx;
		static float fy;
		static float cx;
		static float cy;
		static float invfx;
		static float invfy;
		static float mnMinX;
		static float mnMaxX;
		static float mnMinY;
		static float mnMaxY;
		static bool mbInitialComputations;

		int mnScaleLevels;
		float mfScaleFactor;
		float mfLogScaleFactor;
		std::vector<float> mvScaleFactors;
		std::vector<float> mvInvScaleFactors;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;
		static float mfGridElementWidthInv;
		static float mfGridElementHeightInv;
		std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

		void Init(ORBextractor* _e, cv::Mat _k, cv::Mat _d);
		void ExtractORB(const cv::Mat &im, std::vector<cv::KeyPoint>& vKPs, cv::Mat& desc);
		void UndistortKeyPoints();
		void ComputeImageBounds(const cv::Mat &imLeft);
		void AssignFeaturesToGrid();
		std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel);
		bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
	protected:

	};
}
#endif //ANDROIDOPENCVPLUGINPROJECT_FEATUREFRAME_H
