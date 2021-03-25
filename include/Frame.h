

//
// Created by UVR-KAIST on 2019-01-08.
//

#ifndef UVR_SLAM_FRAME_H
#define UVR_SLAM_FRAME_H
#pragma once

#include <DBoW3.h>
#include <System.h>
#include <map>
#include <functional>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <PointLess.h>
#include <SegmentationData.h>

//#include "KeyFrame.h"

namespace UVR_SLAM {
#define FRAME_GRID_ROWS 24
#define FRAME_GRID_COLS 32

	const unsigned char FLAG_KEY_FRAME = 0x1;
	const unsigned char FLAG_SEGMENTED_FRAME = 0x2; 
	const unsigned char FLAG_LAYOUT_FRAME = 0x4;
	const unsigned char FLAG_INIT_FRAME = 0x8;

	//class MapPoint;
	class MapPoint;
	class CandidatePoint;
	class PlaneInformation;
	class PlaneProcessInformation;
	class Line;
	class FrameGrid;
	class FrameGridLevelKey;

	class Frame {
	public:
		Frame(System* pSys, int id, int w, int h, cv::Mat K, cv::Mat invK, double ts);
		Frame(System* pSys, int id, int w, int h, float _fx, float _fy, float _cx, float _cy, double ts);
		virtual ~Frame();
		void close();

	public:
		void AddMapPoint(MapPoint* pMP, int idx);
		MapPoint* GetMapPoint(int idx);
		void EraseMapPoint(int idx);
		void SetMapPoints(int n);
		std::vector<MapPoint*> GetMapPoints();

	public:
		std::string mstrMapName;
		std::vector<cv::Point2f> mvPts; //Ű����Ʈ�� ����Ʈ�� ������ ����.

	private:
		std::mutex mMutexMPs;
		std::vector<MapPoint*> mvpMPs;

////////////////////////////////////////////////////////////////////////////////////////////////////////
	public:
		//////BOW
		////DBOW
		DBoW3::BowVector mBowVec;
		DBoW3::FeatureVector mFeatVec;
		void ComputeBoW();

		////FBOW
		void process(cv::Ptr<cv::Feature2D> detector); //????????�̰� �̿����ΰ�?
		/*fbow::fBow GetBowVec();
		void SetBowVec(fbow::Vocabulary* pfvoc);
		double Score(UVR_SLAM::Frame* pF);
		fbow::fBow mBowVec;
		fbow::fBow2 mFeatureVec;*/
		////FBOW
	public:
		Frame *mpPrev, *mpNext;
		cv::Mat GetFrame();
		cv::Mat GetOriginalImage();
		void SetPose(cv::Mat _R, cv::Mat _t);
		void GetPose(cv::Mat&_R, cv::Mat& _t);
		void GetInversePose(cv::Mat&_Rinv, cv::Mat& _Tinv);
		void GetRelativePoseFromTargetFrame(Frame* pTargetFrame, cv::Mat& Rft, cv::Mat& Tft);
		cv::Mat ComputeFundamentalMatrix(Frame* pTarget);
		void ComputeSceneDepth();
		float GetDepth(cv::Mat X3D);
		cv::Mat GetRotation();
		cv::Mat GetTranslation();
		void AddMP(UVR_SLAM::MapPoint* pMP, int idx);
		void RemoveMP(int idx); 
		
		void Reset();
		float CalcDiffAngleAxis(UVR_SLAM::Frame* pF);

		//UVR_SLAM::MapPoint* GetMapPoint(int idx);
		//void SetMapPoint(UVR_SLAM::MapPoint* pMP, int idx);
		//bool GetBoolInlier(int idx);
		//void SetBoolInlier(bool flag, int idx);

		int GetNumInliers();
		int TrackedMapPoints(int minObservation);
		
		bool CheckBaseLine(Frame* pTargetKF);
		bool ComputeSceneMedianDepth(std::vector<UVR_SLAM::MapPoint*> vpMPs, cv::Mat R, cv::Mat t, float& fMedianDepth);
		
		cv::Mat GetCameraCenter();
		void SetInliers(int nInliers);
		int GetInliers();

		bool isDeleted();
		bool isInImage(float u, float v, float w = 0);
		bool isInFrustum(MapPoint *pMP, float viewingCosLimit);
		cv::Point2f Projection(cv::Mat w3D);

		//////////////////////////////
		void SetBoolMapping(bool b);
		bool GetBoolMapping();
		std::mutex mMutexMapping;
		bool mbMapping;
		/////////////////////////////

		///
		void Delete();	//KF ������ 2���� ��ġ�� ������. mbNotErase�� false�̰�, mbToBeErased�� true�� �Ǿ�� ��. setbadflag�� seterase�� �Ѵ� ȣ���� �Ǿ�� ��. setbadflag�� mbnoterase�� false���� �����ϸ�, �̰� Ʈ���� ��쿡�� mbtobeerase�� ����. mbtobeerase�� setbadflag������ ������. 
						//mbnoterase�� loop ������ ã�� �� ���� ��Ī�� �ȵ� ��� seterase������ false�� ������.
						//setbadflag�� ������Ʈ ����Ʈ�� ���� ��쿡�� ȣ��
						//seterase�� setnoterase ���� Ŭ��¡���� ȣ��.
		void AddKF(UVR_SLAM::Frame* pKF, int weight);
		void RemoveKF(Frame* pKF);
		void RemoveKF(UVR_SLAM::Frame* pKF, int weight);
		//std::vector<UVR_SLAM::Frame*> GetConnectedKFs();
		std::vector<UVR_SLAM::Frame*> GetConnectedKFs(int n = 0);
		std::set<UVR_SLAM::Frame*> GetConnectedKeyFrameSet(int n = 0);
		std::multimap<int, UVR_SLAM::Frame*, std::greater<int>> GetConnectedKFsWithWeight();
////////////////
////Depth & Seg
	public:
		cv::Mat mRawDepth, mSegLabel, mSegImage, mDepthImage;
		bool bDepth, bSeg;
////FrameGrid2
	public:
		int mnGridSize;
		int mnMaxGridWidth, mnMaxGridHeight;
		void SetGrids();
		std::vector<int> GetPointIndices(cv::Point2f pt);
		cv::Point2i GetGridIndex(cv::Point2f pt);
		std::vector<std::vector<FrameGrid*>> mvpFrameGrids;
	private:
		
////FrameGrid


////FrameGrid
	public:
		void ComputeGradientImage(cv::Mat src, cv::Mat& dst, int ksize = 1);
		//void SetGrids();
		cv::Point2f GetExtendedRect(cv::Point2f pt, int size);
		cv::Point2f GetGridBasePt(cv::Point2f pt, int size);

		std::map<cv::Point2f, FrameGrid*, Point2fLess> mmpFrameGrids;
		//std::map<cv::Point2f, bool, Point2fLess> mmbFrameGrids;
////FrameGrid

//////Pyramid test
	public:
		std::map<FrameGridLevelKey, FrameGrid*> mmpFrameLevelGrids;
		std::vector<cv::Point2f> mvPyramidPts;
//////Pyramid test

////////////////
	public:
		float mfMeanDepth, mfMedianDepth, mfMinDepth, mfMaxDepth, mfStdDev, mfRange;
	////////////////
	public:
		int mnLocalMapFrameID, mnTrackingID;
		int mnLocalBAID, mnFixedBAID;
		int mnFuseFrameID;

	public:
		///////////////////////////////
		////200526
		System* mpSystem;
		int mnWidth, mnHeight;
		cv::Mat mEdgeImg;
		std::vector<cv::Point2f> mvEdgePts;
		////200423
		////200410
		////Optical flow�� ������ ���
		//�̹��� �ȼ��� Ű����Ʈ ������ ����.
		
		std::vector<int> mvnOctaves;
		//��Ī ������ ����
		///////////////////////////////
		std::vector<cv::KeyPoint> mvKeyPoints, mvKeyPointsUn, mvkInliers, mvTempKPs;
		cv::Mat matDescriptor;
		cv::Mat undistorted;
		
		int mnFrameID;  //������ ���̵�� ����
		int mnKeyFrameID;

		////loop closing
		int mnLoopClosingID;
		
		int mnLoopBowWords;
		float mfLoopScore;
		////loop closing
	public:
		//objectype
		cv::Mat mOriginalImage;
		cv::Mat matFrame;
		std::set<MapPoint*> mspFloorMPs, mspCeilMPs, mspWallMPs;
		std::vector<std::multimap<ObjectType, int, std::greater<int>>> mvMapObjects;
		double mdTimestamp;
		//matching
		cv::Mat mPlaneDescriptor;
		cv::Mat mWallDescriptor;
		cv::Mat mObjectDescriptor;
		cv::Mat matSegmented, matLabeled;
		std::vector<int> mPlaneIdxs, mWallIdxs, mObjectIdxs;
		cv::Mat mLabelStatus; //������Ʈ ��ũ���� ���� ���� üũ�ϱ� ���� ��. //1 = floor, 2 = wall, 3 = object //object labeling�� ������ �׳�.

		std::vector<PlaneInformation*> mvpPlanes;
	private:
		//void Increase();
		//void Decrease();
		//object
	public:
		void SetObjectType(UVR_SLAM::ObjectType type, int idx);
		ObjectType GetObjectType(int idx);
		std::vector<ObjectType> GetObjectVector();
		void SetObjectVector(std::vector<ObjectType> vObjTypes);
		void SetBoolSegmented(bool b);
		bool isSegmented();
	private:
		std::mutex mMutexObjectTypes;
		std::vector<ObjectType> mvObjectTypes; //��� Ű����Ʈ�� ���ؼ� �̸� ���ǵ� ���̺����� ���Ҵ�
		bool bSegmented;
		std::mutex mMutexSegmented;
	private:
		
		std::mutex mMutexNumInliers;
		std::mutex mMutexFrame, mMutexPose, mMutexConnection;
		
		std::multimap<int,UVR_SLAM::Frame*, std::greater<int>> mmpConnectedKFs;
		cv::Mat matOri;
		cv::Mat R, t;
		bool mbDeleted;
		int mnInliers;
	public:
		std::map<Frame*, int> mmKeyFrameCount;
		/*
		std::mutex mMutexID;
		std::mutex mMutexMPs, mMutexBoolInliers, mMutexNumInliers, mMutexPose, mMutexType;
		std::mutex mMutexImage;*/

//////////////////////////
////LINE
	public:
		PlaneProcessInformation* mpPlaneInformation;
		void SetLines(std::vector<Line*> lines);
		std::vector<Line*> Getlines();
		std::vector<cv::Mat> GetWallParams();
		void SetWallParams(std::vector<cv::Mat> vParams);
	private:
		std::mutex mMutexLines;
		std::vector<Line*> mvLines;
		std::mutex mMutexWallParams;
		std::vector<cv::Mat> mvWallParams;
////LINE
//////////////////////////

	public:
		std::vector<cv::Mat> mvPyramidImages;
		//from ORB_SLAM2
		
		cv::Mat mDistCoef, mK, mInvK;
		float fx;
		float fy;
		float cx;
		float cy;
		float invfx;
		float invfy;
		float mnMinX;
		float mnMaxX;
		float mnMinY;
		float mnMaxY;
		
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

		void DetectEdge();
		void UndistortKeyPoints();
		void ComputeImageBounds(const cv::Mat &imLeft);
		
	protected:

	};
}
#endif //ANDROIDOPENCVPLUGINPROJECT_FEATUREFRAME_H
