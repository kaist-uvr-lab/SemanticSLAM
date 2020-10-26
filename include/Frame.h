

//
// Created by UVR-KAIST on 2019-01-08.
//

#ifndef UVR_SLAM_FRAME_H
#define UVR_SLAM_FRAME_H
#pragma once
#include <fbow.h> //include windows header.
#include <map>
#include <functional>
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
	const unsigned char FLAG_LAYOUT_FRAME = 0x4;
	const unsigned char FLAG_INIT_FRAME = 0x8;

	//class MapPoint;
	class System;
	class CandidatePoint;
	class ORBextractor;
	class PlaneInformation;
	class PlaneProcessInformation;
	class Line;
	class MatchInfo {
	public:
		
		MatchInfo();
		MatchInfo(System*  pSys, Frame* pRef, Frame* pTarget, int w, int h);
		virtual ~MatchInfo();
		void ConnectAll(); //이거 체크
		void DisconnectAll(); //이것도 체크
		void UpdateKeyFrame();
		bool UpdateFrameQuality();
		void SetMatchingPoints(); //초기화나 매핑시 포인트 매칭을 위한 포인트 추가 과정.
		void SetLabel();
		
		
	public:
		int AddCP(CandidatePoint* pCP, cv::Point2f pt);
		void RemoveCP(int idx);
		std::vector<cv::Point2f> GetMatchingPtsMapping(std::vector<UVR_SLAM::CandidatePoint*>& vpCPs);
		int CheckOpticalPointOverlap(cv::Point2f pt, int radius, int margin = 30); //확인 후 삭제.
		bool CheckOpticalPointOverlap(cv::Mat& overlap, cv::Point2f pt, int radius, int margin = 30); //확인 후 삭제.

		std::vector<UVR_SLAM::CandidatePoint*> mvpMatchingCPs; //KF-KF 매칭에서 삼각화시 베이스라인을 충분히 확보하기 위함.
		std::vector<bool> mvbMapPointInliers; //프레임 업데이트에만 이용
		//std::vector<UVR_SLAM::MapPoint*> mvpMatchingMPs;
		std::vector<cv::Point2f> mvMatchingPts; //CPPt에서 변경함
	public:
		System* mpSystem;
		//오브젝트 관련
		std::map<int, int> mmLabelAcc; //std::list<cv::Point2f>
		std::map<int, cv::Mat> mmLabelMasks; //마스크 이미지
		//std::multimap<int, cv::Rect> mmLabelRects; //동일 물체에 여러개가 나올 수 있기 때문에 멀티맵으로
		//std::multimap<int, std::list<CandidatePoint*>> mmLabelCPs;
		std::multimap<int,std::pair<cv::Rect, std::list<CandidatePoint*>>> mmLabelRectCPs;
	private:
		std::mutex mMutexCPs; //cp vector는 이미 완성된 상태임.
		cv::Mat mMapCP; //현재 이미지 내에 CP의 포인트 위치 & 인덱스, ushort, 16US1
	public:
		int GetNumMPs();
		
		void InitMapPointInlierVector(int N);
		/*
		void AddMP();
		void RemoveMP();
		void AddMP(MapPoint* pMP, int idx);
		void RemoveMP(int idx);
		MapPoint* GetMP(int idx);*/
	private:
		std::mutex mMutexMPs;
		int mnNumMapPoint;

//////////////////
	public:
		//현재 매칭된 값을 저장함.
		float mfLowQualityRatio;
		int mnWidth, mnHeight;
		UVR_SLAM::Frame* mpTargetFrame, *mpRefFrame, *mpNextFrame;
		////매칭 된 정보를 저장하는건데 사용 안할듯함.
		std::vector<int> mvnVisibles, mvnMatches;
		////디버깅용 이미지를 저장함.
		cv::Mat mMatchedImage;
		//엣지에서 뽑은 피티 저장.
		std::vector<cv::Point2f> mvEdgePts;

	public:

	private:

	};

	class Frame {
	public:
		Frame(cv::Mat _src, int w, int h, cv::Mat mK, double ts);
		Frame(void* ptr, int id, int w, int h, cv::Mat mK);
		Frame(void* ptr, int id, int w, int h, cv::Mat _R, cv::Mat _t, cv::Mat mK);
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
		void SetPose(cv::Mat _R, cv::Mat _t);
		void GetPose(cv::Mat&_R, cv::Mat& _t);
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

		bool isInImage(float u, float v, float w = 0);
		bool isInFrustum(MapPoint *pMP, float viewingCosLimit);
		cv::Point2f Projection(cv::Mat w3D);

		//////////////////////////////
		void SetBoolMapping(bool b);
		bool GetBoolMapping();
		std::mutex mMutexMapping;
		bool mbMapping;
		/////////////////////////////

		void SetDepthRange(float min, float max);
		void GetDepthRange(float& min, float& max);

		///
		void AddKF(UVR_SLAM::Frame* pKF, int weight);
		void RemoveKF(UVR_SLAM::Frame* pKF, int weight);
		//std::vector<UVR_SLAM::Frame*> GetConnectedKFs();
		std::vector<UVR_SLAM::Frame*> GetConnectedKFs(int n = 0);
		std::multimap<int, UVR_SLAM::Frame*, std::greater<int>> GetConnectedKFsWithWeight();
////////////////
	public:
		void ComputeSceneMedianDepth();
		float GetSceneMedianDepth();
	private:
		float mfMedianDepth;
		std::mutex mMutexMedianDepth;
	////////////////
	public:
		int mnLocalMapFrameID;
		int mnLocalBAID, mnFixedBAID;
		int mnFuseFrameID;
	public:
		void SetRecentTrackedFrameID(int id);
		int GetRecentTrackedFrameID();
	private:
		int mnRecentTrackedFrameId;
		std::mutex mMutexTrackedFrame;

	public:
		///////////////////////////////
		////200526
		int mnWidth, mnHeight;
		cv::Mat mEdgeImg;
		std::vector<cv::Point2f> mvEdgePts;
		////200423
		MatchInfo* mpMatchInfo;
		////200410
		////Optical flow를 적용한 방식
		//이미지 픽셀에 키포인트 순서를 저장.
		std::vector<cv::Point2f> mvPts; //키포인트의 포인트만 별도로 빼냄.
		std::vector<int> mvnOctaves;
		//매칭 정보를 저장
		///////////////////////////////
		std::vector<cv::KeyPoint> mvKeyPoints, mvKeyPointsUn, mvkInliers, mvTempKPs;
		cv::Mat matDescriptor;
		cv::Mat undistorted;
		fbow::fBow mBowVec;
		int mnFrameID;  //프레임 아이디로 저장
		int mnKeyFrameID;
	public:
		//objectype
		std::set<MapPoint*> mspFloorMPs, mspCeilMPs, mspWallMPs;
		std::vector<std::multimap<ObjectType, int, std::greater<int>>> mvMapObjects;
		double mdTimestamp;
		//matching
		cv::Mat mPlaneDescriptor;
		cv::Mat mWallDescriptor;
		cv::Mat mObjectDescriptor;
		cv::Mat matSegmented, matLabeled;
		std::vector<int> mPlaneIdxs, mWallIdxs, mObjectIdxs;
		cv::Mat mLabelStatus; //오브젝트 디스크립터 포함 유무 체크하기 위한 것. //1 = floor, 2 = wall, 3 = object //object labeling을 따르자 그냥.

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
		std::vector<ObjectType> mvObjectTypes; //모든 키포인트에 대해서 미리 정의된 레이블인지 재할당
		bool bSegmented;
		std::mutex mMutexSegmented;
	private:
		
		
		std::mutex mMutexNumInliers;
		std::mutex mMutexFrame, mMutexPose;
		std::mutex mMutexDepthRange;
		float mfMinDepth, mfMaxDepth;
		
		std::multimap<int,UVR_SLAM::Frame*, std::greater<int>> mmpConnectedKFs;
		cv::Mat matFrame, matOri;
		cv::Mat R, t;
		int mnInliers;
		
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
		void DetectFeature();
		void DetectEdge();
		void ExtractORB(const cv::Mat &im, std::vector<cv::KeyPoint>& vKPs, cv::Mat& desc);
		void UndistortKeyPoints();
		void ComputeImageBounds(const cv::Mat &imLeft);
		void AssignFeaturesToGrid();
		std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel = -1, const int maxLevel = -1);
		bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
	protected:

	};
}
#endif //ANDROIDOPENCVPLUGINPROJECT_FEATUREFRAME_H
