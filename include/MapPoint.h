
#ifndef UVR_SLAM_MAP_POINT_H
#define UVR_SLAM_MAP_POINT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <SegmentationData.h>

namespace UVR_SLAM {

	const enum MapPointType {
		NORMAL_MP,
		PLANE_MP,
		PLANE_DENSE_MP
	};

	class Map;
	class Frame;
	class MatchInfo;
	class FrameWindow;
	class MapPoint {
	public:
		//초기 포즈 만들 때는 double형으로 형변환
		MapPoint();
		MapPoint(Map* pMap, UVR_SLAM::Frame* pRefKF, cv::Mat _p3D, cv::Mat _desc, int label, int nOctave = 0);
		MapPoint(Map* pMap, UVR_SLAM::Frame* pRefKF, cv::Mat _p3D, cv::Mat _desc, MapPointType ntype, int label, int nOctave = 0);
		virtual ~MapPoint();
	public:
		void SetWorldPos(cv::Mat X);
		cv::Mat GetWorldPos();
		void SetNewMP(bool _b);
		bool isNewMP();
		bool isInFrame(MatchInfo* pF);
		int GetPointIndexInFrame(MatchInfo* pF);

		void UpdateNormalAndDepth();
		int PredictScale(const float &currentDist, Frame* pKF);
		
		void SetDescriptor(cv::Mat _desc);
		cv::Mat GetDescriptor();
		void IncreaseVisible(int n = 1);
		void IncreaseFound(int n = 1);
		float GetFVRatio();
		void Fuse(MapPoint* pMP);

		bool Projection(cv::Point2f& _P2D, cv::Mat& _Pcam, cv::Mat R, cv::Mat t, cv::Mat K, int w, int h);
		bool isSeen();

		MapPointType GetMapPointType();
		void SetMapPointType(MapPointType type);

		void SetPlaneID(int nid);
		int GetPlaneID();
		int GetMapPointID();
		
		//void SetDelete(bool b);
		bool isDeleted();
		
		float GetMaxDistance();
		float GetMinDistance();
		cv::Mat GetNormal();
		
	public:
		int mnMapPointID;
		int mnFirstKeyFrameID;
		int mnLocalBAID;
		int mnOctave;

	private:
		Map* mpMap;
		Frame* mpRefKF;
		std::mutex mMutexMP;
		bool mbDelete;
		int mnPlaneID;
		MapPointType mnType;
		

		float mfDepth;
		bool mbSeen;
		bool mbNewMP;
		cv::Mat p3D;
		int mnDenseFrames;
		
		cv::Mat desc;
		
		std::map<UVR_SLAM::Frame*, cv::Point2f> mmpDenseFrames;
		std::mutex mMutexFeatures;
		int mnVisible;
		int mnFound;

		float mfMaxDistance, mfMinDistance;
		cv::Mat mNormalVector;

	//local map 및 최근 트래킹 관련 index 관련
	public:
		int GetRecentLocalMapID();
		void SetRecentLocalMapID(int nLocalMapID);
		int GetRecentTrackingFrameID();
		void SetRecentTrackingFrameID(int nFrameID);
		int GetRecentLayoutFrameID();
		void SetRecentLayoutFrameID(int nFrameID);
	private:
		std::mutex mMutexRecentLocalMapID, mMutexRecentTrackedFrameID, mMutexRecentLayoutFrameID;
		int mnLocalMapID, mnTrackedFrameID, mnLayoutFrameID;
	//Object Type
	public:
		void SetObjectType(ObjectType nType);
		ObjectType  GetObjectType();
	private:
		std::mutex mMutexObjectType;
		ObjectType mObjectType;

		//////////////////////프레임과 관련된 것들
	public:
		void AddFrame(UVR_SLAM::MatchInfo* pF, cv::Point2f pt); //index in frame
		void RemoveFrame(UVR_SLAM::MatchInfo* pKF);
		std::map<MatchInfo*, int> GetConnedtedFrames();
		int GetNumConnectedFrames();
		void Delete();
	private:
		std::map<UVR_SLAM::MatchInfo*, int> mmpFrames;
		int mnConnectedFrames;
		//////////////////////프레임과 관련된 것들

	};
}

#endif