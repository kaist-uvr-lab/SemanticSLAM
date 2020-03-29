
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
		PLANE_SPARSE_MP,
		PLANE_DENSE_MP
	};

	class Frame;
	class FrameWindow;
	class MapPoint {
	public:
		//초기 포즈 만들 때는 double형으로 형변환
		MapPoint();
		MapPoint(UVR_SLAM::Frame* pRefKF, cv::Mat _p3D, cv::Mat _desc);
		MapPoint(UVR_SLAM::Frame* pRefKF, cv::Mat _p3D, cv::Mat _desc, MapPointType ntype);
		virtual ~MapPoint();
	public:
		void SetWorldPos(cv::Mat X);
		cv::Mat GetWorldPos();
		void SetNewMP(bool _b);
		bool isNewMP();
		bool isInFrame(Frame* pF);
		void AddFrame(UVR_SLAM::Frame* pF, int idx); //index in frame
		void RemoveFrame(UVR_SLAM::Frame* pKF);
		int GetIndexInFrame(UVR_SLAM::Frame* pF);
		void UpdateNormalAndDepth();
		int PredictScale(const float &currentDist, Frame* pKF);
		void Delete();
		void SetDescriptor(cv::Mat _desc);
		cv::Mat GetDescriptor();
		void IncreaseVisible(int n = 1);
		void IncreaseFound(int n = 1);
		float GetFVRatio();
		void Fuse(MapPoint* pMP);

		bool Projection(cv::Point2f& _P2D, cv::Mat& _Pcam, cv::Mat R, cv::Mat t, cv::Mat K, int w, int h);
		bool isSeen();

		void SetMapPointType(UVR_SLAM::MapPointType type);
		MapPointType GetMapPointType();

		void SetPlaneID(int nid);
		int GetPlaneID();
		int GetMapPointID();
		std::map<Frame*, int> GetConnedtedFrames();
		int GetNumConnectedFrames();

		void SetDelete(bool b);
		bool isDeleted();
		
		float GetMaxDistance();
		float GetMinDistance();
		cv::Mat GetNormal();

	public:
		int mnMapPointID;
		int mnFirstKeyFrameID;
		int mnLocalBAID;

	private:
		Frame* mpRefKF;
		std::mutex mMutexMP;
		bool mbDelete;
		int mnPlaneID;
		MapPointType mnType;
		

		float mfDepth;
		bool mbSeen;
		bool mbNewMP;
		cv::Mat p3D;
		int mnConnectedFrames;
		
		cv::Mat desc;
		std::map<UVR_SLAM::Frame*, int> mmpFrames;

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
	};
}

#endif