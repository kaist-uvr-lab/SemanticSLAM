
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
		PLANE_MP
	};

	class Frame;
	class FrameWindow;
	class MapPoint {
	public:
		//초기 포즈 만들 때는 double형으로 형변환
		MapPoint();
		MapPoint(cv::Mat _p3D, Frame* pTargetKF, int idx, cv::Mat _desc);
		MapPoint(cv::Mat _p3D, Frame* pTargetKF, int idx, cv::Mat _desc, MapPointType ntype);
		virtual ~MapPoint();
	public:
		void SetWorldPos(cv::Mat X);
		cv::Mat GetWorldPos();
		void SetNewMP(bool _b);
		bool isNewMP();
		void AddFrame(UVR_SLAM::Frame* pF, int idx); //index in frame
		void RemoveFrame(UVR_SLAM::Frame* pKF);
		void Delete();
		void SetDescriptor(cv::Mat _desc);
		cv::Mat GetDescriptor();
		bool Projection(cv::Point2f& _P2D, cv::Mat& _Pcam, cv::Mat R, cv::Mat t, cv::Mat K, int w, int h);
		bool isSeen();

		void SetObjectType(ObjectType nType);
		ObjectType  GetObjectType();
		MapPointType GetMapPointType();

		void SetFrameWindowIndex(int nIdx);
		int GetFrameWindowIndex();
		void SetPlaneID(int nid);
		int GetPlaneID();
		int GetMapPointID();
		std::map<Frame*, int> GetConnedtedFrames();
		int GetNumConnectedFrames();

		void SetDelete(bool b);
		bool isDeleted();
		
	public:
		int mnVisibleCount;
		int mnMatchingCount;
	private:
		void Init(UVR_SLAM::Frame* pTargetKF, int idx);
	private:
		std::mutex mMutexMP;
		bool mbDelete;
		int mnMapPointID;
		int mnPlaneID;
		MapPointType mnType;
		ObjectType mObjectType;
		int mnFrameWindowIndex;

		float mfDepth;
		bool mbSeen;
		bool mbNewMP;
		cv::Mat p3D;
		int mnConnectedFrames;
		
		float mfMinDistance;
		float mfMaxDistance;

		cv::Mat desc;
		std::map<UVR_SLAM::Frame*, int> mmpFrames;
	};
}

#endif