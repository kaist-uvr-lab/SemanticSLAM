
#ifndef UVR_SLAM_MAP_POINT_H
#define UVR_SLAM_MAP_POINT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <SegmentationData.h>

namespace UVR_SLAM {
	class Frame;
	class FrameWindow;
	class MapPoint {
	public:
		//초기 포즈 만들 때는 double형으로 형변환
		MapPoint();
		MapPoint(cv::Mat _p3D, cv::Mat _desc);
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

		void SetFrameWindowIndex(int nIdx);
		int GetFrameWindowIndex();
		int GetMapPointID();
		std::map<Frame*, int> GetConnedtedFrames();

		void SetDelete(bool b);
		bool isDeleted();
		
	public:
		int mnVisibleCount;
		int mnMatchingCount;

	private:
		bool mbDelete;
		std::mutex mMutexMPID;
		int mnMapPointID;

		std::mutex mMutexObjectType; //mnObjectType
		ObjectType mObjectType;
		
		std::mutex mMutexFrameWindowIndex;
		int mnFrameWindowIndex;

		float mfDepth;
		bool mbSeen;
		bool mbNewMP;
		cv::Mat p3D;
		int mnConnectedFrames;
		std::mutex mMutexMP;
		cv::Mat desc;
		std::map<UVR_SLAM::Frame*, int> mmpFrames;
	};
}

#endif