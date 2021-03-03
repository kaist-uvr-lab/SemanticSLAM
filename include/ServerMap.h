#ifndef UVR_SLAM_SERVER_MAP_H
#define UVR_SLAM_SERVER_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class MapPoint;
	class Frame;
	class MapGrid;
	class ServerMap {
	public:
		ServerMap();
		virtual ~ServerMap();
	public:
		bool mbInitialized;
		std::list<MapPoint*> mlpNewMapPoints;
	public:
		void SetInitialKeyFrame(UVR_SLAM::Frame* pKF1, UVR_SLAM::Frame* pKF2);
		void AddFrame(Frame* pF);
		void RemoveFrame(Frame* pF);
		std::vector<Frame*> GetFrames();
		void AddMapPoint(MapPoint* pMP);
		void RemoveMapPoint(MapPoint* pMP);
		std::vector<MapPoint*> GetMapPoints();
	private:
		std::mutex mMutexMPs, mMutexKFs;
		std::set<MapPoint*> mspMapMPs;
		std::set<Frame*> mspMapFrames;
	public:
		Frame* mpCurrKF, *mpPrevKF;
	public:
		int nServerMapPointID; //로드시 값 변경 필요
		int nServerKeyFrameID;
	private:

		////데이터 받고 보내는것도 구현

	};
}
#endif