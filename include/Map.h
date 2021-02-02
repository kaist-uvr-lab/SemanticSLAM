#ifndef UVR_SLAM_MAP_H
#define UVR_SLAM_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <list>
#include <PointLess.h>

namespace UVR_SLAM {

	class Frame;
	class System;
	class MapPoint;
	class WallPlane;
	class PlaneInformation;
	class PlaneProcessInformation;
	class MapGrid;
	
	class Map {
		/////////////////////////////////////
		//////임시 포인트 확인용
	public:
		std::vector<cv::Point2f> GetTrackingPoints();
		void SetTrackingPoints(std::vector<cv::Point2f> vPTs);
	private:
		std::mutex mMutexTrackingPTs;
		std::vector<cv::Point2f> mvTrackingPTs;
	public:
		std::vector<cv::Mat> GetReinit();
		void ClearReinit();
		void AddReinit(cv::Mat m);
	private:
		std::mutex mMutexReinit;
		std::vector<cv::Mat> mvReinit;
		/////////
	public:
		std::vector<cv::Mat> GetTempMPs();
		void ClearTempMPs();
		void AddTempMP(cv::Mat m);
	private:
		std::vector<cv::Mat> mvTempMPs;
		//////임시 포인트 확인용
		/////////////////////////////////////
		
	public:
		//keyframes
		//전체 맵포인트
		//바닥
		//벽을 가져야 한다.
		//바닥과 벽의 초기화 여부를 알아야 한다.
		Map();
		Map(System* pSystem, int nConnected = 8, int nCandiate = 4);
		virtual ~Map();
	public:
		void AddFrame(Frame* pF);
		void RemoveFrame(Frame* pF);
		std::vector<Frame*> GetFrames();

		Frame* GetLastWindowFrame();
		Frame* GetReverseWindowFrame(int idx);
		Frame* AddWindowFrame(Frame* pF);
		//level = 1이면 첫번째 레벨의큐, 2이면 두번째 레벨의 큐 접근, 3이면 세번째 레벨 큐 접근
		std::vector<Frame*> GetTrajectoryFrames();
		std::vector<Frame*> GetWindowFramesVector(int level = 3);
		std::set<Frame*> GetWindowFramesSet(int level = 3);
		std::vector<Frame*> GetGraphFrames();
		int mnMaxConnectedKFs, mnHalfConnectedKFs, mnQuarterConnectedKFs;
		int mnMaxCandidateKFs;
	private:
		int mnHalfCandidate;
		std::mutex mMutexFrames;
		std::mutex mMutexWindowFrames;
		std::list<Frame*> mQueueFrameWindows1, mQueueFrameWindows2, mQueueFrameWindows3;
		std::list<Frame*> mQueueCandidateGraphFrames;
		std::set<Frame*> mspGraphFrames, mspFrames;
		std::vector<Frame*> mvpTrajectoryKFs;
		System* mpSystem;
		////////////////////////////////
		////Dense Flow 관리
	public:
		void AddFlow(int nFrameID, cv::Mat flow);
		cv::Mat GetFlow(int nFrameID);
		std::vector<cv::Mat> GetFlows(int nStartID, int nEndID);
	private:
		std::mutex mMutexFlows;
		std::map<int, cv::Mat> mmFlows;
		////Dense Flow 관리

		////맵포인트 관리
	public:
		std::mutex mMutexMapUpdate, mMutexMapOptimization;
		void AddMap(MapPoint* pMP, int label);
		void RemoveMap(MapPoint* pMP);
		std::map<MapPoint*, int> GetMap();
		void AddDeleteMP(MapPoint* pMP);
		void DeleteMPs();
		void SetNumDeleteMP();
	private:
		std::mutex mMutexMap;
		std::map<MapPoint*, int> mmpMapMPs;
		std::mutex mMutexDeleteMapPointSet;
		std::set<MapPoint*> mspDeleteMPs;
		std::queue<int> mQueueNumDelete;
		int mnDeleteMPs;
		////맵포인트 관리
		////구역별 맵 저장
		
////구역별 맵 저장

////평면 관리
	public:
		void AddPlaneInfo(PlaneProcessInformation* pPlaneInfo);
		std::vector<PlaneProcessInformation*> GetPlaneInfos();
	private:
		std::mutex mMutexPlaneInfo;
		std::vector<PlaneProcessInformation*> mvpPlaneInfos;
		////평면 관리
		////////////////////////////

		////////////////////////////////
		////평면 관리
		//벽평면은 여러개 있을 수 있음
		//바닥 평면은 오직 한개임.
	public:
		bool isFloorPlaneInitialized();
		void SetFloorPlaneInitialization(bool b);
		bool isWallPlaneInitialized();
		void SetWallPlaneInitialization(bool b);
		void ClearWalls();
		//추가 & 획득
		std::vector<WallPlane*> GetWallPlanes();
		void AddWallPlane(WallPlane* pWall);
	private:
		std::mutex mMutexInitFloorPlane;
		bool mbInitFloorPlane;
		std::mutex mMutexInitWallPlane;
		bool mbInitWallPlane;

		std::mutex mMutexWallPlanes;
		std::vector<WallPlane*> mvpWallPlanes;

	public:
		Frame* mpFirstKeyFrame;
		UVR_SLAM::PlaneInformation* mpFloorPlane;
		////평면 관리
		////////////////////////////////

		/////////MapGrid
		public:
			MapGrid* AddMapGrid(cv::Point3f key);
			MapGrid* GetMapGrid(cv::Point3f key);
			std::vector<MapGrid*> GetMapGrids();
		private:
			std::mutex mMutexMapGrids;
			std::map<cv::Point3f, MapGrid*, Point3fLess> mmpMapGrids;
		/////////MapGrid

	};
}
#endif


