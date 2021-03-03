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
		//////�ӽ� ����Ʈ Ȯ�ο�
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
		//////�ӽ� ����Ʈ Ȯ�ο�
		/////////////////////////////////////
		
	public:
		//keyframes
		//��ü ������Ʈ
		//�ٴ�
		//���� ������ �Ѵ�.
		//�ٴڰ� ���� �ʱ�ȭ ���θ� �˾ƾ� �Ѵ�.
		Map();
		Map(System* pSystem, int nConnected = 8, int nCandiate = 4);
		
		virtual ~Map();
	public:
		////���� ���� �ڵ�
		void LoadMapDataFromServer(std::string mapname, std::vector<Frame*>& vpMapFrames);
		std::vector<Frame*> mvpMapFrames;
	public:
		void Reset();
		void AddFrame(Frame* pF);
		void RemoveFrame(Frame* pF);
		std::vector<Frame*> GetFrames();

		Frame* GetLastWindowFrame();
		Frame* GetReverseWindowFrame(int idx);
		Frame* AddWindowFrame(Frame* pF);
		//level = 1�̸� ù��° ������ť, 2�̸� �ι�° ������ ť ����, 3�̸� ����° ���� ť ����
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
		////Dense Flow ����
	public:
		void AddFlow(int nFrameID, cv::Mat flow);
		cv::Mat GetFlow(int nFrameID);
		std::vector<cv::Mat> GetFlows(int nStartID, int nEndID);
	private:
		std::mutex mMutexFlows;
		std::map<int, cv::Mat> mmFlows;
		////Dense Flow ����

		////������Ʈ ����
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
		////������Ʈ ����
		////������ �� ����
		
////������ �� ����

////��� ����
	public:
		void AddPlaneInfo(PlaneProcessInformation* pPlaneInfo);
		std::vector<PlaneProcessInformation*> GetPlaneInfos();
	private:
		std::mutex mMutexPlaneInfo;
		std::vector<PlaneProcessInformation*> mvpPlaneInfos;
		////��� ����
		////////////////////////////

		////////////////////////////////
		////��� ����
		//������� ������ ���� �� ����
		//�ٴ� ����� ���� �Ѱ���.
	public:
		bool isFloorPlaneInitialized();
		void SetFloorPlaneInitialization(bool b);
		bool isWallPlaneInitialized();
		void SetWallPlaneInitialization(bool b);
		void ClearWalls();
		//�߰� & ȹ��
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
		////��� ����
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


