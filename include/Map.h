#ifndef UVR_SLAM_MAP_H
#define UVR_SLAM_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <list>

namespace UVR_SLAM {
	class Frame;
	class System;
	class MapPoint;
	class WallPlane;
	class PlaneInformation;
	class PlaneProcessInformation;
	class MapGrid;
	class Map {
		///////////
	public:
		void ClearReinit();
		std::vector<MapPoint*> GetReinit();
		void AddReinit(MapPoint* pMP);
		std::vector<cv::Mat> GetReinitParallax();
		void AddReinitParallax(cv::Mat m);
	private:
		std::mutex mMutexReinit;
		std::vector<cv::Mat> mvReinitParallax;
		std::vector<MapPoint*> mvReinit;
		/////////
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

		std::mutex mMutexWindowFrames;
		std::list<Frame*> mQueueFrameWindows1, mQueueFrameWindows2, mQueueFrameWindows3;
		std::list<Frame*> mQueueCandidateGraphFrames;
		std::set<Frame*> mspGraphFrames;
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
		struct Point2fLess
		{
			bool operator()(cv::Point2f const&lhs, cv::Point2f const& rhs) const
			{
				return lhs.x == rhs.x ? lhs.y < rhs.y : lhs.x < rhs.x;
			}
		};
		struct Point3fLess
		{
			bool operator()(cv::Point3f const&lhs, cv::Point3f const& rhs) const
			{
				return lhs.x == rhs.x ? lhs.y == rhs.y ? lhs.z < rhs.z : lhs.y < rhs.y : lhs.x < rhs.x;
			}
		};
	public: //�Լ�
		cv::Point3f ProjectMapPoint(UVR_SLAM::MapPoint* pMP, float fSize);
		//������Ʈ�� �߰�, ����, �񱳸� ���� 
		void InsertMapPoint(UVR_SLAM::MapPoint* pMP, UVR_SLAM::MapGrid* pMG);
		void DeleteMapPoint(UVR_SLAM::MapPoint* pMP);
		void UpdateMapPoint(UVR_SLAM::MapPoint* pMP, UVR_SLAM::MapGrid* pMG);

		bool CheckGrid(cv::Point3f pt);
		bool CheckGrid(cv::Point3f pt1, cv::Point3f pt2);
		UVR_SLAM::MapGrid* GetGrid(UVR_SLAM::MapPoint* pMP);
		UVR_SLAM::MapGrid* GetGrid(cv::Point3f pt);
		UVR_SLAM::MapGrid* InsertGrid(cv::Point3f pt); //�߰��� �׸��� �ε��� ����
		std::vector<MapGrid*> GetMapGrids();
	public: //����
		float mfMapGridSize;
	private:
		std::mutex mMutexMapGrid;
		std::map<cv::Point3f, UVR_SLAM::MapGrid*, Point3fLess> mmMapGrids;//int1 : subspace idx, int2 : ����� vv�� index, �ش� ��ġ�� ���꽺���̽��� �����Ǿ����� Ȯ��
		std::map<UVR_SLAM::MapPoint*, UVR_SLAM::MapGrid*> mmMapPointAndMapGrids; //map point�� ��� �׸��忡 ���ԵǾ� �ִ��� �̸� Ȯ����.
		//std::vector<MapGrid*> mvMapGrids; //�� �κ��� �ϴ� ����
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
	};
}
#endif


