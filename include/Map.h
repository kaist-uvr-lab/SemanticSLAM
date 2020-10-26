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
	class Map {
		///////////
	public:
		std::vector<cv::Mat> GetReinit();
		void ClearReinit();
		void AddReinit(cv::Mat m);
	private:
		std::mutex mMutexReinit;
		std::vector<cv::Mat> mvReinit;
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
		std::mutex mMutexMapUdpate;
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


