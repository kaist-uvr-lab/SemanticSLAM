#ifndef UVR_SLAM_MAP_H
#define UVR_SLAM_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class Frame;
	class MapPoint;
	class WallPlane;
	class PlaneInformation;
	class PlaneProcessInformation;

	class Map {
	public:
		//keyframes
		//��ü ������Ʈ
		//�ٴ�
		//���� ������ �Ѵ�.
		//�ٴڰ� ���� �ʱ�ȭ ���θ� �˾ƾ� �Ѵ�.
		Map();
		virtual ~Map();
	public:
		void AddFrame(Frame* pF);
		std::vector<Frame*> GetFrames();
		void SetCurrFrame(Frame* pF);
		Frame* GetCurrFrame();
		void ClearFrames();
		
	private:
		std::mutex mMutexGlobalFrames;
		std::vector<Frame*> mvpGlobalFrames;
////////////////////////////////
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
////////////////////////////////
/////////trajectory ��¿�
	public:
		void AddTraFrame(Frame* pF);
		std::vector<Frame*> GetAllTrajectoryFrames();
	private:
		std::vector<Frame*> mvpAllTrajectoryFrames;
		std::mutex mMutexAllFrames;
/////////trajectory ��¿�
////////////////////////////
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

		Frame* mpCurrFrame;
		std::mutex mMutexCurrFrame;
	public:
		Frame* mpFirstKeyFrame;
		UVR_SLAM::PlaneInformation* mpFloorPlane;
////��� ����
////////////////////////////////
////////////////////////////////
/////Loop�� ���� �� �ϴ� ����
	public:
		std::vector<UVR_SLAM::Frame*> GetLoopFrames();
	private:
		std::vector<Frame*> mvpLoopFrames;
		std::mutex mMutexLoopFrames;
	};
/////Loop�� ���� �� �ϴ� ����
////////////////////////////////
	
}
#endif


