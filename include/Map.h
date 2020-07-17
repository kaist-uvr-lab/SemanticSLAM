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
	class MapGrid;
	class Map {
	public:
		//keyframes
		//전체 맵포인트
		//바닥
		//벽을 가져야 한다.
		//바닥과 벽의 초기화 여부를 알아야 한다.
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
////맵포인트 관리
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
////맵포인트 관리
////구역별 맵 저장
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
	public: //함수
		cv::Point3f ProjectMapPoint(UVR_SLAM::MapPoint* pMP, float fSize);
		//맵포인트는 추가, 삭제, 비교를 위한 
		void InsertMapPoint(UVR_SLAM::MapPoint* pMP, UVR_SLAM::MapGrid* pMG);
		void DeleteMapPoint(UVR_SLAM::MapPoint* pMP);
		void UpdateMapPoint(UVR_SLAM::MapPoint* pMP, UVR_SLAM::MapGrid* pMG);


		bool CheckGrid(cv::Point3f pt);
		bool CheckGrid(cv::Point3f pt1, cv::Point3f pt2);
		UVR_SLAM::MapGrid* GetGrid(UVR_SLAM::MapPoint* pMP);
		UVR_SLAM::MapGrid* GetGrid(cv::Point3f pt);
		UVR_SLAM::MapGrid* InsertGrid(cv::Point3f pt); //추가된 그리드 인덱스 리턴
		std::vector<MapGrid*> GetMapGrids();
	public: //변수
		float mfMapGridSize;
	private:
		std::mutex mMutexMapGrid;
		std::map<cv::Point3f, UVR_SLAM::MapGrid*, Point3fLess> mmMapGrids;//int1 : subspace idx, int2 : 연결된 vv의 index, 해당 위치에 서브스페이스가 생성되었는지 확인
		std::map<UVR_SLAM::MapPoint*, UVR_SLAM::MapGrid*> mmMapPointAndMapGrids; //map point가 어떠한 그리드에 포함되어 있는지 미리 확인함.
		//std::vector<MapGrid*> mvMapGrids; //이 부분은 일단 제거
////구역별 맵 저장
////////////////////////////////
/////////trajectory 출력용
	public:
		void AddTraFrame(Frame* pF);
		std::vector<Frame*> GetAllTrajectoryFrames();
	private:
		std::vector<Frame*> mvpAllTrajectoryFrames;
		std::mutex mMutexAllFrames;
/////////trajectory 출력용
////////////////////////////
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

		Frame* mpCurrFrame;
		std::mutex mMutexCurrFrame;
	public:
		Frame* mpFirstKeyFrame;
		UVR_SLAM::PlaneInformation* mpFloorPlane;
////평면 관리
////////////////////////////////
////////////////////////////////
/////Loop를 위한 것 일단 보류
	public:
		std::vector<UVR_SLAM::Frame*> GetLoopFrames();
	private:
		std::vector<Frame*> mvpLoopFrames;
		std::mutex mMutexLoopFrames;
	};
/////Loop를 위한 것 일단 보류
////////////////////////////////
	
}
#endif


