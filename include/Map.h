#ifndef UVR_SLAM_MAP_H
#define UVR_SLAM_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class Frame;

	class WallPlane;
	class PlaneInformation;

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
		void ClearWalls();
	private:
		std::mutex mMutexGlobalFrames;
		std::vector<Frame*> mvpGlobalFrames;

////////////////////////////////
////평면 관리
//벽평면은 여러개 있을 수 있음
//바닥 평면은 오직 한개임.
	public:
		bool isFloorPlaneInitialized();
		void SetFloorPlaneInitialization(bool b);
		bool isWallPlaneInitialized();
		void SetWallPlaneInitialization(bool b);

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


