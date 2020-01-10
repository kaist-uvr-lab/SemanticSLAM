#ifndef UVR_SLAM_FRAME_WINDOW_H
#define UVR_SLAM_FRAME_WINDOW_H
#pragma once

#include <deque>
#include <queue>
#include <mutex>
#include <Frame.h>

namespace UVR_SLAM {
	class MapPoint;
	class System;
	class FrameWindow{
	public:
		FrameWindow();
		FrameWindow(int _size);
		virtual ~FrameWindow();
	public:
		void SetSystem(System* pSystem);
	public:
		//deque에서 list or vector or set으로 변경하기
		size_t size();
		void clear();
		std::vector<Frame*> GetAllFrames();
		std::set<UVR_SLAM::Frame*> GetAllFrameSets();
		std::deque<Frame*>::iterator GetBeginIterator();
		std::deque<Frame*>::iterator GetEndIterator();
		bool isEmpty();
		void push_front(Frame* pFrame);
		void push_back(Frame* pFrame);
		void pop_front();
		void pop_back();
		Frame* front();
		Frame* back();
		Frame* GetFrame(int idx);
	public:
		//local map
		void SetLocalMapInliers(std::vector<bool> vInliers);
		std::vector<bool> GetLocalMapInliers();
	private:
		std::vector<bool> mvbLocalMaPInliers;
		std::mutex mMutexLocalMapInliers;
	public:
		void SetPose(cv::Mat _R, cv::Mat _t);
		void GetPose(cv::Mat &_R, cv::Mat& _t);
		cv::Mat GetRotation();
		cv::Mat GetTranslation();
		bool CalcFrameDistanceWithBOW(Frame* pF);
		MapPoint* GetMapPoint(int idx);
		void AddMapPoint(MapPoint* pMP);
		void SetMapPoint(MapPoint* pMP, int idx);
		int GetLocalMapSize();
		//void SetBoolInlier(bool b,int idx);
		//bool GetBoolInlier(int idx);
		//void SetVectorInlier(int size, bool b);
		int TrackedMapPoints(int minObservation);
		std::vector<MapPoint*> GetLocalMap();
		void SetLocalMap(int nTargetID);
		cv::Mat GetLocalMapDescriptor();

		void SetLastFrameID(int id);
		int  GetLastFrameID();

		void SetLastSemanticFrameIndex();
		int GetLastSemanticFrameIndex();

		//void AddFrame(UVR_SLAM::Frame* pFrame);
		//void RemoveFrame(UVR_SLAM::Frame* pFrame);
	public:
		
		//std::vector<cv::DMatch> mvMatchingInfo;
		//std::vector<
		//여기서 여기까지 삭제 예정
		int mnLastMatches;
		//std::vector<std::pair<cv::DMatch, bool>> mvPairMatchingInfo; //타겟 프레임과 로컬 맵 사이의 매칭 정보를 기록.
		//여기서 여기까지 삭제 예정
		std::vector<cv::DMatch> mvMatchInfos; //create mp시 두 프레임 사이의 매칭 정보를 기록. query가 최근 키프레임, train이 이전 키프레임
	public:
		void AddFrame(Frame* pF);
		void ClearLocalMapFrames();
		std::vector<Frame*> GetLocalMapFrames();
		double mdFuseTime;
		void SetFuseTime(double d);
		double GetFuseTime();
		std::mutex mMutexFuseTime;

	private:
		//deque에서 list로 변경
		std::list<Frame*> mlpFrames;

	private:
		int LocalMapSize;

		int mnLastSemanticFrame; 
		int mnLastLayoutFrame; //윈도우 내에서 마지막 레이아웃 프레임의 인덱스를 나타냄. 아직 사용 안함.

		int mnLastFrameID;
		std::mutex mMutexLastFrameID;

		System* mpSystem;
		std::mutex mMutexPose;
		std::mutex mMutexDeque;
		std::mutex mMutexLocaMPs;
		
		cv::Mat R, t;
		int mnWindowSize;
		std::deque<Frame*> mpDeque;
		cv::Mat descLocalMap;
		std::vector<UVR_SLAM::MapPoint*> mvpLocalMPs;
		//std::set<UVR_SLAM::MapPoint*>    mspLocalMPs; //local map을 구성할 때 이용
		

		//포즈 그래프 최적화를 위한 Queue와 관련된 자료들
	public:
		int GetQueueSize();
		Frame* GetQueueLastFrame();
	private:
		std::queue<Frame*> mpQueue; //Window에서 나온 프레임을 추가함. 포즈 그래프 옵티마이제이션에 이용. 나중에 별도의 클래스에 빼낼 것임.
		int mnQueueSize;
	};
}

#endif

