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
		//deque���� list or vector or set���� �����ϱ�
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
		//���⼭ ������� ���� ����
		int mnLastMatches;
		//std::vector<std::pair<cv::DMatch, bool>> mvPairMatchingInfo; //Ÿ�� �����Ӱ� ���� �� ������ ��Ī ������ ���.
		//���⼭ ������� ���� ����
		std::vector<cv::DMatch> mvMatchInfos; //create mp�� �� ������ ������ ��Ī ������ ���. query�� �ֱ� Ű������, train�� ���� Ű������
	public:
		void AddFrame(Frame* pF);
		void ClearLocalMapFrames();
		std::vector<Frame*> GetLocalMapFrames();
		double mdFuseTime;
		void SetFuseTime(double d);
		double GetFuseTime();
		std::mutex mMutexFuseTime;

	private:
		//deque���� list�� ����
		std::list<Frame*> mlpFrames;

	private:
		int LocalMapSize;

		int mnLastSemanticFrame; 
		int mnLastLayoutFrame; //������ ������ ������ ���̾ƿ� �������� �ε����� ��Ÿ��. ���� ��� ����.

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
		//std::set<UVR_SLAM::MapPoint*>    mspLocalMPs; //local map�� ������ �� �̿�
		

		//���� �׷��� ����ȭ�� ���� Queue�� ���õ� �ڷ��
	public:
		int GetQueueSize();
		Frame* GetQueueLastFrame();
	private:
		std::queue<Frame*> mpQueue; //Window���� ���� �������� �߰���. ���� �׷��� ��Ƽ�������̼ǿ� �̿�. ���߿� ������ Ŭ������ ���� ����.
		int mnQueueSize;
	};
}

#endif

