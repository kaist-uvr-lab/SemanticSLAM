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
		size_t size();
		void clear();
		std::vector<Frame*> GetAllFrames();
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
		void SetPose(cv::Mat _R, cv::Mat _t);
		void GetPose(cv::Mat &_R, cv::Mat& _t);
		cv::Mat GetRotation();
		cv::Mat GetTranslation();
		bool CalcFrameDistanceWithBOW(UVR_SLAM::Frame* pF);
		UVR_SLAM::MapPoint* GetMapPoint(int idx);
		void SetMapPoint(MapPoint* pMP, int idx);
		void SetBoolInlier(bool b,int idx);
		bool GetBoolInlier(int idx);
		void SetVectorInlier(int size, bool b);
		void SetLocalMap();
		void IncrementFrameCount();
		void SetFrameCount(int nCount);
		int GetFrameCount();

		void SetLastSemanticFrameIndex();
		int GetLastSemanticFrameIndex();
		//void AddFrame(UVR_SLAM::Frame* pFrame);
		//void RemoveFrame(UVR_SLAM::Frame* pFrame);
	public:
		int LocalMapSize;
		//std::vector<cv::DMatch> mvMatchingInfo;
		//std::vector<
		std::vector<std::pair<cv::DMatch, bool>> mvPairMatchingInfo;
		cv::Mat descLocalMap;
	private:
		
	private:

		int mnFrameCount;
		std::mutex mMutexFrameCount;
		int mnLastSemanticFrame; 
		
		System* mpSystem;
		std::mutex mMutexPose;
		std::mutex mMutexDeque;
		std::mutex mMutexLocaMPs;
		std::mutex mMutexBoolInliers;
		cv::Mat R, t;
		int mnWindowSize;
		std::deque<Frame*> mpDeque;



		std::vector<UVR_SLAM::MapPoint*> mvpLocalMPs;
		std::set<UVR_SLAM::MapPoint*>    mspLocalMPs; //local map�� ������ �� �̿�
		std::vector<bool> mvbLocalMPInliers;

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

