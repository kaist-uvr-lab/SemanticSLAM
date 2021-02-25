#ifndef UVR_SLAM_LOOP_CLOSER_H
#define UVR_SLAM_LOOP_CLOSER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include "g2o/types/types_seven_dof_expmap.h"
#include "DBoW3.h"

namespace UVR_SLAM {
	class Frame;
	class System;
	class Map;
	class MapPoint;
	class KeyframeDatabase;
	class Matcher;

	class LoopCloser {
	public:
		typedef std::pair<std::set<Frame*>, int> ConsistentGroup;
		typedef std::map<Frame*, g2o::Sim3, std::less<Frame*>,
			Eigen::aligned_allocator<std::pair<const Frame*, g2o::Sim3> > > KeyFrameAndPose;

	public:
		LoopCloser();
		LoopCloser(System* pSys);
		virtual ~LoopCloser();
		
		void Init();
		void SetBoolProcessing(bool b);
		bool isProcessing();
		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();
		void Run();
		void RunWithMappingServer();
		bool DetectLoop();
		bool ComputeSim3();
		void CorrectLoop();
		void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);
		void ConstructBowDB(std::vector<Frame*> vpFrames);
		void LoadMapData(std::string map);
	public:
		int mLastLoopKFid;
		bool mbLoadData;
		std::string mapName;
	private:
		std::queue<UVR_SLAM::Frame*> mKFQueue;
		
		// Loop detector parameters
		float mnCovisibilityConsistencyTh;

		// Loop detector variables
		Frame* mpTargetFrame;
		Frame* mpMatchedKF;
		std::vector<ConsistentGroup> mvConsistentGroups;
		std::vector<Frame*> mvpEnoughConsistentCandidates;
		std::vector<Frame*> mvpCurrentConnectedKFs;
		std::vector<MapPoint*> mvpCurrentMatchedPoints;
		std::vector<MapPoint*> mvpLoopMapPoints;
		cv::Mat mScw;
		g2o::Sim3 mg2oScw;

	private:
		std::mutex mMutexNewKFs, mMutexLoopClosing, mMutexProcessing;
		bool mbFixScale;//monocular이면 false, 스테레오와 뎁스카메라는 픽스
		int mnWidth;
		int mnHeight;
		cv::Mat mK, mInvK;
		System* mpSystem;
		Map* mpMap;
		KeyframeDatabase* mpKeyFrameDatabase;
		DBoW3::Vocabulary* mpVoc;
		Matcher* mpMatcher;
		bool mbProcessing;
		float mfTime;
	};
}
#endif