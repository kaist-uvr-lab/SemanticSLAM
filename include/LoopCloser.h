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
	class KeyframeDatabase;
	class Matcher;

	class LoopCloser {
	public:
		typedef std::pair<std::set<Frame*>, int> ConsistentGroup;
		typedef std::map<Frame*, g2o::Sim3, std::less<Frame*>,
			Eigen::aligned_allocator<std::pair<const Frame*, g2o::Sim3> > > KeyFrameAndPose;

	public:
		LoopCloser();
		LoopCloser(System* pSys, int w, int h, cv::Mat K);
		virtual ~LoopCloser();

		void Init();
		void SetBoolProcessing(bool b);
		bool isProcessing();
		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();
		void Run();

		bool DetectLoop();
		bool ComputeSim3();
		void CorrectLoop();
		void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

	public:
		int mLastLoopKFid;
	private:
		std::queue<UVR_SLAM::Frame*> mKFQueue;
		
	private:
		std::mutex mMutexNewKFs, mMutexLoopClosing, mMutexProcessing;
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
		Frame* mpTargetFrame;
	};
}
#endif