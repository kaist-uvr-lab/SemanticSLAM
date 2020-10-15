#ifndef KEYFRAME_DATABASE_H
#define KEYFRAME_DATABASE_H
#pragma once

#include "fbow.h"
#include <list>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace UVR_SLAM {
	class System;
	class Frame;
	class Map;
	class KeyframeDatabase {

	public:
		KeyframeDatabase(System* pSys, fbow::Vocabulary* voc, cv::Mat words);
		virtual ~KeyframeDatabase();
	public:
		void Init();
		void Add(Frame* pKF);
		void Remove(Frame* pKF);
		void Reset();

		std::vector<Frame*> DetectLoopCandidates(Frame* pKF, float minScore);

	private:
		fbow::Vocabulary* fvoc;
		System* mpSystem;
		Map* mpMap;
		cv::Mat mBowWords;
		std::vector<std::list<Frame*>> mvInvertedFile;
		std::mutex mMutex;
	};
}
#endif
