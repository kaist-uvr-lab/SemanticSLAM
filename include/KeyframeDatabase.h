#ifndef KEYFRAME_DATABASE_H
#define KEYFRAME_DATABASE_H
#pragma once

#include <DBoW3.h>
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
		KeyframeDatabase(System* pSys, DBoW3::Vocabulary* voc);
		virtual ~KeyframeDatabase();
	public:
		void Init();
		void Add(Frame* pKF);
		void Remove(Frame* pKF);
		void Reset();

		std::vector<Frame*> DetectPlaceCandidates(Frame* pKF);
		std::vector<Frame*> DetectLoopCandidates(Frame* pKF, float minScore);

	private:
		DBoW3::Vocabulary* mpVocabulary;
		System* mpSystem;
		Map* mpMap;
		std::vector<std::list<Frame*>> mvInvertedFile;
		std::mutex mMutex;
	};
}
#endif
