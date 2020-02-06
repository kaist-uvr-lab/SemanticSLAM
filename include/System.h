
#ifndef UVR_SLAM_SYSTEM_H
#define UVR_SLAM_SYSTEM_H
#pragma once

#include <queue>
#include <thread>
#include <fbow.h>
#include <mutex>
#include <ConcurrentList.h>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/calib3d.hpp>

#include <Frame.h>
#include <ORBextractor.h>
#include <Optimization.h>
#include <Matcher.h>
#include <Tracker.h>


namespace fbow {
	class VocabularyCreator {
	public:

		static    cv::Mat getVocabularyLeafNodes(fbow::Vocabulary &voc) {

			//analyze all blocks and count  the leafs
			uint32_t nleafs = 0;
			for (uint32_t b = 0; b<voc._params._nblocks; b++) {
				fbow::Vocabulary::Block block = voc.getBlock(b);
				int nnodes = block.getN();
				for (int n = 0; n<nnodes; n++)
					if (block.getBlockNodeInfo(n)->isleaf()) nleafs++;
			}
			//reserve memory
			cv::Mat features;
			if (voc.getDescType() == CV_8UC1)
				features.create(nleafs, voc.getDescSize(), CV_8UC1);
			else
				features.create(nleafs, voc.getDescSize() / sizeof(float), CV_32FC1);
			//start copy data
			nleafs = 0;
			for (uint32_t b = 0; b<voc._params._nblocks; b++) {
				fbow::Vocabulary::Block block = voc.getBlock(b);
				int nnodes = block.getN();
				for (int n = 0; n<nnodes; n++)
					if (block.getBlockNodeInfo(n)->isleaf())  block.getFeature(n, features.row(nleafs++));
			}
			return features;
		}
	};
}

namespace UVR_SLAM {
	class Tracker;
	class FrameWindow;
	class SemanticSegmentator;
	class LocalMapper;
	class PlaneEstimator;
	class Visualizer;
	class System {
	public:

		System();
		System(std::string strFilePath);
		System(int nWidth, int nHeight, cv::Mat _K,cv::Mat _K2, cv::Mat _D, int _nFeatures, float _fScaleFactor, int _nLevels, int _fIniThFAST, int _fMinThFAST, std::string _strVOCPath);
		virtual ~System();
		void LoadParameter(std::string strPath);
		bool LoadVocabulary();
		void Init();

	public:
		void SetBoolInit(bool b);
		void Reset();
		void SetCurrFrame(cv::Mat img);
		void Track();
		void InitDirPath();
		void SetDirPath(int id);
		std::string GetDirPath(int id);
		//std::list<UVR_SLAM::MapPoint*> GetListNewMPs();
		//void AddNewMP(MapPoint* pMP);

	private:
		ORBextractor* mpInitORBExtractor;
		ORBextractor* mpPoseORBExtractor;
		ORBextractor* mpORBExtractor;

		Frame* mpCurrFrame;
		Frame* mpPrevFrame;
		//Frame* mpInitFrame;
		
		Optimization *mpOptimizer;
		Matcher* mpMatcher;
		Initializer* mpInitializer;

		SemanticSegmentator* mpSegmentator;
		std::thread *mptLayoutEstimator;

		LocalMapper* mpLocalMapper;
		std::thread *mptLocalMapper;

		PlaneEstimator* mpPlaneEstimator;
		std::thread *mptPlaneEstimator;

		Visualizer* mpVisualizer;
		std::thread* mptVisualizer;

		Tracker* mpTracker;
		//std::thread *mptTracker;
		
		std::string mstrFilePath;

		//management created map points
		//std::mutex mMutexListMPs;
		//std::list<UVR_SLAM::MapPoint*> mlpNewMPs;
	private:
		//외부에서 불러온 파라메터
		int mnFeatures;
		float mfScaleFactor;
		int mnLevels; 
		int mfIniThFAST;
		int mfMinThFAST;
		int mnWidth, mnHeight;

		std::mutex mMutexDirPath;
		std::string mStrBasePath;
		std::string mStrDirPath;

	public:
		
		ConcurrentList<UVR_SLAM::MapPoint*> mlpNewMPs;
		std::string strVOCPath;
		fbow::Vocabulary* fvoc;
		FrameWindow* mpFrameWindow;
		cv::Mat mK, mKforPL, mD;
		bool mbInitialized;
		int mnVisScale;
		std::string ip;
		int port;
		static int nKeyFrameID;

	public:
		//lock tracking and localmap
		std::mutex mMutexUseLocalMap;
		std::condition_variable cvUseLocalMap;
		bool mbTrackingEnd;
		bool mbLocalMapUpdateEnd;

//time 계산 관련해서 함수 만들기.
	public:
		void SetSegmentationTime(float t);
		float GetSegmentationTime();
		void SetLocalMappingTime(float t1, float t2);
		void GetLocalMappingTime(float& t1, float& t2);
		void SetLayoutTime(float t1);
		void GetLayoutTime(float& t1);
		void SetSegFrameID(int n);
		int GetSegFrameID();
		void SetLocalMapperFrameID(int n);
		int GetLocalMapperFrameID();
		void SetPlaneFrameID(int n);
		int GetPlaneFrameID();
	private:
		std::mutex mMutexSegmentationTime;
		float mfSegTime;
		std::mutex mMutexLocalMappingTime;
		float mfLocalMappingTime1, mfLocalMappingTime2;
		std::mutex mMutexLayoutTime;
		float mfLayoutTime;
		std::mutex mMutexSegID, mMutexLMID, mMutexPlaneID;
		int mnSegID, mnLoalMapperID, mnPlaneID;
	};
}

#endif

