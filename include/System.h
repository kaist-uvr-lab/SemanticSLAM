
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
	class LoopCloser;
	class Visualizer;
	class FrameVisualizer;
	class MapOptimizer;
	class Map;
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
		void SaveTrajectory();
		void SetBoolInit(bool b);
		bool isInitialized();
		void Reset();
		void SetCurrFrame(cv::Mat img, double t);
		void Track();
		void InitDirPath();
		void SetDirPath(int id = 0);
		std::string GetDirPath(int id = 0);
		//std::list<UVR_SLAM::MapPoint*> GetListNewMPs();
		//void AddNewMP(MapPoint* pMP);

	private:
		ORBextractor* mpInitORBExtractor;
		ORBextractor* mpPoseORBExtractor;
		
		Frame* mpCurrFrame;
		Frame* mpPrevFrame;
		//Frame* mpInitFrame;
		
		Optimization *mpOptimizer;
		Matcher* mpMatcher;
		Initializer* mpInitializer;

		Map* mpMap;

		SemanticSegmentator* mpSegmentator;
		std::thread *mptLayoutEstimator;

		LocalMapper* mpLocalMapper;
		std::thread *mptLocalMapper;

		LoopCloser* mpLoopCloser;
		std::thread *mptLoopCloser;

		PlaneEstimator* mpPlaneEstimator;
		std::thread *mptPlaneEstimator;

		Visualizer* mpVisualizer;
		std::thread* mptVisualizer;

		FrameVisualizer* mpFrameVisualizer;
		std::thread* mptFrameVisualizer;

		MapOptimizer* mpMapOptimizer;
		std::thread* mptMapOptimizer;

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
		int mnMaxConnectedKFs, mnMaxCandidateKFs;

		std::mutex mMutexDirPath;
		std::string mStrBasePath;
		std::string mStrDirPath;

	public:
		ORBextractor* mpORBExtractor;
		ConcurrentList<UVR_SLAM::MapPoint*> mlpNewMPs;
		std::string strVOCPath;
		fbow::Vocabulary* fvoc;
		FrameWindow* mpFrameWindow;
		cv::Mat mK, mKforPL, mD;
		bool mbInitialized;
		int mnPatchSize;
		int mnHalfWindowSize;
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

		std::mutex mMutexUseSegmentation;
		std::condition_variable cvUseSegmentation;
		bool mbSegmentationEnd;

		std::mutex mMutexUseLocalMapping;
		std::condition_variable cvUseLocalMapping;
		bool mbLocalMappingEnd;

		std::mutex mMutexInitialized;

		//layoutestimation
		//1 max depth
		//2 create planar points
		std::mutex mMutexUsePlaneEstimation, mMutexUsePlanarMP;
		std::condition_variable cvUsePlaneEstimation, cvUsePlanarMP;
		bool mbPlaneEstimationEnd, mbPlanarMPEnd;

//time 계산 관련해서 함수 만들기.
	public:
		
	private:
		std::mutex mMutexLocalMappingTime;
		float mfLocalMappingTime1, mfLocalMappingTime2;
		
		std::mutex mMutexSegID, mMutexLMID, mMutexPlaneID, mMutexMapOptimizerID;
		int mnSegID, mnLoalMapperID, mnPlaneID, mnMapOptimizerID;

		//std::mutex mMutexLayoutTime;
		//float mfLayoutTime;
	//time에서 출력하는 스트링으로 변환
	public:
		void SetPlaneString(std::string str);
		std::string GetPlaneString();

		void SetTrackerString(std::string str);
		std::string GetTrackerString();

		void SetLocalMapperString(std::string str);
		std::string GetLocalMapperString();

		void SetSegmentationString(std::string str);
		std::string GetSegmentationString();

		void SetMapOptimizerString(std::string str);
		std::string GetMapOptimizerString();

	private:
		std::mutex mMutexPlaneString;
		std::string mStrPlaneString;
		std::mutex mMutexTrackerString;
		std::string mStrTrackerString;
		std::mutex mMutexSegmentationString;
		std::string mStrSegmentationString;
		std::mutex mMutexLocalMapperString;
		std::string mStrLocalMapperString;
		std::mutex mMutexMapOptimizer;
		std::string mStrMapOptimizer;
	////////////////////////////////////////////////////////////////////////
	
	};
}

#endif

