
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

	struct Point2fLess
	{
		bool operator()(cv::Point2f const&lhs, cv::Point2f const& rhs) const
		{
			return lhs.x == rhs.x ? lhs.y < rhs.y : lhs.x < rhs.x;
		}
	};
	struct Point3fLess
	{
		bool operator()(cv::Point3f const&lhs, cv::Point3f const& rhs) const
		{
			return lhs.x == rhs.x ? lhs.y == rhs.y ? lhs.z < rhs.z : lhs.y < rhs.y : lhs.x < rhs.x;
		}
	};

	class Initializer;
	class Optimization;
	class Tracker;
	class Matcher;
	class ORBextractor;
	class MapPoint;
	class Frame;
	class SemanticSegmentator;
	class LocalMapper;
	class PlaneEstimator;
	class LoopCloser;
	class DepthFilter;
	class Visualizer;
	class FrameVisualizer;
	class MapOptimizer;
	class Map;
	class Database;
	class LocalBinaryPatternProcessor;
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
		Optimization *mpOptimizer;
		Matcher* mpMatcher;
		Initializer* mpInitializer;
		Map* mpMap;
		LocalMapper* mpLocalMapper;
		DepthFilter* mpDepthFilter;
		MapOptimizer* mpMapOptimizer;
		LoopCloser* mpLoopCloser;
		SemanticSegmentator* mpSegmentator;
		PlaneEstimator* mpPlaneEstimator;
		Visualizer* mpVisualizer;
		FrameVisualizer* mpFrameVisualizer;
		LocalBinaryPatternProcessor* mpLBPProcessor;
		Database* mpDatabase;
	public:
		////parameter 파일 관련
		std::string mstrFilePath;
		bool mbSegmentation;
		bool mbPlaneEstimation;
		int mnDisplayX;
		int mnDisplayY;
		int mnThreshMinKF;
		int mnWidth, mnHeight;
		int mnMaxMP;
		int mnRadius;
		cv::Point2f mRectPt;
	public:
		void SaveTrajectory(std::string filename);
		void SetBoolInit(bool b);
		bool isInitialized();
		void Reset();
		void SetCurrFrame(cv::Mat img, double t);
		void Track();
		void InitDirPath();
		void SetDirPath(int id = 0);
		std::string GetDirPath(int id = 0);

	private:
		ORBextractor* mpInitORBExtractor;
		ORBextractor* mpPoseORBExtractor;
		
		Frame* mpCurrFrame;
		Frame* mpPrevFrame;
		//Frame* mpInitFrame;

		std::thread *mptLocalMapper;
		std::thread* mptMapOptimizer;
		std::thread *mptLoopCloser;
		std::thread *mptLayoutEstimator;
		std::thread *mptPlaneEstimator;
		std::thread* mptVisualizer;
		std::thread* mptFrameVisualizer;

		Tracker* mpTracker;
		
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
		int mnMaxConnectedKFs, mnMaxCandidateKFs;

		std::mutex mMutexDirPath;
		std::string mStrBasePath;
		std::string mStrDirPath;

	public:
		ORBextractor* mpORBExtractor;
		//ConcurrentList<UVR_SLAM::MapPoint*> mlpNewMPs;
		std::list<UVR_SLAM::MapPoint*> mlpNewMPs;
		std::string strVOCPath;
		fbow::Vocabulary* fvoc;
		cv::Mat mK, mInvK, mD;
		cv::Mat mKforPL;
		bool mbInitialized;
		int mnPatchSize;
		int mnHalfWindowSize;
		int mnVisScale;
		std::string ip;
		int port;
		static int nKeyFrameID;
		static int nFrameID;

	public:
		//lock tracking and localmap
		std::mutex mMutexUseLocalMapOptimization;
		std::condition_variable cvUseLocalMapOptimization;
		bool mbTrackingEnd;
		bool mbLocalMapOptimizationEnd;

		std::mutex mMutexUseSegmentation;
		std::condition_variable cvUseSegmentation;
		bool mbSegmentationEnd;

		std::mutex mMutexUseCreateCP;
		std::condition_variable cvUseCreateCP;
		bool mbCreateCP;

		std::mutex mMutexUseCreateMP;
		std::condition_variable cvUseCreateMP;
		bool mbCreateMP;

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
	
	//////TEST
	public:
		cv::Mat mPlaneHist;
		std::mutex mMutexPlaneHist;
	};
}

#endif

