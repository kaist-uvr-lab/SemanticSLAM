#ifndef UVR_SLAM_MAP_GRID_H
#define UVR_SLAM_MAP_GRID_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class MatchInfo;
	class Frame;
	class MapPoint;

	class MapGrid {
	public:
		MapGrid();
		MapGrid(cv::Point3f init, float size);
		virtual ~MapGrid();
	public:
		float mfSize;
		cv::Mat Xw;
		cv::Point3f mInitPt;
		
		std::vector<Frame*> mvKFs;
		void InsertMapPoint(MapPoint*);
		void RemoveMapPoint(MapPoint*);
		int Count();
		std::vector<UVR_SLAM::MapPoint*> GetMPs();
		std::map<UVR_SLAM::Frame*, int> GetKFs();
	private:
		std::mutex mMutexMapGrid;
		std::set<MapPoint*> mspMPs;
		std::map<UVR_SLAM::Frame*, int> mmCountKeyframes;
	};
	
	//////스태틱으로 다루기.
	//class MapGridProcessor {
	//	//MP->Map Grid Projection
	//public:
	//	
	//public:
	//private:
	//};
}
#endif