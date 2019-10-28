#ifndef UVR_SLAM_LOCAL_MAP_H
#define UVR_SLAM_LOCAL_MAP_H
#pragma once

#include <deque>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace UVR_SLAM {
	class Frame;
	class FrameWindow;
	class LocalMap{
	public:
		LocalMap();
		LocalMap(int _size);
		virtual ~LocalMap();
	public:
		
	public:
	private:
		
	};
}

#endif

