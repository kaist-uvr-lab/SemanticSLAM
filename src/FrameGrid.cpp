#include <FrameGrid.h>
#include <CandidatePoint.h>

namespace UVR_SLAM {
	FrameGrid::FrameGrid(){}
	FrameGrid::FrameGrid(cv::Point2f base, cv::Rect r):basePt(std::move(base)), rect(std::move(r)),mbMatched(false){
		
	}
	FrameGrid::~FrameGrid(){}
}