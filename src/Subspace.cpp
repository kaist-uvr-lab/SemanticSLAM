#include <Subspace.h>
#include <Frame.h>
#include <MapPoint.h>

namespace UVR_SLAM {
	int Subspace::nSubspaceID = 0;
	Subspace::Subspace():mpStartFrame(nullptr), mpEndFrame(nullptr), mnID(++nSubspaceID), avgParam(cv::Mat::zeros(3,1,CV_32FC1)){
	}
	Subspace::~Subspace(){}

	cv::Mat Subspace::GetData() {
		cv::Mat data = cv::Mat::zeros(0, 3, CV_32FC1);
		for (auto it = mspSubspaceMapPoints.begin(), iend = mspSubspaceMapPoints.end(); it != iend; it++) {
			auto pMPi = *it;
			if (!pMPi || pMPi->isDeleted())
				continue;

		}
		return data.clone();
	}
	bool Subspace::CheckNeedNewSubspace(std::vector<MapPoint*> vpTempMPs) {

		////average

		////vector parameters
		cv::Mat lastParam = mvParams[mvParams.size() - 1];

		return false;
	}
	//parallel floor detection
}