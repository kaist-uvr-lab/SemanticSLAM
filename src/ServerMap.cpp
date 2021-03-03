#include <ServerMap.h>
#include <MapPoint.h>
#include <Frame.h>

namespace UVR_SLAM {
	
	ServerMap::ServerMap():mbInitialized(false), nServerMapPointID(0), nServerKeyFrameID(0){}
	ServerMap::~ServerMap(){}
	void ServerMap::SetInitialKeyFrame(UVR_SLAM::Frame* pKF1, UVR_SLAM::Frame* pKF2) {
		nServerKeyFrameID = 0;
		pKF1->mnKeyFrameID = nServerKeyFrameID++;
		pKF2->mnKeyFrameID = nServerKeyFrameID++;
		mpPrevKF = pKF1;
		mpCurrKF = pKF2;
	}
	void ServerMap::AddFrame(Frame* pF) {
		std::unique_lock<std::mutex> lock(mMutexKFs);
		if(!mspMapFrames.count(pF)){
			mspMapFrames.insert(pF);
			pF->mnKeyFrameID = nServerKeyFrameID++;
			mpPrevKF = mpCurrKF;
			mpCurrKF = pF;
		}
	}
	void ServerMap::RemoveFrame(Frame* pF){
		std::unique_lock<std::mutex> lock(mMutexKFs);
		if (mspMapFrames.count(pF))
			mspMapFrames.erase(pF);
	}
	std::vector<Frame*> ServerMap::GetFrames(){
		std::unique_lock<std::mutex> lock(mMutexKFs);
		return std::vector<Frame*>(mspMapFrames.begin(), mspMapFrames.end());
	}
	void ServerMap::AddMapPoint(MapPoint* pMP){
		std::unique_lock<std::mutex> lock(mMutexMPs);
		if (!mspMapMPs.count(pMP))
			mspMapMPs.insert(pMP);
	}
	void ServerMap::RemoveMapPoint(MapPoint* pMP){
		std::unique_lock<std::mutex> lock(mMutexMPs);
		if (mspMapMPs.count(pMP))
			mspMapMPs.erase(pMP);
	}
	std::vector<MapPoint*> ServerMap::GetMapPoints(){
		std::unique_lock<std::mutex> lock(mMutexMPs);
		return std::vector<MapPoint*>(mspMapMPs.begin(), mspMapMPs.end());
	}

}