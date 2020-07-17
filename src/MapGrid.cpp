#include <MapGrid.h>
#include <Frame.h>
#include <MapPoint.h>
#include <Frame.h>


//////////////////////////////////////////////
////생성자
UVR_SLAM::MapGrid::MapGrid(){}
UVR_SLAM::MapGrid::MapGrid(cv::Point3f init, float size):mInitPt(init), mfSize(size){
	Xw = cv::Mat::zeros(3, 1, CV_32FC1);
	Xw.at<float>(0) = init.x;
	Xw.at<float>(1) = init.y;
	Xw.at<float>(2) = init.z;
}
UVR_SLAM::MapGrid::~MapGrid(){}
////생성자
/////////접근 함수
void UVR_SLAM::MapGrid::InsertMapPoint(MapPoint* pMP){
	//KF 연결
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	auto findres = mspMPs.find(pMP);
	if (findres == mspMPs.end())
	{
		mspMPs.insert(pMP);
		auto tempKFs = pMP->GetConnedtedFrames();
		auto mmpMP = pMP->GetConnedtedFrames();
		for (auto biter = mmpMP.begin(), eiter = mmpMP.end(); biter != eiter; biter++) {
			auto pMatchInfo = biter->first;
			auto pkF = pMatchInfo->mpRefFrame;
			mmCountKeyframes[pkF]++;
		}
	}
}
void UVR_SLAM::MapGrid::RemoveMapPoint(MapPoint* pMP){
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	auto findres = mspMPs.find(pMP);
	if (findres != mspMPs.end())
	{
		auto tempKFs = pMP->GetConnedtedFrames();
		auto mmpMP = pMP->GetConnedtedFrames();
		for (auto biter = mmpMP.begin(), eiter = mmpMP.end(); biter != eiter; biter++) {
			auto pMatchInfo = biter->first;
			auto pkF = pMatchInfo->mpRefFrame;
			mmCountKeyframes[pkF]--;
		}
		mspMPs.erase(pMP);
	}
}
int UVR_SLAM::MapGrid::Count(){
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	return mspMPs.size();
}
std::vector<UVR_SLAM::MapPoint*> UVR_SLAM::MapGrid::GetMPs(){
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	return std::vector<MapPoint*>(mspMPs.begin(), mspMPs.end());
}
std::map<UVR_SLAM::Frame*, int> UVR_SLAM::MapGrid::GetKFs() {
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	return std::map<UVR_SLAM::Frame*, int>(mmCountKeyframes.begin(), mmCountKeyframes.end());
}
/////////접근 함수
//////////////////////////////////////////////
