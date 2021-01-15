#include <MapGrid.h>
#include <Frame.h>
#include <MapPoint.h>
#include <Frame.h>


//////////////////////////////////////////////
////생성자
namespace UVR_SLAM {
	//0.5
	float MapGrid::mfGridSizeX = 0.2f;
	float MapGrid::mfGridSizeY = 0.2f;
	float MapGrid::mfGridSizeZ = 0.2f;

	float fHalfX = MapGrid::mfGridSizeX * 0.5f;
	float fHalfY = MapGrid::mfGridSizeY * 0.5f;
	float fHalfZ = MapGrid::mfGridSizeZ * 0.5f;

	cv::RNG rng(12345);

	MapGrid::MapGrid() :mnTrackingID(-1), mnMapGridID(++System::nMapGridID){
		mGridColor = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	}

	MapGrid::~MapGrid(){}

	cv::Point3f MapGrid::ComputeKey(cv::Mat cam_pos) {
		int a = cam_pos.at<float>(0) / mfGridSizeX;
		int b = cam_pos.at<float>(1) / mfGridSizeY;
		int c = cam_pos.at<float>(2) / mfGridSizeZ;
		float nx = a*mfGridSizeX;
		float ny = b*mfGridSizeY;
		float nz = c*mfGridSizeZ;
		return std::move(cv::Point3f(nx, ny, nz));
	}

	cv::Point3f MapGrid::ComputeKey(cv::Mat cam_pos, float& x, float& y, float& z){
		int a = cam_pos.at<float>(0) / mfGridSizeX;
		int b = cam_pos.at<float>(1) / mfGridSizeY;
		int c = cam_pos.at<float>(2) / mfGridSizeZ;
		float nx = a*mfGridSizeX;
		float ny = b*mfGridSizeY;
		float nz = c*mfGridSizeZ;
		float dx = cam_pos.at<float>(0) - nx;
		float dy = cam_pos.at<float>(1) - ny;
		float dz = cam_pos.at<float>(2) - nz;

		bool bx = dx > fHalfX;
		bool by = dy > fHalfY;
		bool bz = dz > fHalfZ;

		if (dx > fHalfX) {
			x = mfGridSizeX;
		}
		else {
			x = -mfGridSizeX;
		}

		if (dy > fHalfY) {
			y = mfGridSizeY;
		}
		else {
			y = -mfGridSizeY;
		}

		if (dz > fHalfZ) {
			z = mfGridSizeZ;
		}
		else {
			z = -mfGridSizeZ;
		}

		return std::move(cv::Point3f(nx, ny, nz));
	}

	void MapGrid::AddKeyFrame(Frame* pF){
		std::unique_lock<std::mutex> lock(mMutexKeyFrames);
		mvpKeyFrames.push_back(pF);
	}
	std::vector<Frame*> MapGrid::GetKeyFrames(){
		std::vector<Frame*> res;
		{
			std::unique_lock<std::mutex> lock(mMutexKeyFrames);
			res = mvpKeyFrames;
		}
		return std::vector<Frame*>(res.begin(), res.end());
	}
	void MapGrid::AddMapPoint(MapPoint* pMP) {
		std::unique_lock<std::mutex> lock(mMutexMapPoints);
		mspMapPoints.insert(pMP);
		pMP->SetMapGridID(this->mnMapGridID);
		//pMP->mnMapGridID = this->mnMapGridID;
	}
	std::vector<MapPoint*> MapGrid::GetMapPoints() {
		std::set<MapPoint*> res;
		{
			std::unique_lock<std::mutex> lock(mMutexMapPoints);
			res = mspMapPoints;
		}
		return std::vector<MapPoint*>(res.begin(), res.end());
	}
	void MapGrid::RemoveMapPoint(MapPoint* pMP){
		std::unique_lock<std::mutex> lock(mMutexMapPoints);
		if (mspMapPoints.count(pMP)) {
			mspMapPoints.erase(pMP);
		}
	}
}


//UVR_SLAM::MapGrid::MapGrid(cv::Point3f init, float size):mInitPt(init), mfSize(size){
//	Xw = cv::Mat::zeros(3, 1, CV_32FC1);
//	Xw.at<float>(0) = init.x;
//	Xw.at<float>(1) = init.y;
//	Xw.at<float>(2) = init.z;
//}
////생성자
///////////접근 함수
//void UVR_SLAM::MapGrid::InsertMapPoint(MapPoint* pMP){
//	//KF 연결
//	std::unique_lock<std::mutex> lock(mMutexMapGrid);
//	auto findres = mspMPs.find(pMP);
//	if (findres == mspMPs.end())
//	{
//		mspMPs.insert(pMP);
//		auto tempKFs = pMP->GetConnedtedFrames();
//		auto mmpMP = pMP->GetConnedtedFrames();
//		for (auto biter = mmpMP.begin(), eiter = mmpMP.end(); biter != eiter; biter++) {
//			auto pMatchInfo = biter->first;
//			auto pkF = pMatchInfo->mpRefFrame;
//			mmCountKeyframes[pkF]++;
//		}
//	}
//}
//void UVR_SLAM::MapGrid::RemoveMapPoint(MapPoint* pMP){
//	std::unique_lock<std::mutex> lock(mMutexMapGrid);
//	auto findres = mspMPs.find(pMP);
//	if (findres != mspMPs.end())
//	{
//		auto tempKFs = pMP->GetConnedtedFrames();
//		auto mmpMP = pMP->GetConnedtedFrames();
//		for (auto biter = mmpMP.begin(), eiter = mmpMP.end(); biter != eiter; biter++) {
//			auto pMatchInfo = biter->first;
//			auto pkF = pMatchInfo->mpRefFrame;
//			mmCountKeyframes[pkF]--;
//		}
//		mspMPs.erase(pMP);
//	}
//}
//int UVR_SLAM::MapGrid::Count(){
//	std::unique_lock<std::mutex> lock(mMutexMapGrid);
//	return mspMPs.size();
//}
//std::vector<UVR_SLAM::MapPoint*> UVR_SLAM::MapGrid::GetMPs(){
//	std::unique_lock<std::mutex> lock(mMutexMapGrid);
//	return std::vector<MapPoint*>(mspMPs.begin(), mspMPs.end());
//}
//std::map<UVR_SLAM::Frame*, int> UVR_SLAM::MapGrid::GetKFs() {
//	std::unique_lock<std::mutex> lock(mMutexMapGrid);
//	return std::map<UVR_SLAM::Frame*, int>(mmCountKeyframes.begin(), mmCountKeyframes.end());
//}
///////////접근 함수
////////////////////////////////////////////////
