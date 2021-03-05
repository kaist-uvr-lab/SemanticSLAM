#include <ServerMap.h>
#include <MapPoint.h>
#include <Frame.h>
#include <WebAPI.h>
#include <future>

namespace UVR_SLAM {
	ServerMap::ServerMap():mbInitialized(false), nServerMapPointID(0), nServerKeyFrameID(0), mbLoad(false){}
	ServerMap::ServerMap(System* pSys):mpSystem(pSys),mbInitialized(false), nServerMapPointID(0), nServerKeyFrameID(0), mbLoad(false){}
	ServerMap::~ServerMap(){}
	void ServerMap::SetInitialKeyFrame(UVR_SLAM::Frame* pKF1, UVR_SLAM::Frame* pKF2) {
		nServerKeyFrameID = 0;
		pKF1->mnKeyFrameID = nServerKeyFrameID++;
		pKF2->mnKeyFrameID = nServerKeyFrameID++;
		mpPrevKF = pKF1;
		mpCurrKF = pKF2;
	}
	void ServerMap::SetMapLoad(bool bLoad) {
		std::unique_lock<std::mutex> lock(mMutexMapLoad);
		mbLoad = bLoad;
	}
	bool ServerMap::GetMapLoad(){
		std::unique_lock<std::mutex> lock(mMutexMapLoad);
		return mbLoad;
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
	void ServerMap::LoadMapDataFromServer(std::string mapname, std::string ip, int port) {
		SetMapLoad(true);
		auto f1 = std::async(std::launch::async, [](std::string ip, int port, std::string mapname) {
			std::stringstream ss;
			ss << "/SendData?map=" << mapname << "&attr=MapPoints&id=-1&key=bmpids";
			WebAPI* wapi = new WebAPI(ip, port);
			auto resdata = wapi->Send(ss.str(), "");
			////贸府 饶 傈价
			int n = resdata.size() / 4;
			cv::Mat mpids = cv::Mat::zeros(1, resdata.size() / 4, CV_32SC1);
			std::memcpy(mpids.data, resdata.data(), n * sizeof(int));
			return mpids;
		}, ip, port, mapname);
		auto f2 = std::async(std::launch::async, [](std::string ip, int port, std::string mapname) {
			std::stringstream ss;
			ss << "/SendData?map=" << mapname << "&id=-1&key=bkfids";
			WebAPI* wapi = new WebAPI(ip, port);
			auto resdata = wapi->Send(ss.str(), "");
			////贸府 饶 傈价
			int n = resdata.size() / 4;
			cv::Mat mpids = cv::Mat::zeros(1, resdata.size() / 4, CV_32SC1);
			std::memcpy(mpids.data, resdata.data(), n * sizeof(int));
			return mpids;
		}, ip, port, mapname);
		auto f3 = std::async(std::launch::async, [](std::string ip, int port, std::string mapname) {
			std::stringstream ss;
			ss << "/SendData?map=" << mapname << "&id=-1&attr=MapPoints&key=bx3ds";
			WebAPI* wapi = new WebAPI(ip, port);
			auto resdata = wapi->Send(ss.str(), "");
			////贸府 饶 傈价
			int n = resdata.size() / 4;
			cv::Mat mpids = cv::Mat::zeros(resdata.size() / 4, 1, CV_32FC1);
			std::memcpy(mpids.data, resdata.data(), n * sizeof(int));
			return mpids;
		}, ip, port, mapname);
		
		//auto f4 = std::async(std::launch::async, [](std::string ip, int port, std::string mapname) {
		//	std::stringstream ss;
		//	ss << "/SendData?map=" << mapname << "&id=-1&key=bmpidxs";
		//	WebAPI* wapi = new WebAPI(ip, port);
		//	auto resdata = wapi->Send(ss.str(), "");
		//	////贸府 饶 傈价
		//	int n = resdata.size() / 4;
		//	cv::Mat mpids = cv::Mat::zeros(1, resdata.size() / 4, CV_32SC1);
		//	std::memcpy(mpids.data, resdata.data(), n * sizeof(int));
		//	return mpids;
		//}, ip, port, mapname);
		//auto f5 = std::async(std::launch::async, [](std::string ip, int port, std::string mapname) {
		//	std::stringstream ss;
		//	ss << "/SendData?map=" << mapname << "&id=-1&key=bposes";
		//	WebAPI* wapi = new WebAPI(ip, port);
		//	auto resdata = wapi->Send(ss.str(), "");
		//	////贸府 饶 傈价
		//	int n = resdata.size() / 4;
		//	cv::Mat mpids = cv::Mat::zeros(1, resdata.size() / 4, CV_32FC1);
		//	std::memcpy(mpids.data, resdata.data(), n * sizeof(int));
		//	return mpids;
		//}, ip, port, mapname);

		auto mpIDs = f1.get();
		auto kfIDs = f2.get();
		auto mpXws = f3.get();
		//auto kfPoses = f5.get();
		//auto kfMpIdxs = f4.get();

		std::map<int, MapPoint*> mmpMPs;
		std::map<int, cv::Mat> mmMapPointData;
		for (int i = 0, iend = mpIDs.cols; i < iend; i++) {
			int id = mpIDs.at<int>(i);
			cv::Mat Xw = mpXws.rowRange(3 * i, 3 * (i + 1));
			mmMapPointData[id] = Xw;
			/*auto pNewMP = new MapPoint(this, Xw, cv::Mat());
			mmpMPs[id] = pNewMP;*/
		}

		std::map<int, Frame*> mmpFrames;
		for (int i = 0, iend = kfIDs.cols, mpIndex = 0; i < iend; i++) {
			int id = kfIDs.at<int>(i);
			std::cout << "Frame ID = " << id << std::endl;
			float fx, fy, cx, cy;
			int w, h;
			[](std::string ip, int port, int id, std::string map, float& fx, float& fy, float& cx, float& cy, int& w, int& h) {
				WebAPI* mpAPI = new WebAPI(ip, port);
				std::stringstream ss;
				ss << "/SendData?map=" << map << "&id=" << id << "&key=binfo";
				auto res = mpAPI->Send(ss.str(), "");
				int n = res.size();
				cv::Mat temp = cv::Mat::zeros(n / sizeof(float), 1, CV_32FC1);
				std::memcpy(temp.data, res.data(), n * sizeof(char));
				
				fx = temp.at<float>(0);
				fy = temp.at<float>(1);
				cx = temp.at<float>(2);
				cy = temp.at<float>(3);
				w = (int)temp.at<float>(4);
				h = (int)temp.at<float>(5);
			}(ip, port, id, mapname, fx, fy, cx, cy, w, h);

			auto fimg = std::async(std::launch::async, [](std::string ip, int port, int id, std::string map, int w, int h) {
				WebAPI* mpAPI = new WebAPI(ip, port);
				std::stringstream ss;
				ss << "/SendData?map=" << map << "&id=" << id << "&key=bimage";
				auto res = mpAPI->Send(ss.str(), "");
				int n = res.size();
				cv::Mat temp = cv::Mat::zeros(n, 1, CV_8UC1);
				std::memcpy(temp.data, res.data(), n * sizeof(uchar));
				cv::Mat img = cv::imdecode(temp, cv::IMREAD_COLOR);
				return img;
			}, ip, port, id, mapname, w, h);

			auto fpts = std::async(std::launch::async, [](std::string ip, int port, int id, std::string map) {
				WebAPI* mpAPI = new WebAPI(ip, port);
				std::stringstream ss;
				ss << "/SendData?map=" << map << "&id=" << id << "&key=bkpts";
				auto res = mpAPI->Send(ss.str(), "");
				int n = res.size() / 8;
				cv::Mat seg = cv::Mat::zeros(n, 2, CV_32FC1);
				std::memcpy(seg.data, res.data(), n * 2 * sizeof(float));
				std::vector<cv::Point2f> resPts;
				for (int j = 0; j < n; j++) {
					cv::Point2f pt(seg.at<float>(2 * j), seg.at<float>(2 * j + 1));
					resPts.push_back(pt);
				}
				return resPts;
			}, ip, port, id, mapname);

			auto fdesc = std::async(std::launch::async, [](std::string ip, int port, int id, std::string map) {
				cv::Mat desc;
				WebAPI* mpAPI = new WebAPI(ip, port);
				std::stringstream ss;
				ss << "/SendData?map=" << map << "&id=" << id << "&key=bdesc";
				auto res = mpAPI->Send(ss.str(), "");
				int n = res.size() / (256 * 4);
				WebAPIDataConverter::ConvertBytesToDesc(res.c_str(), n, desc);
				return desc;

			}, ip, port, id, mapname);

			auto fpose = std::async(std::launch::async, [](std::string ip, int port, int id, std::string map) {
				WebAPI* mpAPI = new WebAPI(ip, port);
				std::stringstream ss;
				ss << "/SendData?map=" << map << "&id=" << id << "&key=bpose";
				auto res = mpAPI->Send(ss.str(), "");
				cv::Mat P = cv::Mat::zeros(4,3,CV_32FC1);
				std::memcpy(P.data, res.data(), 12 * sizeof(float));
				//WebAPIDataConverter::ConvertBytesToDesc(res.c_str(), n, desc);
				return P;

			}, ip, port, id, mapname);

			auto fmpidx = std::async(std::launch::async, [](std::string ip, int port, int id, std::string map) {
				WebAPI* mpAPI = new WebAPI(ip, port);
				std::stringstream ss;
				ss << "/SendData?map=" << map << "&id=" << id << "&key=bmpidx";
				auto res = mpAPI->Send(ss.str(), "");
				cv::Mat idx = cv::Mat::zeros(1, res.size()/4, CV_32SC1);
				std::memcpy(idx.data, res.data(), res.size() * sizeof(char));
				//WebAPIDataConverter::ConvertBytesToDesc(res.c_str(), n, desc);
				return idx;

			}, ip, port, id, mapname);

			auto vPTs = fpts.get();
			auto desc = fdesc.get();
			auto img = fimg.get();

			Frame* newFrame = new UVR_SLAM::Frame(mpSystem, id, fx, fy, cx, cy, w, h, 0.0);
			newFrame->mnFrameID = id;
			newFrame->SetMapPoints(vPTs.size());
			newFrame->mvPts = std::vector<cv::Point2f>(vPTs.begin(), vPTs.end());
			newFrame->matDescriptor = desc.clone();
			newFrame->ComputeBoW();
			
			auto pose = fpose.get();

			cv::Mat R = pose.rowRange(0, 3);
			cv::Mat t = pose.row(3).t();
			newFrame->SetPose(R, t);

			auto idx = fmpidx.get();
			for (int j = 0, jend = idx.cols; j < jend; j++) {
				int mpID = idx.at<int>(j);
				if (mpID == -1) {
					newFrame->AddMapPoint(nullptr, j);
					continue;
				}
				if (!mmpMPs.count(mpID)) {
					//积己
					cv::Mat Xw = mmMapPointData[mpID];
					cv::Mat desc = newFrame->matDescriptor.row(j);
					MapPoint* pNewMP = new UVR_SLAM::MapPoint(this, newFrame, Xw, desc);
					mmpMPs[mpID] = pNewMP;
				}
				auto pMP = mmpMPs[mpID];
				newFrame->AddMapPoint(pMP, j);
				pMP->AddObservation(newFrame, j);
			}

			//mp客 pose
			//std::cout << "a" << std::endl;
			//cv::Mat R = kfPoses.colRange(12 * i, 12 * i + 9).reshape(1, 3);
			//cv::Mat t = kfPoses.colRange(12 * i + 9, 12 * i + 12).t();
			//newFrame->SetPose(R, t);
			//std::cout << "b" << std::endl;
			//int mpEndIndex = mpIndex + vPTs.size();
			//cv::Mat idxs = kfMpIdxs.colRange(mpIndex, mpEndIndex);
			//for (int j = 0, jend = idxs.cols; j < jend; j++) {
			//	int mpID = idxs.at<int>(j);
			//	if (mpID == -1) {
			//		newFrame->AddMapPoint(nullptr, j);
			//		continue;
			//	}
			//	if (!mmpMPs.count(mpID)) {
			//		//积己
			//		cv::Mat Xw = mmMapPointData[mpID];
			//		cv::Mat desc = newFrame->matDescriptor.row(j);
			//		MapPoint* pNewMP = new UVR_SLAM::MapPoint(this, newFrame, Xw, desc);
			//		mmpMPs[mpID] = pNewMP;
			//	}
			//	auto pMP = mmpMPs[mpID];
			//	newFrame->AddMapPoint(pMP, j);
			//	pMP->AddObservation(newFrame, j);
			//}
			//std::cout << "c" << std::endl;
			//mpIndex = mpEndIndex;
			newFrame->ComputeSceneDepth();
			mmpFrames.insert(std::make_pair(id, newFrame));
		}
		for (auto iter = mmpFrames.begin(), itend = mmpFrames.end(); iter != itend; iter++) {
			auto pKFi = iter->second;
			int id = iter->first;
			this->AddFrame(pKFi);

			cv::Mat data = [](std::string ip, int port, int id, std::string map) {
				WebAPI* mpAPI = new WebAPI(ip, port);
				std::stringstream ss;
				ss << "/SendData?map=" << map << "&id=" << id << "&key=bconnectedkfs";
				auto res = mpAPI->Send(ss.str(), "");
				cv::Mat idx = cv::Mat::zeros(1, res.size() / 4, CV_32SC1);
				std::memcpy(idx.data, res.data(), res.size() * sizeof(char));
				return idx;
			}(ip, port, id, mapname);
			
			for (int i = 0; i < data.cols / 2; i++) {
				int kid = data.at<int>(2 * i);
				int weight = data.at<int>(2 * i + 1);
				auto pNeighKF = mmpFrames[kid];
				pKFi->AddKF(pNeighKF, weight);
				pNeighKF->AddKF(pKFi, weight);
			}
		}
		this->mbInitialized = true;
		SetMapLoad(false);
		std::cout << "ServerMap::Load::End" << std::endl;

		/*
		for (int i = 0, iend = mpIDs.cols; i < iend; i++) {
			int id = mpIDs.at<int>(i);
			cv::Mat Xw = mpXws.rowRange(3 * i, 3 * (i + 1));
			auto pNewMP = new MapPoint(this, Xw, cv::Mat());
			mmpMPs[id] = pNewMP;
		}*/
	}
}