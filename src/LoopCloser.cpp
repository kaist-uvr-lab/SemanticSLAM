#include "LoopCloser.h"
#include "Frame.h"
#include <MapPoint.h>
#include "System.h"
#include <Converter.h>
#include <Matcher.h>
#include <KeyframeDatabase.h>
#include <Sim3Solver.h>
#include <Optimization.h>
#include <Visualizer.h>
#include "Map.h"
#include <ServerMap.h>
#include <future>
#include <WebAPI.h>

/*
프레임의 seterase, setnoterase, setbadflag 관련된 처리가 아직 안되어있음.
matcher의 SearchBySim3 구현
CorrectLoop 관련 구현 필요
SearchAndFuse 구현 필요
RunGlobalBundleAdjustment 구현 필요
matcher의 fuse 구현 필요
맵포인트 Replace 구현 필요
키프레임 updateconnection 구현 필요
Optimization의 OptimizeEssentialGraph 내부 구현
*/

namespace UVR_SLAM {

	auto lambda_api_kf_match_loop_closing = [](std::string ip, int port, std::string map,int id1, int id2, int n) {
		WebAPI* mpAPI = new WebAPI(ip, port);
		std::stringstream ss;
		ss << "/featurematch?map=" << map << "&id1=" << id1 << "&id2=" << id2;
		auto res = mpAPI->Send(ss.str(), "");
		cv::Mat matches = cv::Mat::zeros(res.size() / sizeof(int), 1, CV_32SC1);
		std::memcpy(matches.data, res.data(), matches.rows * sizeof(int));
		return matches;
	};

	LoopCloser::LoopCloser() {}
	LoopCloser::LoopCloser(System* pSys):mpSystem(pSys), mbLoadData(false), mbFixScale(false), mbProcessing(false), mnCovisibilityConsistencyTh(3){
	}
	LoopCloser::~LoopCloser() {}
	void LoopCloser::Init() {
		mpMap = mpSystem->mpMap;
		mpKeyFrameDatabase = mpSystem->mpKeyframeDatabase;
		mpVoc = mpSystem->mpDBoWVoc;
		mpMatcher = mpSystem->mpMatcher;
	}
	void LoopCloser::LoadMapData(std::string map) {
		mbLoadData = true;
		mapName = map;
	}
	void LoopCloser::ConstructBowDB(std::vector<Frame*> vpFrames) {
		for (size_t i = 0, iend = vpFrames.size(); i < iend; i++) {
			mpKeyFrameDatabase->Add(vpFrames[i]);
		}
	}

	bool LoopCloser::Relocalization(Frame* pF) {
		auto vpCandidateKFs = mpKeyFrameDatabase->DetectPlaceCandidates(pF);
		int nMax = 0;
		cv::Mat maxMatch;
		Frame* maxFrame = nullptr;
		for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++) {
			auto pCandidate = vpCandidateKFs[i];
			cv::Mat matches = lambda_api_kf_match_loop_closing(mpSystem->ip, mpSystem->port, pF->mstrMapName, pF->mnFrameID, pCandidate->mnFrameID, pF->mvPts.size());
			auto vpMPs = pCandidate->GetMapPoints();
			int nMatch = 0;
			int nTempMatch = 0;
			for (int j = 0, jend = matches.rows; j < jend; j++) {
				int idx1 = j;
				int idx2 = matches.at<int>(idx1);
				if (idx2 == 10000)
					continue;
				nTempMatch++;
				auto pMP = vpMPs[idx2];
				if (!pMP || pMP->isDeleted())
					continue;
				nMatch++;
			}
			if (nMatch > nMax) {
				nMax = nMatch;
				maxMatch = matches.clone();
				maxFrame = pCandidate;
			}
		}
		if (nMax < 30 || !maxFrame)
			return false;

		//Pose Optimization Test
		cv::Mat R, t;
		maxFrame->GetPose(R, t);

		std::vector<cv::Point2f> vPTs;
		std::vector<cv::Point3f> vXws;
		std::vector<MapPoint*> vMPs;
		std::vector <bool> vInliers;
		std::vector<int> vnIDXs;
		auto vpMPs = maxFrame->GetMapPoints();
		for (int j = 0, jend = maxMatch.rows; j < jend; j++) {
			int idx1 = j;
			int idx2 = maxMatch.at<int>(idx1);
			if (idx2 == 10000)
				continue;
			auto pMP = vpMPs[idx2];
			if (!pMP || pMP->isDeleted())
				continue;
			vPTs.push_back(pF->mvPts[j]);
			vMPs.push_back(pMP);
			vInliers.push_back(true);
			vnIDXs.push_back(j);
		}

		pF->SetPose(R, t);
		int nPoseRecovery = Optimization::PoseOptimization(mpMap, pF, vMPs, vPTs, vInliers);
		if (nPoseRecovery < 20)
			return false;

		for (size_t i = 0; i < vInliers.size(); i++)
		{
			if (vInliers[i]) {
				int idx = vnIDXs[i];
				pF->AddMapPoint(vMPs[i], idx);
			}
		}
		return true;
	}

	void LoopCloser::RunWithMappingServer() {
		std::cout << "MappingServer::LoopCloser::Start" << std::endl;

		int nPrevSegFrame = -1;
		int nCurrSegFrame;
		int nPrevDepthFrame = -1;
		int nCurrDepthFrame;

		while (true) {

			if (mbLoadData) {
				std::cout << "Load Data" << std::endl;
				auto pTargetMap = mpSystem->GetMap(mapName);
				if (pTargetMap) {
					pTargetMap->LoadMapDataFromServer(mapName, mpSystem->ip, mpSystem->port);
					////디비를 공유하게 되면 전혀 다른 맵데이터가 합쳐질 수 있음.ㅁ
					ConstructBowDB(pTargetMap->GetFrames());
				}
				//mpMap->LoadMapDataFromServer(mapName, mpMap->mvpMapFrames);
				
				mbLoadData = false;
			}

			if (CheckNewKeyFrames()) {
				std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
				ProcessNewKeyFrame();
#ifdef DEBUG_LOOP_CLOSING_LEVEL_1
				std::cout << "MappingServer::LoopClosing::" << mpTargetFrame->mnFrameID << "::Start" << std::endl;
#endif
				mpKeyFrameDatabase->Add(mpTargetFrame);
				
				////////relocaalization
				//auto vpCandidateKFs = mpKeyFrameDatabase->DetectPlaceCandidates(mpTargetFrame);
				//std::cout << "Place Recognizer = " << vpCandidateKFs.size() << std::endl;
				//int nMax = 0; 
				//cv::Mat maxMatch;
				//Frame* maxFrame = nullptr;
				//for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++) {
				//	auto pCandidate = vpCandidateKFs[i];
				//	cv::Mat matches = lambda_api_kf_match_loop_closing(mpSystem->ip, mpSystem->port, mpTargetFrame->mstrMapName, mpTargetFrame->mnFrameID, pCandidate->mnFrameID, mpTargetFrame->mvPts.size());
				//	auto vpMPs = pCandidate->GetMapPoints();
				//	int nMatch = 0;
				//	int nTempMatch = 0;
				//	for (int j = 0, jend = matches.cols; j < jend; j++) {
				//		int idx1 = j;
				//		int idx2 = matches.at<int>(idx1);
				//		if (idx2 == 10000)
				//			continue;
				//		nTempMatch++;
				//		auto pMP = vpMPs[idx2];
				//		if (!pMP || pMP->isDeleted())
				//			continue;
				//		nMatch++;
				//	}
				//	if (nMatch > nMax) {
				//		nMax = nMatch;
				//		maxMatch = matches.clone();
				//		maxFrame = pCandidate;
				//	}
				//	//std::cout <<"ID = "<<pCandidate->mnFrameID<<", "<< nTempMatch << std::endl;
				//	//cv::imshow("candidate", pCandidate->GetOriginalImage()); cv::waitKey(1);
				//}
				//if (maxFrame)
				//{
				//	//cv::imshow("candidate", maxFrame->GetOriginalImage()); cv::waitKey(10);
				//	//Pose Optimization Test
				//	cv::Mat R, t;
				//	maxFrame->GetPose(R, t);
				//	mpTargetFrame->SetPose(R, t);
				//	cv::Mat targetImg = [](std::string ip, int port, int id, std::string map, int w, int h) {
				//		WebAPI* mpAPI = new WebAPI(ip, port);
				//		std::stringstream ss;
				//		ss << "/SendData?map=" << map << "&id=" << id << "&key=bimage";
				//		auto res = mpAPI->Send(ss.str(), "");
				//		int n = res.size();
				//		cv::Mat temp = cv::Mat::zeros(n, 1, CV_8UC1);
				//		std::memcpy(temp.data, res.data(), n * sizeof(uchar));
				//		cv::Mat img = cv::imdecode(temp, cv::IMREAD_COLOR);
				//		return img;
				//	}(mpSystem->ip, mpSystem->port, mpTargetFrame->mnFrameID, mpTargetFrame->mstrMapName, mpTargetFrame->mnWidth, mpTargetFrame->mnHeight);

				//	//cv::Mat targetImg = mpTargetFrame->GetOriginalImage().clone();
				//	cv::Mat resimg = maxFrame->GetOriginalImage().clone();
				//	
				//	
				//	std::vector<cv::Point2f> vPTs;
				//	std::vector<cv::Point3f> vXws;
				//	std::vector<MapPoint*> vMPs;
				//	std::vector <bool> vInliers;
				//	auto vpMPs = maxFrame->GetMapPoints();
				//	for (int j = 0, jend = maxMatch.cols; j < jend; j++) {
				//		int idx1 = j;
				//		int idx2 = maxMatch.at<int>(idx1);
				//		if (idx2 == 10000)
				//			continue;
				//		auto pMP = vpMPs[idx2];
				//		if (!pMP || pMP->isDeleted())
				//			continue;
				//		vPTs.push_back(mpTargetFrame->mvPts[j]);
				//		vMPs.push_back(pMP);
				//		vInliers.push_back(true);
				//		cv::Mat Xw = pMP->GetWorldPos();
				//		
				//		vXws.push_back(cv::Point3f(Xw.at<float>(0), Xw.at<float>(1), Xw.at<float>(2)));

				//		cv::Mat proj = mpTargetFrame->mK*(R*Xw + t);
				//		cv::Point2f projPt(proj.at<float>(0) / proj.at<float>(2), proj.at<float>(1) / proj.at<float>(2));
				//		cv::circle(targetImg, mpTargetFrame->mvPts[idx1], 3, cv::Scalar(255, 255, 0), -1);
				//		cv::circle(resimg, maxFrame->mvPts[idx2], 3, cv::Scalar(255, 0, 255), -1);
				//		cv::circle(resimg, projPt, 3, cv::Scalar(0, 255, 255), -1);
				//	}
				//	cv::resize(targetImg, targetImg, cv::Size(mpTargetFrame->mnWidth / 2, mpTargetFrame->mnHeight / 2));
				//	cv::resize(resimg, resimg, cv::Size(mpTargetFrame->mnWidth / 2, mpTargetFrame->mnHeight / 2));
				//	mpSystem->mpVisualizer->SetOutputImage(targetImg, 0);
				//	mpSystem->mpVisualizer->SetOutputImage(resimg, 1);
				//	/*cv::Mat rvec, tvec;
				//	cv::solvePnPRansac(vXws, vPTs, mpSystem->mK, cv::Mat(), rvec, tvec);
				//	cv::Rodrigues(rvec, R);*/
				//	mpTargetFrame->SetPose(R, t);
				//	int nPoseRecovery = Optimization::PoseOptimization(mpMap, mpTargetFrame, vMPs, vPTs, vInliers);
				//	mpMap->SetUserPosition(mpTargetFrame->GetCameraCenter());
				//	std::cout << "Place Recognizer::res::" << "::"<< nMax << std::endl;
				//}
				////////relocaalization

				//if (mpTargetFrame->GetConnectedKFs().size() < 3) {
				//	for (size_t i = 0, iend = vpNeighKFs.size(); i < iend; i++) {
				//		auto pKF = vpNeighKFs[i];
				//		/*auto ftest = std::async(std::launch::async, UVR_SLAM::lambda_api_kf_match, "143.248.96.81", 35005, mpTargetFrame->mnFrameID, pKF->mnFrameID, mpTargetFrame->mvPts.size());
				//		cv::Mat temp = ftest.get();*/
				//		cv::Mat temp = lambda_api_kf_match_loop_closing(mpSystem->ip, mpSystem->port, mpTargetFrame->mnFrameID, pKF->mnFrameID, mpTargetFrame->mvPts.size());
				//		if (mpTargetFrame->mvPts.size() != temp.cols) {
				//			std::cout << "Error::Matching::Invalid Matching Size::" << temp.cols << ", " << mpTargetFrame->mvPts.size() << std::endl;
				//		}
				//		std::vector<bool> vecBoolOverlap(pKF->mvPts.size(), false);
				//		for (size_t j = 0, jend = temp.cols; j < jend; j++) {
				//			int idx1 = j;
				//			int idx2 = temp.at<int>(idx1);
				//			if (idx2 == -1)
				//				continue;
				//			if (idx2 >= pKF->mvPts.size() || idx2 < -1) {
				//				temp.at<int>(idx1) = -1;
				//				std::cout << "Error::Matching::Invalid Frame2 Indexs = " << idx2 << ", " << pKF->mvPts.size() << "::" << j << std::endl;
				//				continue;
				//			}
				//			if (vecBoolOverlap[idx2])
				//			{
				//				temp.at<int>(idx1) = -1;
				//				continue;
				//			}
				//			vecBoolOverlap[idx2] = true;
				//		}
				//	}
				//	//mMatches.push_back(temp);
				//}


				{
					//segmentation test
					WebAPI* mpAPI = new WebAPI(mpSystem->ip, mpSystem->port);
					std::stringstream ss;
					ss << "/GetLastFrameID?map="<<mpTargetFrame->mstrMapName<<"&key=bsegmentation";
					WebAPIDataConverter::ConvertStringToNumber(mpAPI->Send(ss.str(), "").c_str(), nCurrSegFrame);
					if (nCurrSegFrame >= 0 && nCurrSegFrame != nPrevSegFrame) {
						ss.str("");
						ss << "/SendData?map=" << mpTargetFrame->mstrMapName << "&id="<< nCurrSegFrame << "&key=bsegmentation";
						auto f = std::async(std::launch::async, [](WebAPI* wapi,std::string method, int w, int h) {
							auto resdata = wapi->Send(method, "");
							////처리 후 전송
							cv::Mat seg = cv::Mat::zeros(h, w, CV_8UC1);
							std::memcpy(seg.data, resdata.data(), w*h*sizeof(uchar));
							cv::Mat seg_color = cv::Mat::zeros(seg.size(), CV_8UC3);
							for (int y = 0; y < seg_color.rows; y++) {
								for (int x = 0; x < seg_color.cols; x++) {
									int label = seg.at<uchar>(y, x);
									seg_color.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
								}
							}
							return seg_color;
							//imshow("segmentation", seg_color); cv::waitKey(1);
						}, mpAPI, ss.str(), mpTargetFrame->mnWidth/2, mpTargetFrame->mnHeight/2);
						nPrevSegFrame = nCurrSegFrame;
						auto res = f.get();
						mpSystem->mpVisualizer->SetOutputImage(res, 2);
					}
				}

				{
					//segmentation test
					WebAPI* mpAPI = new WebAPI(mpSystem->ip, mpSystem->port);
					std::stringstream ss;
					ss << "/GetLastFrameID?map=" << mpTargetFrame->mstrMapName << "&key=bdepth";
					WebAPIDataConverter::ConvertStringToNumber(mpAPI->Send(ss.str(), "").c_str(), nCurrDepthFrame);
					if (nCurrDepthFrame >= 0 && nCurrDepthFrame != nPrevDepthFrame) {
						ss.str("");
						ss << "/SendData?map=" << mpTargetFrame->mstrMapName << "&id=" << nCurrDepthFrame << "&key=bdepth";
						auto f = std::async(std::launch::async, [](WebAPI* wapi, std::string method, int w, int h, Frame* pF) {
							auto resdata = wapi->Send(method, "");
							
							////처리 후 전송
							cv::Mat depth = cv::Mat::zeros(h, w, CV_32FC1);
							std::memcpy(depth.data, resdata.data(), w*h * sizeof(float));

							std::vector<std::tuple<cv::Point2f, float, int>> vecTuples;

							cv::Mat R, t;
							pF->GetPose(R, t);

							////depth 정보 저장 및 포인트와 웨이트 정보를 튜플로 저장
							cv::Mat Rcw2 = R.row(2);
							Rcw2 = Rcw2.t();
							float zcw = t.at<float>(2);
							auto vpMPs = pF->GetMapPoints();
							for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
								auto pMPi = vpMPs[i];
								if (!pMPi || pMPi->isDeleted())
									continue;
								auto pt = pF->mvPts[i];
								cv::Mat x3Dw = pMPi->GetWorldPos();
								float z = (float)Rcw2.dot(x3Dw) + zcw;
								std::tuple<cv::Point2f, float, int> data = std::make_tuple(std::move(pt), 1.0 / z, pMPi->GetNumObservations());//cv::Point2f(pt.x / 2, pt.y / 2)
								vecTuples.push_back(data);
							}

							////웨이트와 포인트 정보로 정렬
							std::sort(vecTuples.begin(), vecTuples.end(),
								[](std::tuple<cv::Point2f, float, int> const &t1, std::tuple<cv::Point2f, float, int> const &t2) {
									if (std::get<2>(t1) == std::get<2>(t2)) {
										return std::get<0>(t1).x != std::get<0>(t2).x ? std::get<0>(t1).x > std::get<0>(t2).x : std::get<0>(t1).y > std::get<0>(t2).y;
									}
									else {
										return std::get<2>(t1) > std::get<2>(t2);
									}
								}
							);

							////파라메터 검색 및 뎁스 정보 복원
							int nTotal = 20;
							if (vecTuples.size() > nTotal) {
								int nData = nTotal;
								cv::Mat A = cv::Mat::ones(nData, 2, CV_32FC1);
								cv::Mat B = cv::Mat::zeros(nData, 1, CV_32FC1);

								for (size_t i = 0; i < nData; i++) {
									auto data = vecTuples[i];
									auto pt = std::get<0>(data);
									auto invdepth = std::get<1>(data);
									auto nConnected = std::get<2>(data);

									float p = depth.at<float>(pt);
									A.at<float>(i, 0) = p;//invdepth;
									B.at<float>(i) = invdepth;//p;
								}

								//cv::Mat X = A.inv(cv::DECOMP_QR)*B;
								cv::Mat S = A.t()*A;
								cv::Mat X = S.inv()*A.t()*B;
								float a = X.at<float>(0);
								float b = X.at<float>(1);

								/*cv::Mat C = A*X - B;
								std::cout << "depth val = " << cv::sum(C) / C.rows << std::endl;*/

								//depth = a*depth + b; //(depth - b) / a;
								for (int x = 0, cols = depth.cols; x < cols; x++) {
									for (int y = 0, rows = depth.rows; y < rows; y++) {
										float val = a*depth.at<float>(y, x) + b;//1.0 / depth.at<float>(y, x);
										/*if (val < 0.0001)
										val = 0.5;*/
										depth.at<float>(y, x) = 1.0/val;
									}
								}
								
							}

							return depth;
						}, mpAPI, ss.str(), mpTargetFrame->mnWidth, mpTargetFrame->mnHeight, mpTargetFrame);
						nPrevDepthFrame = nCurrDepthFrame;
						auto res = f.get();
						cv::Mat tempDepth = res.clone();

						cv::normalize(res, res, 0, 255, cv::NORM_MINMAX, CV_8UC1);
						cv::cvtColor(res, res, CV_GRAY2BGR);
						cv::resize(res, res, res.size() / 2);
						mpSystem->mpVisualizer->SetOutputImage(res, 3);

						////시각화 테스트
						/*cv::Mat Rinv, Tinv;
						mpTargetFrame->GetInversePose(Rinv, Tinv);
						mpSystem->mpMap->ClearTempMPs();
						int inc = 10;
						for (size_t row = inc, rows = tempDepth.rows; row < rows; row += inc) {
							for (size_t col = inc, cols = tempDepth.cols; col < cols; col += inc) {
								cv::Point2f pt(col, row);
								float depth = tempDepth.at<float>(pt);
								if (depth < 0.0001)
									continue;
								cv::Mat a = Rinv*(mpTargetFrame->mInvK*(cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.0)*depth) + Tinv;
								mpSystem->mpMap->AddTempMP(a);
							}
						}*/

						ss.str("");
						ss << "/ReceiveData?map=" << mpTargetFrame->mstrMapName << "&id=" << nCurrDepthFrame << "&key=rdepth";
						mpAPI->Send(ss.str(), tempDepth.data, tempDepth.rows*tempDepth.cols*sizeof(float));
					}
				}

				std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
				auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				float t_test1 = du_test1 / 1000.0;
#ifdef DEBUG_LOOP_CLOSING_LEVEL_1
				std::cout << "MappingServer::LoopClosing::" << mpTargetFrame->mnFrameID << "::"<< vpNeighKFs.size()<<"::"<<t_test1 << "::End" << std::endl;
#endif
			}
		}
	}
	void LoopCloser::Run() {

		while (true) {
			if (CheckNewKeyFrames()) {
				SetBoolProcessing(true);
				ProcessNewKeyFrame();

				if (DetectLoop()) {
					std::cout << "Loop Frame Detection!!" << std::endl;
					if (ComputeSim3())
					{
						std::cout << "Loop Closing!!" << std::endl;
						CorrectLoop();
					}
				}

				///////////////VoW 매칭
				//auto vpGrahWindows = mpMap->GetGraphFrames();
				//for (int i = 0; i < vpGrahWindows.size(); i++) {
				//	auto pKFi = vpGrahWindows[i];
				//	auto score = mpTargetFrame->Score(pKFi);
				//	
				//	if (score > 0.01) {
				//		std::cout << "Loop::Score::" << score << std::endl;
				//		imshow("Loop::1", mpTargetFrame->GetOriginalImage());
				//		imshow("Loop::2", pKFi->GetOriginalImage());
				//		cv::waitKey(500);
				//	}
				//}
				///////////////VoW 매칭
				mpKeyFrameDatabase->Add(mpTargetFrame);
				SetBoolProcessing(false);
			}//visualize
		}
	}
	void LoopCloser::InsertKeyFrame(UVR_SLAM::Frame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		mKFQueue.push(pKF);
	}

	bool LoopCloser::CheckNewKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		return(!mKFQueue.empty());
	}

	void LoopCloser::ProcessNewKeyFrame()
	{
		{
			std::unique_lock<std::mutex> lock(mMutexNewKFs);
			mpTargetFrame = mKFQueue.front();
			mKFQueue.pop();
		}
		/*[](std::string ip, int port, Frame* pF) {
			WebAPI* mpAPI = new WebAPI(ip, port);
			std::stringstream ss;
			ss << "/SendData?map=" << pF->mstrMapName << "&id=" << pF->mnFrameID << "&key=bdesc";
			WebAPIDataConverter::ConvertBytesToDesc(mpAPI->Send(ss.str(), "").c_str(), pF->mvPts.size(), pF->matDescriptor);
			pF->ComputeBoW();
		}(mpSystem->ip, mpSystem->port, mpTargetFrame);*/
	}


	bool LoopCloser::isProcessing() {
		std::unique_lock<std::mutex> lock(mMutexProcessing);
		return mbProcessing;
	}
	void LoopCloser::SetBoolProcessing(bool b) {
		std::unique_lock<std::mutex> lock(mMutexProcessing);
		mbProcessing = b;
	}

	bool LoopCloser::DetectLoop() {

		std::vector<Frame*> vpKFs = mpTargetFrame->GetConnectedKFs();
		float minScore = 1;
		const DBoW3::BowVector &CurrentBowVec = mpTargetFrame->mBowVec;
		for (auto iter = vpKFs.begin(), iter2 = vpKFs.end(); iter != iter2; iter++) {
			auto pKF = *iter;
			if (mpTargetFrame == pKF)
				continue;
			const DBoW3::BowVector &BowVec = pKF->mBowVec;
			float score = mpVoc->score(CurrentBowVec, BowVec);
			if (score < minScore)
				minScore = score;
		}

		auto vpCandidateKFs = mpKeyFrameDatabase->DetectLoopCandidates(mpTargetFrame, minScore);
		if(vpCandidateKFs.empty())
		{
			
			mvConsistentGroups.clear();
			//mpKeyFrameDB->add(mpCurrentKF);
			//mpCurrentKF->SetErase();
			return false;
		}
		mvpEnoughConsistentCandidates.clear();

		//하나의 ConsistentGroup은 frame의 집합과 int(해당 그룹과 연결된 그룹의 수를 의미)로 구성. ConsistentGroup의 벡터
		//mvConsistentGroup 내의 각 그룹들은 서로 연결되지 않음.
		//갑자기 한번 많이 매칭된 프레임이 아닌 지속적으로 매칭이 되어야 루프 프레임을 선택. 다만, 무조건 동일한 프레임이 아닌 다른 프레임과 연결된 경우에도 가능함.
		std::vector<ConsistentGroup> vCurrentConsistentGroups;
		std::vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);
		for (size_t i = 0, iend = vpCandidateKFs.size(); i<iend; i++)
		{
			Frame* pCandidateKF = vpCandidateKFs[i];
			
			//bow vector를 이용해서 획득한 후보 키프레임 들의 연결된 프레임을 본인 포함해서 캔디데이트 그룹으로 설정
			//이게 현재 키프레임에 대한 ConsistentGroup,이며 프레임 집합
			std::set<Frame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrameSet();
			spCandidateGroup.insert(pCandidateKF);

			bool bEnoughConsistent = false;
			bool bConsistentForSomeGroup = false;
			//현재 연결된 모든 ConsistentGroup를 체크함.
			for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG<iendG; iG++)
			{
				//현재 검색하고자 하는 ConsistentGroup의 프레임 집합
				std::set<Frame*> sPreviousGroup = mvConsistentGroups[iG].first;

				bool bConsistent = false;
				for (std::set<Frame*>::iterator sit = spCandidateGroup.begin(), send = spCandidateGroup.end(); sit != send; sit++)
				{
					//이전 ConsistentGroup 그룹 중에 현재 키프레임의 ConsistentGroup과 겹치는게 있는지 찾고 있으면 바로 스탑
					if (sPreviousGroup.count(*sit))
					{
						bConsistent = true;
						bConsistentForSomeGroup = true;
						break;
					}
				}
				//ConsistentGroup을 찾은 경우
				if (bConsistent)
				{
					//현재 키프레임의 ConsistentGroup 벡터에 연결 정보를 갱신한 후 추가. 
					//후보 프레임이 일정 조건을 찾으면 방금 찾은 그룹에 아예 추가. 이 과정은 후보 프레임들마다 딱 한번만 수행 됨
					//3개의 프레임과 연결되면 됨.
					int nPreviousConsistency = mvConsistentGroups[iG].second;
					int nCurrentConsistency = nPreviousConsistency + 1;
					if (!vbConsistentGroup[iG])
					{
						ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
						vCurrentConsistentGroups.push_back(cg);
						vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
					}
					if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
					{
						mvpEnoughConsistentCandidates.push_back(pCandidateKF);
						bEnoughConsistent = true; //this avoid to insert the same candidate more than once
					}
				}
			}

			//현재 검색중인 후보 프레임과 연결되는 ConsistentGroup이 없으면 0으로 설정
			// If the group is not consistent with any previous group insert with consistency counter set to zero
			if (!bConsistentForSomeGroup)
			{
				ConsistentGroup cg = make_pair(spCandidateGroup, 0);
				vCurrentConsistentGroups.push_back(cg);
			}
		}

		// Update Covisibility Consistent Groups
		mvConsistentGroups = vCurrentConsistentGroups;


		// Add Current Keyframe to database
		if (mvpEnoughConsistentCandidates.empty())
		{
			//mpCurrentKF->SetErase();
			return false;
		}
		else
		{
			return true;
		}

		//mpCurrentKF->SetErase();
		return false;
	}
	bool LoopCloser::ComputeSim3() {
		
		////후보 루프 프레임의 Sim3 계산
		const int nInitialCandidates = mvpEnoughConsistentCandidates.size();
		std::vector<Sim3Solver*> vpSim3Solvers;
		vpSim3Solvers.resize(nInitialCandidates);

		//map point의 2차원 벡터로 현재 타겟 프레임과 후보 루프 프레임 사이의 매칭 정보를 저장함.
		//따라서 타겟 프레임의 포인트 사이즈 x 후보 루프 프레임 수가 크기가 됨
		std::vector<std::vector<MapPoint*> > vvpMapPointMatches;
		vvpMapPointMatches.resize(nInitialCandidates);
		
		std::vector<bool> vbDiscarded;
		vbDiscarded.resize(nInitialCandidates);

		int nCandidates = 0; //candidates with enough matches

		for (int i = 0; i<nInitialCandidates; i++)
		{
			Frame* pKF = mvpEnoughConsistentCandidates[i];

			// avoid that local mapping erase it while it is being processed in this thread
			//pKF->SetNotErase();

			if (pKF->isDeleted())
			{
				vbDiscarded[i] = true;
				continue;
			}

			////bow 기반 매칭
			int nmatches = mpMatcher->BagOfWordsMatching(mpTargetFrame, pKF, vvpMapPointMatches[i]);
			if (nmatches<20)
			{
				vbDiscarded[i] = true;
				continue;
			}
			else
			{
				//두 프레임의 자세와 매칭 정보를 RANSAC을 이용해서 Sim3Solver 설정
				Sim3Solver* pSolver = new Sim3Solver(mpTargetFrame, pKF, vvpMapPointMatches[i], mbFixScale);
				pSolver->SetRansacParameters(0.99, 20, 300);
				vpSim3Solvers[i] = pSolver;
			}

			nCandidates++;
		}
		bool bMatch = false;

		// Perform alternatively RANSAC iterations for each candidate
		// until one is succesful or all fail
		while (nCandidates>0 && !bMatch)
		{
			for (int i = 0; i<nInitialCandidates; i++)
			{
				if (vbDiscarded[i])
					continue;

				Frame* pKF = mvpEnoughConsistentCandidates[i];

				// Perform 5 Ransac Iterations
				std::vector<bool> vbInliers;
				int nInliers;
				bool bNoMore;

				Sim3Solver* pSolver = vpSim3Solvers[i];
				cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

				// If Ransac reachs max. iterations discard keyframe
				if (bNoMore)
				{
					vbDiscarded[i] = true;
					nCandidates--;
				}

				// If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
				if (!Scm.empty())
				{
					std::vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
					for (size_t j = 0, jend = vbInliers.size(); j<jend; j++)
					{
						if (vbInliers[j])
							vpMapPointMatches[j] = vvpMapPointMatches[i][j];
					}

					cv::Mat R = pSolver->GetEstimatedRotation();
					cv::Mat t = pSolver->GetEstimatedTranslation();
					const float s = pSolver->GetEstimatedScale();
					
					//matcher.SearchBySim3(mpTargetFrame, pKF, vpMapPointMatches, s, R, t, 7.5);

					g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);
					const int nInliers = Optimization::OptimizeSim3(mpTargetFrame, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

					// If optimization is succesful stop ransacs and continue
					if (nInliers >= 20)
					{
						bMatch = true;
						mpMatchedKF = pKF;
						g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()), Converter::toVector3d(pKF->GetTranslation()), 1.0);
						mg2oScw = gScm*gSmw;
						mScw = Converter::toCvMat(mg2oScw);

						mvpCurrentMatchedPoints = vpMapPointMatches;
						break;
					}
				}
			}
		}

		////Erase 관련 
		if (!bMatch)
		{
			/*for (int i = 0; i<nInitialCandidates; i++)
				mvpEnoughConsistentCandidates[i]->SetErase();
			mpCurrentKF->SetErase();*/
			return false;
		}

		// Retrieve MapPoints seen in Loop Keyframe and neighbors
		std::vector<Frame*> vpLoopConnectedKFs = mpMatchedKF->GetConnectedKFs();
		vpLoopConnectedKFs.push_back(mpMatchedKF);
		mvpLoopMapPoints.clear();
		for (std::vector<Frame*>::iterator vit = vpLoopConnectedKFs.begin(); vit != vpLoopConnectedKFs.end(); vit++)
		{
			Frame* pKF = *vit;
			std::vector<MapPoint*> vpMapPoints = pKF->GetMapPoints();
			for (size_t i = 0, iend = vpMapPoints.size(); i<iend; i++)
			{
				MapPoint* pMP = vpMapPoints[i];
				if (pMP)
				{
					if (!pMP->isDeleted() && pMP->mnLoopPointForKF != mpTargetFrame->mnKeyFrameID)
					{
						mvpLoopMapPoints.push_back(pMP);
						pMP->mnLoopPointForKF = mpTargetFrame->mnKeyFrameID;
					}
				}
			}
		}

		//// Find more matches projecting with the computed Sim3
		//matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints, 10);

		//// If enough matches accept Loop
		int nTotalMatches = 0;
		for (size_t i = 0; i<mvpCurrentMatchedPoints.size(); i++)
		{
			if (mvpCurrentMatchedPoints[i])
				nTotalMatches++;
		}

		if (nTotalMatches >= 40)
		{
			/*for (int i = 0; i<nInitialCandidates; i++)
				if (mvpEnoughConsistentCandidates[i] != mpMatchedKF)
					mvpEnoughConsistentCandidates[i]->SetErase();*/
			return true;
		}
		else
		{
			/*for (int i = 0; i<nInitialCandidates; i++)
				mvpEnoughConsistentCandidates[i]->SetErase();
			mpCurrentKF->SetErase();*/
			return false;
		}
	}
	void LoopCloser::CorrectLoop() {
		
		////로컬 매핑에 스탑 요청

		////GBA 동작 멈추기 && 구현하기

		////KF update connection 구현 및 관리 필요
		//mpTargetFrame->UpdateConnections();

		// Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
		mvpCurrentConnectedKFs = mpTargetFrame->GetConnectedKFs();
		mvpCurrentConnectedKFs.push_back(mpTargetFrame);

		KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
		CorrectedSim3[mpTargetFrame] = mg2oScw;
		cv::Mat Rwc, twc;
		cv::Mat Twc;
		mpTargetFrame->GetInversePose(Rwc, twc);
		cv::hconcat(Rwc, twc, Twc);


		{
			// Get Map Mutex
			std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

			for (std::vector<Frame*>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
			{
				Frame* pKFi = *vit;

				cv::Mat Riw, tiw;
				cv::Mat Tiw;
				pKFi->GetPose(Riw, tiw);
				cv::hconcat(Riw, tiw, Tiw);

				if (pKFi != mpTargetFrame)
				{
					cv::Mat Tic = Tiw*Twc;
					cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
					cv::Mat tic = Tic.rowRange(0, 3).col(3);
					g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
					g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
					//Pose corrected with the Sim3 of the loop closure
					CorrectedSim3[pKFi] = g2oCorrectedSiw;
				}

				g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
				//Pose without correction
				NonCorrectedSim3[pKFi] = g2oSiw;
			}

			// Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
			for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; mit++)
			{
				Frame* pKFi = mit->first;
				g2o::Sim3 g2oCorrectedSiw = mit->second;
				g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

				g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

				std::vector<MapPoint*> vpMPsi = pKFi->GetMapPoints();
				for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
				{
					MapPoint* pMPi = vpMPsi[iMP];
					if (!pMPi || pMPi->isDeleted() || pMPi->mnCorrectedByKF == mpTargetFrame->mnKeyFrameID)
						continue;

					// Project with non-corrected pose and project back with corrected pose
					cv::Mat P3Dw = pMPi->GetWorldPos();
					Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
					Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

					cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
					pMPi->SetWorldPos(cvCorrectedP3Dw);
					pMPi->mnCorrectedByKF = mpTargetFrame->mnKeyFrameID;
					pMPi->mnCorrectedReference = pKFi->mnKeyFrameID;
					//pMPi->UpdateNormalAndDepth();
				}

				// Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
				Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
				Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
				double s = g2oCorrectedSiw.scale();

				eigt *= (1. / s); //[R t/s;0 1]

				cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

				pKFi->SetPose(correctedTiw.colRange(0,3).rowRange(0,3), correctedTiw.col(3).rowRange(0,3));

				// Make sure connections are updated
				//pKFi->UpdateConnections();
			}

			// Start Loop Fusion
			// Update matched map points and replace if duplicated
			for (size_t i = 0; i<mvpCurrentMatchedPoints.size(); i++)
			{
				if (mvpCurrentMatchedPoints[i])
				{
					MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
					MapPoint* pCurMP = mpTargetFrame->GetMapPoint(i);
					if (pCurMP)
					{
						//pCurMP->Replace(pLoopMP);
					}
					else
					{
						mpTargetFrame->AddMapPoint(pLoopMP, i);
						pLoopMP->AddObservation(mpTargetFrame, i);
						//pLoopMP->ComputeDistinctiveDescriptors();
					}
				}
			}

		}

		// Project MapPoints observed in the neighborhood of the loop keyframe
		// into the current keyframe and neighbors using corrected poses.
		// Fuse duplications.
		SearchAndFuse(CorrectedSim3);


		// After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
		std::map<Frame*, std::set<Frame*> > LoopConnections;

		for (std::vector<Frame*>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
		{
			Frame* pKFi = *vit;
			std::vector<Frame*> vpPreviousNeighbors = pKFi->GetConnectedKFs();

			// Update connections. Detect new links.
			//pKFi->UpdateConnections();
			LoopConnections[pKFi] = pKFi->GetConnectedKeyFrameSet();
			for (std::vector<Frame*>::iterator vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end(); vit_prev != vend_prev; vit_prev++)
			{
				LoopConnections[pKFi].erase(*vit_prev);
			}
			for (std::vector<Frame*>::iterator vit2 = mvpCurrentConnectedKFs.begin(), vend2 = mvpCurrentConnectedKFs.end(); vit2 != vend2; vit2++)
			{
				LoopConnections[pKFi].erase(*vit2);
			}
		}

		// Optimize graph
		Optimization::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpTargetFrame, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

		//mpMap->InformNewBigChange();

		//// Add loop edge
		//mpMatchedKF->AddLoopEdge(mpCurrentKF);
		//mpCurrentKF->AddLoopEdge(mpMatchedKF);

		//// Launch a new thread to perform Global Bundle Adjustment
		//mbRunningGBA = true;
		//mbFinishedGBA = false;
		//mbStopGBA = false;
		//mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);

		//// Loop closed. Release Local Mapping.
		//mpLocalMapper->Release();

		mLastLoopKFid = mpTargetFrame->mnKeyFrameID;

	}
	void LoopCloser::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap) {

	}
}