#include <MappingServer.h>
#include <System.h>
#include <User.h>
#include <Map.h>
#include <LocalMapper.h>
#include <LoopCloser.h>
#include <MapOptimizer.h>
#include <Visualizer.h>
#include <Initializer.h>
#include <Matcher.h>
#include <Frame.h>
#include <MapPoint.h>
#include <WebAPI.h>

#include <future>

namespace UVR_SLAM {

	int		CODE_MATCH_ERROR_MAPPING_SERVER = 10000;
	float	TH_RATIO_MP_KP = 0.7f;
	int		TH_NUM_KP_MATCH = 80;
	int		TH_NUM_MP_MATCH = 50;

	auto lambda_api_detect = [](std::string ip, int port, Frame* pTarget, std::string map) {
		WebAPI* mpAPI = new WebAPI(ip, port);
		std::stringstream ss;
		ss << "/SendData?map="<< map <<"&id=" << pTarget->mnFrameID << "&key=bkpts";
		auto res = mpAPI->Send(ss.str(), "");
		int n = res.size() / 8;
		cv::Mat seg = cv::Mat::zeros(n, 2, CV_32FC1);
		std::memcpy(seg.data, res.data(), n * 2 * sizeof(float));
		for (int i = 0; i < n; i++) {
			cv::Point2f pt(seg.at<float>(2 * i), seg.at<float>(2 * i + 1));
			pTarget->mvPts.push_back(pt);
		}

		//auto strID1 = WebAPIDataConverter::ConvertNumberToString(pTarget->mnFrameID);
		//WebAPIDataConverter::ConvertStringToPoints(mpAPI->Send("/getPts", strID1).c_str(), pTarget->mvPts);
		pTarget->SetMapPoints(pTarget->mvPts.size());
	};
	auto lambda_api_detectAndmatch = [](std::string ip, int port, Frame* pRef, Frame* pTarget, std::string map) {
		WebAPI* mpAPI = new WebAPI(ip, port);
		std::stringstream ss;
		ss << "/SendData?map=" << map << "&id=" << pTarget->mnFrameID << "&key=bkpts";
		auto res = mpAPI->Send(ss.str(), "");
		int n = res.size() / 8;
		cv::Mat seg = cv::Mat::zeros(n, 2, CV_32FC1);
		std::memcpy(seg.data, res.data(), n * 2 * sizeof(float));
		for (int i = 0; i < n; i++) {
			cv::Point2f pt(seg.at<float>(2 * i), seg.at<float>(2 * i + 1));
			pTarget->mvPts.push_back(pt);
		}
		//auto strID2 = WebAPIDataConverter::ConvertNumberToString(pTarget->mnFrameID);
		//WebAPIDataConverter::ConvertStringToPoints(mpAPI->Send("/getPts", strID2).c_str(), pTarget->mvPts);
		pTarget->SetMapPoints(pTarget->mvPts.size());
		std::vector<int> matches;
		ss.str("");
		ss << "/featurematch?map=" << map <<"&id1="<<pRef->mnFrameID<<"&id2=" << pTarget->mnFrameID;
		//auto strID12 = WebAPIDataConverter::ConvertNumberToString(pRef->mnFrameID, pTarget->mnFrameID);
		WebAPIDataConverter::ConvertStringToMatches(mpAPI->Send(ss.str(),"").c_str(), pRef->mvPts.size(), matches);
		return matches;
	};
	auto lambda_api_Initialization = [](std::string ip, int port, System* pSystem, Frame* pRef, Frame* pTarget, int& nRefID, std::vector<int> vMatches, std::string map, bool& bReplace) {

		////캐칭 관련 쓰레시홀딩 값 정리
		int nFeatureSize = pRef->mvPts.size();
		int nThreshInit = 0.40*nFeatureSize;
		int nThreshReplace = 0.30*nFeatureSize;

		//매칭 정보를 담을 곳
		std::vector<cv::Point2f> vTempPts1, vTempPts2;
		std::vector<int> vTempIndexs;

		////중복 체크
		std::vector<bool> vecBoolOverlap(pTarget->mvPts.size(), false);
		int nMatch = 0;
		for (size_t i = 0, iend = vMatches.size(); i < iend; i++) {
			if (vMatches[i] == CODE_MATCH_ERROR_MAPPING_SERVER)
				continue;
			int idx1 = i;
			int idx2 = vMatches[i];

			if (vecBoolOverlap[idx2])
			{
				continue;
			}
			vecBoolOverlap[idx2] = true;
			nMatch++;

			auto pt1 = pRef->mvPts[idx1];
			auto pt2 = pTarget->mvPts[idx2];

			vTempIndexs.push_back(i);
			vTempPts1.push_back(pt1);
			vTempPts2.push_back(pt2);

		}

		//매칭 값으로 이후 과정 진행 체크 또는 레퍼런스 프레임 교체
		if (nMatch < nThreshReplace) {
#ifdef DEBUG_TRACKING_LEVEL_3
			std::cout << "Initializer::replace::keyframe1::" << pTarget->mnFrameID << "::" << nMatch << ", MIN = " << nThreshInit << ", " << nThreshReplace << std::endl;
#endif
			bReplace = true;
			return false;
		}
		else if (nMatch < nThreshInit) {
#ifdef DEBUG_TRACKING_LEVEL_3
			std::cout << "Initializer::match::부족" << nMatch << ", MIN = " << nThreshInit << ", " << nThreshReplace << std::endl;
#endif
			return false;
		}

		/////////////////////Fundamental Matrix Decomposition & Triangulation
		std::vector<uchar> vFInliers;
		std::vector<cv::Point2f> vTempMatchPts1, vTempMatchPts2;
		std::vector<int> vTempMatchOctaves, vTempMatchIDXs; //vTempMatchPts2와 대응되는 매칭 인덱스를 저장.
		std::vector<bool> vbFtest;
		cv::Mat F12;
		float score;
		////E  찾기 : OpenCV
		cv::Mat E12 = cv::findEssentialMat(vTempPts1, vTempPts2, pRef->mK, cv::FM_RANSAC, 0.999, 1.0, vFInliers);
		for (unsigned long i = 0; i < vFInliers.size(); i++) {
			if (vFInliers[i]) {
				vTempMatchPts1.push_back(vTempPts1[i]);
				vTempMatchPts2.push_back(vTempPts2[i]);
				vTempMatchIDXs.push_back(i);//vTempIndexs[i]
			}
		}

		nMatch = vTempMatchPts1.size();
		//////F, E를 통한 매칭 결과 반영

		///////삼각화 : OpenCV
		cv::Mat R1, t1;
		cv::Mat matTriangulateInliers;
		cv::Mat Map3D;
		cv::Mat K;
		pRef->mK.convertTo(K, CV_64FC1);
		int res2 = cv::recoverPose(E12, vTempMatchPts1, vTempMatchPts2, K, R1, t1, 50.0, matTriangulateInliers, Map3D);
		R1.convertTo(R1, CV_32FC1);
		t1.convertTo(t1, CV_32FC1);
		///////////////////////Fundamental Matrix Decomposition & Triangulation

		////////////삼각화 결과에 따른 초기화 판단
		if (res2 < 0.9*nMatch) {
#ifdef DEBUG_TRACKING_LEVEL_3
			std::cout << "Initializer::triangulation::부족" << res2 << ", MIN = " << 0.9*nMatch << std::endl;
#endif
			return false;
		}

		/////여기까지 오면 초기화는 성공한 것
		pSystem->mpLocalMapper->SetInitialKeyFrame(pRef, pTarget);

		////KF는 디스크립터를 획득하고 BoWVec 계산해야 함
		WebAPI* mpAPI = new WebAPI(ip, port);
		std::stringstream ss;
		ss << "/SendData?map=" << map << "&id=" << pRef->mnFrameID << "&key=bdesc";
		WebAPIDataConverter::ConvertBytesToDesc(mpAPI->Send(ss.str(), "").c_str(), pRef->mvPts.size(), pRef->matDescriptor);
		ss.str("");
		ss << "/SendData?map=" << map << "&id=" << pTarget->mnFrameID << "&key=bdesc";
		WebAPIDataConverter::ConvertBytesToDesc(mpAPI->Send(ss.str(), "").c_str(), pTarget->mvPts.size(), pTarget->matDescriptor);
		//auto strID1 = WebAPIDataConverter::ConvertNumberToString(pRef->mnFrameID);
		//auto strID2 = WebAPIDataConverter::ConvertNumberToString(pTarget->mnFrameID);
		//WebAPIDataConverter::ConvertBytesToDesc(mpAPI->Send("/getDesc", strID1).c_str(), pRef->mvPts.size(), pRef->matDescriptor);
		//WebAPIDataConverter::ConvertBytesToDesc(mpAPI->Send("/getDesc", strID2).c_str(), pTarget->mvPts.size(), pTarget->matDescriptor);

		std::vector<UVR_SLAM::MapPoint*> tempMPs;
		std::vector<cv::Point2f> vTempMappedPts1, vTempMappedPts2; //맵포인트로 생성된 포인트 정보를 저장
		std::vector<int> vTempMappedOctaves, vTempMappedIDXs; //vTempMatch에서 다시 일부분을 저장. 초기 포인트와 대응되는 위치를 저장.
		int res3 = 0;
		for (int i = 0; i < matTriangulateInliers.rows; i++) {
			int val = matTriangulateInliers.at<uchar>(i);
			if (val == 0)
				continue;
			///////////////뎁스값 체크
			cv::Mat X3D = Map3D.col(i).clone();
			X3D.convertTo(X3D, CV_32FC1);
			X3D /= X3D.at<float>(3);
			if (X3D.at<float>(2) < 0.0) {
				std::cout << X3D.t() << ", " << val << std::endl;
				continue;
			}
			///////////////reprojection error
			X3D = X3D.rowRange(0, 3);
			cv::Mat proj1 = X3D.clone();
			cv::Mat proj2 = R1*X3D + t1;
			
			proj1 = pRef->mK*proj1;
			proj2 = pTarget->mK*proj2;
			
			cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
			cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));
			auto pt1 = vTempMatchPts1[i];
			auto pt2 = vTempMatchPts2[i];
			auto diffPt1 = projected1 - pt1;
			auto diffPt2 = projected2 - pt2;
			float err1 = (diffPt1.dot(diffPt1));
			float err2 = (diffPt2.dot(diffPt2));
			if (err1 > 4.0 || err2 > 4.0){
				//std::cout << "asdfasdfasdF::" <<err1<<"< "<<err2<< std::endl;
				continue;
			}
			///////////////reprojection error

			int idx = vTempMatchIDXs[i];
			int idx2 = vTempIndexs[idx];
			int idx3 = vMatches[idx2];

			//int label1 = mpInitFrame1->matLabeled.at<uchar>(pt1.y / 2, pt1.x / 2);
			auto pMP = new UVR_SLAM::MapPoint(pSystem->mpMap, pRef, X3D, pTarget->matDescriptor.row(idx3), 0);
			pMP->SetOptimization(true); //삭제하기
			tempMPs.push_back(pMP);
			vTempMappedPts1.push_back(vTempMatchPts1[i]);
			vTempMappedPts2.push_back(vTempMatchPts2[i]);
			vTempMappedIDXs.push_back(i);//vTempMatchIDXs[i]

			pRef->AddMapPoint(pMP, idx2);
			pMP->AddObservation(pRef, idx2);

			pTarget->AddMapPoint(pMP, idx3);
			pMP->AddObservation(pTarget, idx3);

			pMP->IncreaseVisible(2);
			pMP->IncreaseFound(2);
			pSystem->mlpNewMPs.push_back(pMP);

		}
		std::cout << tempMPs.size() << std::endl;
		//////////////////////////////////////
		/////median depth 
		float medianDepth;
		pRef->ComputeSceneMedianDepth(tempMPs, R1, t1, medianDepth);
		float invMedianDepth = 1.0f / medianDepth;
		if (medianDepth < 0.0) {
#ifdef DEBUG_TRACKING_LEVEL_3
			std::cout << "Mapping::Initialization::Median Depth Error=" << std::endl;
#endif
			return false;
		}
		for (int i = 0; i < tempMPs.size(); i++) {
			UVR_SLAM::MapPoint* pMP = tempMPs[i];
			if (!pMP)
				continue;
			if (pMP->isDeleted())
				continue;
			pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);

		}
		t1 *= invMedianDepth;
		//////////////////////////////////////

		//////////////////////////키프레임 생성
		//////////카메라 자세 변환 안하는 경우
		pRef->SetPose(cv::Mat::eye(3, 3, CV_32FC1), cv::Mat::zeros(3, 1, CV_32FC1));
		pTarget->SetPose(R1, t1); //두번째 프레임은 median depth로 변경해야 함.
								  //////////카메라 자세 변환 안하는 경우
								  //////////카메라 자세 변환 하는 경우
								  //mpInitFrame1->SetPose(cv::Mat::eye(3, 3, CV_32FC1)*Rcw, cv::Mat::zeros(3, 1, CV_32FC1));
								  //mpInitFrame2->SetPose(R1*Rcw, t1); //두번째 프레임은 median depth로 변경해야 함.
								  //////////카메라 자세 변환 하는 경우

								  ////이거 2개는 키프레임으로 설정.
								  ////키프레임 쓰레드는 필요하다.
								  ////매핑과 루프 클로징도 다시 추가가 필요하다.
								  ////매핑에서 일단 그 쓰레드를 변경한 새로운 쓰레드를 만든다.
								  ////매핑, 루프 클로징, 옵티마이제이션, 비쥬얼라이제이션 총 4개의 쓰레드 추가 필요
								  ////일단 pts와 관련 mp를 보내야 한다.

		pRef->AddKF(pTarget, tempMPs.size());
		pTarget->AddKF(pRef, tempMPs.size());
		pRef->ComputeBoW();
		pTarget->ComputeBoW();

		pSystem->mpMap->AddFrame(pRef);
		pSystem->mpMap->AddFrame(pTarget);

		//nRefID = pTarget->mnFrameID;
		//pRef->Init(mpSystem->mpORBExtractor, mpSystem->mK, mpSystem->mD);
		
#ifdef DEBUG_TRACKING_LEVEL_3
		std::cout << "Mapping::Initialization::Success=" << pRef->mnFrameID << ", " << pTarget->mnFrameID << "::" << tempMPs.size() << std::endl;
#endif
		bReplace = true;
		return true;
	};
	auto lambda_api_tracking = [](System* pSystem, Frame* pRef, Frame* pTarget, int& nRefID, std::vector<int> vMatches) {
		/*cv::Mat debugMatch;
		cv::Mat prevImg = pRef->mOriginalImage.clone();
		cv::Mat currImg = pTarget->mOriginalImage.clone();
		cv::Point2f ptBottom = cv::Point2f(prevImg.cols, 0);
		cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
		cv::Rect mergeRect2 = cv::Rect(prevImg.cols, 0, prevImg.cols, prevImg.rows);
		debugMatch = cv::Mat::zeros(prevImg.rows, prevImg.cols * 2, prevImg.type());
		prevImg.copyTo(debugMatch(mergeRect1));
		currImg.copyTo(debugMatch(mergeRect2));*/

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		std::vector < cv::Point2f> vTempPts;
		std::vector<MapPoint*> vpTempMPs;
		std::vector<bool> vbTempInliers;
		std::vector<int> vnTempIDXs;
		int nMatch = 0;
		auto vpMPs = pRef->GetMapPoints();
		std::vector<bool> vecBoolOverlap(pTarget->mvPts.size(), false);

		cv::Mat R, t;
		pRef->GetPose(R, t);
		pTarget->SetPose(R, t);

		int nTargetID = pTarget->mnFrameID;
		for (size_t i = 0, iend = vMatches.size(); i < iend; i++) {
			int idx1 = i;
			int idx2 = vMatches[i];
			if (idx2 == CODE_MATCH_ERROR_MAPPING_SERVER)
				continue;
			if (idx2 > CODE_MATCH_ERROR_MAPPING_SERVER || idx2 < -1) {
				vMatches[idx1] = CODE_MATCH_ERROR_MAPPING_SERVER;
				//std::cout << "Error::Matching::Invalid Frame2 Indexs = " << idx2 << ", " << pKF->mvPts.size() << "::" << j << std::endl;
				continue;
			}
			
			auto pMP = vpMPs[idx1];
			if (!pMP || pMP->isDeleted())
				continue;
			if (pMP->mnTrackingID == nTargetID)
				continue;
			if (vecBoolOverlap[idx2])
			{
				vMatches[idx1] = CODE_MATCH_ERROR_MAPPING_SERVER;
				continue;
			}
			vecBoolOverlap[idx2] = true;
			pMP->mnTrackingID = nTargetID;
			vpTempMPs.push_back(pMP);
			vTempPts.push_back(pTarget->mvPts[idx2]);
			vbTempInliers.push_back(true);
			vnTempIDXs.push_back(idx2);
			nMatch++;

			//auto pt1 = pRef->mvPts[idx1];
			//auto pt2 = pTarget->mvPts[idx2];
			///*cv::Mat X3D = pMP->GetWorldPos();
			//cv::Mat temp = K*(R*X3D + t);
			//float depth = temp.at < float>(2);
			//cv::Point2f proj(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
			//std::cout << X3D.t() << " ," << pt1 << ", " << pt2 << "< " << proj << std::endl;*/

			//cv::line(debugMatch, pt1, pt2+ptBottom, cv::Scalar(255, 0, 255), 1);
			//cv::circle(debugMatch, pt1, 3, cv::Scalar(255, 255, 0), -1);
			//cv::circle(debugMatch, pt2+ ptBottom, 3, cv::Scalar(255, 255, 0), -1);
		}
		int nPoseRecovery = Optimization::PoseOptimization(pSystem->mpMap, pTarget, vpTempMPs, vTempPts, vbTempInliers);

		for (size_t i = 0; i < vbTempInliers.size(); i++)
		{
			vpTempMPs[i]->IncreaseVisible();
			if (vbTempInliers[i]) {
				vpTempMPs[i]->IncreaseFound();
			}
		}

		float ratio = ((float)nPoseRecovery) / nMatch;

		bool b1 = ratio < TH_RATIO_MP_KP;
		bool b2 = nPoseRecovery < TH_NUM_MP_MATCH;
		bool b3 = nMatch < TH_NUM_KP_MATCH;
		bool b4 = nRefID + 15 <= nTargetID;

		if (b1 || b2 || b3 || b4) {
			////KF update
			//nRefID = pTarget->mnFrameID;
			if (!pSystem->mpLocalMapper->isDoingProcess()) {
				for (size_t i = 0; i < vbTempInliers.size(); i++)
				{
					if (vbTempInliers[i]) {
						int idx = vnTempIDXs[i];
						pTarget->AddMapPoint(vpTempMPs[i], idx);
						//vpTempMPs[i]->AddObservation(pTarget, idx);
					}
				}
				////KF update
				pSystem->mpLocalMapper->InsertKeyFrame(pTarget);
			}
		}
		std::chrono::high_resolution_clock::time_point end3 = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start).count();
		float t_test1 = du_test1 / 1000.0;
#ifdef DEBUG_TRACKING_LEVEL_3
		std::cout << "MappingServer::Tracking::MapPoints = " << nPoseRecovery << ", KeyPoints" << nMatch << "::" << t_test1 << std::endl;
#endif
		//imshow("match test", debugMatch); cv::waitKey(1);
	};

	MappingServer::MappingServer() {}
	MappingServer::MappingServer(System* pSys):mpSystem(pSys), mbInitialized(false), mnReferenceID(-1){}
	MappingServer::~MappingServer() {}
	void MappingServer::Reset() {
		mmFrames.clear();
		mmKeyFrames.clear();
		mnReferenceID = -1;
		mbInitialized = false;
	}
	void MappingServer::Init() {
		mpMap = mpSystem->mpMap;
		mpMatcher = mpSystem->mpMatcher;
		mpInitializer = mpSystem->mpInitializer;
		mpLocalMapper = mpSystem->mpLocalMapper;
		mpLoopCloser = mpSystem->mpLoopCloser;
		mpMapOptimizer = mpSystem->mpMapOptimizer;
		mpVisualizer = mpSystem->mpVisualizer;
		UVR_SLAM::ObjectColors::Init();
	}
	void MappingServer::InsertFrame(std::pair<std::string, int> pairInfo) {
		std::unique_lock<std::mutex> lock(mMutexQueue);
		mQueue.push(pairInfo);
	}
	bool MappingServer::CheckNewFrame() {
		std::unique_lock<std::mutex> lock(mMutexQueue);
		return(!mQueue.empty());
	}
	void MappingServer::AcquireFrame() {
		{
			std::unique_lock<std::mutex> lock(mMutexQueue);
			mPairFrameInfo = mQueue.front();
			mQueue.pop();
		}
	}
	void MappingServer::ProcessNewFrame() {
		std::string strUser = mPairFrameInfo.first;
		auto user = mpSystem->mmpConnectedUserList[strUser];
		std::string strMap = user->mapName;
		int nID = mPairFrameInfo.second;
		Frame* pNewF = new UVR_SLAM::Frame(mpSystem, nID, user->mnWidth, user->mnHeight, user->K, user->InvK, 0.0);
		pNewF->mstrMapName = strMap;
		mmFrames[nID] = pNewF;
		
		if (mnReferenceID == -1) {
			lambda_api_detect(mpSystem->ip, mpSystem->port, pNewF, strMap);
			//auto ftest = std::async(std::launch::async, lambda_api_detect, mpSystem->ip, mpSystem->port, pNewF);
			WebAPI* mpAPI = new WebAPI(mpSystem->ip, mpSystem->port);
			std::stringstream ss;
			ss << "/SetLastFrameID?map="<<strMap<<"&id=" << pNewF->mnFrameID << "&key=reference";
			mpAPI->Send(ss.str(),"");
			/*ss << "{\"id\":" << (int)pNewF->mnFrameID << ",\"key\":\"reference\"}";
			mpAPI->Send("/SetLastFrameID", ss.str());*/
			mnReferenceID = pNewF->mnFrameID;
		} 
		else 
		{
			WebAPI* mpAPI = new WebAPI(mpSystem->ip, mpSystem->port);
			std::stringstream ss;
			ss << "/GetLastFrameID?map="<<strMap<<"&key=reference";
			WebAPIDataConverter::ConvertStringToNumber(mpAPI->Send(ss.str(), "").c_str(), mnReferenceID);
			//std::cout << "Last Reference = " << mnReferenceID << std::endl;
			auto pRef = mmFrames[mnReferenceID];
			auto matches = lambda_api_detectAndmatch(mpSystem->ip, mpSystem->port, pRef, pNewF, strMap);
			if (mbInitialized) {
				std::async(std::launch::async, [](std::string ip, int port, Frame* pF) {
					WebAPI* mpAPI = new WebAPI(ip, port);
					std::stringstream ss;
					ss << "/SendData?map=" << pF->mstrMapName << "&id=" << pF->mnFrameID << "&key=bdesc";
					WebAPIDataConverter::ConvertBytesToDesc(mpAPI->Send(ss.str(), "").c_str(), pF->mvPts.size(), pF->matDescriptor);
					pF->ComputeBoW();
				},mpSystem->ip, mpSystem->port, pNewF);
				UVR_SLAM::lambda_api_tracking(mpSystem, pRef, pNewF, mnReferenceID, matches);//user->mpLastFrame->mnFrameID
				////그리드 기반 로컬 맵 계산 후 매칭
				mpVisualizer->SetBoolDoingProcess(true);
				
			}
			else {
				bool bReplace = false;
				mbInitialized = UVR_SLAM::lambda_api_Initialization(mpSystem->ip, mpSystem->port, mpSystem, pRef, pNewF, mnReferenceID, matches, strMap, bReplace);
				if (bReplace) {
					std::stringstream ss;
					ss << "/SetLastFrameID?map=" << strMap << "&id=" << pNewF->mnFrameID << "&key=reference";
					mpAPI->Send(ss.str(), "");
				}
				/*if (mbInitialized) {
					mpMapOptimizer->InsertKeyFrame(pNewF);
				}*/
			}
			////서버에 결과 전송하기
		}
		user->mpLastFrame = pNewF;
	}
	void MappingServer::RunWithMappingServer() {

		std::cout << "MappingServer::Thread::Start" << std::endl;
		while (true) {
			if (CheckNewFrame()) {
				AcquireFrame();
				std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
#ifdef DEBUG_TRACKING_LEVEL_1
				std::cout << "MappingServer::Thread::TargetID = " << mPairFrameInfo.first <<", "<<mnReferenceID<<"::"<< mQueue .size()<<"::Start"<< std::endl;
#endif
				////user 마다 다르게
				ProcessNewFrame();
				//if(mpSystem->mbMapping)
				//else {
				//	//일단 초기 위치 추정
				//	std::string strUser = mPairFrameInfo.first;
				//	auto user = mpSystem->mmpConnectedUserList[strUser];
				//	std::string strMap = user->mapName;
				//	int nID = mPairFrameInfo.second;
				//	Frame* pNewF = new UVR_SLAM::Frame(mpSystem, user, nID, user->mnWidth, user->mnHeight, user->K, 0.0);
				//	pNewF->mstrMapName = strMap;
				//	lambda_api_detect(mpSystem->ip, mpSystem->port, pNewF, strMap);
				//	mpLoopCloser->InsertKeyFrame(pNewF);
				//}

				std::chrono::high_resolution_clock::time_point end3 = std::chrono::high_resolution_clock::now();
				auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start).count();
				float t_test1 = du_test1 / 1000.0;
#ifdef DEBUG_TRACKING_LEVEL_1
				std::cout << "MappingServer::Thread::TargetID = " << mPairFrameInfo.first <<"::"<< t_test1 <<"::End"<< std::endl;
#endif
			}//checknewframe
		}//while
	}
}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               