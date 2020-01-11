#include <LocalMapper.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <MapPoint.h>
#include <Matcher.h>
#include <Optimization.h>
#include <PlaneEstimator.h>
#include <SemanticSegmentator.h>

UVR_SLAM::LocalMapper::LocalMapper(){}
UVR_SLAM::LocalMapper::LocalMapper(int w, int h):mnWidth(w), mnHeight(h), mbStopBA(false), mbDoingProcess(false){}
UVR_SLAM::LocalMapper::~LocalMapper() {}

void UVR_SLAM::LocalMapper::InsertKeyFrame(UVR_SLAM::Frame *pKF)
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mKFQueue.push(pKF);
	std::cout << "insertkeyframe::queue size = " << mKFQueue.size() << std::endl;
	mbStopBA = true;
}

bool UVR_SLAM::LocalMapper::CheckNewKeyFrames()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	return(!mKFQueue.empty());
}

void UVR_SLAM::LocalMapper::ProcessNewKeyFrame()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mpTargetFrame = mKFQueue.front();
	mKFQueue.pop();
	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
	mpFrameWindow->SetLastFrameID(mpTargetFrame->GetFrameID());
	mbStopBA = false;

	////이게 필요한지?
	//이전 키프레임 정보 획득 후 현재 프레임을 윈도우에 추가
	//mpPrevKeyFrame = mpFrameWindow->back();
	//mpFrameWindow->push_back(mpTargetFrame);
	
}

bool UVR_SLAM::LocalMapper::isDoingProcess(){
	std::unique_lock<std::mutex> lock(mMutexDoingProcess);
	return mbDoingProcess;
}
void UVR_SLAM::LocalMapper::SetDoingProcess(bool flag){
	std::unique_lock<std::mutex> lock(mMutexDoingProcess);
	mbDoingProcess = flag;
}

//void UVR_SLAM::LocalMapper::InterruptLocalMapping() {
//	std::unique_lock<std::mutex> lock(mMutexNewKFs);
//	mbStopBA = true;
//}

void UVR_SLAM::LocalMapper::Run() {
	while (1) {

		//???????????????????????
		SetDoingProcess(true);

		if (CheckNewKeyFrames()) {
			std::cout << "LocalMap::1" << std::endl;
			ProcessNewKeyFrame();
			std::cout << "LocalMap::2" << std::endl;
			CalculateKFConnections();
			std::cout << "LocalMap::3" << std::endl;
			UpdateKFs();
			std::cout << "LocalMap::4" << std::endl;
			////이전 프레임에서 생성된 맵포인트 중 삭제
			//프레임 윈도우 내의 로컬 맵 포인트 중 new인 애들만 수행
			NewMapPointMaginalization();

			UpdateMPs();

			std::cout << "LocalMap::5" << std::endl;
			//프레임 내에서 삭제 되는 녀석과 업데이트 되는 녀석의 분리가 필요함.

			std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
			int nCreateMP = Test();
			std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
			auto du_test = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
			double t_test = du_test / 1000.0;
			std::cout << "LocalMap::kf::TEST::" << t_test<<"::Create "<< nCreateMP<< std::endl;

			//marginalization test
			/*std::chrono::high_resolution_clock::time_point kf_mar_start = std::chrono::high_resolution_clock::now();
			KeyframeMarginalization();
			std::chrono::high_resolution_clock::time_point kf_mar_end = std::chrono::high_resolution_clock::now();
			auto du_kf_mar = std::chrono::duration_cast<std::chrono::milliseconds>(kf_mar_end - kf_mar_start).count();
			double t_kf_mar = du_kf_mar / 1000.0;
			std::cout << "LocalMap::kf::marginalization::"<<t_kf_mar<< std::endl;*/
			
			std::cout << "LocalMap::6" << std::endl;
			////새로운 맵포인트 생성
			//여기부터 다시 검증이 필요
			//CreateMapPoints(mpTargetFrame, mpPrevKeyFrame);

			//create map points
			std::chrono::high_resolution_clock::time_point cm_start = std::chrono::high_resolution_clock::now();
			//CreateMapPoints();
			std::chrono::high_resolution_clock::time_point cm_end = std::chrono::high_resolution_clock::now();
			auto du_cm = std::chrono::duration_cast<std::chrono::milliseconds>(cm_end - cm_start).count();
			double t_cm = du_cm / 1000.0;
			std::cout << "LocalMap::CM::" << t_cm << std::endl;
			std::cout << "LocalMap::7" << std::endl;
			//fuse
			if (!CheckNewKeyFrames())
			{
				std::cout << "LocalMap::81" << std::endl;
				std::chrono::high_resolution_clock::time_point fuse_start = std::chrono::high_resolution_clock::now();
				FuseMapPoints();
				std::chrono::high_resolution_clock::time_point fuse_end = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(fuse_end - fuse_start).count();
				double tttt = duration / 1000.0;
				mpFrameWindow->SetFuseTime(tttt);
				std::cout << "LocalMap::82" << std::endl;
			}
			////BA
			//BA에서는 최근 생성된 맵포인트까지 반영을 해야 함.
			bool btest = false;
			if(!CheckNewKeyFrames()){
				std::cout << "LocalMap::91" << std::endl;
				Optimization::LocalBundleAdjustment(mpFrameWindow, mpTargetFrame->GetFrameID(), btest, 2, 5, false);
				std::cout << "LocalMap::92" << std::endl;
			}
			//while (mpFrameWindow->isUseLocalMap()){}
			std::cout << "LocalMap::931" << std::endl;
			//mpFrameWindow->SetUseLocalMap(true);
			mpFrameWindow->SetLocalMap(mpTargetFrame->GetFrameID());
			//mpFrameWindow->SetUseLocalMap(false);
			std::cout << "LocalMap::932" << std::endl;
			SetDoingProcess(false);
			//std::cout << "Create KeyFrame::End!!!!" << std::endl;
		}
	}//while
}

void UVR_SLAM::LocalMapper::SetMatcher(UVR_SLAM::Matcher* pMatcher) {
	mpMatcher = pMatcher;
}

void UVR_SLAM::LocalMapper::SetLayoutEstimator(SemanticSegmentator* pEstimator) {
	mpSegmentator = pEstimator;
}

void UVR_SLAM::LocalMapper::SetPlaneEstimator(PlaneEstimator* pPlaneEstimator) {
	mpPlaneEstimator = pPlaneEstimator;
}

void UVR_SLAM::LocalMapper::SetFrameWindow(UVR_SLAM::FrameWindow* pFrameWindow) {
	mpFrameWindow = pFrameWindow;
}

//맵포인트가 삭제 되면 현재 프레임에서도 해당 맵포인트를 삭제 해야 하며, 
//이게 수행되기 전에는 트래킹이 동작하지 않도록 막아야 함.
//
void UVR_SLAM::LocalMapper::NewMapPointMaginalization() {
	//std::cout << "Maginalization::Start" << std::endl;
	//mvpDeletedMPs.clear();
	int nMarginalized = 0;

	std::list<UVR_SLAM::MapPoint*>::iterator lit = mlpNewMPs.begin();
	while (lit != mlpNewMPs.end()) {
		UVR_SLAM::MapPoint* pMP = *lit;
		bool bBad = false;
		if (pMP->isDeleted()) {
			//already deleted
			lit = mlpNewMPs.erase(lit);
		}
		else if (pMP->GetFVRatio() < 0.25f) {
			//pMP->Delete();
			bBad = true;
			lit = mlpNewMPs.erase(lit);
		}
		else if (pMP->mnFirstKeyFrameID + 2 <= mpTargetFrame->GetKeyFrameID() && pMP->GetNumConnectedFrames() <= 2) {
			bBad = true;
			lit = mlpNewMPs.erase(lit);
		}
		else if (pMP->mnFirstKeyFrameID + 3 <= mpTargetFrame->GetKeyFrameID())
			lit = mlpNewMPs.erase(lit);
		else
			lit++;
		if (bBad) {
			//mpFrameWindow->SetMapPoint(nullptr, i);
			//mpFrameWindow->SetBoolInlier(false, i);
			//frame window와 현재 키프레임에 대해서 삭제될 포인트 처리가 필요할 수 있음.
			pMP->SetDelete(true);
			pMP->Delete();
		}
	}

	return;
}

//window에 포함되는 KF를 설정하기.
//너무 많은 KF가 포함안되었으면 하고, 
//MP들이 잘 분배되었으면 함.
//lastframeid의 역할은?
void UVR_SLAM::LocalMapper::UpdateKFs() {
	mpFrameWindow->ClearLocalMapFrames();
	auto mvpConnectedKFs = mpTargetFrame->GetConnectedKFs();
	mpFrameWindow->AddFrame(mpTargetFrame);
	for (auto iter = mvpConnectedKFs.begin(); iter != mvpConnectedKFs.end(); iter++) {
		mpFrameWindow->AddFrame(*iter);
	}
}

void UVR_SLAM::LocalMapper::UpdateMPs() {
	int nUpdated = 0;
	for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mpTargetFrame->mvpMPs[i];
		if (pMP) {
			if (pMP->isDeleted()) {
				mpTargetFrame->RemoveMP(i);
			}
			else {
				nUpdated++;
				pMP->AddFrame(mpTargetFrame, i);
			}
		}
	}
	//std::cout << "Update MPs::" << nUpdated << std::endl;
}

void UVR_SLAM::LocalMapper::DeleteMPs() {
	for (int i = 0; i < mvpDeletedMPs.size(); i++) {
		delete mvpDeletedMPs[i];
	}
}

void UVR_SLAM::LocalMapper::FuseMapPoints()
{
	int nn = 15;
	int nTargetID = mpTargetFrame->GetFrameID();
	const auto vpNeighKFs = mpTargetFrame->GetConnectedKFs(nn);

	std::vector<UVR_SLAM::Frame*> vpTargetKFs;
	for (std::vector<UVR_SLAM::Frame*>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
	{
		UVR_SLAM::Frame* pKF = *vit;
		if (pKF->mnFuseFrameID == nTargetID)
			continue;
		pKF->mnFuseFrameID = nTargetID;
		vpTargetKFs.push_back(pKF);

		const auto vpTempNeighKFs = mpTargetFrame->GetConnectedKFs(15);
		for (std::vector<UVR_SLAM::Frame*>::const_iterator vit2 = vpTempNeighKFs.begin(), vend2 = vpTempNeighKFs.end(); vit2 != vend2; vit2++)
		{
			UVR_SLAM::Frame* pKF2 = *vit2;
			if (pKF2->mnFuseFrameID == nTargetID || pKF2->GetFrameID() == nTargetID)
				continue;
			pKF2->mnFuseFrameID = nTargetID;
			vpTargetKFs.push_back(pKF2);
		}
	}
	std::cout << "LocalMapper::Fuse::" << vpTargetKFs.size() << std::endl;
	std::vector<MapPoint*> vpMapPointMatches = mpTargetFrame->GetMapPoints();
	for (int i = 0; i < vpTargetKFs.size(); i++) {
		if (CheckNewKeyFrames())
			break;
		int n1 = mpMatcher->MatchingForFuse(vpMapPointMatches, vpTargetKFs[i]);
		std::vector<MapPoint*> vpMapPointMatches2 = vpTargetKFs[i]->GetMapPoints();
		int n2 = mpMatcher->MatchingForFuse(vpMapPointMatches2, mpTargetFrame);
		std::cout << "LocalMapper::MatchingFuse::" << n1<<", "<<n2 << std::endl;
	}
}
int UVR_SLAM::LocalMapper::CreateMapPoints() {
	int nRes = 0;
	auto mvpConnectedKFs = mpTargetFrame->GetConnectedKFs();
	for (int i = 0; i < mvpConnectedKFs.size(); i++) {
		if (CheckNewKeyFrames()){
			std::cout << "LocalMapping::CreateMPs::Break::" << std::endl;
			break;
		}
		nRes += CreateMapPoints(mpTargetFrame, mvpConnectedKFs[i]);
	}
	std::cout << "LocalMapping::CreateMPs::End::" << nRes << std::endl;
	return nRes;
}

int UVR_SLAM::LocalMapper::CreateMapPoints(UVR_SLAM::Frame* pCurrKF, UVR_SLAM::Frame* pLastKF) {
	//debugging image
	cv::Mat lastImg = pLastKF->GetOriginalImage();
	cv::Mat currImg = pCurrKF->GetOriginalImage();
	//cv::Rect mergeRect1 = cv::Rect(0, 0, lastImg.cols, lastImg.rows);
	//cv::Rect mergeRect2 = cv::Rect(lastImg.cols, 0, lastImg.cols, lastImg.rows);
	//cv::Mat debugging = cv::Mat::zeros(lastImg.rows, lastImg.cols * 2, lastImg.type());
	cv::Rect mergeRect1 = cv::Rect(0, 0,			lastImg.cols, lastImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, lastImg.rows, lastImg.cols, lastImg.rows);
	cv::Mat debugging = cv::Mat::zeros(lastImg.rows * 2, lastImg.cols, lastImg.type());
	lastImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));
	//cv::cvtColor(debugging, debugging, CV_RGBA2BGR);
	//debugging.convertTo(debugging, CV_8UC3);

	//preprocessing
	bool bNearBaseLine = false;
	if (!pCurrKF->CheckBaseLine(pCurrKF, pLastKF)) {
		std::cout << "CreateMapPoints::Baseline error" << std::endl;
		bNearBaseLine = true;
		return 0;
	}

	cv::Mat Rprev, Tprev, Rcurr, Tcurr;
	pLastKF->GetPose(Rprev, Tprev);
	pCurrKF->GetPose(Rcurr, Tcurr);

	cv::Mat mK = pCurrKF->mK.clone();

	cv::Mat RprevInv = Rprev.t();
	cv::Mat RcurrInv = Rcurr.t();
	float invfx = 1.0 / mK.at<float>(0, 0);
	float invfy = 1.0 / mK.at<float>(1, 1);
	float cx = mK.at<float>(0, 2);
	float cy = mK.at<float>(1, 2);
	float ratioFactor = 1.5f*pCurrKF->mfScaleFactor;

	cv::Mat P0 = cv::Mat::zeros(3, 4, CV_32FC1);
	Rprev.copyTo(P0.colRange(0, 3));
	Tprev.copyTo(P0.col(3));
	cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32FC1);
	Rcurr.copyTo(P1.colRange(0, 3));
	Tcurr.copyTo(P1.col(3));
	
	cv::Mat O1 = pLastKF->GetCameraCenter();
	cv::Mat O2 = pCurrKF->GetCameraCenter();

	cv::Mat F12 = mpMatcher->CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);
	int thresh_epi_dist = 50;
	float thresh_reprojection = 16.0;
	int count = 0;

	cv::RNG rng(12345);

	//두 키프레임 사이의 매칭 정보 초기화
	//mpFrameWindow->mvMatchInfos.clear();

	for (int i = 0; i < pCurrKF->mvKeyPoints.size(); i++) {
		if (pCurrKF->mvbMPInliers[i])
			continue;
		int matchIDX, kidx;
		cv::KeyPoint kp = pCurrKF->mvKeyPoints[i];
		cv::Point2f pt = pCurrKF->mvKeyPoints[i].pt;
		cv::Mat desc = pCurrKF->matDescriptor.row(i);

		float sigma = pCurrKF->mvLevelSigma2[kp.octave];
		bool bMatch = mpMatcher->FeatureMatchingWithEpipolarConstraints(matchIDX, pLastKF, F12, kp, desc, sigma, thresh_epi_dist);
		if (bMatch) {
			if (!pLastKF->mvbMPInliers[matchIDX]) {
				
				cv::KeyPoint kp2 = pLastKF->mvKeyPoints[matchIDX];
				cv::Mat X3D     = Triangulate(kp2.pt, kp.pt, mK*P0, mK*P1);
				cv::Mat Xcam1 = Rprev*X3D + Tprev;
				cv::Mat Xcam2 = Rcurr*X3D + Tcurr;
				//SetLogMessage("Triangulation\n");
				if (!CheckDepth(Xcam1.at<float>(2)) || !CheckDepth(Xcam2.at<float>(2))) {
					continue;
				}

				if (!CheckReprojectionError(Xcam1, mK, kp2.pt, thresh_reprojection) || !CheckReprojectionError(Xcam2, mK, kp.pt, thresh_reprojection))
				{
					continue;
				}

				if (!CheckScaleConsistency(X3D, O2, O1, ratioFactor, pCurrKF->mvScaleFactors[kp.octave], pLastKF->mvScaleFactors[kp2.octave]))
				{
					continue;
				}

				UVR_SLAM::MapPoint* pMP2 = new UVR_SLAM::MapPoint(X3D, desc);
				pMP2->AddFrame(pLastKF, matchIDX);
				pMP2->AddFrame(pCurrKF, i);
				pMP2->mnFirstKeyFrameID = pCurrKF->GetKeyFrameID();
				mlpNewMPs.push_back(pMP2);
				//mvpNewMPs.push_back(pMP2);
				//mDescNewMPs.push_back(pMP2->GetDescriptor());

				//매칭 정보 추가
				cv::DMatch tempMatch;
				tempMatch.queryIdx = i;
				tempMatch.trainIdx = matchIDX;
				//mpFrameWindow->mvMatchInfos.push_back(tempMatch);

				cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				//cv::line(debugging, pLastKF->mvKeyPoints[matchIDX].pt, pCurrKF->mvKeyPoints[i].pt + cv::Point2f(lastImg.cols, 0), cv::Scalar(255, 0, 255), 1);
				cv::line(debugging, pLastKF->mvKeyPoints[matchIDX].pt, pCurrKF->mvKeyPoints[i].pt + cv::Point2f(0, lastImg.rows), color, 1);
				count++;
			}
		}
	}
	cv::imshow("LocalMapping::CreateMPs", debugging); cv::waitKey(10);
	//std::cout << "CreateMapPoints=" << count << std::endl;
	return count;
}


cv::Mat UVR_SLAM::LocalMapper::Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2) {

	cv::Mat A(4, 4, CV_32F);
	A.row(0) = pt1.x*P1.row(2) - P1.row(0);
	A.row(1) = pt1.y*P1.row(2) - P1.row(1);
	A.row(2) = pt2.x*P2.row(2) - P2.row(0);
	A.row(3) = pt2.y*P2.row(2) - P2.row(1);

	cv::Mat u, w, vt;
	cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	cv::Mat x3D = vt.row(3).t();
	x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
	return x3D;
}

bool UVR_SLAM::LocalMapper::CheckDepth(float depth) {
	if (depth < 0)
		return false;
	return true;
}

bool UVR_SLAM::LocalMapper::CheckReprojectionError(cv::Mat x3D, cv::Mat K, cv::Point2f pt, float thresh) {
	cv::Mat reproj1 = K*x3D;
	reproj1 /= x3D.at<float>(2);
	float squareError1 = (reproj1.at<float>(0) - pt.x)*(reproj1.at<float>(0) - pt.x) + (reproj1.at<float>(1) - pt.y)*(reproj1.at<float>(1) - pt.y);
	if (squareError1>thresh)
		return false;
	return true;
}

bool UVR_SLAM::LocalMapper::CheckScaleConsistency(cv::Mat x3D, cv::Mat Ow1, cv::Mat Ow2, float fRatioFactor, float fScaleFactor1, float fScaleFactor2) {
	cv::Mat normal1 = x3D - Ow1;
	float dist1 = cv::norm(normal1);

	cv::Mat normal2 = x3D - Ow2;
	float dist2 = cv::norm(normal2);

	if (dist1 == 0 || dist2 == 0)
		return false;

	const float ratioDist = dist2 / dist1;
	const float ratioOctave = fScaleFactor1 / fScaleFactor2;

	if (ratioDist*fRatioFactor<ratioOctave || ratioDist>ratioOctave*fRatioFactor)
		return false;
	return true;
}

void UVR_SLAM::LocalMapper::CalculateKFConnections() {
	std::map<UVR_SLAM::Frame*, int> mmpCandidateKFs;
	int nTargetID = mpTargetFrame->GetFrameID();
	auto mvpTemporalCandidateKFs = mpFrameWindow->GetLocalMapFrames();
	std::cout << "LocalMapper::" << mvpTemporalCandidateKFs.size() << std::endl;
	std::cout << "LocalMapper::TargetKF::" << nTargetID << std::endl;
	
	for (int i = 0; i < mvpTemporalCandidateKFs.size(); i++) {
		if (nTargetID == mvpTemporalCandidateKFs[i]->GetFrameID())
			continue;
		//std::cout << "LocalMapper::Window::" <<i<<"::"<< mvpTemporalCandidateKFs[i]->GetFrameID() <<", "<<mvpTemporalCandidateKFs[i]->GetKeyFrameID()<< std::endl;
		
		mmpCandidateKFs[mvpTemporalCandidateKFs[i]] = 0;
		//mspCandidateKFs.insert(mvpTemporalCandidateKFs[i]);
		auto mvpTemp2 = mvpTemporalCandidateKFs[i]->GetConnectedKFs();
		for (int j = 0; j < mvpTemp2.size(); j++) {
			if (nTargetID == mvpTemp2[j]->GetFrameID())
				continue;
			mmpCandidateKFs[mvpTemp2[j]] = 0;
		}
	}
	
	int Nkf = mmpCandidateKFs.size();
	auto mvpLocalMPs = mpTargetFrame->GetMapPoints();

	for (int i = 0; i < mvpLocalMPs.size(); i++) {
		
		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		auto mmpMP = pMP->GetConnedtedFrames();
		for (auto biter = mmpMP.begin(), eiter = mmpMP.end(); biter != eiter; biter++) {
			UVR_SLAM::Frame* pCandidateKF = biter->first;
			if (nTargetID == pCandidateKF->GetFrameID())
				continue;
			/*if (mmpCandidateKFs.find(pCandidateKF) == mmpCandidateKFs.end()) {
				std::cout << "LocalMapping::Not connected kf" << std::endl;
			}*/
			mmpCandidateKFs[pCandidateKF]++;
		}
	}
	//sort mmp
	std::vector<std::pair<int,UVR_SLAM::Frame*>> vPairs;

	for (auto biter = mmpCandidateKFs.begin(), eiter = mmpCandidateKFs.end(); biter != eiter; biter++) {
		UVR_SLAM::Frame* pKF = biter->first;
		int nCount = biter->second;
		if (nCount > 20) {
			//mpTargetFrame->AddKF(pKF);
			vPairs.push_back(std::make_pair(nCount, pKF));
		}
	}
	int nKF = 0;
	int nThreshKF = 10;
	sort(vPairs.begin(), vPairs.end());
	//store frame and wegiths
	for (int i = vPairs.size()-1; i >=0 ; i--) {
		UVR_SLAM::Frame* pKF = vPairs[i].second;
		int nCount = vPairs[i].first;
		mpTargetFrame->AddKF(pKF);
		std::cout << "LocalMapping::Connection::" << pKF->GetFrameID() <<" | "<<pKF->GetKeyFrameID()<< ":: " << nCount << std::endl;

		nKF++;
		if (nKF > nThreshKF)
			break;
	}
	std::cout << "LocalMapping::Connected KFs::" << mpTargetFrame->GetConnectedKFs().size() << std::endl;
}
int UVR_SLAM::LocalMapper::Test() {
	auto mvpLocalFrames = mpTargetFrame->GetConnectedKFs();


	cv::Mat mK = mpTargetFrame->mK.clone();
	cv::Mat Rcurr, Tcurr;
	mpTargetFrame->GetPose(Rcurr, Tcurr);
	cv::Mat O2 = mpTargetFrame->GetCameraCenter();
	cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32FC1);
	Rcurr.copyTo(P1.colRange(0, 3));
	Tcurr.copyTo(P1.col(3));

	int nTotal = 0;

	for (int i = 0; i < mvpLocalFrames.size(); i++) {
		
		UVR_SLAM::Frame* pKF = mvpLocalFrames[i];

		//preprocessing
		bool bNearBaseLine = false;
		if (!pKF->CheckBaseLine(pKF, mpTargetFrame)) {
			std::cout << "CreateMapPoints::Baseline error" << std::endl;
			bNearBaseLine = true;
			continue;
		}

		cv::Mat Rprev, Tprev;
		pKF->GetPose(Rprev, Tprev);

		cv::Mat RprevInv = Rprev.t();
		cv::Mat RcurrInv = Rcurr.t();
		float invfx = 1.0 / mK.at<float>(0, 0);
		float invfy = 1.0 / mK.at<float>(1, 1);
		float cx = mK.at<float>(0, 2);
		float cy = mK.at<float>(1, 2);
		float ratioFactor = 1.5f*mpTargetFrame->mfScaleFactor;

		cv::Mat P0 = cv::Mat::zeros(3, 4, CV_32FC1);
		Rprev.copyTo(P0.colRange(0, 3));
		Tprev.copyTo(P0.col(3));
		
		cv::Mat O1 = pKF->GetCameraCenter();

		//cv::Mat F12 = mpMatcher->CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);
		//int thresh_epi_dist = 50;
		float thresh_reprojection = 16.0;
		int count = 0;


		pKF->mTrackedDescriptor = cv::Mat::zeros(0,pKF->matDescriptor.rows, pKF->matDescriptor.type());
		pKF->mNotTrackedDescriptor = cv::Mat::zeros(0, pKF->matDescriptor.rows, pKF->matDescriptor.type());
		
		pKF->mvTrackedIdxs.clear();
		pKF->mvNotTrackedIdxs.clear();

		for (int j = 0; j < pKF->mvpMPs.size(); j++) {
			UVR_SLAM::MapPoint* pMP = pKF->mvpMPs[j];
			bool bMatch = false;
			if (pMP) {
				if (!pMP->isDeleted()) {
					bMatch = true;
				}
			}
			if (bMatch) {
				pKF->mTrackedDescriptor.push_back(pKF->matDescriptor.row(j));
				pKF->mvTrackedIdxs.push_back(j);
			}
			else {
				pKF->mNotTrackedDescriptor.push_back(pKF->matDescriptor.row(j));
				pKF->mvNotTrackedIdxs.push_back(j);
			}
		}
		std::vector<cv::DMatch> vMatches;
		mpMatcher->KeyFrameFeatureMatching(mpTargetFrame, pKF, vMatches);
		int nTemp = 0;
		for (int j = 0; j < vMatches.size(); j++) {
			int idx1 = vMatches[j].queryIdx;
			int idx2 = vMatches[j].trainIdx;
			cv::KeyPoint kp1 = mpTargetFrame->mvKeyPoints[idx1];
			cv::KeyPoint kp2 = pKF->mvKeyPoints[idx2];
			cv::Mat X3D = Triangulate(kp2.pt, kp1.pt, mK*P0, mK*P1);
			cv::Mat Xcam1 = Rprev*X3D + Tprev;
			cv::Mat Xcam2 = Rcurr*X3D + Tcurr;
			//SetLogMessage("Triangulation\n");
			if (!CheckDepth(Xcam1.at<float>(2)) || !CheckDepth(Xcam2.at<float>(2))) {
				continue;
			}

			if (!CheckReprojectionError(Xcam1, mK, kp2.pt, thresh_reprojection) || !CheckReprojectionError(Xcam2, mK, kp1.pt, thresh_reprojection))
			{
				continue;
			}

			if (!CheckScaleConsistency(X3D, O2, O1, ratioFactor, mpTargetFrame->mvScaleFactors[kp1.octave], pKF->mvScaleFactors[kp2.octave]))
			{
				continue;
			}

			cv::Mat desc = mpTargetFrame->matDescriptor.row(idx1);
			UVR_SLAM::MapPoint* pMP = new UVR_SLAM::MapPoint(X3D, desc);
			pMP->AddFrame(pKF, idx2);
			pMP->AddFrame(mpTargetFrame, idx1);
			pMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
			mlpNewMPs.push_back(pMP);

			mpTargetFrame->mTrackedDescriptor.push_back(mpTargetFrame->matDescriptor.row(idx1));
			mpTargetFrame->mvTrackedIdxs.push_back(idx1);
			pKF->mTrackedDescriptor.push_back(pKF->matDescriptor.row(idx2));
			pKF->mvTrackedIdxs.push_back(idx2);

			nTemp++;
		}
		nTotal += nTemp;
	}
	return nTotal;
}

void UVR_SLAM::LocalMapper::KeyframeMarginalization() {

	int nThreshKF = 5;

	//auto mvpLocalMPs = mpTargetFrame->GetMapPoints();
	auto mvpLocalFrames = mpTargetFrame->GetConnectedKFs();
	int nKFs = mvpLocalFrames.size();
	int nMPs = 0;
	if (nKFs < nThreshKF)
		return;
	//여기에 true는 계속 나오는 MP이고 false는 별로 나오지 않는 MP이다.
	//없애는게 나을지도 모르는 것들
	/*std::vector<bool> mvbMPs(mvpLocalMPs.size(), false);
	for (int i = 0; i < mvpLocalMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		int nObs = pMP->GetNumConnectedFrames();
		double ratio = ((double)nObs) / nKFs;
		if (nObs > 2) {
			mvbMPs[i] = true;
			
		}
		else {
			nMPs++;
		}
	}*/

	auto mvpConnectedKFs = mpTargetFrame->GetConnectedKFs();
	for (int i = 0; i < mvpConnectedKFs.size(); i++) {
		mpMatcher->KeyFrameFuseFeatureMatching(mpTargetFrame, mvpConnectedKFs[i]);
	}

	std::cout << "TESt:::" << nMPs <<", "<< mvpConnectedKFs.size()<< std::endl;
}
