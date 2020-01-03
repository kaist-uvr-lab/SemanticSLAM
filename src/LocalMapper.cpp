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
	mbStopBA = false;

	////이게 필요한지?
	//이전 키프레임 정보 획득 후 현재 프레임을 윈도우에 추가
	//mpPrevKeyFrame = mpFrameWindow->back();
	//mpFrameWindow->push_back(mpTargetFrame);
	//mpFrameWindow->SetLastFrameID(mpTargetFrame->GetFrameID());
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

		SetDoingProcess(true);

		if (CheckNewKeyFrames()) {
			ProcessNewKeyFrame();
			CalculateKFConnections();
			UpdateKFs();

			////이전 프레임에서 생성된 맵포인트 중 삭제
			//프레임 윈도우 내의 로컬 맵 포인트 중 new인 애들만 수행
			NewMapPointMaginalization();
			//프레임 내에서 삭제 되는 녀석과 업데이트 되는 녀석의 분리가 필요함.
			UpdateMPs();
			////새로운 맵포인트 생성
			//여기부터 다시 검증이 필요
			//CreateMapPoints(mpTargetFrame, mpPrevKeyFrame);
			CreateMapPoints();

			//fuse
			if (!CheckNewKeyFrames())
			{
				std::chrono::high_resolution_clock::time_point fuse_start = std::chrono::high_resolution_clock::now();
				FuseMapPoints();
				std::chrono::high_resolution_clock::time_point fuse_end = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(fuse_end - fuse_start).count();
				double tttt = duration / 1000.0;
				mpFrameWindow->SetFuseTime(tttt);
			}
			////BA
			//BA에서는 최근 생성된 맵포인트까지 반영을 해야 함.
			bool btest = false;
			if(!CheckNewKeyFrames())
				Optimization::LocalBundleAdjustment(mpFrameWindow, mpTargetFrame->GetFrameID(), btest, 2, 5, false);
			mpFrameWindow->SetLocalMap(mpTargetFrame->GetFrameID());
			SetDoingProcess(false);
			//std::cout << "Create KeyFrame::End!!!!" << std::endl;
		}

		if (false) {
			std::cout << "Create KeyFrame::Start!!!!" << std::endl;
			std::cout << "Window size = " << mpFrameWindow->size() << std::endl;
			//키프레임으로 설정하고 프레임 윈도우에 추가
			
			////이전 프레임에서 생성된 맵포인트 중 삭제
			//프레임 윈도우 내의 로컬 맵 포인트 중 new인 애들만 수행
			NewMapPointMaginalization();
			//프레임 내에서 삭제 되는 녀석과 업데이트 되는 녀석의 분리가 필요함.
			UpdateMPs();
			////새로운 맵포인트 생성
			//여기부터 다시 검증이 필요
			CreateMapPoints(mpTargetFrame, mpPrevKeyFrame);

			////BA
			//BA에서는 최근 생성된 맵포인트까지 반영을 해야 함.
			//Optimization::LocalBundleAdjustment(mpFrameWindow, 2, 5, false);
			
			//Delete MPs
			//DeleteMPs();

			//평면 검출 쓰레드에 추가

			if (mpSegmentator->isRun() && !mpSegmentator->isDoingProcess()) {
				mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_SEGMENTED_FRAME);
				mpSegmentator->SetBoolDoingProcess(true);
				mpSegmentator->SetTargetFrame(mpTargetFrame);
				
			}/*else if (!mpPlaneEstimator->isDoingProcess()) {
				mpPlaneEstimator->SetBoolDoingProcess(true, 2);
				mpPlaneEstimator->SetTargetFrame(mpTargetFrame);
			}*/

			//이 시점에서는 로컬맵이 완성이 되고 삭제되는 일이 없음.
			//뮤텍스로 여기에서 한번 막으면 달라지는 것은?
			mpFrameWindow->SetLocalMap(mpTargetFrame->GetFrameID());
			//SetBoolDoingProcess(false);
			std::cout << "Create KeyFrame::End!!!!" << std::endl;
		}//if

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
	for (int i = 0; i < mpFrameWindow->GetLocalMapSize(); i++) {
		UVR_SLAM::MapPoint* pMP = mpFrameWindow->GetMapPoint(i);
		if (!pMP)
			continue;
		if (!pMP->isNewMP())
			continue;
		pMP->SetNewMP(false);
		if (pMP->GetMapPointType() == MapPointType::NORMAL_MP) {
			//if (!mpFrameWindow->GetBoolInlier(i))
			float ratio = pMP->GetFVRatio();
			if (ratio < 0.15) {
				mpFrameWindow->SetMapPoint(nullptr, i);
				mpFrameWindow->SetBoolInlier(false, i);
				pMP->SetDelete(true);
				pMP->Delete();
				//mvpDeletedMPs.push_back(pMP);
				nMarginalized++;
			}
		}
		else {
			//if (pMP->mnMatchingCount == 0) {
			//	mpFrameWindow->SetMapPoint(nullptr, i);
			//	mpFrameWindow->SetBoolInlier(false, i);
			//	pMP->SetDelete(true);
			//	pMP->Delete();
			//	//mvpDeletedMPs.push_back(pMP);
			//	nMarginalized++;
			//}
			//평면으로 생성한 맵포인트의 마지날 라이제이션

		}
		
		/*if (pMP->GetMapPointType() == MapPointType::PLANE_MP)
			std::cout << "Plane MP = " << pMP->mnMatchingCount << ", " << pMP->mnVisibleCount << std::endl;*/
		/*if(pMP->mnVisibleCount > nFrameCount || pMP->mnVisibleCount < nFrameCount*0.5 || ratio < 0.5)
			std::cout <<"ID="<<pMP->GetMapPointID()<< "::Count=" << nFrameCount << ", " << pMP->mnMatchingCount << ", " << pMP->mnVisibleCount << std::endl;*/
	}
	//std::cout << "Maginalization::End::"<< nMarginalized << std::endl;
}

void UVR_SLAM::LocalMapper::UpdateKFs() {
	mpFrameWindow->ClearLocalMapFrames();
	auto mvpConnectedKFs = mpTargetFrame->GetConnectedKFs();
	mpFrameWindow->AddFrame(mpTargetFrame);
	mpFrameWindow->SetLastFrameID(mpTargetFrame->GetFrameID());
	for (auto iter = mvpConnectedKFs.begin(); iter != mvpConnectedKFs.end(); iter++) {
		mpFrameWindow->AddFrame(*iter);
	}
	auto mvpKFs = mpFrameWindow->GetLocalMapFrames();
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
	mpFrameWindow->mvMatchInfos.clear();

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
				mpFrameWindow->mvMatchInfos.push_back(tempMatch);

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
	/*if (mmpCandidateKFs.find(mpTargetFrame) == mmpCandidateKFs.end()) {
		std::cout << "kf map error" << std::endl;
	}*/
	int Nkf = mmpCandidateKFs.size();
	
	auto mvpLocalMPs = mpFrameWindow->GetLocalMap();

	for (int i = 0; i < mvpLocalMPs.size(); i++) {
		if (!mpFrameWindow->GetBoolInlier(i)) {
			continue;
		}
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
		if (nCount > 10) {
			//mpTargetFrame->AddKF(pKF);
			vPairs.push_back(std::make_pair(nCount, pKF));
		}
		//
		
	}
	sort(vPairs.begin(), vPairs.end());
	//store frame and wegiths
	for (int i = vPairs.size()-1; i >=0 ; i--) {
		UVR_SLAM::Frame* pKF = vPairs[i].second;
		int nCount = vPairs[i].first;
		mpTargetFrame->AddKF(pKF);
		std::cout << "LocalMapping::Connection::" << pKF->GetFrameID() <<" | "<<pKF->GetKeyFrameID()<< ":: " << nCount << std::endl;
	}
	std::cout << "LocalMapping::Connected KFs::" << mpTargetFrame->GetConnectedKFs().size() << std::endl;
}