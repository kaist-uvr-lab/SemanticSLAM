#include <LocalMapper.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <MapPoint.h>
#include <Matcher.h>
#include <Optimization.h>
#include <PlaneEstimator.h>
#include <SemanticSegmentator.h>

UVR_SLAM::LocalMapper::LocalMapper(){}
UVR_SLAM::LocalMapper::LocalMapper(int w, int h):mnWidth(w), mnHeight(h){}
UVR_SLAM::LocalMapper::~LocalMapper() {}


void UVR_SLAM::LocalMapper::Run() {
	while (1) {
		if (isDoingProcess()) {
			std::cout << "Create KeyFrame::Start!!!!" << std::endl;
			std::cout << "Window size = " << mpFrameWindow->size() << std::endl;
			//키프레임으로 설정하고 프레임 윈도우에 추가
			int nFrameCount = mpFrameWindow->GetFrameCount();
			mpFrameWindow->SetFrameCount(0);
			UVR_SLAM::Frame* mpPrevKeyFrame = mpFrameWindow->back();
			mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
			mpFrameWindow->push_back(mpTargetFrame);

			////이전 프레임에서 생성된 맵포인트 중 삭제
			//프레임 윈도우 내의 로컬 맵 포인트 중 new인 애들만 수행
			NewMapPointMaginalization(nFrameCount);
			//프레임 내에서 삭제 되는 녀석과 업데이트 되는 녀석의 분리가 필요함.
			UpdateMPs();
			////새로운 맵포인트 생성
			//여기부터 다시 검증이 필요
			CreateMapPoints(mpTargetFrame, mpPrevKeyFrame);

			////BA
			//BA에서는 최근 생성된 맵포인트까지 반영을 해야 함.
			Optimization::LocalBundleAdjustment(mpFrameWindow, 2, 5, false);
			
			//Delete MPs
			//DeleteMPs();

			//평면 검출 쓰레드에 추가

			if (!mpSegmentator->isDoingProcess()) {
				mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_SEGMENTED_FRAME);
				mpSegmentator->SetBoolDoingProcess(true);
				mpSegmentator->SetTargetFrame(mpTargetFrame);
				
			}else if (!mpPlaneEstimator->isDoingProcess()) {
				mpPlaneEstimator->SetBoolDoingProcess(true, 2);
				mpPlaneEstimator->SetTargetFrame(mpTargetFrame);
			}

			//이 시점에서는 로컬맵이 완성이 되고 삭제되는 일이 없음.
			//뮤텍스로 여기에서 한번 막으면 달라지는 것은?
			mpFrameWindow->SetLocalMap();
			SetBoolDoingProcess(false);
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
void UVR_SLAM::LocalMapper::SetTargetFrame(Frame* pFrame) {
	mpTargetFrame = pFrame;
}

void UVR_SLAM::LocalMapper::SetBoolDoingProcess(bool b) {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	mbDoingProcess = b;
}
bool UVR_SLAM::LocalMapper::isDoingProcess() {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	return mbDoingProcess;
}

//맵포인트가 삭제 되면 현재 프레임에서도 해당 맵포인트를 삭제 해야 하며, 
//이게 수행되기 전에는 트래킹이 동작하지 않도록 막아야 함.
//
void UVR_SLAM::LocalMapper::NewMapPointMaginalization(int nFrameCount) {
	std::cout << "Maginalization::Start" << std::endl;
	mvpDeletedMPs.clear();
	int nMarginalized = 0;
	for (int i = 0; i < mpFrameWindow->GetLocalMapSize(); i++) {
		UVR_SLAM::MapPoint* pMP = mpFrameWindow->GetMapPoint(i);
		if (!pMP)
			continue;
		if (!pMP->isNewMP())
			continue;
		pMP->SetNewMP(false);
		if (pMP->GetMapPointType() == MapPointType::NORMAL_MP) {
			//if (!mpFrameWindow->GetBoolInlier(i))
			float ratio1 = ((float)pMP->mnMatchingCount) / pMP->mnVisibleCount;
			float ratio2 = ((float)pMP->mnVisibleCount) / nFrameCount;
			if (ratio1 < 0.5) {
				mpFrameWindow->SetMapPoint(nullptr, i);
				mpFrameWindow->SetBoolInlier(false, i);
				pMP->SetDelete(true);
				pMP->Delete();
				mvpDeletedMPs.push_back(pMP);
				nMarginalized++;
			}
		}
		else {
			if (pMP->mnMatchingCount == 0) {
				mpFrameWindow->SetMapPoint(nullptr, i);
				mpFrameWindow->SetBoolInlier(false, i);
				pMP->SetDelete(true);
				pMP->Delete();
				mvpDeletedMPs.push_back(pMP);
				nMarginalized++;
			}
			//평면으로 생성한 맵포인트의 마지날 라이제이션

		}
		
		/*if (pMP->GetMapPointType() == MapPointType::PLANE_MP)
			std::cout << "Plane MP = " << pMP->mnMatchingCount << ", " << pMP->mnVisibleCount << std::endl;*/
		/*if(pMP->mnVisibleCount > nFrameCount || pMP->mnVisibleCount < nFrameCount*0.5 || ratio < 0.5)
			std::cout <<"ID="<<pMP->GetMapPointID()<< "::Count=" << nFrameCount << ", " << pMP->mnMatchingCount << ", " << pMP->mnVisibleCount << std::endl;*/
	}
	std::cout << "Maginalization::End::"<< nMarginalized << std::endl;
}

void UVR_SLAM::LocalMapper::UpdateMPs() {
	int nUpdated = 0;
	for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mpTargetFrame->GetMapPoint(i);
		if (pMP) {
			if (pMP->isDeleted()) {
				mpTargetFrame->RemoveMP(i);
			}
		}
		if (mpTargetFrame->GetBoolInlier(i)) {
			if (pMP){
				nUpdated++;
				pMP->AddFrame(mpTargetFrame, i);
			}
		}
	}
	std::cout << "Update MPs::" << nUpdated << std::endl;
}

void UVR_SLAM::LocalMapper::DeleteMPs() {
	for (int i = 0; i < mvpDeletedMPs.size(); i++) {
		delete mvpDeletedMPs[i];
	}
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
		if (pCurrKF->GetBoolInlier(i))
			continue;
		int matchIDX, kidx;
		cv::KeyPoint kp = pCurrKF->mvKeyPoints[i];
		cv::Point2f pt = pCurrKF->mvKeyPoints[i].pt;
		cv::Mat desc = pCurrKF->matDescriptor.row(i);

		float sigma = pCurrKF->mvLevelSigma2[kp.octave];
		bool bMatch = mpMatcher->FeatureMatchingWithEpipolarConstraints(matchIDX, pLastKF, F12, kp, desc, sigma, thresh_epi_dist);
		if (bMatch) {
			if (!pLastKF->GetBoolInlier(matchIDX)) {
				if (bNearBaseLine)
					continue;
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
	cv::imshow("LocalMapping::CreateMPs", debugging); cv::waitKey(100);
	std::cout << "CreateMapPoints=" << count << std::endl;
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