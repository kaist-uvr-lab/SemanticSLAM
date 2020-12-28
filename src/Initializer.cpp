#include <Initializer.h>
#include <LocalMapper.h>
#include <System.h>
#include <Map.h>
#include <MatrixOperator.h>
#include <SemanticSegmentator.h>
#include <CandidatePoint.h>
#include <PlaneEstimator.h>
#include <Visualizer.h>
#include <Plane.h>
#include <FrameGrid.h>
#include <direct.h>
#include <DepthFilter.h>

//추후 파라메터화. 귀찮아.
int N_matching_init_therah = 120; //80
int N_thresh_init_triangulate = 60; //80

UVR_SLAM::Initializer::Initializer() :mbInit(false), mpInitFrame1(nullptr), mpInitFrame2(nullptr), mpTempFrame(nullptr){
}
UVR_SLAM::Initializer::Initializer(System* pSystem, Map* pMap, cv::Mat _K, int w, int h) : mpSystem(pSystem), mK(_K), mbInit(false), mpInitFrame1(nullptr), mpInitFrame2(nullptr), mpTempFrame(nullptr),
mnWidth(w), mnHeight(h)
{
	mpMap = pMap;
	//mK.convertTo(mK,CV_64FC1);
}
UVR_SLAM::Initializer::Initializer(cv::Mat _K):mK(_K),mbInit(false), mpInitFrame1(nullptr), mpInitFrame2(nullptr), mpTempFrame(nullptr) {
	//mK.convertTo(mK, CV_64FC1);
}
UVR_SLAM::Initializer::~Initializer(){

}

void UVR_SLAM::Initializer::Init() {
	mpSegmentator = mpSystem->mpSegmentator;
	mpLocalMapper = mpSystem->mpLocalMapper;
	mpPlaneEstimator = mpSystem->mpPlaneEstimator;
	mpMatcher = mpSystem->mpMatcher;
	mpVisualizer = mpSystem->mpVisualizer;
}

void UVR_SLAM::Initializer::Reset() {
	mpInitFrame1->Reset();
	mpInitFrame2 = nullptr;
	mbInit = false;

	mpTempFrame = mpInitFrame1;
}

bool UVR_SLAM::Initializer::Initialize(Frame* pFrame, bool& bReset, int w, int h) {
	//std::cout << "Initializer::Initialize::Start" << std::endl;
	
	if (!mpInitFrame1) {
		mpInitFrame1 = pFrame;
		mpTempFrame = mpInitFrame1;
		mpInitFrame1->Init(mpSystem->mpORBExtractor, mK, mpSystem->mD);
		mpInitFrame1->mpMatchInfo = new UVR_SLAM::MatchInfo(mpSystem, mpInitFrame1, nullptr, mnWidth, mnHeight);
		mpInitFrame1->DetectFeature();
		/*mpInitFrame1->DetectEdge();
		mpInitFrame1->SetBowVec(mpSystem->fvoc);
		mpInitFrame1->mpMatchInfo->SetMatchingPoints();*/
		mpInitFrame1->SetGrids();
		mpSegmentator->InsertKeyFrame(mpInitFrame1);
		return mbInit;
	}
	else {
		///////////////////////////////////////////////////////////////////////////////////////
		//200419
		//두번째 키프레임은 초기화가 성공하거나 키프레임을 교체할 때 세그멘테이션을 수행함.
		mpInitFrame2 = pFrame;
		//////매칭 정보 생성
		mpInitFrame2->mpMatchInfo = new UVR_SLAM::MatchInfo(mpSystem, mpInitFrame2, mpInitFrame1, mnWidth, mnHeight);
		
		//////매칭 정보 생성
		bool bSegment = false;
		int nSegID = mpInitFrame1->mnFrameID;
		int nMatchingThresh = 0;//mpInitFrame1->mpMatchInfo->mvTempPts.size()*0.6;
		std::vector<cv::Point2f> vTempPts1, vTempPts2;
		std::vector<bool> vTempInliers;
		std::vector<int> vTempIndexs;
		std::vector<std::pair<cv::Point2f, cv::Point2f>> tempMatches2, resMatches;
		cv::Mat debugging;
		
		std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
		int count = mpMatcher->OpticalMatchingForInitialization(mpInitFrame1, mpInitFrame2, vTempPts1, vTempPts2, vTempInliers, vTempIndexs, debugging);
		std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
		double tttt = duration / 1000.0;

		/////////매칭 확인
		if (count < nMatchingThresh) {
			while (mpSegmentator->isDoingProcess());
			delete mpInitFrame1;
			mpInitFrame1 = nullptr;
			std::cout << "Initializer::replace::keyframe1" << std::endl;
			return mbInit;
		}
		
		/////////////////////Fundamental Matrix Decomposition & Triangulation
		std::vector<uchar> vFInliers;
		std::vector<cv::Point2f> vTempMatchPts1, vTempMatchPts2;
		std::vector<int> vTempMatchOctaves,vTempMatchIDXs; //vTempMatchPts2와 대응되는 매칭 인덱스를 저장.
		////F 찾기 : 기존 방법
		std::vector<bool> vbFtest;
		cv::Mat F12;
		float score;
		////E  찾기 : OpenCV
		cv::Mat E12 = cv::findEssentialMat(vTempPts1, vTempPts2, mK, cv::FM_RANSAC, 0.999, 1.0, vFInliers);
		////E  찾기 : OpenCV

		//////F, E를 통한 매칭 결과 반영
		for (unsigned long i = 0; i < vFInliers.size(); i++) {
			if (vFInliers[i]) {
				resMatches.push_back(std::make_pair(vTempPts1[i], vTempPts2[i]));
				vTempMatchPts1.push_back(vTempPts1[i]);
				vTempMatchPts2.push_back(vTempPts2[i]);
				vTempMatchIDXs.push_back(i);//vTempIndexs[i]
			}
		}
		count = resMatches.size();

		//////F, E를 통한 매칭 결과 반영
		///////삼각화 : OpenCV
		cv::Mat R1, t1;
		cv::Mat matTriangulateInliers;
		cv::Mat Map3D;
		cv::Mat K;
		mK.convertTo(K, CV_64FC1);
		int res2 = cv::recoverPose(E12, vTempMatchPts1, vTempMatchPts2, K, R1, t1, 50.0, matTriangulateInliers,Map3D);
		R1.convertTo(R1, CV_32FC1);
		t1.convertTo(t1, CV_32FC1);
		///////////////////////Fundamental Matrix Decomposition & Triangulation

		////////////삼각화 결과에 따른 초기화 판단
		if (res2 < 0.9*count) {
			mpTempFrame = mpInitFrame2;
			return mbInit;
		}
		////////////삼각화 결과에 따른 초기화 판단

		//////////////////////////////////////
		cv::Point2f ptBottom(0, mnHeight);
		std::vector<UVR_SLAM::MapPoint*> tempMPs;
		std::vector<cv::Point2f> vTempMappedPts1, vTempMappedPts2; //맵포인트로 생성된 포인트 정보를 저장
		std::vector<int> vTempMappedOctaves,vTempMappedIDXs; //vTempMatch에서 다시 일부분을 저장. 초기 포인트와 대응되는 위치를 저장.
		int nGridSize = mpSystem->mnRadius * 2;
		int res3 = 0;
		for (int i = 0; i < matTriangulateInliers.rows; i++) {
			int val = matTriangulateInliers.at<uchar>(i);
			if (val == 0)
				continue;
			///////////////뎁스값 체크
			cv::Mat X3D = Map3D.col(i).clone();
			X3D.convertTo(X3D, CV_32FC1);
			X3D /= X3D.at<float>(3);
			if(X3D.at<float>(2) < 0.0){
				std::cout << X3D.t() << ", " << val << std::endl;
				continue;
			}
			///////////////reprojection error
			X3D = X3D.rowRange(0, 3);
			cv::Mat proj1 = X3D.clone();
			cv::Mat proj2 = R1*X3D + t1;
			proj1 = mK*proj1;
			proj2 = mK*proj2;
			cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
			cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));
			auto pt1 = vTempMatchPts1[i];
			auto pt2 = vTempMatchPts2[i];
			auto diffPt1 = projected1 - pt1;
			auto diffPt2 = projected2 - pt2;
			float err1 = (diffPt1.dot(diffPt1));
			float err2 = (diffPt2.dot(diffPt2));
			if (err1 > 4.0 || err2 > 4.0)
				continue;
			///////////////reprojection error
			
			////그리드 체크
			auto gridPt = mpInitFrame1->GetGridBasePt(pt2, nGridSize);
			if (mpInitFrame2->mmbFrameGrids[gridPt]) {
				continue;
			}
			
			int idx = vTempMatchIDXs[i];
			int idx2 = vTempIndexs[idx];
			//std::cout << vTempPts1[idx] << ", " << mpInitFrame1->mpMatchInfo->GetCPPt(idx2) <<"::"<< vTempMatchPts1[i] << std::endl;

			res3++;
			auto pCP = mpInitFrame1->mpMatchInfo->mvpMatchingCPs[idx2];
			int label1 = mpInitFrame1->matLabeled.at<uchar>(pt1.y / 2, pt1.x / 2);
			auto pMP = new UVR_SLAM::MapPoint(mpMap, mpInitFrame2, pCP, X3D, cv::Mat(), label1);
			pMP->SetOptimization(true);
			tempMPs.push_back(pMP);
			vTempMappedPts1.push_back(vTempMatchPts1[i]);
			vTempMappedPts2.push_back(vTempMatchPts2[i]);
			vTempMappedIDXs.push_back(i);//vTempMatchIDXs[i]
			
			////그리드 매칭
			auto rect = cv::Rect(gridPt, std::move(cv::Point2f(gridPt.x + nGridSize, gridPt.y + nGridSize)));
			auto prevGridPt = mpInitFrame1->GetGridBasePt(pt1, nGridSize);
			auto prevGrid = mpInitFrame1->mmpFrameGrids[prevGridPt];
			if (!prevGrid) {
				std::cout << "initialization::error" << std::endl;
				continue;
			}
			auto prevRect = mpInitFrame1->GetOriginalImage()(prevGrid->rect);
			auto currRect = mpInitFrame2->GetOriginalImage()(rect);
			std::vector<cv::Point2f> vPrevGridPTs, vGridPTs;
			/*bool bGridMatch = mpMatcher->OpticalGridMatching(prevGrid, prevRect, currRect, vPrevGridPTs, vGridPTs);
			if (!bGridMatch)
				continue;*/

			//InitFrame2에 CP를 추가
			int idx3 = mpInitFrame2->mpMatchInfo->AddCP(pCP, vTempMatchPts2[i]);
			pCP->ConnectFrame(mpInitFrame2->mpMatchInfo, idx3);

			////grid 추가
			mpInitFrame2->mmbFrameGrids[gridPt] = true;
			auto currGrid = new FrameGrid(gridPt, rect);
			//currGrid->vecPTs = vGridPTs;
			mpInitFrame2->mmpFrameGrids[gridPt] = currGrid;
			mpInitFrame2->mmpFrameGrids[gridPt]->mpCP = pCP;
			mpInitFrame2->mmpFrameGrids[gridPt]->pt = pt2;
			prevGrid->mpNext = currGrid;
			currGrid->mpPrev = prevGrid;
			////grid 추가


			//MP 등록
			pMP->ConnectFrame(mpInitFrame1->mpMatchInfo, idx2);
			pMP->ConnectFrame(mpInitFrame2->mpMatchInfo, idx3);

			cv::circle(debugging, pt1, 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(debugging, pt2+ ptBottom, 2, cv::Scalar(0, 255, 0), -1);
			cv::line(debugging, pt1, projected1, cv::Scalar(255, 0, 0));
			cv::line(debugging, pt2 + ptBottom, projected2 + ptBottom, cv::Scalar(255, 0, 0));

		}
		//mpInitFrame1->mpMatchInfo->UpdateFrame();
		//mpInitFrame2->mpMatchInfo->UpdateFrame();
		
		cv::resize(debugging, debugging, cv::Size(debugging.cols / 2, debugging.rows / 2));
		cv::Rect rect1(0, 0, mnWidth / 2, mnHeight / 2);
		cv::Rect rect2(0, mnHeight/2, mnWidth / 2, mnHeight / 2);
		mpVisualizer->SetOutputImage(debugging(rect1), 0);
		mpVisualizer->SetOutputImage(debugging(rect2), 1);

		cv::waitKey(1);
		//////////////////////////////////////
		/////median depth 
		float medianDepth;
		mpInitFrame1->ComputeSceneMedianDepth(tempMPs, R1, t1, medianDepth);
		
		float invMedianDepth = 1.0f / medianDepth;
		if (medianDepth < 0.0) {
			mpTempFrame = mpInitFrame2;
			return mbInit;
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
		mpInitFrame2->Init(mpSystem->mpORBExtractor, mK, mpSystem->mD);
		//////////카메라 자세 변환 안하는 경우
		mpInitFrame1->SetPose(cv::Mat::eye(3,3,CV_32FC1), cv::Mat::zeros(3,1,CV_32FC1));
		mpInitFrame2->SetPose(R1, t1); //두번째 프레임은 median depth로 변경해야 함.
		//////////카메라 자세 변환 안하는 경우
		//////////카메라 자세 변환 하는 경우
		//mpInitFrame1->SetPose(cv::Mat::eye(3, 3, CV_32FC1)*Rcw, cv::Mat::zeros(3, 1, CV_32FC1));
		//mpInitFrame2->SetPose(R1*Rcw, t1); //두번째 프레임은 median depth로 변경해야 함.
		//////////카메라 자세 변환 하는 경우

		mpInitFrame1->mnKeyFrameID = UVR_SLAM::System::nKeyFrameID++;
		mpInitFrame2->mnKeyFrameID = UVR_SLAM::System::nKeyFrameID++;
		
		////맵포인트 정보 설정
		for (int i = 0; i < tempMPs.size(); i++) {
			UVR_SLAM::MapPoint* pNewMP = tempMPs[i];
			auto pt1 = vTempMappedPts1[i];
			auto pt2 = vTempMappedPts2[i];
			int idx2 = vTempMappedIDXs[i];
			
			pNewMP->mnFirstKeyFrameID = mpInitFrame2->mnKeyFrameID;
			////이 사이는 이제 이용 안하는데??
			pNewMP->IncreaseVisible(2);
			pNewMP->IncreaseFound(2);
			mpSystem->mlpNewMPs.push_back(pNewMP);
		}
		////맵포인트 정보 설정

		/////////////레이아웃 추정
		mpSegmentator->InsertKeyFrame(mpInitFrame2);
		/////////////레이아웃 추정

		////CP 추가
		mpInitFrame1->ComputeSceneDepth();
		mpInitFrame2->ComputeSceneDepth();
		////시드를 프레임마다 생성할 시
		for (auto iter = mpInitFrame1->mpMatchInfo->mvpMatchingCPs.begin(), iend = mpInitFrame1->mpMatchInfo->mvpMatchingCPs.end(); iter != iend; iter++) {
			auto pCPi = *iter;
			auto pMPi = pCPi->GetMP();
			if (pMPi && !pMPi->isDeleted())
				continue;
			auto pSeed = pCPi->mpSeed;
			cv::Mat ray = pSeed->ray.clone();
			float err = pSeed->px_err_angle;
			delete pCPi->mpSeed;
			pCPi->mpSeed = new UVR_SLAM::Seed(ray, mpInitFrame1->mfMedianDepth, mpInitFrame1->mfMinDepth);
			//pSeed->mf
		}
		////시드를 프레임마다 생성할 시

		mpInitFrame2->SetGrids();
		/*mpInitFrame1->mpMatchInfo->UpdateKeyFrame();
		mpInitFrame2->mpMatchInfo->UpdateKeyFrame();*/
		if(mpInitFrame2->mpMatchInfo->mvpMatchingCPs.size() < mpSystem->mnMaxMP){
			/*mpInitFrame2->DetectFeature();
			mpInitFrame2->DetectEdge();
			mpInitFrame2->SetBowVec(mpSystem->fvoc);
			mpInitFrame2->mpMatchInfo->SetMatchingPoints();*/
			std::cout <<"INITIALIZER::TEST::"<< mpInitFrame2->mpMatchInfo->mvpMatchingCPs.size() << std::endl;
		}
		////CP 추가

		////////////////////시각화에 카메라 포즈를 출력하기 위해
		///////////이것도 차후 없애야 함.
		////여기서 무슨일 하는지 정리 후 삭제
		///////////10개 중에 한개씩 저장. 그냥 평면 값 비교하기 위해
		mpLocalMapper->SetInitialKeyFrame(mpInitFrame1, mpInitFrame2);
		mpMap->AddWindowFrame(mpInitFrame1);
		mpMap->AddWindowFrame(mpInitFrame2);

		mpInitFrame1->AddKF(mpInitFrame2, tempMPs.size()); //여기도
		mpInitFrame2->AddKF(mpInitFrame1, tempMPs.size());
		////////////////////시각화에 카메라 포즈를 출력하기 위해
		mpMap->mpFirstKeyFrame = mpInitFrame1;
		
		mbInit = true;
		mpInitFrame1->mpMatchInfo->mMatchedImage = debugging.clone();
		std::cout << "Initializer::Success::" << tempMPs .size()<< std::endl << std::endl << std::endl;
		//////////////////////////키프레임 생성
		
		//////////////////////////////시각화 설정
		/*mpVisualizer->SetMatchInfo(mpInitFrame2->mpMatchInfo);
		if (!mpVisualizer->isDoingProcess()) {
			mpVisualizer->SetBoolDoingProcess(true);
		}*/
		//////////////////////////////시각화 설정

		//init초기화가 안되면 이렇게 해야 함
		mpTempFrame = mpInitFrame2;

		mpSystem->SetDirPath(0);
		std::string base = mpSystem->GetDirPath(0);
		std::stringstream sababs;
		sababs << base << "/kfmatching";
		_mkdir(sababs.str().c_str());
		sababs.str("");
		sababs << base << "/map";
		_mkdir(sababs.str().c_str());
		sababs.str("");
		sababs << base << "/testmatching";
		_mkdir(sababs.str().c_str());
		sababs.str("");
		sababs << base << "/fuse";
		_mkdir(sababs.str().c_str());
		sababs.str("");
		sababs << base << "/seg";
		_mkdir(sababs.str().c_str());
		return mbInit;
		//200419
		///////////////////////////////////////////////////////////////////////////////////////
	}
	return mbInit;
}

void UVR_SLAM::Initializer::SetCandidatePose(cv::Mat F, std::vector<cv::DMatch> Matches, std::vector<UVR_SLAM::InitialData*>& vCandidates) {
	//E
	Mat E = mK.t()*F*mK;;
	//Decompose E

	float th = 4.0f;
	Mat R1, R2, t1, t2;
	DecomposeE(E, R1, R2, t1, t2);
	vCandidates[0]->SetRt(R1, t1);
	vCandidates[1]->SetRt(R2, t1);
	vCandidates[2]->SetRt(R1, t2);
	vCandidates[3]->SetRt(R2, t2);

#pragma  omp parallel for
	for (int i = 0; i < 4; i++) {
		CheckRT(Matches, vCandidates[i], th);
	}
	/*CheckRT(Matches, vCandidates[0], th);
	CheckRT(Matches, vCandidates[1], th);
	CheckRT(Matches, vCandidates[2], th);
	CheckRT(Matches, vCandidates[3], th);*/
}

void UVR_SLAM::Initializer::SetCandidatePose(cv::Mat F, std::vector<std::pair<cv::Point2f, cv::Point2f>> Matches, std::vector<UVR_SLAM::InitialData*>& vCandidates) {
	//E
	Mat E = mK.t()*F*mK;;
	//Decompose E

	float th = 4.0f;
	Mat R1, R2, t1, t2;
	DecomposeE(E, R1, R2, t1, t2);
	vCandidates[0]->SetRt(R1, t1);
	vCandidates[1]->SetRt(R2, t1);
	vCandidates[2]->SetRt(R1, t2);
	vCandidates[3]->SetRt(R2, t2);

#pragma  omp parallel for
	for (int i = 0; i < 4; i++) {
		CheckRT(Matches, vCandidates[i], th);
	}
	/*CheckRT(Matches, vCandidates[0], th);
	CheckRT(Matches, vCandidates[1], th);
	CheckRT(Matches, vCandidates[2], th);
	CheckRT(Matches, vCandidates[3], th);*/
}

void UVR_SLAM::Initializer::DecomposeE(cv::Mat E, cv::Mat &R1, cv::Mat& R2, cv::Mat& t1, cv::Mat& t2){
	cv::Mat u, w, vt;
	cv::SVD::compute(E, w, u, vt);

	 u.col(2).copyTo(t1); // or UZU.t()xt=가 0이여서
	t1 = t1 / cv::norm(t1);
	t2 = -1.0f*t1;
	
	cv::Mat W = cv::Mat::zeros(3, 3, CV_32FC1);
	W.at<float>(0, 1) = -1.0f;
	W.at<float>(1, 0) = 1.0f;
	W.at<float>(2, 2) = 1.0f;

	R1 = u*W*vt;
	if (cv::determinant(R1)<0.0){
		R1 = -R1;
	}
	R2 = u*W.t()*vt;
	if (cv::determinant(R2)<0.0){
		R2 = -R2;
	}
}
void UVR_SLAM::Initializer::CheckRT(std::vector<cv::DMatch> Matches, UVR_SLAM::InitialData* candidate, float th2) {

	//vector map을 대신할 무엇인가가 필요함.

	candidate->vbTriangulated = std::vector<bool>(Matches.size(), false);
	candidate->mvX3Ds = std::vector<cv::Mat>(Matches.size(), cv::Mat::zeros(3, 1, CV_32FC1));
	//candidate->vMap3D = std::vector<UVR::MapPoint*>(pInitFrame->mvnCPMatchingIdx.size(), nullptr);
	//candidate->vP3D.resize(pKF->mvnMatchingIdx.size());

	std::vector<float> vCosParallax;
	//vCosParallax.reserve(pKF->mvnMatchingIdx.size());

	//cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
	//cv::Mat t = cv::Mat::zeros(3, 1, CV_32FC1);

	// Camera 1 Projection Matrix K[I|0]
	cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
	mK.copyTo(P1.rowRange(0, 3).colRange(0, 3));
	cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

	// Camera 2 Projection Matrix K[R|t]
	cv::Mat P2(3, 4, CV_32F);
	candidate->R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
	candidate->t.copyTo(P2.rowRange(0, 3).col(3));
	P2 = mK*P2;
	
	cv::Mat O2 = -candidate->R.t()*candidate->t;

	for (unsigned long i = 0; i < Matches.size(); i++)
	{

		const cv::KeyPoint &kp1 = mpInitFrame1->mvKeyPoints[Matches[i].queryIdx];
		const cv::KeyPoint &kp2 = mpInitFrame2->mvKeyPoints[Matches[i].trainIdx];
		cv::Mat X3D;

		if (!Triangulate(kp1.pt, kp2.pt, P1, P2, X3D))
			continue;
		
		float cosParallax;

		bool res = CheckCreatedPoints(X3D, kp1.pt, kp2.pt, O1, O2, candidate->R0, candidate->t0, candidate->R, candidate->t, cosParallax, th2,th2);
		if (res) {
			vCosParallax.push_back(cosParallax);
			//candidate->vMap3D[i] = new MapPoint(X3D);
			/*
			{
			//항상 초기화 시에 수행해야 할 듯.
			cv::Mat Ow = -R.t()*t;
			cv::Mat PC = X3D-Ow;
			float dist = cv::norm(PC);
			int level = kp2.octave;
			int nLevels = pF->mnScaleLevels;
			float levelScaleFactor = pF->mvScaleFactors[level];
			candidate->vMap3D[i]->mfMaxDistance = dist*levelScaleFactor;
			candidate->vMap3D[i]->mfMinDistance = candidate->vMap3D[i]->mfMaxDistance / pF->mvScaleFactors[nLevels-1];
			candidate->vMap3D[i]->mNormalVector = X3D-Ow;
			candidate->vMap3D[i]->mNormalVector = candidate->vMap3D[i]->mNormalVector / cv::norm(candidate->vMap3D[i]->mNormalVector);
			}
			*/
			candidate->vbTriangulated[i] = true;
			candidate->mvX3Ds[i] = X3D.clone();
			candidate->nGood++;
		}
	}
	
	if (candidate->nGood>0)
	{
		std::sort(vCosParallax.begin(), vCosParallax.end());
		int idx = 50;
		int nParallaxSize = (int)vCosParallax.size() - 1;
		if (idx > nParallaxSize) {
			idx = nParallaxSize;
		}
		candidate->parallax = (float)(acos(vCosParallax[idx])*UVR_SLAM::MatrixOperator::rad2deg);
	}
	else {
		candidate->parallax = 0.0f;
	}
}

void UVR_SLAM::Initializer::CheckRT(std::vector<std::pair<cv::Point2f, cv::Point2f>> Matches, UVR_SLAM::InitialData* candidate, float th2) {

	//vector map을 대신할 무엇인가가 필요함.

	candidate->vbTriangulated = std::vector<bool>(Matches.size(), false);
	candidate->mvX3Ds = std::vector<cv::Mat>(Matches.size(), cv::Mat::zeros(3, 1, CV_32FC1));
	//candidate->vMap3D = std::vector<UVR::MapPoint*>(pInitFrame->mvnCPMatchingIdx.size(), nullptr);
	//candidate->vP3D.resize(pKF->mvnMatchingIdx.size());

	std::vector<float> vCosParallax;
	//vCosParallax.reserve(pKF->mvnMatchingIdx.size());

	//cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
	//cv::Mat t = cv::Mat::zeros(3, 1, CV_32FC1);

	// Camera 1 Projection Matrix K[I|0]
	cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
	mK.copyTo(P1.rowRange(0, 3).colRange(0, 3));
	cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

	// Camera 2 Projection Matrix K[R|t]
	cv::Mat P2(3, 4, CV_32F);
	candidate->R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
	candidate->t.copyTo(P2.rowRange(0, 3).col(3));
	P2 = mK*P2;

	cv::Mat O2 = -candidate->R.t()*candidate->t;

	for (unsigned long i = 0; i < Matches.size(); i++)
	{
		const cv::Point2f pt1 = Matches[i].first;
		const cv::Point2f pt2 = Matches[i].second;
		/*const cv::KeyPoint &kp1 = mpInitFrame1->mvKeyPoints[Matches[i].queryIdx];
		const cv::KeyPoint &kp2 = mpInitFrame2->mvKeyPoints[Matches[i].trainIdx];*/
		cv::Mat X3D;

		if (!Triangulate(pt1,pt2, P1, P2, X3D))
			continue;

		float cosParallax;

		bool res = CheckCreatedPoints(X3D, pt1,pt2, O1, O2, candidate->R0, candidate->t0, candidate->R, candidate->t, cosParallax, th2, th2);
		if (res) {
			vCosParallax.push_back(cosParallax);
			//candidate->vMap3D[i] = new MapPoint(X3D);
			/*
			{
			//항상 초기화 시에 수행해야 할 듯.
			cv::Mat Ow = -R.t()*t;
			cv::Mat PC = X3D-Ow;
			float dist = cv::norm(PC);
			int level = kp2.octave;
			int nLevels = pF->mnScaleLevels;
			float levelScaleFactor = pF->mvScaleFactors[level];
			candidate->vMap3D[i]->mfMaxDistance = dist*levelScaleFactor;
			candidate->vMap3D[i]->mfMinDistance = candidate->vMap3D[i]->mfMaxDistance / pF->mvScaleFactors[nLevels-1];
			candidate->vMap3D[i]->mNormalVector = X3D-Ow;
			candidate->vMap3D[i]->mNormalVector = candidate->vMap3D[i]->mNormalVector / cv::norm(candidate->vMap3D[i]->mNormalVector);
			}
			*/
			candidate->vbTriangulated[i] = true;
			candidate->mvX3Ds[i] = X3D.clone();
			candidate->nGood++;
		}
	}
	//std::cout << "check rt = " << candidate->nGood << std::endl;
	if (candidate->nGood>0)
	{
		std::sort(vCosParallax.begin(), vCosParallax.end());
		int idx = 50;
		int nParallaxSize = (int)vCosParallax.size() - 1;
		if (idx > nParallaxSize) {
			idx = nParallaxSize;
		}
		//std::cout<<"parallax::"<< (float)(acos(vCosParallax[idx])*UVR_SLAM::MatrixOperator::rad2deg)<<", "<< (float)(acos(vCosParallax[idx/2])*UVR_SLAM::MatrixOperator::rad2deg)<<std::endl;
		candidate->parallax = (float)(acos(vCosParallax[idx])*UVR_SLAM::MatrixOperator::rad2deg);
	}
	else {
		candidate->parallax = 0.0f;
	}
}

bool UVR_SLAM::Initializer::Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2, cv::Mat& x3D){
	cv::Mat A(4, 4, CV_32F);

	A.row(0) = pt1.x*P1.row(2) - P1.row(0);
	A.row(1) = pt1.y*P1.row(2) - P1.row(1);
	A.row(2) = pt2.x*P2.row(2) - P2.row(0);
	A.row(3) = pt2.y*P2.row(2) - P2.row(1);

	cv::Mat u, w, vt;
	cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	x3D = vt.row(3).t();

	float a = w.at<float>(3);
	if (a < 0.001)
		return false;
	//if (abs(x3D.at<float>(3)) <= 0.001)
	//	return false;

	//if (abs(x3D.at<float>(3)) < 0.01)
	//	std::cout << "abc:" << x3D.at<float>(3) << std::endl;

	x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
	
	return true;
}
bool UVR_SLAM::Initializer::CheckCreatedPoints(cv::Mat X3D, cv::Point2f kp1, cv::Point2f kp2, cv::Mat O1, cv::Mat O2, cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, float& cosParallax, float th1, float th2) {
	if (!std::isfinite(X3D.at<float>(0)) || !std::isfinite(X3D.at<float>(1)) || !std::isfinite(X3D.at<float>(2)))
	{
		return false;
	}
	
	cv::Mat p3dC1 = X3D;
	cv::Mat p3dC2 = R2*X3D + t2;
	// Check parallax
	cv::Mat normal1 = p3dC1 - O1;
	float dist1 = (float)cv::norm(normal1);
	cv::Mat normal2 = p3dC1 - O2;
	float dist2 = (float)cv::norm(normal2);

	cosParallax = ((float)normal1.dot(normal2)) / (dist1*dist2);

	if (cosParallax >= 0.99998f)
		return false;
	//std::cout << p3dC1 << ", " << p3dC2 << ", " << th1 <<", "<< th2 << std::endl;
	// Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
	if (p3dC1.at<float>(2) <= 0.0f || p3dC2.at<float>(2) <= 0.0f )
		return false;

	// Check reprojection error in first image
	cv::Mat reproj1 = mK*p3dC1;
	reproj1 /= p3dC1.at<float>(2);
	float squareError1 = (reproj1.at<float>(0) - kp1.x)*(reproj1.at<float>(0) - kp1.x) + (reproj1.at<float>(1) - kp1.y)*(reproj1.at<float>(1) - kp1.y);
	
	//std::cout << squareError1 << std::endl;
	if (squareError1>th1)
		return false;

	// Check reprojection error in second image
	cv::Mat reproj2 = mK*p3dC2;
	reproj2 /= p3dC2.at<float>(2);
	float squareError2 = (reproj2.at<float>(0) - kp2.x)*(reproj2.at<float>(0) - kp2.x) + (reproj2.at<float>(1) - kp2.y)*(reproj2.at<float>(1) - kp2.y);
	
	if (squareError2>th2)
		return false;
	return true;
}

int UVR_SLAM::Initializer::SelectCandidatePose(std::vector<UVR_SLAM::InitialData*>& vCandidates) {
	//int SelectCandidatePose(UVR::InitialData* c1, UVR::InitialData* c2, UVR::InitialData* c3, UVR::InitialData* c4){
	float minParallax = 0.5f;//1.0f;
	int   minTriangulated = 50;

	unsigned long maxIdx = (unsigned long)-1;
	int nMaxGood = -1;
	for (unsigned long i = 0; i < vCandidates.size(); i++) {
		if (vCandidates[i]->nGood > nMaxGood) {
			maxIdx = i;
			nMaxGood = vCandidates[i]->nGood;
		}
	}
	
	int nsimilar = 0;
	int th_good = (int)(0.7f*(float)nMaxGood);
	//int nMinGood = 600;
	int nMinGood = (int)(0.7f*(float)vCandidates[0]->nMinGood); //0.8
	if (nMinGood < minTriangulated) {
		nMinGood = minTriangulated;
	}
	for (unsigned long i = 0; i < vCandidates.size(); i++) {
		if (vCandidates[i]->nGood>th_good) {
			nsimilar++;
		}
	}
	std::cout << "paralaxx::" << vCandidates[maxIdx]->parallax << ", " << minParallax <<"::"<< nMaxGood <<", "<<nMinGood<<"::"<<nsimilar<< std::endl;
	int res = -1;
	if (vCandidates[maxIdx]->parallax > minParallax && nMaxGood > nMinGood && nsimilar == 1) {
		res = (int)maxIdx;
	}
	return res;
}