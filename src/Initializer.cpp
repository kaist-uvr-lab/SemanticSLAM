#include <Initializer.h>
#include <FrameWindow.h>
#include <LocalMapper.h>
#include <System.h>
#include <Map.h>
#include <MatrixOperator.h>
#include <SemanticSegmentator.h>
#include <CandidatePoint.h>
#include <PlaneEstimator.h>
#include <Visualizer.h>
#include <Plane.h>
#include <direct.h>

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
	mpInitFrame1 = nullptr;
	mpInitFrame2 = nullptr;
	mbInit = false;
}

void UVR_SLAM::Initializer::Reset() {
	mpInitFrame1->Reset();
	mpInitFrame2 = nullptr;
	mbInit = false;

	mpTempFrame = mpInitFrame1;
}



void UVR_SLAM::Initializer::SetLocalMapper(LocalMapper* pMapper) {
	mpLocalMapper = pMapper;
}

void UVR_SLAM::Initializer::SetMatcher(UVR_SLAM::Matcher* pMatcher) {
	mpMatcher = pMatcher;
}

void UVR_SLAM::Initializer::SetFrameWindow(UVR_SLAM::FrameWindow* pWindow) {
	mpFrameWindow = pWindow;
}

void UVR_SLAM::Initializer::SetSegmentator(SemanticSegmentator* pEstimator) {
	mpSegmentator = pEstimator;
}

void UVR_SLAM::Initializer::SetPlaneEstimator(PlaneEstimator* pEstimator) {
	mpPlaneEstimator = pEstimator;
}

void UVR_SLAM::Initializer::SetVisualizer(Visualizer* pVis) {
	mpVisualizer = pVis;
}

bool UVR_SLAM::Initializer::Initialize(Frame* pFrame, bool& bReset, int w, int h) {
	//std::cout << "Initializer::Initialize::Start" << std::endl;
	
	if (!mpInitFrame1) {
		
		mpInitFrame1 = pFrame;
		mpTempFrame = mpInitFrame1;
		mpInitFrame1->Init(mpSystem->mpORBExtractor, mK, mpSystem->mD);
		mpInitFrame1->mpMatchInfo = new UVR_SLAM::MatchInfo(mpInitFrame1, nullptr, mnWidth, mnHeight);
		std::cout << "1" << std::endl;
		mpInitFrame1->mpMatchInfo->SetMatchingPoints();
		mpSegmentator->InsertKeyFrame(mpInitFrame1);
		std::cout << "2" << std::endl;
		return mbInit;
	}
	else {
		///////////////////////////////////////////////////////////////////////////////////////
		//200419
		//두번째 키프레임은 초기화가 성공하거나 키프레임을 교체할 때 세그멘테이션을 수행함.
		mpInitFrame2 = pFrame;
		//////매칭 정보 생성
		mpInitFrame2->mpMatchInfo = new UVR_SLAM::MatchInfo(mpInitFrame2, mpInitFrame1, mnWidth, mnHeight);
		
		//////매칭 정보 생성
		bool bSegment = false;
		int nSegID = mpInitFrame1->GetFrameID();
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
		
		////////현재 프레임의 매칭 정보 복사 및 초기 프레임의 포인트 저장
		//mpInitFrame2->mpMatchInfo->mvpMatchingMPs = std::vector<UVR_SLAM::MapPoint*>(vTempPts2.size(), nullptr);
		/*for (int i = 0; i < vTempPts2.size(); i++) {
			int idx = vTempIndexs[i];
			mpInitFrame2->mpMatchInfo->mvMatchingPts.push_back(vTempPts2[i]);
			mpInitFrame2->mpMatchInfo->mvnMatchingPtIDXs.push_back(idx);
			mpInitFrame2->mpMatchInfo->mvObjectLabels.push_back(0);
			mpInitFrame2->mpMatchInfo->mvnOctaves.push_back(mpInitFrame1->mpMatchInfo->mvnOctaves[idx]);
			vTempPts1.push_back(mpInitFrame1->mpMatchInfo->mvMatchingPts[idx]);
			cv::circle(mpInitFrame2->mpMatchInfo->used, vTempPts2[i], 2, cv::Scalar(255), -1);
		}*/
		////////현재 프레임의 매칭 정보 복사

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
				//vTempMatchOctaves.push_back(mpInitFrame1->mpMatchInfo->mvTempOctaves[vTempIndexs[i]]);
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
			int idx = vTempMatchIDXs[i];
			int idx2 = vTempIndexs[idx];
			//std::cout << vTempPts1[idx] << ", " << mpInitFrame1->mpMatchInfo->GetCPPt(idx2) <<"::"<< vTempMatchPts1[i] << std::endl;

			res3++;
			auto pCP = mpInitFrame1->mpMatchInfo->GetCP(idx2);
			int label1 = mpInitFrame1->matLabeled.at<uchar>(pt1.y / 2, pt1.x / 2);
			auto pMP = new UVR_SLAM::MapPoint(mpMap, mpInitFrame2, pCP, X3D, cv::Mat(), label1);
			tempMPs.push_back(pMP);
			vTempMappedPts1.push_back(vTempMatchPts1[i]);
			vTempMappedPts2.push_back(vTempMatchPts2[i]);
			//vTempMappedOctaves.push_back(vTempMatchOctaves[i]);
			vTempMappedIDXs.push_back(i);//vTempMatchIDXs[i]
			
			//pCP->AddFrame(mpInitFrame2->mpMatchInfo, vTempMatchPts2[i]);
			mpInitFrame2->mpMatchInfo->AddCP(pCP, vTempMatchPts2[i]);
			//pMP->AddFrame(mpInitFrame1->mpMatchInfo, idx2);
			//pMP->AddFrame(mpInitFrame2->mpMatchInfo, idx2);

			cv::circle(debugging, pt1, 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(debugging, pt2+ ptBottom, 2, cv::Scalar(0, 255, 0), -1);
			cv::line(debugging, pt1, projected1, cv::Scalar(255, 0, 0));
			cv::line(debugging, pt2 + ptBottom, projected2 + ptBottom, cv::Scalar(255, 0, 0));

		}
		mpInitFrame1->mpMatchInfo->UpdateFrame();
		mpInitFrame2->mpMatchInfo->UpdateFrame();
		cv::imshow("Init::proj", debugging);
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

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////평면 관련 기능들
		/////////////////////바닥 초기화를 위한 세그멘테이션 정보를 이용한 평면 포인트 나누기
		//std::vector<UVR_SLAM::MapPoint*> mvpFloorMPs;
		//std::vector<cv::Point2f> vTempFloorPts; //호모 그래피를 이용할 경우 사용

		//for (int i = 0; i < tempMPs.size(); i++) {
		//	auto pt1 = vTempMappedPts1[i];
		//	//auto pt2 = vTempMappedPts2[i];
		//	int label1 = mpInitFrame1->matLabeled.at<uchar>(pt1.y / 2, pt1.x / 2);

		//	if (label1 == 150) {
		//		mvpFloorMPs.push_back(tempMPs[i]);
		//		//vTempFloorPts.push_back(pt2);
		//	}

		//}
		/////////////////////바닥 초기화를 위한 세그멘테이션 정보를 이용한 평면 포인트 나누기
		/////////////////////////////평면 초기화
		//UVR_SLAM::PlaneInformation* pFloor = new UVR_SLAM::PlaneInformation();
		//bool bRes = UVR_SLAM::PlaneInformation::PlaneInitialization(pFloor, mvpFloorMPs, mpInitFrame2->GetFrameID(), 1500, 0.01, 0.4);
		//cv::Mat param = pFloor->GetParam();
		//if (!bRes || abs(param.at<float>(1)) < 0.98)//98
		//{
		//	mpTempFrame = mpInitFrame2;
		//	return mbInit;
		//}
		/////////////////////////////평면 초기화

		/////////////////////////////평면 정보 생성
		//////초기 평면 MP 설정 필요
		//mpInitFrame1->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpInitFrame1, pFloor);
		//mpInitFrame2->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpInitFrame2, pFloor);
		//cv::Mat invP, invT, invK;
		//mpInitFrame2->mpPlaneInformation->Calculate();
		//mpInitFrame2->mpPlaneInformation->GetInformation(invP, invT, invK);
		/////////////////////////////평면 정보 생성

		/////////////////////////////평면 정보를 이용한 alignment
		////cv::Mat Rcw = UVR_SLAM::PlaneInformation::CalcPlaneRotationMatrix(param).clone();
		////cv::Mat normal;
		////float dist;
		////pFloor->GetParam(normal, dist);
		////cv::Mat tempP = Rcw.t()*normal;
		////if (tempP.at<float>(0) < 0.00001)
		////	tempP.at<float>(0) = 0.0;
		////if (tempP.at<float>(2) < 0.00001)
		////	tempP.at<float>(2) = 0.0;

		//////전체 맵포인트 변환
		////for (int i = 0; i < tempMPs.size(); i++) {
		////	UVR_SLAM::MapPoint* pMP = tempMPs[i];
		////	/*if (!pMP)
		////		continue;
		////	if (pMP->isDeleted())
		////		continue;*/
		////	cv::Mat tempX = Rcw.t()*pMP->GetWorldPos();
		////	pMP->SetWorldPos(tempX);
		////}
		//////평면 파라메터 변환
		////pFloor->SetParam(tempP, dist);
		/////////////////////////////평면 정보를 이용한 alignment
		//std::cout << mpMap->mpFirstKeyFrame->mpPlaneInformation->GetFloorPlane()->GetParam() << std::endl << std::endl << std::endl;
		//////////////////////////////////////////////////////////평면 관련 기능들
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
		mpInitFrame1->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME, 0);
		mpInitFrame2->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
		//맵포인트 정보 설정
		for (int i = 0; i < tempMPs.size(); i++) {
			UVR_SLAM::MapPoint* pNewMP = tempMPs[i];
			auto pt1 = vTempMappedPts1[i];
			auto pt2 = vTempMappedPts2[i];
			
			int idx2 = vTempMappedIDXs[i];
			
			//mpInitFrame2->mpMatchInfo->mvnMatchingIDXs.push_back(mpInitFrame1->mpMatchInfo->mvnMatchingIDXs.size());
			//mpInitFrame1->mpMatchInfo->mvnMatchingIDXs.push_back(-1);

			//pNewMP->mnOctave = vTempMappedOctaves[i];//mpInitFrame1->mpMatchInfo->mvnOctaves[idx1];
			
			pNewMP->mnFirstKeyFrameID = mpInitFrame2->GetKeyFrameID();
			pNewMP->IncreaseVisible(2);
			pNewMP->IncreaseFound(2);
			//pNewMP->UpdateNormalAndDepth();
			mpSystem->mlpNewMPs.push_back(pNewMP);
			auto pt3D = mpMap->ProjectMapPoint(pNewMP, mpMap->mfMapGridSize);
			auto pMG = mpMap->GetGrid(pt3D);
			if (!pMG) {
				pMG = mpMap->InsertGrid(pt3D);
			}
			mpMap->InsertMapPoint(pNewMP, pMG);
		}
		
		/////////////////////Object labeling
		mpInitFrame1->mpMatchInfo->SetLabel();
		
		/*for (int i = 0; i < mpInitFrame2->mpMatchInfo->mvMatchingPts.size(); i++) {
			int idx = mpInitFrame2->mpMatchInfo->mvnMatchingPtIDXs[i];
			mpInitFrame2->mpMatchInfo->mvObjectLabels[i] = mpInitFrame1->mpMatchInfo->mvObjectLabels[idx];
		}*/
		/////////////////////Object labeling

		//////키프레임으로 업데이트 과정
		//타겟 프레임과의 매칭 정보 저장
		//mpInitFrame1->mpMatchInfo->mnTargetMatch = mpInitFrame1->mpMatchInfo->mvMatchingPts.size();
		//mpInitFrame2->mpMatchInfo->SetKeyFrame();
		//////키프레임으로 업데이트 과정

		/////////////레이아웃 추정
		//mpPlaneEstimator->InsertKeyFrame(mpInitFrame2);
		mpSegmentator->InsertKeyFrame(mpInitFrame2);
		/////////////레이아웃 추정

		////////////////////시각화에 카메라 포즈를 출력하기 위해
		///////////이것도 차후 없애야 함.
		////여기서 무슨일 하는지 정리 후 삭제
		///////////10개 중에 한개씩 저장. 그냥 평면 값 비교하기 위해
		mpLocalMapper->SetInitialKeyFrame(mpInitFrame1, mpInitFrame2);
		mpMap->AddFrame(mpInitFrame1);
		mpMap->AddFrame(mpInitFrame2);
		mpMap->mQueueFrameWindows.push_back(mpInitFrame1);
		mpMap->mQueueFrameWindows.push_back(mpInitFrame2);

		mpInitFrame1->AddKF(mpInitFrame2, tempMPs.size());
		mpInitFrame2->AddKF(mpInitFrame1, tempMPs.size());
		////////////////////시각화에 카메라 포즈를 출력하기 위해
		mpMap->mpFirstKeyFrame = mpInitFrame1;
		mpVisualizer->SetMatchInfo(mpInitFrame2->mpMatchInfo);
		mbInit = true;
		mpInitFrame1->mpMatchInfo->mMatchedImage = debugging.clone();
		std::cout << "Initializer::Success::" << tempMPs .size()<< std::endl << std::endl << std::endl;
		//////////////////////////키프레임 생성
		
		///////////////호모그래피 테스트
		/*
		1) 바닥, 벽으로 포인트 벡터 분리(초기 optical flow 매칭 결과에 대해서 수행)
		2) 각각에 대해서 호모그래피 돌리기
		3) 디컴포지션 테스트
		4) 맵포인트와의 연결도 필요할 듯.
		5) 벽의 경우 미리 나누는 것도 중요함.
		*/
		//std::vector<cv::Point2f> vFloorPts1, vFloorPts2, vWallPts1, vWallPts2;
		//for (int i = 0; i < vTempPts2.size(); i++) {
		//	auto pt1 = vTempPts1[i];
		//	auto pt2 = vTempPts2[i];
		//	int label1 = mpInitFrame1->matLabeled.at<uchar>(pt1.y / 2, pt1.x / 2);
		//	/*int label2 = mpInitFrame2->matLabeled.at<uchar>(pt2.y / 2, pt2.x / 2);
		//	if (label1 != label2)
		//		continue;*/
		//	if (label1 == 150) {
		//		vFloorPts1.push_back(pt1);
		//		vFloorPts2.push_back(pt2);
		//	}
		//	else if (label1 == 255) {
		//		vWallPts1.push_back(pt1);
		//		vWallPts2.push_back(pt2);
		//	}
		//}
		//cv::Mat inlierH1, inlierH2;
		//cv::Mat H1 = cv::findHomography(vFloorPts1, vFloorPts2, inlierH1, cv::RHO, 3.0);
		//cv::Mat H2 = cv::findHomography( vWallPts1,  vWallPts2, inlierH2, cv::RHO, 3.0);
		//std::vector<cv::Mat> Rs1, Ts1, Ns1;
		//std::vector<cv::Mat> Rs2, Ts2, Ns2;
		//cv::decomposeHomographyMat(H1, mK, Rs1, Ts1, Ns1);
		//cv::decomposeHomographyMat(H2, mK, Rs2, Ts2, Ns2);
		//std::cout << "R::" << R1 << std::endl;
		//for (ipnt i = 0; i < Ns2.size(); i++) {
		//	std::cout <<"Normal::"<<Ns1[i].t()<< Ns2[i].t() << std::endl;
		//	std::cout << "Rot::" << Rs1[i].t() << std::endl << Rs2[i].t() << std::endl;
		//}
		/////////////////호모그래피 테스트

		//////////////////////////////시각화 설정
		//mpVisualizer->SetMPs(tempMPs);
		if (!mpVisualizer->isDoingProcess()) {
			mpVisualizer->SetBoolDoingProcess(true);
		}
		//////////////////////////////시각화 설정

		/////////////debug
		//cv::Mat prevImg = mpInitFrame1->GetOriginalImage();
		//cv::Mat currImg = mpInitFrame2->GetOriginalImage();
		//cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
		//cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
		//cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
		//debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
		//prevImg.copyTo(debugging(mergeRect1));
		//currImg.copyTo(debugging(mergeRect2));

		//int nTest = 0;
		//for (int i = 0; i < mpInitFrame2->mpMatchInfo->mvnMatchingIDXs.size(); i++) {
		//	int idx = mpInitFrame2->mpMatchInfo->mvnMatchingIDXs[i];
		//	
		//	cv::Point2f pt1 = mpInitFrame1->mpMatchInfo->mvMatchingPts[idx];
		//	cv::Point2f pt2 = mpInitFrame2->mpMatchInfo->mvMatchingPts[i] + ptBottom;
		//	if(mpInitFrame2->mpMatchInfo->mvpMatchingMPs[i]){
		//		cv::Mat X3D = mpInitFrame2->mpMatchInfo->mvpMatchingMPs[i]->GetWorldPos();

		//		cv::Mat proj1 = X3D.clone();
		//		cv::Mat proj2 = R1*X3D + t1;
		//		proj1 = mK*proj1;
		//		proj2 = mK*proj2;
		//		cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
		//		cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));
		//		projected2 += ptBottom;
		//		cv::line(debugging, pt1, projected1, cv::Scalar(255, 0, 255), 1);
		//		cv::line(debugging, pt2, projected2, cv::Scalar(255, 0, 255), 1);
		//	}
		//	/*cv::Point2f pt1 = vTempPts1[i];
		//	cv::Point2f pt2 = vTempPts2[i] + ptBottom;*/

		//	/*if(vFInliers[i]){
		//		cv::line(debugging, pt1, pt2, cv::Scalar(255, 0, 255), 1);
		//		nTest++;
		//	}
		//	else
		//		cv::line(debugging, pt1, pt2, cv::Scalar(0, 0, 255), 1);*/

		//	cv::circle(debugging, pt1, 1, cv::Scalar(255, 255, 0), -1);

		//	cv::circle(debugging, pt2, 1, cv::Scalar(255, 255, 0), -1);
		//}

		//////호모그래피 시각화
		////for (int i = 0; i < vFloorPts1.size(); i++) {
		////	if (inlierH1.at<uchar>(i)) {
		////		cv::circle(debugging, vFloorPts1[i], 3, cv::Scalar(255, 0, 0), -1);
		////		cv::circle(debugging, vFloorPts2[i] + ptBottom, 3, cv::Scalar(255, 0, 0), -1);
		////	}
		////	/*else {
		////		cv::line(debugging, vFloorPts1[i], vFloorPts2[i] + ptBottom, cv::Scalar(255, 255, 0), 1);
		////	}*/
		////}
		////for (int i = 0; i < vWallPts1.size(); i++) {
		////	if (inlierH2.at<uchar>(i)) {
		////		//cv::line(debugging, vWallPts1[i], vWallPts2[i]+ptBottom, cv::Scalar(0, 255, 0), 1);
		////		cv::circle(debugging, vWallPts1[i], 3, cv::Scalar(0, 255, 0), -1);
		////		cv::circle(debugging, vWallPts2[i] + ptBottom, 3, cv::Scalar(0, 255, 0), -1);
		////	}
		////	else {
		////		//cv::line(debugging, vWallPts1[i], vWallPts2[i] + ptBottom, cv::Scalar(0, 255, 255), 1);
		////	}
		////}
		//////호모그래피 시각화
		//
		//std::stringstream ss;
		//ss << "Optical flow init= " << nTest <<", "<<"||"<< nSegID <<", "<<mpInitFrame2->GetFrameID()<<"::"<<mpInitFrame2->mpMatchInfo->mvMatchingPts.size() << ", "<<tttt;
		//cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
		//cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
		//imshow("Init::OpticalFlow ", debugging);
		/////////////debug

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
		return mbInit;
		//200419
		///////////////////////////////////////////////////////////////////////////////////////

		
		/////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////BACKUP CODE
		////////////////////////rotation test
		//cv::Mat Rcw = UVR_SLAM::PlaneInformation::CalcPlaneRotationMatrix(param).clone();
		//cv::Mat normal;
		//float dist;
		//pFloor->GetParam(normal, dist);
		//cv::Mat tempP = Rcw.t()*normal;
		//if (tempP.at<float>(0) < 0.00001)
		//	tempP.at<float>(0) = 0.0;
		//if (tempP.at<float>(2) < 0.00001)
		//	tempP.at<float>(2) = 0.0;

		////카메라 자세 변환
		//mpInitFrame1->GetPose(R, t);
		//mpInitFrame1->SetPose(R*Rcw, t);
		//mpInitFrame2->GetPose(R, t);
		//mpInitFrame2->SetPose(R*Rcw, t);

		////전체 맵포인트 변환
		//for (int i = 0; i < mvpMPs.size(); i++) {
		//	UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		//	if (!pMP)
		//		continue;
		//	if (pMP->isDeleted())
		//		continue;
		//	cv::Mat tempX = Rcw.t()*pMP->GetWorldPos();
		//	pMP->SetWorldPos(tempX);
		//}
		////평면 파라메터 변환
		//pFloor->SetParam(tempP, dist);
		////////////////////////rotation test
		//바닥 맵포인트 바로 생성
		//인포메이션 바로 생성하기.


		//////매칭 테스트
		//cv::Mat debugImg;
		//std::vector<cv::DMatch> vMatches;
		//std::vector<cv::Mat> vPlanarMaps;
		//std::vector<bool> vbInliers;
		//std::vector<std::pair<int, cv::Point2f>> vPairs;
		//vPlanarMaps = std::vector<cv::Mat>(mpInitFrame2->mvKeyPoints.size(), cv::Mat::zeros(0, 0, CV_8UC1));
		//UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(mpInitFrame2, pFloor, vPlanarMaps);
		//mpMatcher->MatchingWithEpiPolarGeometry(mpInitFrame1, mpInitFrame2, vPlanarMaps, vbInliers, vMatches, vPairs, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugImg);
		////mpMatcher->DenseMatchingWithEpiPolarGeometry(mpInitFrame1, mpInitFrame2, vPlanarMaps, vPairs, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugImg);

		//std::stringstream ss;
		//ss << mpSystem->GetDirPath(0) << "/init.jpg";
		//imwrite(ss.str(), debugImg);

		//for (int i = 0; i < vPairs.size(); i++) {
		//	//기존 평면인지 확인이 어려움.
		//	
		//	auto idx = vPairs[i].first;
		//	auto pt = vPairs[i].second;
		//	if (mpInitFrame2->mvpMPs[idx]) {
		//		mpInitFrame2->mvpMPs[idx]->Delete();
		//	}
		//	UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpInitFrame2, vPlanarMaps[idx],mpInitFrame2->matDescriptor.row(idx), UVR_SLAM::PLANE_DENSE_MP);
		//	pNewMP->SetPlaneID(pFloor->mnPlaneID);
		//	pNewMP->SetObjectType(pFloor->mnPlaneType);
		//	pNewMP->AddDenseFrame(mpInitFrame1, pt);
		//	pNewMP->AddDenseFrame(mpInitFrame2, mpInitFrame2->mvKeyPoints[idx].pt);
		//	//pNewMP->AddFrame(mpInitFrame2, idx);
		//	pNewMP->UpdateNormalAndDepth();
		//	pNewMP->mnFirstKeyFrameID = mpInitFrame2->GetKeyFrameID();
		//	mpSystem->mlpNewMPs.push_back(pNewMP);
		//	pFloor->tmpMPs.push_back(pNewMP);
		//}

		//for (int i = 0; i < vMatches.size(); i++) {
		//	if (vbInliers[i]) {
		//		int idx1 = vMatches[i].trainIdx;
		//		int idx2 = vMatches[i].queryIdx;
		//		UVR_SLAM::MapPoint* pNewMP = mpInitFrame2->mvpMPs[idx2];
		//		if (pNewMP && pNewMP->isDeleted()) {
		//			continue;
		//		}
		//		if (mpInitFrame2->mvpMPs[idx2] && mpInitFrame1->mvpMPs[idx1]) {
		//			std::cout << "init::case::1" << std::endl;
		//			pNewMP = mpInitFrame2->mvpMPs[idx2];
		//			pNewMP->SetWorldPos(vPlanarMaps[idx2]);
		//		}else if (mpInitFrame2->mvpMPs[idx2]) {
		//			std::cout << "init::case::2" << std::endl;
		//			pNewMP = mpInitFrame2->mvpMPs[idx2];
		//			pNewMP->SetWorldPos(vPlanarMaps[idx2]);
		//		}
		//		else if (mpInitFrame1->mvpMPs[idx1]) {
		//			std::cout << "init::case::3" << std::endl;
		//			pNewMP = mpInitFrame1->mvpMPs[idx1];
		//			pNewMP->SetWorldPos(vPlanarMaps[idx2]);
		//		}
		//		else {
		//			std::cout << "init::case::4" << std::endl;
		//			pNewMP = new UVR_SLAM::MapPoint(mpInitFrame2, vPlanarMaps[idx2], mpInitFrame2->matDescriptor.row(idx2), UVR_SLAM::PLANE_MP);
		//			pNewMP->SetPlaneID(pFloor->mnPlaneID);
		//			pNewMP->SetObjectType(pFloor->mnPlaneType);
		//			pNewMP->AddFrame(mpInitFrame1, idx1);
		//			pNewMP->AddFrame(mpInitFrame2, idx2);
		//			pNewMP->UpdateNormalAndDepth();
		//			pNewMP->mnFirstKeyFrameID = mpInitFrame2->GetKeyFrameID();
		//		}
		//		mpSystem->mlpNewMPs.push_back(pNewMP);
		//		pFloor->tmpMPs.push_back(pNewMP);
		//		//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
		//	}
		//	else {
		//		/*if (vPlanarMaps[i].rows == 0)
		//			continue;
		//		UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpInitFrame2, vPlanarMaps[i], mpInitFrame2->matDescriptor.row(i), UVR_SLAM::PLANE_MP);
		//		pNewMP->SetPlaneID(pFloor->mnPlaneID);
		//		pNewMP->SetObjectType(pFloor->mnPlaneType);
		//		pNewMP->AddFrame(mpInitFrame2, i);
		//		pNewMP->UpdateNormalAndDepth();
		//		pNewMP->mnFirstKeyFrameID = mpInitFrame2->GetKeyFrameID();
		//		mpSystem->mlpNewMPs.push_back(pNewMP);
		//		pFloor->tmpMPs.push_back(pNewMP);*/
		//	}

		//	
		//	
		//}

		////매칭 테스트
		//UVR_SLAM::PlaneInformation::CreatePlanarMapPoints(mpInitFrame2, mpSystem);

		////plane 변환 테스트
		//float planeDist = 0.0;
		//pFloor->SetParam(tempP, dist);
		//std::cout << "plane : " << pFloor->mvpMPs.size() <<", "<<mvpFloorMPs.size()<< std::endl;
		//for (int i = 0; i < pFloor->mvpMPs.size(); i++) {
		//	UVR_SLAM::MapPoint* pMP = pFloor->mvpMPs[i];
		//	planeDist += (dist+ pMP->GetWorldPos().dot(tempP));
		//}
		//std::cout << "sum plane dist : " << planeDist << ", " << std::endl;

		//cv::Mat newR, newT;
		//mpInitFrame2->GetPose(newR, newT);
		//cv::Mat tempVis = mpInitFrame2->GetOriginalImage();
		//for (int i = 0; i < mvpMPs.size(); i++) {
		//	UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		//	if (!pMP)
		//		continue;
		//	if (pMP->isDeleted())
		//		continue;
		//	cv::Mat p3D = mvpMPs[i]->GetWorldPos();
		//	cv::Mat temp = newR*p3D + newT;
		//	std::cout << temp.t() << mK << std::endl;
		//	temp = mK*temp;
		//	std::cout << temp << std::endl;
		//	cv::Point2f pt(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
		//	cv::circle(tempVis, pt, 2, cv::Scalar(255, 0, 0), -1);
		//	cv::circle(tempVis, mpInitFrame2->mvKeyPoints[i].pt, 2, cv::Scalar(0, 0, 255), -1);
		//}
		//imshow("testestestset::", tempVis); cv::waitKey(1000);

		////////////////////rotation test
		//////////////////////////////////////////////////////BACKUP CODE
		/////////////////////////////////////////////////////////////////////////////////////////////////////
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