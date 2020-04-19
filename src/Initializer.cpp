#include <Initializer.h>
#include <FrameWindow.h>
#include <LocalMapper.h>
#include <System.h>
#include <Map.h>
#include <MatrixOperator.h>
#include <SemanticSegmentator.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <direct.h>

//추후 파라메터화. 귀찮아.
int N_matching_init_therah = 120; //80
int N_thresh_init_triangulate = 60; //80

UVR_SLAM::Initializer::Initializer() :mbInit(false), mpInitFrame1(nullptr), mpInitFrame2(nullptr) {
}
UVR_SLAM::Initializer::Initializer(System* pSystem, Map* pMap, cv::Mat _K) : mpSystem(pSystem), mK(_K), mbInit(false), mpInitFrame1(nullptr), mpInitFrame2(nullptr) {
	mpMap = pMap;
}
UVR_SLAM::Initializer::Initializer(cv::Mat _K):mK(_K),mbInit(false), mpInitFrame1(nullptr), mpInitFrame2(nullptr){
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

bool UVR_SLAM::Initializer::Initialize(Frame* pFrame, bool& bReset, int w, int h) {
	//std::cout << "Initializer::Initialize::Start" << std::endl;
	
	if (!mpInitFrame1) {
		mpInitFrame1 = pFrame;
		mpSegmentator->InsertKeyFrame(mpInitFrame1);
		
		return mbInit;
	}
	else {

		//세그멘테이션 중인 동안 대기
		if (!mpInitFrame1->isSegmented())
			return mbInit;
		
		//매칭이 적으면 mpInitFrame1을 mpInitFrame2로 교체
		cv::Mat F;
		//std::vector<cv::DMatch> tempMatches, resMatches;
		mpInitFrame2 = pFrame;
		//if (mpInitFrame2->GetFrameID() - mpInitFrame1->GetFrameID() < 3)
		//	return mbInit;

		//int count = mpMatcher->MatchingProcessForInitialization(mpInitFrame1, mpInitFrame2, F, tempMatches);
		std::vector<std::pair<cv::Point2f, cv::Point2f>> tempMatches2, resMatches2;
		int count = mpMatcher->OpticalMatchingForInitialization(mpInitFrame1, mpInitFrame2, tempMatches2);
		//int count = mpMatcher->SearchForInitialization(mpInitFrame1, mpInitFrame2, tempMatches, 100);
		if (count < 600) {//N_matching_init_therah
			delete mpInitFrame1;
			mpInitFrame1 = mpInitFrame2;
			if(!mpInitFrame1->CheckFrameType(UVR_SLAM::FLAG_SEGMENTED_FRAME))
				mpSegmentator->InsertKeyFrame(mpInitFrame1);
			return mbInit;
		}
		
		////F를 이용한 매칭
		//std::vector<bool> mvInliers;
		//float score;

		//if ((int)tempMatches.size() >= 8) {
		//	mpMatcher->FindFundamental(mpInitFrame1, mpInitFrame2, tempMatches, mvInliers, score, F);
		//	F.convertTo(F, CV_32FC1);
		//}
		//
		//if ((int)tempMatches.size() < 8 || F.empty()) {
		//	F.release();
		//	F = cv::Mat::zeros(0, 0, CV_32FC1);
		//	//delete mpInitFrame1;
		//	//mpInitFrame1 = mpInitFrame2;
		//	return mbInit;
		//}

		//for (unsigned long i = 0; i < tempMatches.size(); i++) {
		//	if (mvInliers[i]) {
		//		resMatches.push_back(tempMatches[i]);
		//	}
		//}
		//count = resMatches.size();
		////Optical flow 매칭 관련
		std::vector<uchar> mvInliers2;
		//float score2;
		//if ((int)tempMatches2.size() >= 8) {
		//	mpMatcher->FindFundamental(mpInitFrame1, mpInitFrame2, tempMatches2, mvInliers2, score2, F);
		//	F.convertTo(F, CV_32FC1);
		//}

		//if ((int)tempMatches2.size() < 8 || F.empty()) {
		//	F.release();
		//	F = cv::Mat::zeros(0, 0, CV_32FC1);
		//	//delete mpInitFrame1;
		//	//mpInitFrame1 = mpInitFrame2;
		//	return mbInit;
		//}

		//for (unsigned long i = 0; i < tempMatches2.size(); i++) {
		//	if (mvInliers2[i]) {
		//		resMatches2.push_back(tempMatches2[i]);
		//	}
		//}
		std::vector<cv::Point2f> vPts1, vPts2;
		for (int i = 0; i < tempMatches2.size(); i++) {
			vPts1.push_back(tempMatches2[i].first);
			vPts2.push_back(tempMatches2[i].second);
		}

		cv::Mat F12 = cv::findFundamentalMat(vPts1, vPts2, mvInliers2, cv::FM_RANSAC);
		std::cout << F12 << F12.type() <<", "<<CV_32FC1<< std::endl;
		F12.convertTo(F12, CV_32FC1);
		for (unsigned long i = 0; i < tempMatches2.size(); i++) {
			if (mvInliers2[i]) {
				resMatches2.push_back(tempMatches2[i]);
			}
		}
		count = resMatches2.size();
		std::cout << "Init::Opt::" << resMatches2.size()<<"::"<< tempMatches2.size() << std::endl;
		//count = resMatches2.size();

		////F를 이용한 매칭


		std::vector<UVR_SLAM::InitialData*> vCandidates;
		UVR_SLAM::InitialData *mC1 = new UVR_SLAM::InitialData(count);
		UVR_SLAM::InitialData *mC2 = new UVR_SLAM::InitialData(count);
		UVR_SLAM::InitialData *mC3 = new UVR_SLAM::InitialData(count);
		UVR_SLAM::InitialData *mC4 = new UVR_SLAM::InitialData(count);
		vCandidates.push_back(mC1);
		vCandidates.push_back(mC2);
		vCandidates.push_back(mC3);
		vCandidates.push_back(mC4);
		SetCandidatePose(F12, resMatches2, vCandidates);
		int resIDX = SelectCandidatePose(vCandidates);

		cv::Mat vis1 = mpInitFrame1->GetOriginalImage();
		cv::Mat vis2 = mpInitFrame2->GetOriginalImage();
		cv::Point2f ptBottom = cv::Point2f(0, vis1.rows);
		cv::Rect mergeRect1 = cv::Rect(0, 0, vis1.cols, vis1.rows);
		cv::Rect mergeRect2 = cv::Rect(0, vis1.rows, vis1.cols, vis1.rows);
		cv::Mat debugging = cv::Mat::zeros(vis1.rows * 2, vis1.cols, vis1.type());
		vis1.copyTo(debugging(mergeRect1));
		vis2.copyTo(debugging(mergeRect2));
		//cvtColor(vis1, vis1, CV_8UC3);
		cv::RNG rng = cv::RNG(12345);

		if (resIDX > 0)
			std::cout << "init::triangulation::" << vCandidates[resIDX]->nGood << std::endl;
		else
			std::cout << "init::triangulation::fail!!" << std::endl;

		if (resIDX > 0 && vCandidates[resIDX]->nGood > N_thresh_init_triangulate) {

			mpSegmentator->InsertKeyFrame(mpInitFrame2);

			std::cout << vCandidates[resIDX]->nGood << std::endl;
			mpInitFrame1->SetPose(vCandidates[resIDX]->R0, vCandidates[resIDX]->t0);
			mpInitFrame2->SetPose(vCandidates[resIDX]->R, vCandidates[resIDX]->t); //두번째 프레임은 median depth로 변경해야 함.

			//////키프레임으로 설정
			mpInitFrame1->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME, 0);
			mpInitFrame2->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
			/*mpInitFrame1->SetKeyFrameID(0);
			mpInitFrame2->SetKeyFrameID();*/
			
			//윈도우에 두 개의 키프레임 넣기
			//20.01.02 deque에서 list로 변경함.
			mpFrameWindow->AddFrame(mpInitFrame1);
			mpFrameWindow->AddFrame(mpInitFrame2);
			
			//mpInitFrame1->mTrackedDescriptor = cv::Mat::zeros(0, mpInitFrame1->matDescriptor.cols, mpInitFrame1->matDescriptor.type());
			//mpInitFrame2->mTrackedDescriptor = cv::Mat::zeros(0, mpInitFrame2->matDescriptor.cols, mpInitFrame2->matDescriptor.type());
			//mpFrameWindow->push_back(mpInitFrame1);
			//mpFrameWindow->push_back(mpInitFrame2);


			//객체 세그멘테이션 될 때가지 웨이트
			while (!mpInitFrame2->isSegmented()) {
			}
			
			//맵포인트 생성 및 키프레임과 연결
			int nMatch = 0;
			std::vector<MapPoint*> vpMPs;
			std::vector<UVR_SLAM::Frame*> vpKFs;
			vpKFs.push_back(mpInitFrame1);
			vpKFs.push_back(mpInitFrame2);
			
			std::cout << "init::keyframeid::" << mpInitFrame1->GetKeyFrameID() << ", " << mpInitFrame2->GetKeyFrameID() << std::endl;

			//auto mvpOPs1 = mpInitFrame1->GetObjectVector();
			//auto mvpOPs2 = mpInitFrame2->GetObjectVector();
			std::vector<int> idxs;
			for (int i = 0; i < vCandidates[resIDX]->mvX3Ds.size(); i++) {
				if (vCandidates[resIDX]->vbTriangulated[i]) {
					auto pt1 = resMatches2[i].first;
					auto pt2 = resMatches2[i].second;
					UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpInitFrame1, vCandidates[resIDX]->mvX3Ds[i], cv::Mat());
					pNewMP->AddDenseFrame(mpInitFrame1, pt1);
					pNewMP->AddDenseFrame(mpInitFrame2, pt2);

					pNewMP->mnFirstKeyFrameID = mpInitFrame2->GetKeyFrameID();

					nMatch++;
					idxs.push_back(i);
					vpMPs.push_back(pNewMP);

					////matching pt 저장
					mpInitFrame1->mvpMatchingMPs.push_back(pNewMP);
					mpInitFrame1->mvMatchingPts.push_back(pt1);
					mpInitFrame2->mvpMatchingMPs.push_back(pNewMP);
					mpInitFrame2->mvMatchingPts.push_back(pt2);
				}
			}
			std::cout << "init::ba::start::" << mpInitFrame2->TrackedMapPoints(2) <<"::"<< nMatch << std::endl;
			//최적화 수행 후 Map 생성
			//UVR_SLAM::Optimization::InitOptimization(vCandidates[resIDX], resMatches, mpInitFrame1, mpInitFrame2, mK, bInitOpt);
			UVR_SLAM::Optimization::InitBundleAdjustment(vpKFs, vpMPs, 20);
			std::cout << "init::ba::end::"<< mpInitFrame2->TrackedMapPoints(2) << std::endl;
			//calculate median depth
			float medianDepth;
			mpInitFrame1->ComputeSceneMedianDepth(medianDepth);
			float invMedianDepth = 1.0f / medianDepth;
			
			if (medianDepth < 0.0 || mpInitFrame2->TrackedMapPoints(2) < 100){
				mbInit = false;
				bReset = true;
				std::cout << "Reset" << std::endl;
				/*while (mpSegmentator->isDoingProcess()) {
				}*/
				return mbInit;
			}
			//포즈 업데이트
			cv::Mat R, t;
			mpInitFrame2->GetPose(R, t);
			mpInitFrame2->SetPose(R, t*invMedianDepth);

			//맵포인트 업데이트
			for (int i = 0; i < vpMPs.size(); i++) {
				UVR_SLAM::MapPoint* pMP = vpMPs[i];
				if (!pMP)
					continue;
				if (pMP->isDeleted())
					continue;
				pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
				pMP->UpdateNormalAndDepth();
				pMP->IncreaseFound(2);
				pMP->IncreaseVisible(2);
			}
			
			//////////
			//바닥 인식 및 이것 저것 테스트
			
			int count = 0;
			auto mvpMPs = mpInitFrame2->GetMapPoints();
			std::vector<UVR_SLAM::MapPoint*> mvpFloorMPs;
			//auto mvpOPs = mpInitFrame2->GetObjectVector();
			///////////////두 프레임에서 작동하도록 변경.
			/*for (int i = 0; i < mvpMPs.size(); i++) {
				UVR_SLAM::MapPoint* pMP = mvpMPs[i];
				if (!pMP)
					continue;
				if (pMP->isDeleted())
					continue;
				if (mvpOPs2[i] == UVR_SLAM::ObjectType::OBJECT_FLOOR) {
					count++;
					mvpFloorMPs.push_back(pMP);
				}
			}*/
			std::vector<cv::Point2f> vTempFloorPts;
			for (int i = 0; i < vpMPs.size(); i++) {
				UVR_SLAM::MapPoint* pMP = vpMPs[i];
				if (!pMP)
					continue;
				if (pMP->isDeleted())
					continue;
				auto pt1 = resMatches2[idxs[i]].first;
				auto pt2 = resMatches2[idxs[i]].second;
				int label1 = mpInitFrame1->matLabeled.at<uchar>(pt1.y / 2, pt1.x / 2);
				if (label1 == 150) {
					int label2 = mpInitFrame2->matLabeled.at<uchar>(pt2.y / 2, pt2.x / 2);
					if (label1 == label2) {
						count++;
						mvpFloorMPs.push_back(pMP);
						vTempFloorPts.push_back(pt2);
						cv::line(debugging, pt1, pt2 + ptBottom, cv::Scalar(255, 0, 0), 1);
						
					}
				}
			}
			imshow("a;lsdjf;lasjkdf", debugging); cv::waitKey(1);
			std::cout << "floor point ::" << count << std::endl;
			if (count < 20)
				return mbInit;
			/*std::cout << "init1 test : " << mpInitFrame1->isSegmented() << std::endl;
			auto temptttt = mpInitFrame1->GetOriginalImage();
			auto mvpOPs11 = mpInitFrame1->GetObjectVector();
			for (int i = 0; i < mvpOPs11.size(); i++) {
				if (mvpOPs11[i] == UVR_SLAM::ObjectType::OBJECT_FLOOR)
					circle(temptttt, mpInitFrame1->mvKeyPoints[i].pt, 3, cv::Scalar(255, 0, 255), -1);
			}
			imshow("init1tttttt", temptttt);
			cv::waitKey(1);*/

			/////////////////////////
			//평면 검출을 초기화 과정에 추가
			UVR_SLAM::PlaneInformation* pFloor = new UVR_SLAM::PlaneInformation();
			bool bRes = UVR_SLAM::PlaneInformation::PlaneInitialization(pFloor, mvpFloorMPs, mpInitFrame2->GetFrameID(), 1500, 0.01, 0.4);
			cv::Mat param = pFloor->GetParam();
			if(bRes){
				std::cout <<"Init::param::"<< param.t()<<", "<<pFloor->mvpMPs.size()<<", "<<mvpFloorMPs.size() << std::endl;
			}
			/*else {
				mbInit = false;
				bReset = true;
				std::cout << "Reset" << std::endl;
				return mbInit;
			}*/
			if (!bRes || abs(param.at<float>(1)) < 0.98)//98
			{
				mbInit = false;
				bReset = true;
				std::cout << "Reset" << std::endl;
				return mbInit;
			}

			//초기 평면 MP 설정 필요
			mpInitFrame1->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpInitFrame1, pFloor);
			mpInitFrame2->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpInitFrame2, pFloor);
			cv::Mat invP, invT, invK;
			mpInitFrame2->mpPlaneInformation->Calculate();
			mpInitFrame2->mpPlaneInformation->GetInformation(invP, invT, invK);
			for (int i = 0; i < mvpFloorMPs.size(); i++) {
				UVR_SLAM::MapPoint* pMP = mvpFloorMPs[i];
				if (!pMP)
					continue;
				if (pMP->isDeleted())
					continue;
				cv::Mat X3D;
				bool bRes = PlaneInformation::CreatePlanarMapPoint(vTempFloorPts[i], invP, invT, invK, X3D);
				if (bRes)
				{
					pMP->SetPlaneID(pFloor->mnPlaneID);
					pMP->SetWorldPos(X3D);
					pMP->SetMapPointType(UVR_SLAM::PLANE_DENSE_MP);
				}
			}
			/*for (int i = 0; i < pFloor->mvpMPs.size(); i++) {
				pFloor->mvpMPs[i]->SetMapPointType(UVR_SLAM::PLANE_DENSE_MP);
			}*/
			
			//윈도우 로컬맵, 포즈 설정
			mpFrameWindow->SetPose(R, t*invMedianDepth);
			//mpFrameWindow->SetLocalMap(mpInitFrame2->GetFrameID());
			mpFrameWindow->SetLastFrameID(mpInitFrame2->GetFrameID());
			mpFrameWindow->mnLastMatches = nMatch;

			/////////////////////////////////////////////////////////////////////////////////////////////////////
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
			/////////////////////////////////////////////////////////////////////////////////////////////////////
			
			//평면 설정하기
			pFloor->mnFrameID = mpInitFrame2->GetFrameID();
			pFloor->mnPlaneType = ObjectType::OBJECT_FLOOR;
			//mpInitFrame1->mvpPlanes.push_back(pFloor);
			//mpInitFrame2->mvpPlanes.push_back(pFloor);
			mpMap->SetFloorPlaneInitialization(true);
			mpMap->mpFloorPlane = pFloor;

			//mpSegmentator->InsertKeyFrame(mpInitFrame1);

			////두 키프레임 연결
			mpInitFrame1->AddKF(mpInitFrame2, 0);
			mpInitFrame2->AddKF(mpInitFrame1, 0);
			////////두 키프레임 연결

			//////////insert keyframe
			mpLocalMapper->InsertKeyFrame(mpInitFrame1);
			mpLocalMapper->InsertKeyFrame(mpInitFrame2);
			mpPlaneEstimator->InsertKeyFrame(mpInitFrame1);
			mpPlaneEstimator->InsertKeyFrame(mpInitFrame2);
			//////////insert keyframe

			mpMap->SetCurrFrame(mpInitFrame2);
			mbInit = true;

			if (mbInit) {
				mpSystem->SetDirPath(0);
				std::string base = mpSystem->GetDirPath(0);
				std::stringstream ss;
				ss << base << "/dense";
				_mkdir(ss.str().c_str());
				ss.str("");
				ss << base << "/kfmatching";
				_mkdir(ss.str().c_str());
				ss.str("");
				ss << base << "/tracking";
				_mkdir(ss.str().c_str());
			}

			std::cout << "세그멘테이션 끝::" << count <<"::"<<pFloor->GetParam().t()<< std::endl;
			std::cout << "Initializer::" << nMatch <<"::"<<mpFrameWindow->GetLastFrameID()<< std::endl;
		}

		//imshow("Initialization::Frame::1", debugging);
		//imshow("Initialization::Frame::2", vis2);
		waitKey(1);

	}
	//std::cout << "Initializer::Initialize::End" << std::endl;
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