#include <PlaneEstimator.h>
#include <random>
#include <System.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <SegmentationData.h>
#include <MapPoint.h>
#include <Initializer.h>

static int nPlaneID = 0;

UVR_SLAM::PlaneEstimator::PlaneEstimator() :mbDoingProcess(false), mnProcessType(0), mpLayoutFrame(nullptr){
}
UVR_SLAM::PlaneEstimator::PlaneEstimator(std::string strPath,cv::Mat K, cv::Mat K2, int w, int h) : mK(K), mK2(K2),mbDoingProcess(false), mnWidth(w), mnHeight(h), mnProcessType(0), mpLayoutFrame(nullptr),
mpPrevFrame(nullptr), mpTargetFrame(nullptr)
{
	cv::FileStorage fSettings(strPath, cv::FileStorage::READ);
	mnRansacTrial = fSettings["Layout.trial"];
	mfThreshPlaneDistance = fSettings["Layout.dist"];
	mfThreshPlaneRatio = fSettings["Layout.ratio"];
	mfThreshNormal = fSettings["Layout.normal"];

	//mnNeedFloorMPs = fSettings["Layout.nfloor"];
	//mnNeedWallMPs = fSettings["Layout.nwall"];
	//mnNeedCeilMPs = fSettings["Layout.nceil"];
	//mnConnect = fSettings["Layout.nconnect"];
	fSettings.release();
}
UVR_SLAM::PlaneEstimator::~PlaneEstimator() {}

///////////////////////////////////////////////////////////////////////////////
//기본 함수들
void UVR_SLAM::PlaneEstimator::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}
void UVR_SLAM::PlaneEstimator::SetFrameWindow(UVR_SLAM::FrameWindow* pWindow) {
	mpFrameWindow = pWindow;
}
void UVR_SLAM::PlaneEstimator::SetTargetFrame(Frame* pFrame) {
	mpPrevFrame = mpTargetFrame;
	mpTargetFrame = pFrame;
}
void UVR_SLAM::PlaneEstimator::SetInitializer(UVR_SLAM::Initializer* pInitializer) {
	mpInitializer = pInitializer;
}
void UVR_SLAM::PlaneEstimator::SetBoolDoingProcess(bool b, int ptype) {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	mbDoingProcess = b;
	mnProcessType = ptype;
}
bool UVR_SLAM::PlaneEstimator::isDoingProcess() {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	return mbDoingProcess;
}

void UVR_SLAM::PlaneEstimator::InsertKeyFrame(UVR_SLAM::Frame *pKF)
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mKFQueue.push(pKF);
}

bool UVR_SLAM::PlaneEstimator::CheckNewKeyFrames()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	return(!mKFQueue.empty());
}

void UVR_SLAM::PlaneEstimator::ProcessNewKeyFrame()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mpTargetFrame = mKFQueue.front();
	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_LAYOUT_FRAME);
	mpSystem->SetPlaneFrameID(mpTargetFrame->GetKeyFrameID());
	mKFQueue.pop();
}

///////////////////////////////////////////////////////////////////////////////

void UVR_SLAM::PlaneEstimator::Run() {

	std::string mStrPath;

	std::vector<UVR_SLAM::PlaneInformation*> mvpPlanes;

	while (1) {
		if (CheckNewKeyFrames()) {
			//저장 디렉토리 명 획득
			SetBoolDoingProcess(true,0);
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			ProcessNewKeyFrame();
			mStrPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());

			//현재 레이아웃 추정하는 키프레임 설정
			if (mpPrevFrame) {
				std::cout << mpPrevFrame->GetKeyFrameID() << ", " << mpTargetFrame->GetKeyFrameID() << std::endl;
			}
			int nTargetID = mpTargetFrame->GetFrameID();
			mpFrameWindow->SetLastLayoutFrameID(nTargetID);

			cv::Mat R, t;
			mpTargetFrame->GetPose(R, t);
			cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
			R.copyTo(T.rowRange(0, 3).colRange(0, 3));
			t.copyTo(T.col(3).rowRange(0, 3));
			cv::Mat invT = T.inv();

			
			//std::set<UVR_SLAM::MapPoint*> mspLocalFloorMPs, mspLocalWallMPs, mspLocalCeilMPs;
			//std::vector<UVR_SLAM::MapPoint*> mvpLocalFloorMPs, mvpLocalWallMPs, mvpLocalCeilMPs;
			//
			//for (int i = 0; i < mvpMPs.size(); i++) {
			//	UVR_SLAM::MapPoint* pMP = mvpMPs[i];
			//	if (!pMP)
			//		continue;
			//	if (pMP->isDeleted())
			//		continue;
			//	//type check
			//	auto type = mvpOPs[i];
			//	switch (type) {
			//	case UVR_SLAM::ObjectType::OBJECT_FLOOR:
			//		mspLocalFloorMPs.insert(pMP);
			//		break;
			//	case UVR_SLAM::ObjectType::OBJECT_WALL:
			//		mspLocalWallMPs.insert(pMP);
			//		break;
			//	case UVR_SLAM::ObjectType::OBJECT_CEILING:
			//		mspLocalCeilMPs.insert(pMP);
			//		break;
			//	}
			//}
			//평면 변수 선언
			//std::cout << "Local Keyframe ::" << mspLocalFloorMPs.size() << ", " << mspLocalWallMPs.size() << ", " << mspLocalCeilMPs.size() << std::endl;

			UVR_SLAM::PlaneInformation* pPlane1, *pPlane2, *pPlane3;
			int tempFloorID = 0;
			int tempWallID = 0;
			pPlane1 = new UVR_SLAM::PlaneInformation();
			pPlane2 = new UVR_SLAM::PlaneInformation();

			//평면 RANSAC 초기화
			bool bLocalFloor = false;
			/*if (mspLocalFloorMPs.size() > 10)
				bLocalFloor = PlaneInitialization(pPlane2, mspLocalFloorMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
			*/
			if (mpTargetFrame->mspFloorMPs.size() > 10){
				UVR_SLAM::PlaneInformation* pTemp = new UVR_SLAM::PlaneInformation();
				bLocalFloor = PlaneInitialization(pPlane2, mpTargetFrame->mspFloorMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
			}
			if (mpFrameWindow->mspWallMPs.size() > 10) {
				UVR_SLAM::PlaneInformation* pTemp = new UVR_SLAM::PlaneInformation();
				PlaneInitialization(pTemp, mpFrameWindow->mspWallMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
			}
			//평면을 찾은 경우 현재 평면을 교체하던가 수정을 해야 함.
			if (bLocalFloor) {
				tempFloorID = pPlane2->mnPlaneID;
				pPlane2->mnFrameID = nTargetID;
				pPlane2->mnPlaneType = ObjectType::OBJECT_FLOOR;
				pPlane2->mnCount = 1;
				
				//CreatePlanarMapPoints(mvpMPs, mvpOPs, pPlane2, invT);
			}

			///////벽을 찾는 과정
			//벽은 현재 프레임에서 바닥을 찾았거나 바닥을 찾은 상태일 때 수행.
			bool bLocalWall = false;
			if(mpTargetFrame->mspWallMPs.size() > 10 && (bLocalFloor || mvpPlanes.size() > 0 )){
				//bLocalWall = PlaneInitialization(pPlane1, mspLocalWallMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);

				UVR_SLAM::PlaneInformation* groundPlane = nullptr;
				if (mvpPlanes.size() > 0)
					groundPlane = mvpPlanes[0];
				else
					groundPlane = pPlane2;
				bLocalWall = PlaneInitialization(pPlane1, groundPlane, 1, mpTargetFrame->mspWallMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
			}
			//라인 프로젝션
			if (bLocalWall) {
				tempWallID = pPlane1->mnPlaneID;
				pPlane1->mnFrameID = nTargetID;
				pPlane1->mnPlaneType = ObjectType::OBJECT_WALL;
				pPlane1->mnCount = 1;

			}
			///////벽을 찾는 과정
			
			bool bFailCase = false;
			bool bFailCase2 = false;
			//compare & association
			//평면을 교체하게 되면 잘못되는 경우가 생김.
			for (int i = 0; i < mvpPlanes.size(); i++) {
				UVR_SLAM::PlaneInformation* p = mvpPlanes[i];
				float ratio = 0.0;
				switch (p->mnPlaneType) {
				case ObjectType::OBJECT_FLOOR:
					if(bLocalFloor){
						ratio = pPlane2->CalcOverlapMPs(p, nTargetID);
						//std::cout << "Association::Test::" <<p->mnCount<<"::"<< p->mnPlaneID << ", " << pPlane2->mnPlaneID << "=" << ratio << std::endl;
						std::cout <<"PLANE::"<< pPlane2->matPlaneParam.t() << ", " << p->matPlaneParam.t() << std::endl;
						std::cout << pPlane2->CalcCosineSimilarity(p)<<", "<<pPlane2->CalcPlaneDistance(p)<<std::endl;
						if (pPlane2->CalcCosineSimilarity(p) < 0.98){
							bFailCase = true;
							//mvpPlanes[i] = pPlane2;
						}
						if (pPlane2->CalcPlaneDistance(p) >= 0.015)
							bFailCase2 = true;
						/*if (ratio > 0.15) {
							p->Merge(pPlane2, nTargetID, mfThreshPlaneDistance);
							p->mnCount++;
						}
						else {
							mvpPlanes[i] = pPlane2;
						}*/
					}
					break;
				case ObjectType::OBJECT_WALL:
					break;
				}
				
			}
			//merge
			auto mvpMPs = mpTargetFrame->GetMapPoints();
			auto mvpOPs = mpTargetFrame->GetObjectVector();
			if (bFailCase) {
				CreatePlanarMapPoints(mvpMPs, mvpOPs, mvpPlanes[0], invT);
			}
			else if(bLocalFloor){
				CreatePlanarMapPoints(mvpMPs, mvpOPs, pPlane2, invT);
			}
			
			if (mvpPlanes.size() == 0 && bLocalFloor) {
				mvpPlanes.push_back(pPlane2);
			}
			else {
				//save txt
				/*std::ofstream f;
				std::stringstream sss;
				sss << mStrPath.c_str() << "/plane.txt";
				f.open(sss.str().c_str());
				for (int j = 0; j < mvpPlanes[0]->mvpMPs.size(); j++) {
					UVR_SLAM::MapPoint* pMP = mvpPlanes[0]->mvpMPs[j];
					if (!pMP) {
						continue;
					}
					if (pMP->isDeleted()) {
						continue;
					}
					cv::Mat Xw = pMP->GetWorldPos();
					f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) << " 1" << std::endl;
				}
				f.close();*/
			}
			//update local map
			//mpFrameWindow->SetLocalMap(mpTargetFrame->GetFrameID());

			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto leduration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float letime = leduration / 1000.0;
			std::stringstream ss;
			ss << "Layout time = " << letime << ", " << mpTargetFrame->mspWallMPs.size() << ", " << mpTargetFrame->mspFloorMPs.size();
			mpSystem->SetPlaneString(ss.str());

			//////test
			//////save txt
			std::ofstream f;
			std::stringstream sss;
			sss << mStrPath.c_str() << "/plane.txt";
			f.open(sss.str().c_str());
			mvpMPs = mpTargetFrame->GetMapPoints();
			mvpOPs = mpTargetFrame->GetObjectVector();
			if(bLocalFloor)
				for (int j = 0; j < mvpMPs.size(); j++) {
					UVR_SLAM::MapPoint* pMP = mvpMPs[j];
					if (!pMP) {
						continue;
					}
					if (pMP->isDeleted()) {
						continue;
					}
					cv::Mat Xw = pMP->GetWorldPos();
				
					if (pMP->GetPlaneID() > 0) {
						if (pMP->GetPlaneID() == tempFloorID)
						{
							f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) << " 1" << std::endl;
						}
						else if (pMP->GetPlaneID() == tempWallID) {
							f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) << " 2" << std::endl;
						}
					}
					else
						f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) << " 0" << std::endl;
				}
			f.close();

			////////////////image
			cv::Mat vImg = mpTargetFrame->GetOriginalImage();
			for (int j = 0; j < mvpMPs.size(); j++) {
				auto type = mvpOPs[j];
				switch (type) {
				case ObjectType::OBJECT_WALL:
					cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 255), -1);
					break;
				case ObjectType::OBJECT_CEILING:
					cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 0), -1);
					break;
				case ObjectType::OBJECT_NONE:
					cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 0, 0), -1);
					break;
				}
					

				UVR_SLAM::MapPoint* pMP = mvpMPs[j];

				if (!pMP) {
					continue;
				}
				if (pMP->isDeleted()) {
					continue;
				}
				cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(255, 0, 0), -1);
				cv::Mat Xw = pMP->GetWorldPos();

				if(bLocalFloor && pMP->GetPlaneID()==pPlane2->mnPlaneID){
					cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 3, cv::Scalar(255, 0, 255));
				}
				else if (bLocalWall && pMP->GetPlaneID() == pPlane1->mnPlaneID) {
					cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 3, cv::Scalar(255, 255, 0));
				}
				if (!bLocalFloor && mvpPlanes.size() > 0 && pMP->GetPlaneID() == mvpPlanes[0]->mnPlaneID ) {
					cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 3, cv::Scalar(255, 0, 255));
				}
			}
			//line visualization
			if (bLocalWall && (mvpPlanes.size() > 0)) {
				float m;
				cv::Mat mLine = mvpPlanes[0]->FlukerLineProjection(pPlane1, mpTargetFrame->GetRotation(), mpTargetFrame->GetTranslation(), mK2, m);
				cv::Point2f sPt, ePt;
				mvpPlanes[0]->CalcFlukerLinePoints(sPt, ePt, 0.0, mnHeight, mLine);
				cv::line(vImg, sPt, ePt, cv::Scalar(0, 255, 0), 3);
			}

			if (bFailCase) {
				cv::Mat temp = cv::Mat::zeros(vImg.size(), CV_8UC3);
				rectangle(temp, cv::Point(0, 0), cv::Point(vImg.cols, vImg.rows), cv::Scalar(0, 255, 0), -1);
				cv::addWeighted(vImg, 0.7, temp, 0.3, 0.0, vImg);
			}

			if (bFailCase2) {
				cv::Mat temp = cv::Mat::zeros(vImg.size(), CV_8UC3);
				rectangle(temp, cv::Point(0, 0), cv::Point(vImg.cols, vImg.rows), cv::Scalar(255, 0, 0), -1);
				cv::addWeighted(vImg, 0.7, temp, 0.3, 0.0, vImg);
			}

			sss.str("");
			sss << mStrPath.c_str() << "/plane.jpg";
			cv::imwrite(sss.str(), vImg);
			imshow("Output::PlaneEstimation", vImg); cv::waitKey(1);

			SetBoolDoingProcess(false, 1);
		}
	}

	//나중에 클래스에 옮길 것.
	//이제 이것들도 삭제 될 때 처리가 필요함.
	std::set<UVR_SLAM::MapPoint*> mspFloorMPs, mspWallMPs, mspCeilMPs;

	int RANSAC_TRIAL = 2500;
	float thresh_plane_distance = 0.01;
	float thresh_ratio = 0.1;

	bool bInitTest = false;
	bool b1 = false;
	bool b2 = false;
	bool b3 = false;
	UVR_SLAM::PlaneInformation* pPlane1, *pPlane2, *pPlane3;
	pPlane1 = new UVR_SLAM::PlaneInformation();
	pPlane2 = new UVR_SLAM::PlaneInformation();
	pPlane3 = new UVR_SLAM::PlaneInformation();

	while (1) {
		//프로세스 타입 1 : key frame 에서 맵포인트에 대한 처리하고 평면을 추정하는 과정
		//프로세스 타입 2 : frame에서 평면 레이아웃 시각화하고 새로운 평면 맵포인트를 추가하는 과정.
		if (isDoingProcess()) {
			
			if (!bInitTest && mnProcessType == 3) {
				std::cout << "ground plane init test!!!" << std::endl;

			}

			//여기에 들어오면 일단 전부 레이아웃은 보여줄 준비를 해야 함.
			//평면이 추정되었는지 확인해야 함.
			//레이아웃 추정도 필요함.
			cv::Mat vis = mpTargetFrame->GetOriginalImage();
			cv::Mat R, t, P1, P2, P3;
			cv::Mat invP1, invP2, invP3;
			cv::Mat PLw1, Lw1, NLw1, Ni1; //바닥과 벽
			cv::Mat PLw2, Lw2, NLw2, Ni2; //천장과 벽
			float m1, m2;
			Ni1 = cv::Mat::zeros(0, 0, CV_32FC1);
			Ni2 = cv::Mat::zeros(0, 0, CV_32FC1);

			mpTargetFrame->GetPose(R, t);
			cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
			R.copyTo(T.rowRange(0, 3).colRange(0, 3));
			t.copyTo(T.col(3).rowRange(0, 3));
			cv::Mat invT = T.inv();

			if(b1){
				P1 = pPlane1->matPlaneParam.clone();
				invP1 = invT.t()*P1;
			}
			if(b2){
				P2 = pPlane2->matPlaneParam.clone();
				invP2 = invT.t()*P2;
			}
			if(b3){
				P3 = pPlane3->matPlaneParam.clone();
				invP3 = invT.t()*P3;
			}

			if (mnProcessType >= 1) {
				if (b1 && b2) {
					//Ni1 = FlukerLineProjection(P1, P2, R, t, m1);
					//PLw1 = P1*P2.t() - P2*P1.t();
					//Lw1 = cv::Mat::zeros(6, 1, CV_32FC1);
					//Lw1.at<float>(3) = PLw1.at<float>(2, 1);
					//Lw1.at<float>(4) = PLw1.at<float>(0, 2);
					//Lw1.at<float>(5) = PLw1.at<float>(1, 0);
					//NLw1 = PLw1.col(3).rowRange(0, 3);
					//NLw1.copyTo(Lw1.rowRange(0, 3));

					////Line projection test : Ni
					//cv::Mat T2 = cv::Mat::zeros(6, 6, CV_32FC1);
					//R.copyTo(T2.rowRange(0, 3).colRange(0, 3));
					//R.copyTo(T2.rowRange(3, 6).colRange(3, 6));
					//cv::Mat tempSkew = cv::Mat::zeros(3, 3, CV_32FC1);
					//tempSkew.at<float>(0, 1) = -t.at<float>(2);
					//tempSkew.at<float>(1, 0) = t.at<float>(2);
					//tempSkew.at<float>(0, 2) = t.at<float>(1);
					//tempSkew.at<float>(2, 0) = -t.at<float>(1);
					//tempSkew.at<float>(1, 2) = -t.at<float>(0);
					//tempSkew.at<float>(2, 1) = t.at<float>(0);
					//tempSkew *= R;
					//tempSkew.copyTo(T2.rowRange(0, 3).colRange(3, 6));
					//cv::Mat Lc = T2*Lw1;
					//cv::Mat Nc = Lc.rowRange(0, 3);
					//Ni1 = mK2*Nc;

					float x1 = 0;
					float y1 = 0;
					if (Ni1.at<float>(0) != 0)
						x1 = -Ni1.at<float>(2) / Ni1.at<float>(0);

					float x2 = 0;
					float y2 = mnHeight;
					if (Ni1.at<float>(0) != 0)
						x2 = (-Ni1.at<float>(2) - Ni1.at<float>(1)*y2) / Ni1.at<float>(0);

					cv::line(vis, cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Scalar(255, 0, 0), 2);
				}
				if (b3 && b2) {
					//Ni2 = FlukerLineProjection(P2, P3, R, t, m2);

					float x1 = 0;
					float y1 = 0;
					if (Ni2.at<float>(0) != 0)
						x1 = -Ni2.at<float>(2) / Ni2.at<float>(0);

					float x2 = 0;
					float y2 = mnHeight;
					if (Ni2.at<float>(0) != 0)
						x2 = (-Ni2.at<float>(2) - Ni2.at<float>(1)*y2) / Ni2.at<float>(0);
					cv::line(vis, cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Scalar(255, 0, 0), 2);
				}
			}

			//if (mnProcessType == 1 && mpLayoutFrame) {
			//	{
			//		//Last Semantic Frame 처리
			//		cv::Mat R, t;
			//		mpLayoutFrame->GetPose(R, t);

			//		cv::Mat imgLF = mpLayoutFrame->GetOriginalImage();
			//		for (int i = 0; i < mpLayoutFrame->mvKeyPoints.size(); i++) {
			//			UVR_SLAM::MapPoint* pMP = mpLayoutFrame->GetMapPoint(i);
			//			if (!pMP)
			//				continue;
			//			if (pMP->isDeleted())
			//				continue;
			//			cv::Point2f p2D;
			//			cv::Mat pCam;
			//			pMP->Projection(p2D, pCam, mpLayoutFrame->GetRotation(), mpLayoutFrame->GetTranslation(), mK, mnWidth, mnHeight);
			//			if (pMP->GetMapPointType() == MapPointType::PLANE_MP) {
			//				cv::circle(imgLF, p2D, 2, cv::Scalar(255, 0, 255), -1);
			//				//cv::circle(imgLF, mpLayoutFrame->mvKeyPoints[i].pt, 2, cv::Scalar(0, 0, 255), -1);
			//			}
			//		}
			//		cv::imshow("plane label test", imgLF);
			//	}
			//}

			if (mnProcessType >= 2) {
				//현재 프레임에서는 어떻게 전파를 할 것인가?
				//라인을 획득하게 되면 레이블링이 가능해진다.

				//평면이 존재하는지 확인하고, 평면의 타입을 확인하기.

				//set mp로 바닥, 벽, 천장 획득하기
				for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {

					if (mpTargetFrame->mvbMPInliers[i]) {
						UVR_SLAM::MapPoint* pMP = mpTargetFrame->mvpMPs[i];
						if (!pMP)
							continue;
						if (pMP->isDeleted())
							continue;
						//type check
						auto type = pMP->GetObjectType();
						switch (type) {
						case UVR_SLAM::ObjectType::OBJECT_FLOOR:
							mspFloorMPs.insert(pMP);
							break;
						case UVR_SLAM::ObjectType::OBJECT_WALL:
							mspWallMPs.insert(pMP);
							break;
						case UVR_SLAM::ObjectType::OBJECT_CEILING:
							mspCeilMPs.insert(pMP);
							break;
						}
					}
					else {
						//키포인트도 계속 저장하기

					}
				}//for

				 //일정 포인트 넘으면 평면 검출하고, 남은 것 초기화 또는 삭제하기.
				 //맵을 생성한 뒤에 새로운 평면 포인트와 오브젝트 포인트를 어떻게 처리할지 생각해야 함.

				if (mspFloorMPs.size() > 100 && !b1) {
					std::cout << "Floor::Estimation::Start" << std::endl;
					b1 = PlaneInitialization(pPlane1, mspFloorMPs, mpTargetFrame->GetFrameID(), RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);
					
					//if (b1) {
					mspFloorMPs.clear();
					//}
					if(b1){
						P1 = pPlane1->matPlaneParam.clone();
						invP1 = invT.t()*P1;
						pPlane1->mnPlaneType = ObjectType::OBJECT_FLOOR;
					}
				}
				if (mspWallMPs.size() > 100 && !b2) {
					std::cout << "Wall::Estimation::Start" << std::endl;
					b2 = PlaneInitialization(pPlane2, mspWallMPs, mpTargetFrame->GetFrameID(), RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);
					
					//if (b2){
					mspWallMPs.clear();
					//}
					if(b2){
						P2 = pPlane2->matPlaneParam.clone();
						invP2 = invT.t()*P2;
						pPlane2->mnPlaneType = ObjectType::OBJECT_WALL;
					}
				}
				if (mspCeilMPs.size() > 100 && !b3) {
					std::cout << "Ceil::Estimation::Start" << std::endl;
					b3 = PlaneInitialization(pPlane3, mspCeilMPs, mpTargetFrame->GetFrameID(), RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);
					
					//if (b2){
					mspCeilMPs.clear();
					//}
					if (b3) {
						pPlane3->mnPlaneType = ObjectType::OBJECT_CEILING;
						P3 = pPlane3->matPlaneParam.clone();
						invP3 = invT.t()*P3;
					}
				}
			}


			if (mnProcessType >= 3) {
				
				bool bVal1 = false; //벽과 바닥을 나누는지 확인
				bool bVal2 = false; //벽과 천장을 나누는지 확인

				if (b1 && b2){
					bVal1 = true;
					//if (Ni1.rows == 0)
						//Ni1 = FlukerLineProjection(P1, P2, R, t, m1);
				}
				if (b2 && b3){
					bVal2 = true;
					//if (Ni2.rows == 0)
						//Ni2 = FlukerLineProjection(P3, P2, R, t, m2);
				}
				for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
					if (!mpTargetFrame->mvbMPInliers[i]) {
						cv::Point2f pt = mpTargetFrame->mvKeyPoints[i].pt;
						cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
						auto oType = mpTargetFrame->GetObjectType(i);
						float val1, val2;
						
						if (b1 && oType == ObjectType::OBJECT_FLOOR) {
							//bVal1 체크
							if (bVal1) {
								float val = pt.x*Ni1.at<float>(0) + pt.y*Ni1.at<float>(1) + Ni1.at<float>(2);
								if (m1 < 0)
									val *= -1;
								if (val < 0)
									continue;
							}
							
							temp = mK.inv()*temp;
							cv::Mat matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
							float depth = matDepth.at<float>(0);
							temp *= depth;
							temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
							cv::Mat estimated = invT*temp;

							UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame,estimated.rowRange(0, 3), mpTargetFrame->matDescriptor.row(i), MapPointType::PLANE_MP);
							pNewMP->SetPlaneID(pPlane1->mnPlaneID);
							pNewMP->SetObjectType(pPlane1->mnPlaneType);
							pNewMP->AddFrame(mpTargetFrame, i);
							cv::circle(vis, pt, 3, cv::Scalar(255, 0, 255), -1);

							cv::Point2f p2D;
							cv::Mat pCam;
							pNewMP->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);
							cv::circle(vis, p2D, 2, cv::Scalar(0, 0, 0), -1);

						}
						else if (b2 && oType == ObjectType::OBJECT_WALL) {
							//bVal1, bVal2 체크
							if (!bVal1 && !bVal2)
								continue;

							if (bVal1) {
								float val = pt.x*Ni1.at<float>(0) + pt.y*Ni1.at<float>(1) + Ni1.at<float>(2);
								if (m1 < 0)
									val *= -1;
								if (val > 0)
									continue;
							}
							/*if (bVal2) {
								float val = pt.x*Ni2.at<float>(0) + pt.y*Ni2.at<float>(1) + Ni2.at<float>(2);
								if (val < 0)
									continue;
							}*/
							temp = mK.inv()*temp;
							cv::Mat matDepth = -invP2.at<float>(3) / (invP2.rowRange(0, 3).t()*temp);
							float depth = matDepth.at<float>(0);
							if (depth < 0.0)
								continue;
							temp *= depth;
							temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
							cv::Mat estimated = invT*temp;

							UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), mpTargetFrame->matDescriptor.row(i), MapPointType::PLANE_MP);
							pNewMP->AddFrame(mpTargetFrame, i);
							pNewMP->SetPlaneID(pPlane2->mnPlaneID);
							pNewMP->SetObjectType(pPlane2->mnPlaneType);
							cv::circle(vis, pt, 2, cv::Scalar(255, 255, 0), -1);
						}
						//else if (b3 && oType == ObjectType::OBJECT_CEILING) {
						//   //bVal2 체크
						//	if (bVal2) {
						//		float val = pt.x*Ni2.at<float>(0) + pt.y*Ni2.at<float>(1) + Ni2.at<float>(2);
						//		if (val > 0)
						//			continue;
						//	}
						//	temp = mK.inv()*temp;
						//	cv::Mat matDepth = -invP3.at<float>(3) / (invP3.rowRange(0, 3).t()*temp);
						//	float depth = matDepth.at<float>(0);
						//	temp *= depth;
						//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
						//	cv::Mat estimated = invT*temp;

						//	UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(estimated.rowRange(0, 3), mpTargetFrame->matDescriptor.row(i), MapPointType::PLANE_MP);
						//	pNewMP->AddFrame(mpTargetFrame, i);
						//	pNewMP->SetPlaneID(pPlane3->mnPlaneID);
						//	pNewMP->SetObjectType(pPlane3->mnPlaneType);
						//	mpFrameWindow->AddMapPoint(pNewMP);
						//	cv::circle(vis, pt, 2, cv::Scalar(0, 255, 255), -1);
						//}
					}//if inlier
				}//for
				
				//layout frame으로 설정
				mpLayoutFrame = mpTargetFrame;

				////이전 키프레임과 매칭 정보를 확인하기
				//cv::Mat vis2 = vis.clone();
				//cv::Mat R2, T2;
				//UVR_SLAM::Frame* pPrevKF = mpFrameWindow->GetFrame(mpFrameWindow->GetFrameCount() - 1);
				//pPrevKF->GetPose(R2, T2);
				//for (int i = 0; i < mpFrameWindow->mvMatchInfos.size(); i++) {
				//	int qidx = mpFrameWindow->mvMatchInfos[i].queryIdx;
				//	int tidx = mpFrameWindow->mvMatchInfos[i].trainIdx;
				//}
			}
			
			std::stringstream ss;
			ss << "Frame ID = " << mpTargetFrame->GetFrameID() << ", Type = "<<mnProcessType<<" || F=" << b1 << ", W=" << b2 << ", C=" << b3;
			ss << " || " << mspFloorMPs.size() << ", " << mspWallMPs.size() << ", " << mspCeilMPs.size();
			cv::rectangle(vis, cv::Point2f(0, 0), cv::Point2f(vis.cols, 30), cv::Scalar::all(0), -1);
			cv::putText(vis, ss.str(), cv::Point2f(0, 20), 2, 0.5, cv::Scalar::all(255));
			cv::imshow("Output::PlaneEstimation", vis);
			if (mnProcessType == 3) {
				cv::imshow("Output::PlaneEstimation222", vis);
			}
			cv::imwrite("../../bin/segmentation/res/plane.jpg", vis);
			cv::waitKey(10);
			SetBoolDoingProcess(false, 1);
		}
	}
}

//////////////////////////////////////////////////////////////////////
//평면 추정 관련 함수들
bool UVR_SLAM::PlaneEstimator::PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, std::set<UVR_SLAM::MapPoint*> spMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
	//RANSAC
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;

	cv::Mat param, paramStatus;

	//초기 매트릭스 생성
	cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
	for(auto iter = spMPs.begin(); iter != spMPs.end(); iter++) {
		UVR_SLAM::MapPoint* pMP = *iter;
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		cv::Mat temp = pMP->GetWorldPos();
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		mMat.push_back(temp.t());
	}

	std::random_device rn;
	std::mt19937_64 rnd(rn());
	std::uniform_int_distribution<int> range(0, mMat.rows-1);

	for (int n = 0; n < ransac_trial; n++) {

		cv::Mat arandomPts = cv::Mat::zeros(0, 4, CV_32FC1);
		//select pts
		for (int k = 0; k < 3; k++) {
			int randomNum = range(rnd);
			cv::Mat temp = mMat.row(randomNum).clone();
			arandomPts.push_back(temp);
		}//select

		 //SVD
		cv::Mat X;
		cv::Mat w, u, vt;
		cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
		X = vt.row(3).clone();
		cv::transpose(X, X);
		calcUnitNormalVector(X);
		//reversePlaneSign(X);

		/*cv::Mat X2 = vt.col(3).clone();
		calcUnitNormalVector(X2);
		reversePlaneSign(X2);
		std::cout << sum(abs(mMatFromMap*X)) << " " << sum(abs(mMatFromMap*X2)) << std::endl;*/

		//cv::Mat checkResidual = abs(mMatCurrMap*X);
		//threshold(checkResidual, checkResidual, thresh_plane_distance, 1.0, cv::THRESH_BINARY_INV);
		cv::Mat checkResidual = abs(mMat*X) < thresh_distance;
		checkResidual = checkResidual / 255;
		int temp_inlier = cv::countNonZero(checkResidual);

		if (max_num_inlier < temp_inlier) {
			max_num_inlier = temp_inlier;
			param = X.clone();
			paramStatus = checkResidual.clone();
		}
	}//trial

	float planeRatio = ((float)max_num_inlier / mMat.rows);

	if (planeRatio > thresh_ratio) {
		cv::Mat pParam = param.clone();
		pPlane->matPlaneParam = pParam.clone();
		pPlane->mnPlaneID = ++nPlaneID;

		pPlane->normal = pParam.rowRange(0, 3);
		pPlane->distance = pParam.at<float>(3);
		pPlane->norm = sqrt(pPlane->normal.dot(pPlane->normal));

		int idx = 0;
		for (auto iter = spMPs.begin(); iter != spMPs.end(); iter++) {
			int nidx = idx++;
			int checkIdx = paramStatus.at<uchar>(nidx);
			//std::cout << checkIdx << std::endl;
			UVR_SLAM::MapPoint* pMP = *iter;
			if (checkIdx == 0)
				continue;
			if (pMP) {
				//평면에 대한 레이블링이 필요함.
				if (pMP->isDeleted())
					continue;

				pMP->SetRecentLayoutFrameID(nTargetID);
				pMP->SetPlaneID(pPlane->mnPlaneID);
				pPlane->mvpMPs.push_back(pMP);
			}
		}
		//평면 정보 생성.
		return true;
	}
	else
	{
		//std::cout << "failed" << std::endl;
		return false;
	}
}

//GroundPlane은 현재 평면, type == 1이면 벽, 아니면 천장
bool UVR_SLAM::PlaneEstimator::PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, UVR_SLAM::PlaneInformation* GroundPlane, int type, std::set<UVR_SLAM::MapPoint*> spMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
	//RANSAC
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;

	cv::Mat param, paramStatus;

	//초기 매트릭스 생성
	cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
	for (auto iter = spMPs.begin(); iter != spMPs.end(); iter++) {
		UVR_SLAM::MapPoint* pMP = *iter;
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		cv::Mat temp = pMP->GetWorldPos();
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		mMat.push_back(temp.t());
	}
	if (mMat.rows == 0)
		return false;

	std::random_device rn;
	std::mt19937_64 rnd(rn());
	std::uniform_int_distribution<int> range(0, mMat.rows - 1);

	for (int n = 0; n < ransac_trial; n++) {

		cv::Mat arandomPts = cv::Mat::zeros(0, 4, CV_32FC1);
		//select pts
		for (int k = 0; k < 3; k++) {
			int randomNum = range(rnd);
			cv::Mat temp = mMat.row(randomNum).clone();
			arandomPts.push_back(temp);
		}//select

		 //SVD
		cv::Mat X;
		cv::Mat w, u, vt;
		cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
		X = vt.row(3).clone();
		cv::transpose(X, X);
		calcUnitNormalVector(X);

		float val = GroundPlane->CalcCosineSimilarity(X);
		//std::cout << "cos::" << val << std::endl;
		if (type == 1) {
			//바닥과 벽	
			if (abs(val) > mfThreshNormal)
				continue;
		}
		else {
			//바닥과 천장
			if (1.0 - abs(val) > mfThreshNormal)
				continue;
		}

		cv::Mat checkResidual = abs(mMat*X) < thresh_distance;
		checkResidual = checkResidual / 255;
		int temp_inlier = cv::countNonZero(checkResidual);

		if (max_num_inlier < temp_inlier) {
			max_num_inlier = temp_inlier;
			param = X.clone();
			paramStatus = checkResidual.clone();
		}
	}//trial

	if (max_num_inlier == 0)
		return false;

	float planeRatio = ((float)max_num_inlier / mMat.rows);
	//std::cout << "PLANE INIT : " << max_num_inlier << ", " << paramStatus.rows << "::" << cv::countNonZero(paramStatus) << " " << spMPs.size() << "::" << planeRatio << std::endl;

	//cv::Mat checkResidual2 = mMat*param > 2 * thresh_distance; checkResidual2 /= 255; checkResidual2 *= 2;
	//paramStatus += checkResidual2;

	if (planeRatio > thresh_ratio) {
		cv::Mat pParam = param.clone();
		pPlane->matPlaneParam = pParam.clone();
		pPlane->mnPlaneID = ++nPlaneID;

		pPlane->normal = pParam.rowRange(0, 3);
		pPlane->distance = pParam.at<float>(3);
		pPlane->norm = sqrt(pPlane->normal.dot(pPlane->normal));

		int idx = 0;
		for (auto iter = spMPs.begin(); iter != spMPs.end(); iter++) {
			int nidx = idx++;
			int checkIdx = paramStatus.at<uchar>(nidx);
			//std::cout << checkIdx << std::endl;
			UVR_SLAM::MapPoint* pMP = *iter;
			if (checkIdx == 0)
				continue;
			if (pMP) {
				//평면에 대한 레이블링이 필요함.
				if (pMP->isDeleted())
					continue;
				pMP->SetRecentLayoutFrameID(nTargetID);
				pMP->SetPlaneID(pPlane->mnPlaneID);
				pPlane->mvpMPs.push_back(pMP);
			}
		}
		//평면 정보 생성.
		return true;
	}
	else
	{
		//std::cout << "failed" << std::endl;
		return false;
	}
}

bool UVR_SLAM::PlaneEstimator::calcUnitNormalVector(cv::Mat& X) {
	float sum = sqrt(X.at<float>(0, 0)*X.at<float>(0, 0) + X.at<float>(1, 0)*X.at<float>(1, 0) + X.at<float>(2, 0)*X.at<float>(2, 0));
	//cout<<"befor X : "<<X<<endl;
	if (sum != 0) {
		X.at<float>(0, 0) = X.at<float>(0, 0) / sum;
		X.at<float>(1, 0) = X.at<float>(1, 0) / sum;
		X.at<float>(2, 0) = X.at<float>(2, 0) / sum;
		X.at<float>(3, 0) = X.at<float>(3, 0) / sum;
		//cout<<"after X : "<<X<<endl;
		return true;
	}
	return false;
}

void UVR_SLAM::PlaneEstimator::reversePlaneSign(cv::Mat& param) {
	if (param.at<float>(3, 0) < 0.0) {
		param *= -1.0;
	}
}
//평면 추정 관련 함수들
//////////////////////////////////////////////////////////////////////

//플루커 라인 프로젝션 관련 함수
cv::Mat UVR_SLAM::PlaneInformation::FlukerLineProjection(PlaneInformation* P, cv::Mat R, cv::Mat t, cv::Mat K, float& m) {
	cv::Mat PLw1, Lw1, NLw1;
	PLw1 = this->matPlaneParam*P->matPlaneParam.t() - P->matPlaneParam*this->matPlaneParam.t();
	Lw1 = cv::Mat::zeros(6, 1, CV_32FC1);
	Lw1.at<float>(3) = PLw1.at<float>(2, 1);
	Lw1.at<float>(4) = PLw1.at<float>(0, 2);
	Lw1.at<float>(5) = PLw1.at<float>(1, 0);
	NLw1 = PLw1.col(3).rowRange(0, 3);
	NLw1.copyTo(Lw1.rowRange(0, 3));

	//Line projection test : Ni
	cv::Mat T2 = cv::Mat::zeros(6, 6, CV_32FC1);
	R.copyTo(T2.rowRange(0, 3).colRange(0, 3));
	R.copyTo(T2.rowRange(3, 6).colRange(3, 6));
	cv::Mat tempSkew = cv::Mat::zeros(3, 3, CV_32FC1);
	tempSkew.at<float>(0, 1) = -t.at<float>(2);
	tempSkew.at<float>(1, 0) = t.at<float>(2);
	tempSkew.at<float>(0, 2) = t.at<float>(1);
	tempSkew.at<float>(2, 0) = -t.at<float>(1);
	tempSkew.at<float>(1, 2) = -t.at<float>(0);
	tempSkew.at<float>(2, 1) = t.at<float>(0);
	tempSkew *= R;
	tempSkew.copyTo(T2.rowRange(0, 3).colRange(3, 6));
	cv::Mat Lc = T2*Lw1;
	cv::Mat Nc = Lc.rowRange(0, 3);
	cv::Mat res = K*Nc;
	if (res.at<float>(0) < 0)
		res *= -1;
	if (res.at<float>(0) != 0)
		m = res.at<float>(1) / res.at<float>(0);
	else
		m = 9999.0;
	return res.clone();
}

cv::Point2f UVR_SLAM::PlaneInformation::CalcLinePoint(float y, cv::Mat mLine) {
	float x = 0.0;
	if (mLine.at<float>(0) != 0)
		x = (-mLine.at<float>(2) - mLine.at<float>(1)*y) / mLine.at<float>(0);
	return cv::Point2f(x, y);
}

void UVR_SLAM::PlaneInformation::CalcFlukerLinePoints(cv::Point2f& sPt, cv::Point2f& ePt, float f1, float f2, cv::Mat mLine) {
	sPt = CalcLinePoint(f1, mLine);
	ePt = CalcLinePoint(f2, mLine);
}

///////////////
////Pluker Lines
//cv::Mat P1 = (cv::Mat_<float>(4, 1) << 0, 1, 0, 0);
//cv::Mat P2 = (cv::Mat_<float>(4, 1) << 1, 0, 0, -0.36);

//cv::Mat PLw = P1*P2.t() - P2*P1.t();
//cv::Mat Lw = cv::Mat::zeros(6, 1, CV_32FC1);
//Lw.at<float>(3) = PLw.at<float>(2, 1);
//Lw.at<float>(4) = PLw.at<float>(0, 2);
//Lw.at<float>(5) = PLw.at<float>(1, 0);
//cv::Mat NLw = PLw.col(3).rowRange(0, 3);
//NLw.copyTo(Lw.rowRange(0, 3));
//std::cout << PLw << Lw << std::endl;
////Line projection test : Ni
//cv::Mat T2 = cv::Mat::zeros(6, 6, CV_32FC1);
//R.copyTo(T2.rowRange(0, 3).colRange(0, 3));
//R.copyTo(T2.rowRange(3, 6).colRange(3, 6));
//cv::Mat tempSkew = cv::Mat::zeros(3, 3, CV_32FC1);
//tempSkew.at<float>(0, 1) = -t.at<float>(2);
//tempSkew.at<float>(1, 0) = t.at<float>(2);
//tempSkew.at<float>(0, 2) = t.at<float>(1);
//tempSkew.at<float>(2, 0) = -t.at<float>(1);
//tempSkew.at<float>(1, 2) = -t.at<float>(0);
//tempSkew.at<float>(2, 1) = t.at<float>(0);
//tempSkew *= R;
//tempSkew.copyTo(T2.rowRange(0, 3).colRange(3, 6));
//cv::Mat Lc = T2*Lw;
//cv::Mat Nc = Lc.rowRange(0, 3);
//cv::Mat Ni = mK2*Nc;
//std::cout << Ni << std::endl;

//float x1 = 0;
//float y1 = 0;
//if (Ni.at<float>(0) != 0)
//	x1 = -Ni.at<float>(2) / Ni.at<float>(0);

//float x2 = 0;
//float y2 = 480;
//if (Ni.at<float>(0) != 0)
//	x2 = (-Ni.at<float>(2) - Ni.at<float>(1)*y2) / Ni.at<float>(0);
//cv::line(vis, cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Scalar(255, 0, 0), 2);
////Pluker Lines
///////////////

//keyframe(p)에서 현재 local map(this)와 머지
void UVR_SLAM::PlaneInformation::Merge(PlaneInformation* p, int nID, float thresh) {
	//p에 속하는 MP 중 현재 평면에 속하는 것들 추가
	//map point vector 복사
	//update param

	int n1 = p->mvpMPs.size();
	int n2 = mvpMPs.size();

	for (int i = 0; i < mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		/*if (pMP->GetRecentLocalMapID() < nID) {
			continue;
		}*/
		if (pMP->GetPlaneID() == p->mnPlaneID) {
			continue;
		}
		//distance 계산
		cv::Mat X3D = pMP->GetWorldPos();
		cv::Mat normal = p->matPlaneParam.rowRange(0, 3);
		float dist = p->matPlaneParam.at<float>(3);
		float res = abs(normal.dot(X3D) + dist);

		if (res < thresh)
			p->mvpMPs.push_back(pMP);
	}
	mvpMPs = std::vector<MapPoint*>(p->mvpMPs.begin(), p->mvpMPs.end());

	std::cout << "Merge::" << n1 << ", " << n2 << "::" << mvpMPs.size() << std::endl;
}

//this : keyframe
//p : localmap
float UVR_SLAM::PlaneInformation::CalcOverlapMPs(PlaneInformation* p, int nID) {
	std::map<UVR_SLAM::MapPoint*, int> mmpMPs;
	int nCount = 0;
	int nTotal = 0;

	for (int i = 0; i < p->mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = p->mvpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		if (pMP->GetRecentLocalMapID()>=nID) {
			nTotal++;
		}
		if (pMP->GetPlaneID() == mnPlaneID) {
			nCount++;
		}
	}
	std::cout << "Association::Overlap::" << nCount << ", " << nTotal <<"::"<<p->mvpMPs.size()<<", "<<mvpMPs.size()<< std::endl;
	return ((float)nCount) / nTotal;
}

bool CheckZero(float val) {
	if (abs(val) < 1e-6) {
		return true;
	}
	return false;
}

float UVR_SLAM::PlaneInformation::CalcCosineSimilarity(PlaneInformation* p) {
	
	float d1 = this->norm;
	float d2 = p->norm;
	if (CheckZero(d1) || CheckZero(d2))
		return 0.0;
	return normal.dot(p->normal) / (d1*d2);
}

float UVR_SLAM::PlaneInformation::CalcCosineSimilarity(cv::Mat P) {

	float d1 = this->norm;
	cv::Mat tempNormal = P.rowRange(0, 3);

	float d2 = sqrt(tempNormal.dot(tempNormal));
	
	if (CheckZero(d1) || CheckZero(d2))
		return 0.0;
	return normal.dot(tempNormal) / (d1*d2);
}


float UVR_SLAM::PlaneInformation::CalcPlaneDistance(PlaneInformation* p) {
	return abs(distance - p->distance);
}

float UVR_SLAM::PlaneInformation::CalcPlaneDistance(cv::Mat X) {
	return X.dot(this->normal) + distance;
}

void UVR_SLAM::PlaneEstimator::CreatePlanarMapPoints(std::vector<UVR_SLAM::MapPoint*> mvpMPs, std::vector<UVR_SLAM::ObjectType> mvpOPs, UVR_SLAM::PlaneInformation* pPlane, cv::Mat invT) {
	cv::Mat invP1 = invT.t()*pPlane->matPlaneParam.clone();

	float minDepth = FLT_MAX;
	float maxDepth = 0.0f;
	//create new mp in current frame
	for (int j = 0; j < mvpMPs.size(); j++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[j];
		auto oType = mvpOPs[j];

		if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR)
			continue;
		cv::Point2f pt = mpTargetFrame->mvKeyPoints[j].pt;
		cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
		temp = mK.inv()*temp;
		cv::Mat matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
		float depth = matDepth.at<float>(0);
		temp *= depth;
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

		if (maxDepth < depth)
			maxDepth = depth;
		if (minDepth > depth)
			minDepth = depth;

		cv::Mat estimated = invT*temp;
		if (pMP) {
			pMP->SetWorldPos(estimated.rowRange(0, 3));
		}
		else {
			UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), mpTargetFrame->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
			pNewMP->SetPlaneID(pPlane->mnPlaneID);
			pNewMP->SetObjectType(pPlane->mnPlaneType);
			pNewMP->AddFrame(mpTargetFrame, j);
			pNewMP->UpdateNormalAndDepth();
			pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
			mpSystem->mlpNewMPs.push_back(pNewMP);
		}
	}

	cv::Mat R, t;
	mpTargetFrame->GetPose(R, t);

	for (int j = 0; j < mvpMPs.size(); j++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[j];
		auto oType = mvpOPs[j];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR) {
			cv::Mat X3D = pMP->GetWorldPos();
			cv::Mat Xcam = R*X3D + t;
			float depth = Xcam.at<float>(2);
			if (depth < 0.0 || depth > maxDepth)
				pMP->SetDelete(true);
		}
	}

	mpTargetFrame->SetDepthRange(minDepth, maxDepth);
}