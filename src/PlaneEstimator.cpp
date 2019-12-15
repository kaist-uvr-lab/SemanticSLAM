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
UVR_SLAM::PlaneEstimator::PlaneEstimator(cv::Mat K, cv::Mat K2, int w, int h) : mK(K), mK2(K2),mbDoingProcess(false), mnWidth(w), mnHeight(h), mnProcessType(0), mpLayoutFrame(nullptr){
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
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
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

///////////////////////////////////////////////////////////////////////////////

void UVR_SLAM::PlaneEstimator::Run() {

	//나중에 클래스에 옮길 것.
	//이제 이것들도 삭제 될 때 처리가 필요함.
	std::set<UVR_SLAM::MapPoint*> mspFloorMPs, mspWallMPs, mspCeilMPs;

	int RANSAC_TRIAL = 2500;
	float thresh_plane_distance = 0.01;
	float thresh_ratio = 0.1;

	bool bInitTest = false;
	bool bFloor = false;
	bool bWall = false;
	bool bCeil = false;
	UVR_SLAM::PlaneInformation* pPlane1, *pPlane2, *pPlane3;
	std::vector<PlaneInformation*> mvpWalls;
	pPlane1 = new UVR_SLAM::PlaneInformation();
	pPlane2 = new UVR_SLAM::PlaneInformation();
	pPlane3 = new UVR_SLAM::PlaneInformation();
	mvpWalls.push_back(pPlane2);

	int nWalls = 0;
	//벽은 주변의 바닥, 천장, 벽 정보를 담고 표현하는 것이 필요함.

	while (1) {
		//프로세스 타입 1 : key frame 에서 맵포인트에 대한 처리하고 평면을 추정하는 과정
		//프로세스 타입 2 : frame에서 평면 레이아웃 시각화하고 새로운 평면 맵포인트를 추가하는 과정.
		if (isDoingProcess()) {
			continue;
			std::cout << "LayoutEstimation::Start" << std::endl;
			//여기에 들어오면 일단 전부 레이아웃은 보여줄 준비를 해야 함.
			//평면이 추정되었는지 확인해야 함.
			//레이아웃 추정도 필요함.
			cv::Mat vis = mpTargetFrame->GetOriginalImage();
			cv::Mat R, t, P1, P2, P3;
			cv::Mat invP1, invP2, invP3;
			cv::Mat PLw1, Lw1, NLw1, Ni1; //바닥과 벽. 여러개 가능
			cv::Mat PLw2, Lw2, NLw2, Ni2; //천장과 벽. 여러개 가능
			float m1, m2;
			Ni1 = cv::Mat::zeros(0, 0, CV_32FC1);
			Ni2 = cv::Mat::zeros(0, 0, CV_32FC1);

			mpTargetFrame->GetPose(R, t);
			cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
			R.copyTo(T.rowRange(0, 3).colRange(0, 3));
			t.copyTo(T.col(3).rowRange(0, 3));
			cv::Mat invT = T.inv();

			if(bFloor){
				P1 = pPlane1->matPlaneParam.clone();
				invP1 = invT.t()*P1;
			}
			if(bWall){
				P2 = pPlane2->matPlaneParam.clone();
				invP2 = invT.t()*P2;
			}
			if(bCeil){
				P3 = pPlane3->matPlaneParam.clone();
				invP3 = invT.t()*P3;
			}

			if (mnProcessType >= 1) {
				if (bFloor && bWall) {
					Ni1 = FlukerLineProjection(P1, P2, R, t, m1);
					
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
				if (bCeil && bWall) {
					Ni2 = FlukerLineProjection(P2, P3, R, t, m2);

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

					if (mpTargetFrame->GetBoolInlier(i)) {
						UVR_SLAM::MapPoint* pMP = mpTargetFrame->GetMapPoint(i);
						if (!pMP)
							continue;
						if (pMP->isDeleted())
							continue;
						if (pMP->GetPlaneID() > 0)
							continue;
						//type check
						auto type = pMP->GetObjectType();

						switch (type) {
						case UVR_SLAM::ObjectType::OBJECT_FLOOR:
							if (bFloor) {
								double dist = calcPlaneDistance(P1, pMP->GetWorldPos());
								if(dist > thresh_plane_distance*4)
									mspFloorMPs.insert(pMP);
							}else
								mspFloorMPs.insert(pMP);
							break;
						case UVR_SLAM::ObjectType::OBJECT_WALL:
							if (bWall) {
								double dist = calcPlaneDistance(P2, pMP->GetWorldPos());
								if (dist > thresh_plane_distance * 4)
									mspWallMPs.insert(pMP);
							}
							else
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

				if (mspFloorMPs.size() > 100 && !bFloor) {
					std::cout << "Floor::Estimation::Start" << std::endl;
					bFloor = PlaneInitialization(pPlane1, mspFloorMPs, RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);
					
					//if (b1) {
					mspFloorMPs.clear();
					//}
					if(bFloor){
						P1 = pPlane1->matPlaneParam.clone();
						invP1 = invT.t()*P1;
						pPlane1->mnPlaneType = ObjectType::OBJECT_FLOOR;
					}
				}
				if (mspWallMPs.size() > 100) {
					std::cout << "Wall::Estimation::Start" << std::endl;
					//mvpWalls[nWalls]->bInit = PlaneInitialization(mvpWalls[nWalls], P1, 1,mspWallMPs, RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);
					mvpWalls[nWalls]->bInit = PlaneInitialization(mvpWalls[nWalls], mspWallMPs, RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);

					if (!bWall && mvpWalls[nWalls]->bInit) {
						bWall = true;
					}

					if(mvpWalls[nWalls]->bInit){
						nWalls++;
						mvpWalls.push_back(new PlaneInformation());
						mspWallMPs.clear();

						P2 = pPlane2->matPlaneParam.clone();
						invP2 = invT.t()*P2;
						pPlane2->mnPlaneType = ObjectType::OBJECT_WALL;
					}
				}
				if (mspCeilMPs.size() > 100 && !bCeil && bWall) {
					std::cout << "Ceil::Estimation::Start" << std::endl;
					bCeil = PlaneInitialization(pPlane3, P2, 1, mspCeilMPs, RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);
					
					//if (b2){
					
					//}
					if (bCeil) {
						mspCeilMPs.clear();
						pPlane3->mnPlaneType = ObjectType::OBJECT_CEILING;
						P3 = pPlane3->matPlaneParam.clone();
						invP3 = invT.t()*P3;
					}
				}
			}


			if (!bInitTest && mnProcessType == 3) {
				std::cout << "ground plane init test!!!" << std::endl;

				bFloor = PlaneInitialization(pPlane1, mspFloorMPs, RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);
				if (bFloor) {
					P1 = pPlane1->matPlaneParam.clone();
					invP1 = invT.t()*P1;
					pPlane1->mnPlaneType = ObjectType::OBJECT_FLOOR;
					//UVR_SLAM::Optimization::GroundPlaneInitialize(mpInitializer->mpInitData, mpInitializer->mvMatchInfos, mpInitializer->mpInitFrame1, mpInitializer->mpInitFrame2, pPlane1->matPlaneParam, mK);
				}
				mvpWalls[nWalls]->bInit = PlaneInitialization(mvpWalls[nWalls], mspWallMPs, RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);
				if (mvpWalls[nWalls]) {
					P2 = mvpWalls[nWalls]->matPlaneParam.clone();
					invP2 = invT.t()*P2;
					mvpWalls[nWalls]->mnPlaneType = ObjectType::OBJECT_WALL;
					nWalls++;
					mvpWalls.push_back(new PlaneInformation());
					bWall = true;
				}
				bInitTest = true;
				//이걸 먼저 수행해야 하면, 성공 후 플래그 해제하고, 리턴해야 함.
			}

			if (mnProcessType >= 3) {
				
				bool bVal1 = false; //벽과 바닥을 나누는지 확인
				bool bVal2 = false; //벽과 천장을 나누는지 확인

				if (bFloor && bWall){

					float val = calcCosineSimilarity(P1, P2);
					std::cout << "cos sim1 = " << val << std::endl << std::endl << std::endl;

					bVal1 = true;
					if (Ni1.rows == 0)
						Ni1 = FlukerLineProjection(P1, P2, R, t, m1);
				}
				if (bWall && bCeil){

					float val = calcCosineSimilarity(P3, P2);
					std::cout << "cos sim2 = " << val << std::endl << std::endl << std::endl;

					bVal2 = true;
					if (Ni2.rows == 0)
						Ni2 = FlukerLineProjection(P3, P2, R, t, m2);
				}
				for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
					if (!mpTargetFrame->GetBoolInlier(i)) {
						cv::Point2f pt = mpTargetFrame->mvKeyPoints[i].pt;
						cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
						auto oType = mpTargetFrame->GetObjectType(i);
						float val1, val2;
						
						if (bFloor && oType == ObjectType::OBJECT_FLOOR) {
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

							UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(estimated.rowRange(0, 3), mpTargetFrame, i, mpTargetFrame->matDescriptor.row(i), MapPointType::PLANE_MP);
							pNewMP->SetPlaneID(pPlane1->mnPlaneID);
							pNewMP->SetObjectType(pPlane1->mnPlaneType);
							pNewMP->AddFrame(mpTargetFrame, i);
							mpFrameWindow->AddMapPoint(pNewMP);
							cv::circle(vis, pt, 3, cv::Scalar(255, 0, 255), -1);

							cv::Point2f p2D;
							cv::Mat pCam;
							pNewMP->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);
							cv::circle(vis, p2D, 2, cv::Scalar(0, 0, 0), -1);

						}
						//else if (bWall && oType == ObjectType::OBJECT_WALL) {
						//	//bVal1, bVal2 체크
						//	if (!bVal1 && !bVal2)
						//		continue;

						//	if (bVal1) {
						//		float val = pt.x*Ni1.at<float>(0) + pt.y*Ni1.at<float>(1) + Ni1.at<float>(2);
						//		if (m1 < 0)
						//			val *= -1;
						//		if (val > 0)
						//			continue;
						//	}
						//	if (bVal2) {
						//		float val = pt.x*Ni2.at<float>(0) + pt.y*Ni2.at<float>(1) + Ni2.at<float>(2);
						//		if(m2 > 0)
						//			val *= -1;
						//		if (val > 0)
						//			continue;
						//	}
						//	temp = mK.inv()*temp;
						//	cv::Mat matDepth = -invP2.at<float>(3) / (invP2.rowRange(0, 3).t()*temp);
						//	float depth = matDepth.at<float>(0);
						//	temp *= depth;
						//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
						//	cv::Mat estimated = invT*temp;

						//	UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(estimated.rowRange(0, 3), mpTargetFrame, i, mpTargetFrame->matDescriptor.row(i), MapPointType::PLANE_MP);
						//	pNewMP->AddFrame(mpTargetFrame, i);
						//	pNewMP->SetPlaneID(pPlane2->mnPlaneID);
						//	pNewMP->SetObjectType(pPlane2->mnPlaneType);
						//	mpFrameWindow->AddMapPoint(pNewMP);
						//	cv::circle(vis, pt, 2, cv::Scalar(255, 255, 0), -1);
						//}
						//else if (bCeil && oType == ObjectType::OBJECT_CEILING) {
						//   //bVal2 체크
						//	if (bVal2) {
						//		float val = pt.x*Ni2.at<float>(0) + pt.y*Ni2.at<float>(1) + Ni2.at<float>(2);
						//		if(m2 < 0)
						//			val *= -1;
						//		if (val > 0)
						//			continue;
						//	}
						//	temp = mK.inv()*temp;
						//	cv::Mat matDepth = -invP3.at<float>(3) / (invP3.rowRange(0, 3).t()*temp);
						//	float depth = matDepth.at<float>(0);
						//	temp *= depth;
						//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
						//	cv::Mat estimated = invT*temp;

						//	UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(estimated.rowRange(0, 3), mpTargetFrame, i, mpTargetFrame->matDescriptor.row(i), MapPointType::PLANE_MP);
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

				//이전 키프레임과 매칭 정보를 확인하기
				/*cv::Mat vis2 = vis.clone();
				cv::Mat R2, T2;
				UVR_SLAM::Frame* pPrevKF = mpFrameWindow->GetFrame(mpFrameWindow->GetFrameCount() - 1);
				pPrevKF->GetPose(R2, T2);
				for (int i = 0; i < mpFrameWindow->mvMatchInfos.size(); i++) {
					int qidx = mpFrameWindow->mvMatchInfos[i].queryIdx;
					int tidx = mpFrameWindow->mvMatchInfos[i].trainIdx;

				}*/
			}
			
			std::stringstream ss;
			ss << "Frame ID = " << mpTargetFrame->GetFrameID() << ", Type = "<<mnProcessType<<" || F=" << bFloor << ", W=" << nWalls << ", C=" << bCeil;
			ss << " || " << mspFloorMPs.size() << ", " << mspWallMPs.size() << ", " << mspCeilMPs.size();
			cv::rectangle(vis, cv::Point2f(0, 0), cv::Point2f(vis.cols, 30), cv::Scalar::all(0), -1);
			cv::putText(vis, ss.str(), cv::Point2f(0, 20), 2, 0.5, cv::Scalar::all(255));
			cv::imshow("Output::PlaneEstimation", vis);
			if (mnProcessType == 3) {
				cv::imshow("Output::PlaneEstimation222", vis);
			}
			cv::imwrite("../../bin/segmentation/res/plane.jpg", vis);
			cv::waitKey(10);
			std::cout << "LayoutEstimation::End" << std::endl;
			SetBoolDoingProcess(false, 1);
		}
	}
}

//////////////////////////////////////////////////////////////////////
//평면 추정 관련 함수들
bool UVR_SLAM::PlaneEstimator::PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, std::set<UVR_SLAM::MapPoint*> spMPs, int ransac_trial, float thresh_distance, float thresh_ratio) {
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
	std::cout << "PLANE INIT : " << max_num_inlier << ", " << paramStatus.rows << "::" << cv::countNonZero(paramStatus) << " " << spMPs.size() << "::" << planeRatio << std::endl;

	//cv::Mat checkResidual2 = mMat*param > 2 * thresh_distance; checkResidual2 /= 255; checkResidual2 *= 2;
	//paramStatus += checkResidual2;

	if (planeRatio > thresh_ratio) {

		//평면 다시 구하기.
		cv::Mat mMatB = cv::Mat::zeros(0, 4, CV_32FC1);
		for (int i = 0; i < mMat.rows; i++) {
			if (paramStatus.at<uchar>(i) > 0)
				mMatB.push_back(mMat.row(i));
		}
		cv::Mat w, u, vt;
		cv::SVD::compute(mMatB, w, u, vt, cv::SVD::FULL_UV);
		param = vt.row(3).clone();
		cv::transpose(param, param);
		calcUnitNormalVector(param);

		pPlane->matPlaneParam = param.clone();
		pPlane->mnPlaneID = ++nPlaneID;
		std::cout << "PlaneInitParam::" << param << std::endl;
		int idx = 0;

		for (auto iter = spMPs.begin(); iter != spMPs.end(); iter++) {
			int nidx = idx++;
			int checkIdx = paramStatus.at<uchar>(nidx);
			UVR_SLAM::MapPoint* pMP = *iter;
			if (pMP) {
				//평면에 대한 레이블링이 필요함.
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

bool UVR_SLAM::PlaneEstimator::PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, cv::Mat GroundPlane, int type, std::set<UVR_SLAM::MapPoint*> spMPs, int ransac_trial, float thresh_distance, float thresh_ratio) {
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

		float val = calcCosineSimilarity(GroundPlane, X);
		
		if (type == 1) {
			//바닥과 벽	
			if (abs(val) > 0.001)
				continue;
		}
		else {
			//바닥과 천장
			if (1.0 - abs(val) > 0.001)
				continue;
		}

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

	if (max_num_inlier == 0)
		return false;

	float planeRatio = ((float)max_num_inlier / mMat.rows);
	std::cout << "PLANE INIT : " << max_num_inlier << ", " << paramStatus.rows << "::" << cv::countNonZero(paramStatus) << " " << spMPs.size() << "::" << planeRatio << std::endl;

	//cv::Mat checkResidual2 = mMat*param > 2 * thresh_distance; checkResidual2 /= 255; checkResidual2 *= 2;
	//paramStatus += checkResidual2;

	if (planeRatio > thresh_ratio) {
		//평면 다시 구하기.
		/*cv::Mat mMatB = cv::Mat::zeros(0, 4, CV_32FC1);
		for (int i = 0; i < mMat.rows; i++) {
			if (paramStatus.at<uchar>(i) > 0)
				mMatB.push_back(mMat.row(i));
		}
		cv::Mat w, u, vt;
		cv::SVD::compute(mMatB, w, u, vt, cv::SVD::FULL_UV);
		param = vt.row(3).clone();
		cv::transpose(param, param);
		calcUnitNormalVector(param);*/

		pPlane->matPlaneParam = param.clone();
		pPlane->mnPlaneID = ++nPlaneID;
		std::cout << "PlaneInitParam::" << param << std::endl;
		int idx = 0;

		for (auto iter = spMPs.begin(); iter != spMPs.end(); iter++) {
			int nidx = idx++;
			int checkIdx = paramStatus.at<uchar>(nidx);
			UVR_SLAM::MapPoint* pMP = *iter;
			if (pMP) {
				//평면에 대한 레이블링이 필요함.
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
cv::Mat UVR_SLAM::PlaneEstimator::FlukerLineProjection(cv::Mat P1, cv::Mat P2, cv::Mat R, cv::Mat t, float& m) {
	cv::Mat PLw1, Lw1, NLw1;
	PLw1 = P1*P2.t() - P2*P1.t();
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
	cv::Mat res = mK2*Nc;
	if (res.at<float>(0) < 0)
		res *= -1;
	m = res.at<float>(1) / res.at<float>(0);
	return res.clone();
}

bool CheckZero(float val) {
	if (abs(val) < 1e-6) {
		return true;
	}
	return false;
}

float UVR_SLAM::PlaneEstimator::calcCosineSimilarity(cv::Mat p1, cv::Mat p2) {
	cv::Mat n1 = p1.rowRange(0, 3);
	cv::Mat n2 = p2.rowRange(0, 3);

	float d1 = sqrt(n1.dot(n1));
	float d2 = sqrt(n2.dot(n2));

	if (CheckZero(d1) || CheckZero(d2))
		return 0.0;

	return n1.dot(n2)/(d1*d2);
}

double UVR_SLAM::PlaneEstimator::calcPlaneDistance(cv::Mat P, cv::Mat X) {
	X.push_back(cv::Mat::ones(1, 1, CV_32FC1));
	return abs(P.dot(X));
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
