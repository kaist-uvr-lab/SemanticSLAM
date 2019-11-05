#include <PlaneEstimator.h>
#include <random>
#include <System.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <SegmentationData.h>
#include <MapPoint.h>

UVR_SLAM::PlaneEstimator::PlaneEstimator() :mbDoingProcess(false), mnProcessType(0){
}
UVR_SLAM::PlaneEstimator::PlaneEstimator(cv::Mat K, cv::Mat K2, int w, int h) : mK(K), mK2(K2),mbDoingProcess(false), mnWidth(w), mnHeight(h), mnProcessType(0) {
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
	mpTargetFrame = pFrame;
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

	bool b1 = false;
	bool b2 = false;
	UVR_SLAM::PlaneInformation* pPlane1, *pPlane2, *Plane3;
	pPlane1 = new UVR_SLAM::PlaneInformation();
	pPlane2 = new UVR_SLAM::PlaneInformation();

	while (1) {
		//프로세스 타입 1 : key frame 에서 맵포인트에 대한 처리하고 평면을 추정하는 과정
		//프로세스 타입 2 : frame에서 평면 레이아웃 시각화하고 새로운 평면 맵포인트를 추가하는 과정.
		if (isDoingProcess()) {

			if(mnProcessType == 1){
				//현재 프레임에서는 어떻게 전파를 할 것인가?
				//라인을 획득하게 되면 레이블링이 가능해진다.

				//평면이 존재하는지 확인하고, 평면의 타입을 확인하기.

				//set mp로 바닥, 벽, 천장 획득하기
				for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
				
					if (mpTargetFrame->GetBoolInlier(i)){
						UVR_SLAM::MapPoint* pMP = mpTargetFrame->GetMapPoint(i);
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
					}else{
						//키포인트도 계속 저장하기
					
					}
				}

				//일정 포인트 넘으면 평면 검출하고, 남은 것 초기화 또는 삭제하기.
				//맵을 생성한 뒤에 새로운 평면 포인트와 오브젝트 포인트를 어떻게 처리할지 생각해야 함.
			
				if (mspFloorMPs.size() > 50 && !b1) {
					std::cout << "Floor::Estimation::Start" << std::endl;
					b1 = PlaneInitialization(pPlane1, mspFloorMPs, RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);
					//if (b1) {
						mspFloorMPs.clear();
					//}
				}
				if (mspWallMPs.size() > 50 && !b2) {
					std::cout << "Wall::Estimation::Start" << std::endl;
					b2 = PlaneInitialization(pPlane2, mspWallMPs, RANSAC_TRIAL, thresh_plane_distance, thresh_ratio);
					//if (b2){
						mspWallMPs.clear();
					//}
				}
			}
			else if(mnProcessType==2){
				if (b1 && b2) {
					
					//시각화 용도
					std::cout << "PlaneEstimator::Visualize::Line" << std::endl << std::endl << std::endl;

					cv::Mat vis = mpTargetFrame->GetOriginalImage();
					cv::Mat P1 = pPlane1->matPlaneParam.clone();
					cv::Mat P2 = pPlane2->matPlaneParam.clone();
					cv::Mat R, t;
					mpTargetFrame->GetPose(R, t);

					cv::Mat PLw = P1*P2.t() - P2*P1.t();
					cv::Mat Lw = cv::Mat::zeros(6, 1, CV_32FC1);
					Lw.at<float>(3) = PLw.at<float>(2, 1);
					Lw.at<float>(4) = PLw.at<float>(0, 2);
					Lw.at<float>(5) = PLw.at<float>(1, 0);
					cv::Mat NLw = PLw.col(3).rowRange(0, 3);
					NLw.copyTo(Lw.rowRange(0, 3));
									
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
					cv::Mat Lc = T2*Lw;
					cv::Mat Nc = Lc.rowRange(0, 3);
					cv::Mat Ni = mK2*Nc;
					
					float x1 = 0;
					float y1 = 0;
					if (Ni.at<float>(0) != 0)
						x1 = -Ni.at<float>(2) / Ni.at<float>(0);

					float x2 = 0;
					float y2 = mnHeight;
					if (Ni.at<float>(0) != 0)
						x2 = (-Ni.at<float>(2) - Ni.at<float>(1)*y2) / Ni.at<float>(0);
					
					cv::line(vis, cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Scalar(255, 0, 0), 2);

					
					//맵포인트 생성.
					//영역 체크도 필요함.
					cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
					R.copyTo(T.rowRange(0, 3).colRange(0, 3));
					t.copyTo(T.col(3).rowRange(0, 3));
					cv::Mat invT = T.inv();

					cv::Mat tempP1 = invT.t()*P1;
					cv::Mat tempP2 = invT.t()*P2;
					
					for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
						if (!mpTargetFrame->GetBoolInlier(i)) {
							cv::Point2f pt = mpTargetFrame->mvKeyPoints[i].pt;
							float val = pt.x*Ni.at<float>(0) + pt.y*Ni.at<float>(1) + Ni.at<float>(2);
							cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
							if (val < 0 && mpTargetFrame->GetObjectType(i) == ObjectType::OBJECT_FLOOR) {
								
								temp = mK.inv()*temp;
								cv::Mat matDepth = -tempP1.at<float>(3) / (tempP1.rowRange(0, 3).t()*temp);
								float depth = matDepth.at<float>(0);
								temp *= depth;
								temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
								cv::Mat estimated = invT*temp;
								
								UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(estimated.rowRange(0,3), mpTargetFrame->matDescriptor.row(i));
								pNewMP->AddFrame(mpTargetFrame, i);

								cv::circle(vis, pt, 2, cv::Scalar(255, 0, 255), -1);
							}
							else if (val > 0 && mpTargetFrame->GetObjectType(i) == ObjectType::OBJECT_WALL) {
								temp = mK.inv()*temp;
								cv::Mat matDepth = -tempP2.at<float>(3) / (tempP2.rowRange(0, 3).t()*temp);
								float depth = matDepth.at<float>(0);
								temp *= depth;
								temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
								cv::Mat estimated = invT*temp;

								UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(estimated.rowRange(0, 3), mpTargetFrame->matDescriptor.row(i));
								pNewMP->AddFrame(mpTargetFrame, i);

								cv::circle(vis, pt, 2, cv::Scalar(255, 255, 0), -1);
							}
						}//if inlier
					}//for

					imshow("Output::PlaneEstimation", vis);
				}
			}
			cv::waitKey(10);
			SetBoolDoingProcess(false, 2);
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
		reversePlaneSign(X);

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

	cv::Mat checkResidual2 = mMat*param > 2 * thresh_distance; checkResidual2 /= 255; checkResidual2 *= 2;
	paramStatus += checkResidual2;

	if (planeRatio > thresh_ratio) {
		cv::Mat pParam = param.clone();
		pPlane->matPlaneParam = pParam.clone();
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
