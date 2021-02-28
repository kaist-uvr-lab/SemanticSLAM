#include <PlaneEstimator.h>
#include <random>
#include <System.h>
#include <Map.h>
#include <Plane.h>
#include <Frame.h>
#include <FrameGrid.h>
#include <CandidatePoint.h>
#include <SegmentationData.h>
#include <MapPoint.h>
#include <Matcher.h>
#include <Visualizer.h>
#include <Initializer.h>
#include <MatrixOperator.h>

static int nPlaneID = 0;

UVR_SLAM::PlaneEstimator::PlaneEstimator() :mbDoingProcess(false), mpTempFrame(nullptr), mnProcessType(0), mpLayoutFrame(nullptr){
}
UVR_SLAM::PlaneEstimator::PlaneEstimator(System* pSys, std::string strPath) : mpSystem(pSys), mbDoingProcess(false), mpTempFrame(nullptr), mnProcessType(0), mpLayoutFrame(nullptr),
mpPrevFrame(nullptr), mpPPrevFrame(nullptr), mpTargetFrame(nullptr)
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
	mpFloorPlaneInformation = new PlaneInformation();
}
void UVR_SLAM::PlaneEstimator::Init() {
	mpMap = mpSystem->mpMap;
	mpVisualizer = mpSystem->mpVisualizer;
	mpMatcher = mpSystem->mpMatcher;
	mpInitializer = mpSystem->mpInitializer;
	//mK2 = mpSystem->mKforPL.clone();
}
UVR_SLAM::PlaneInformation::PlaneInformation() {
	mbInit = false;
	matPlaneParam = cv::Mat::zeros(4, 1, CV_32FC1);
	normal = matPlaneParam.rowRange(0, 3);
	distance = matPlaneParam.at<float>(3);
}
UVR_SLAM::PlaneInformation::~PlaneInformation() {

}
void UVR_SLAM::PlaneInformation::SetParam(cv::Mat m) {
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
	matPlaneParam = cv::Mat::zeros(4, 1, CV_32FC1);
	matPlaneParam = m.clone();

	normal = matPlaneParam.rowRange(0, 3);
	distance = matPlaneParam.at<float>(3);
	norm = normal.dot(normal);
	normal /= norm;
	distance /= norm;
	norm = 1.0;

	normal.copyTo(matPlaneParam.rowRange(0, 3));
	matPlaneParam.at<float>(3) = distance;
}
void UVR_SLAM::PlaneInformation::SetParam(cv::Mat n, float d){
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
	matPlaneParam = cv::Mat::zeros(4, 1, CV_32FC1);
	normal = n.clone();
	distance = d;
	norm = normal.dot(normal);
	normal.copyTo(matPlaneParam.rowRange(0, 3));
	matPlaneParam.at<float>(3) = d;
}
void UVR_SLAM::PlaneInformation::GetParam(cv::Mat& n, float& d){
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
	n = normal.clone();
	d = distance;
}
cv::Mat UVR_SLAM::PlaneInformation::GetParam() {
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
	return matPlaneParam.clone();
}
UVR_SLAM::PlaneEstimator::~PlaneEstimator() {}

///////////////////////////////////////////////////////////////////////////////
//기본 함수들
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
	mpPPrevFrame = mpPrevFrame;
	mpPrevFrame = mpTargetFrame;
	mpTargetFrame = mKFQueue.front();

	//임시로 주석 처리
	/*if (mpMap->isFloorPlaneInitialized()) {
		mpTargetFrame->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpTargetFrame, mpMap->mpFloorPlane);
		mpTargetFrame->mpPlaneInformation->Calculate();
	}*/

	mKFQueue.pop();
}
void UVR_SLAM::PlaneEstimator::Reset() {
	mpTargetFrame = mpPrevFrame;
	mpPPrevFrame = nullptr;
	mpPrevFrame = nullptr;
}
///////////////////////////////////////////////////////////////////////////////

void UVR_SLAM::PlaneEstimator::Run() {

	mpFloorPlaneInformation = new UVR_SLAM::PlaneInformation();
	mpFloorPlaneInformation->mbInit = false;
	std::string mStrPath;

	std::vector<UVR_SLAM::PlaneInformation*> mvpPlanes;

	UVR_SLAM::Frame* pTestF = nullptr;
	UVR_SLAM::PlaneInformation* mpGlobalFloor = nullptr;
	UVR_SLAM::PlaneInformation* mpGlobalCeil = nullptr;

	std::vector<UVR_SLAM::MapPoint*> mvpFloorMPs, mvpWallMPs, mvpCeilMPs;
	std::vector<int> vIndices;

	int mnFontFace = (2);
	double mfFontScale = (0.6);
	int mnDebugFontSize = 20;
	int nFrameSize;
	int nTrial = 500;
	float pidst = 0.05;//0.01
	float pratio = 0.25;

	int nLayoutFloor = 1;
	int nLayoutCeil = 2;
	int nLayoutWall = 3;

	float sumFloor = 0.0;
	int nFloor = 0;

	float sumCeil = 0.0;
	int nCeil = 0;

	int mnLabel_floor = 4;
	int mnLabel_ceil = 6;
	int mnLabel_wall = 1;

	while (true) {

		if (CheckNewKeyFrames()) {
			
			//저장 디렉토리 명 획득
			SetBoolDoingProcess(true,0);
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			ProcessNewKeyFrame();
			
			if (!mpPrevFrame) {
				SetBoolDoingProcess(false, 0);
				continue;
			}
			else
			{

				cv::Scalar color1(255, 0, 0);
				cv::Scalar color2(0, 255, 0);
				cv::Scalar color3(0, 0, 255);

				std::vector<MapPoint*> vpMPs;
				std::set<MapPoint*> spMPs;

				std::vector<UVR_SLAM::MapPoint*> vpTempFloorMPs, vpTempOutlierFloorMPs;

				auto spGraphKFs = mpMap->GetGraphFrames();//GetWindowFramesSet, GetGraphFrames
				auto spWindowKFs = mpMap->GetWindowFramesSet();
				for (auto iter = spWindowKFs.begin(), iend = spWindowKFs.end(); iter != iend; iter++) {
					auto pKFi = *iter;
					spGraphKFs.push_back(pKFi);
				}
				for (auto iter = spGraphKFs.begin(), iend = spGraphKFs.end(); iter != iend; iter++) {
					auto pKFi = *iter;

					auto vpGrids = pKFi->mmpFrameGrids;

					for (auto iter = vpGrids.begin(), iend = vpGrids.end(); iter != iend; iter++) {
						auto pGrid = iter->second;
						auto pt = iter->first;
						if (!pGrid)
							continue;
						
						int nCountFloor = pGrid->mObjCount.at<int>(mnLabel_floor);//pGrid->mmObjCounts.count(mnLabel_floor);
						float fWallArea = pGrid->mObjArea.at<float>(mnLabel_wall);
						float fFloorArea = pGrid->mObjArea.at<float>(mnLabel_floor);
						bool bFloor = nCountFloor > 0 && fFloorArea > fWallArea;

						for (size_t ci = 0, cend = pGrid->mvpCPs.size(); ci < cend; ci++) {
							auto pCPi = pGrid->mvpCPs[ci];
							if (!pCPi)
								continue;
							auto pMPi = pCPi->GetMP();
							bool bMP = !pMPi || pMPi->isDeleted();
							if (bMP)
								continue;
							if (spMPs.count(pMPi))
								continue;

							if (bFloor) {
								vpTempFloorMPs.push_back(pMPi);
							}
							vpMPs.push_back(pMPi);
							spMPs.insert(pMPi);
						}

						//int nCountFloor = pGrid->mmObjCounts[mnLabel_floor];
					}
				}
				//std::cout << "Plane::MP::"<< vpTempFloorMPs.size()<<"::"<<vpMPs.size() << std::endl;
				
				if (vpTempFloorMPs.size() > 50) {
					//std::cout << vpTempFloorMPs.size() << ", " << vpTempOutlierFloorMPs.size() << std::endl;
					auto tempFloor = new UVR_SLAM::PlaneInformation();
					bool bFloorRes = UVR_SLAM::PlaneInformation::PlaneInitialization(tempFloor, vpTempFloorMPs, vpTempOutlierFloorMPs, mpTargetFrame->mnFrameID, 1500, pidst, 0.1);
					if (bFloorRes) {
						tempFloor->mbInit = true;
						SetPlaneParam(tempFloor);
						//std::cout << "PARAM::" << tempFloor->GetParam().t() << std::endl;
						cv::Mat R, t;
						mpTargetFrame->GetPose(R, t);
						for (size_t i = 0, iend = tempFloor->mvpMPs.size(); i < iend; i++) {
							auto pMPi = tempFloor->mvpMPs[i];
							if (!pMPi || pMPi->isDeleted())
								continue;
							cv::Mat X3D = pMPi->GetWorldPos();
							cv::Mat temp = mpTargetFrame->mK*(R*X3D + t);
							cv::Point2f pt(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
							//cv::circle(testImg, pt, 2, color2, -1);
						}
						for (size_t i = 0, iend = vpTempOutlierFloorMPs.size(); i < iend; i++) {
							auto pMPi = vpTempOutlierFloorMPs[i];
							cv::Mat X3D = pMPi->GetWorldPos();
							cv::Mat temp = mpTargetFrame->mK*(R*X3D + t);
							cv::Point2f pt(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
							//cv::circle(testImg, pt, 2, color3, -1);
						}

						////평면 시각화용
						/*cv::Mat invP, invT, invK;
						mpTargetFrame->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpTargetFrame, tempFloor);
						mpTargetFrame->mpPlaneInformation->Calculate(tempFloor);
						mpTargetFrame->mpPlaneInformation->GetInformation(invP, invT, invK);
						std::vector<cv::Mat> vTempVisPts;
						int x = mpTargetFrame->matFrame.cols / 2;
						for (size_t y = 0, yend = mpTargetFrame->matFrame.rows / 2; y < yend; y += 2) {
							cv::Point2f pt(x, y);
							cv::Mat s;
							bool b = PlaneInformation::CreatePlanarMapPoint(s, pt, invP, invT, invK);
							if (b)
								vTempVisPts.push_back(s);
						}
						SetTempPTs(vTempVisPts);*/
						////평면 시각화용

					}//평면 검출 성공
				}//사이즈
				 //imshow("test::floor", testImg); cv::waitKey(1);
				 /*cv::Mat resized;
				 cv::resize(testImg, resized, cv::Size(testImg.cols / 2, testImg.rows / 2));
				 mpVisualizer->SetOutputImage(resized, 2);*/

				SetBoolDoingProcess(false, 0);
				continue;
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////
cv::Mat UVR_SLAM::PlaneInformation::PlaneLineEstimator(WallPlane* pWall, PlaneInformation* pFloor) {
	cv::Mat mat = cv::Mat::zeros(0,3,CV_32FC1);

	auto lines = pWall->GetLines();
	for (int i = 0; i < lines.size(); i++) {
		mat.push_back(lines[i]->GetLinePts());
	}
	
	std::random_device rn;
	std::mt19937_64 rnd(rn());
	std::uniform_int_distribution<int> range(0, mat.rows - 1);

	//ransac
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;
	cv::Mat param, paramStatus;
	//ransac

	for (int n = 0; n < 1000; n++) {

		cv::Mat arandomPts = cv::Mat::zeros(0, 4, CV_32FC1);
		//select pts
		for (int k = 0; k < 3; k++) {
			int randomNum = range(rnd);
			cv::Mat temp = mat.row(randomNum).clone();
			arandomPts.push_back(temp);
		}//select

		 //SVD
		cv::Mat X;
		cv::Mat w, u, vt;
		cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
		X = vt.row(2).clone();
		cv::transpose(X, X);

		cv::Mat checkResidual = abs(mat*X) < 0.01;
		checkResidual = checkResidual / 255;
		int temp_inlier = cv::countNonZero(checkResidual);

		if (max_num_inlier < temp_inlier) {
			max_num_inlier = temp_inlier;
			param = X.clone();
			paramStatus = checkResidual.clone();
		}
	}
	
	/////////
	cv::Mat newMat = cv::Mat::zeros(0,3, CV_32FC1);
	int nInc = 1;
	if (mat.rows > 1000)
		nInc *= 10;
	for (int i = 0; i < mat.rows; i+= nInc) {
		int checkIdx = paramStatus.at<uchar>(i);
		if (checkIdx == 0)
			continue;
		newMat.push_back(mat.row(i));
	}
	
	cv::Mat w, u, vt;
	cv::SVD::compute(newMat, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	
	param = vt.row(2).clone();
	cv::transpose(param, param);
	
	/////////
	float yval = 0.0;
	cv::Mat pParam = pFloor->matPlaneParam.clone();
	//std::cout << param.t() <<pParam.t()<< std::endl;
	cv::Mat normal;
	float dist;
	pFloor->GetParam(normal, dist);
	cv::Mat s = cv::Mat::zeros(4, 1, CV_32FC1);
	s.at<float>(3) = 1.0;
	s.at<float>(2) = -param.at<float>(2) / param.at<float>(1);
	yval = pParam.dot(s);
	s.at<float>(1) = -yval / pParam.at<float>(1);

	cv::Mat e = cv::Mat::zeros(4, 1, CV_32FC1);
	e.at<float>(0) = 1.0;
	e.at<float>(3) = 1.0;
	e.at<float>(2) = -(param.at<float>(0)+param.at<float>(2)) / param.at<float>(1);
	yval = pParam.dot(e);
	e.at<float>(1) = -yval / pParam.at<float>(1);
	//std::cout << "????" << std::endl;
	//std::cout << s.dot(pParam) << ", " << e.dot(pParam) << std::endl;
	param = UVR_SLAM::PlaneInformation::PlaneWallEstimator(s.rowRange(0,3), e.rowRange(0,3), normal, cv::Mat(), cv::Mat(), cv::Mat());
	//std::cout << "test : " << param.t() << "::" << max_num_inlier << ", " << mat.rows << std::endl;
	//std::cout << "line param : " << param.t() << std::endl;
	return param;
} 

//평면 추정 관련 함수들
bool UVR_SLAM::PlaneInformation::PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::MapPoint*>& vpOutlierMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
	//RANSAC
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;

	cv::Mat param, paramStatus;

	//초기 매트릭스 생성
	cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
	std::vector<int> vIdxs;
	for(int i = 0; i < vpMPs.size(); i++){
		UVR_SLAM::MapPoint* pMP = vpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		cv::Mat temp = pMP->GetWorldPos();
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		mMat.push_back(temp.t());
		vIdxs.push_back(i);
	}
	if (mMat.rows < 10)
		return false;
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
		cv::Mat tempMat = cv::Mat::zeros(0, 4, CV_32FC1);
		cv::Mat pParam = param.clone();
		pPlane->matPlaneParam = pParam.clone();
		pPlane->mnPlaneID = ++nPlaneID;

		cv::Mat normal = pParam.rowRange(0, 3);
		float dist = pParam.at<float>(3);
		pPlane->SetParam(normal, dist);
		//pPlane->norm = sqrt(pPlane->normal.dot(pPlane->normal));

		for (int i = 0; i < mMat.rows; i++) {
			int checkIdx = paramStatus.at<uchar>(i);
			//std::cout << checkIdx << std::endl;
			UVR_SLAM::MapPoint* pMP = vpMPs[vIdxs[i]];
			if (checkIdx == 0){
				vpOutlierMPs.push_back(pMP);
				continue;
			}
			if (pMP && !pMP->isDeleted()) {
				//평면에 대한 레이블링이 필요함.
				//pMP->SetRecentLayoutFrameID(nTargetID);
				//pMP->SetPlaneID(pPlane->mnPlaneID);
				pPlane->mvpMPs.push_back(pMP);
				tempMat.push_back(mMat.row(i));
			}
		}
		//평면 정보 생성.

		cv::Mat X;
		cv::Mat w, u, vt;
		cv::SVD::compute(tempMat, w, u, vt, cv::SVD::FULL_UV);
		X = vt.row(3).clone();
		cv::transpose(X, X);
		UVR_SLAM::PlaneInformation::calcUnitNormalVector(X);
		int idx = pPlane->GetNormalType(X);
		if (X.at<float>(idx) > 0.0)
			X *= -1.0;
		pPlane->mnCount = pPlane->mvpMPs.size();
		//std::cout <<"PLANE::"<< planeRatio << std::endl;
		//std::cout << "Update::" << pPlane->matPlaneParam.t() << ", " << X.t() <<", "<<pPlane->mvpMPs.size()<<", "<<nReject<< std::endl;
		pPlane->SetParam(X.rowRange(0, 3), X.at<float>(3));

		return true;
	}
	else
	{
		//std::cout << "failed" << std::endl;
		return false;
	}
}
bool UVR_SLAM::PlaneInformation::PlaneInitialization2(UVR_SLAM::PlaneInformation* pPlane, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::MapPoint*>& vpOutlierMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
	//RANSAC
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;

	cv::Mat param, paramStatus;

	//초기 매트릭스 생성
	cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
	std::vector<int> vIdxs;
	for (int i = 0; i < vpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = vpMPs[i];
		if (!pMP || pMP->isDeleted())
			continue;
		cv::Mat temp = pMP->GetWorldPos();
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		mMat.push_back(temp.t());
		vIdxs.push_back(i);
	}
	if (mMat.rows < 10)
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
		cv::Mat tempMat = cv::Mat::zeros(0, 4, CV_32FC1);
		cv::Mat pParam = param.clone();
		pPlane->matPlaneParam = pParam.clone();
		pPlane->mnPlaneID = ++nPlaneID;

		cv::Mat normal = pParam.rowRange(0, 3);
		float dist = pParam.at<float>(3);
		pPlane->SetParam(normal, dist);
		//pPlane->norm = sqrt(pPlane->normal.dot(pPlane->normal));

		for (int i = 0; i < mMat.rows; i++) {
			int checkIdx = paramStatus.at<uchar>(i);
			//std::cout << checkIdx << std::endl;
			UVR_SLAM::MapPoint* pMP = vpMPs[vIdxs[i]];
			if (checkIdx == 0) {
				vpOutlierMPs.push_back(pMP);
				continue;
			}
			if (pMP && !pMP->isDeleted()) {
				//평면에 대한 레이블링이 필요함.
				//pMP->SetRecentLayoutFrameID(nTargetID);
				//pMP->SetPlaneID(pPlane->mnPlaneID);
				pPlane->mvpMPs.push_back(pMP);
				tempMat.push_back(mMat.row(i));
			}
		}
		//평면 정보 생성.

		cv::Mat X;
		cv::Mat w, u, vt;
		cv::SVD::compute(tempMat, w, u, vt, cv::SVD::FULL_UV);
		X = vt.row(3).clone();
		cv::transpose(X, X);
		UVR_SLAM::PlaneInformation::calcUnitNormalVector(X);
		int idx = pPlane->GetNormalType(X);
		if (X.at<float>(idx) > 0.0)
			X *= -1.0;
		pPlane->mnCount = pPlane->mvpMPs.size();
		//std::cout <<"PLANE::"<< planeRatio << std::endl;
		//std::cout << "Update::" << pPlane->matPlaneParam.t() << ", " << X.t() <<", "<<pPlane->mvpMPs.size()<<", "<<nReject<< std::endl;
		pPlane->SetParam(X.rowRange(0, 3), X.at<float>(3));

		return true;
	}
	else
	{
		//std::cout << "failed" << std::endl;
		return false;
	}
}
void UVR_SLAM::PlaneEstimator::UpdatePlane(PlaneInformation* pPlane, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
	auto mvpMPs = std::vector<UVR_SLAM::MapPoint*>(pPlane->mvpMPs.begin(), pPlane->mvpMPs.end());
	std::vector<int> vIdxs(0);
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;
	//초기 매트릭스 생성
	cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
	int nStartIdx = 0;
	if (mvpMPs.size() > 5000) {
		nStartIdx = 1000;
	}
	for (int i = nStartIdx; i < mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		cv::Mat temp = pMP->GetWorldPos();
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		mMat.push_back(temp.t());
		vIdxs.push_back(i);
	}
	if (mMat.rows == 0)
		return;

	cv::Mat param, paramStatus;
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
		UVR_SLAM::PlaneInformation::calcUnitNormalVector(X);
		if (X.at<float>(1) > 0.0)
			X *= -1.0;

		cv::Mat checkResidual = abs(mMat*X) < 0.001;
		checkResidual = checkResidual / 255;
		int temp_inlier = cv::countNonZero(checkResidual);

		if (max_num_inlier < temp_inlier) {
			max_num_inlier = temp_inlier;
			param = X.clone();
			paramStatus = checkResidual.clone();
		}
	}//trial

	if (max_num_inlier == 0)
		return;

	float planeRatio = ((float)max_num_inlier / mMat.rows);
	
	if (planeRatio > thresh_ratio) {
		int nReject = 0;
		pPlane->mvpMPs.clear();
		cv::Mat tempMat = cv::Mat::zeros(0, 4, CV_32FC1);
		
		for (int i = 0; i < mMat.rows; i++) {
			int checkIdx = paramStatus.at<uchar>(i);
			UVR_SLAM::MapPoint* pMP = mvpMPs[vIdxs[i]];
			if (checkIdx == 0){
				pMP->SetPlaneID(-1);
				nReject++;
				continue;
			}
			if (pMP) {
				//평면에 대한 레이블링이 필요함.
				if (pMP->isDeleted())
					continue;
				pMP->SetRecentLayoutFrameID(nTargetID);
				pMP->SetPlaneID(pPlane->mnPlaneID);
				pPlane->mvpMPs.push_back(pMP);
				tempMat.push_back(mMat.row(i));
			}
		}
		//평면 정보 생성.
		
		cv::Mat X;
		cv::Mat w, u, vt;
		cv::SVD::compute(tempMat, w, u, vt, cv::SVD::FULL_UV);
		X = vt.row(3).clone();
		cv::transpose(X, X);
		UVR_SLAM::PlaneInformation::calcUnitNormalVector(X);
		if (X.at<float>(1) > 0.0)
			X *= -1.0;

		//std::cout << "Update::" << pPlane->matPlaneParam.t() << ", " << X.t() <<", "<<pPlane->mvpMPs.size()<<", "<<nReject<< std::endl;
		pPlane->SetParam(X.rowRange(0, 3), X.at<float>(3));

		return;
	}
	else
	{
		return;
	}
}

//GroundPlane은 현재 평면, type == 1이면 벽, 아니면 천장
bool UVR_SLAM::PlaneEstimator::PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, UVR_SLAM::PlaneInformation* GroundPlane, int type, std::vector<UVR_SLAM::MapPoint*> vpMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
	//RANSAC
	std::vector<int> vIdxs(0);
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;

	cv::Mat param, paramStatus;

	//초기 매트릭스 생성
	cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
	for (int i = 0; i < vpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = vpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		cv::Mat temp = pMP->GetWorldPos();
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		mMat.push_back(temp.t());
		vIdxs.push_back(i);
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
		UVR_SLAM::PlaneInformation::calcUnitNormalVector(X);

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
		pPlane->mnPlaneID = ++nPlaneID;
		pPlane->SetParam(pParam.rowRange(0, 3), pParam.at<float>(3));

		for (int i = 0; i < mMat.rows; i++) {
			int checkIdx = paramStatus.at<uchar>(i);
			//std::cout << checkIdx << std::endl;
			UVR_SLAM::MapPoint* pMP = vpMPs[vIdxs[i]];
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

bool UVR_SLAM::PlaneInformation::calcUnitNormalVector(cv::Mat& X) {
	float sum = sqrt(X.at<float>(0)*X.at<float>(0) + X.at<float>(1)*X.at<float>(1) + X.at<float>(2)*X.at<float>(2));
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
int UVR_SLAM::PlaneInformation::GetNormalType(cv::Mat X) {
	float maxVal = 0.0;
	int idx;
	for (int i = 0; i < 3; i++) {
		float val = abs(X.at<float>(i));
		if (val > maxVal) {
			maxVal = val;
			idx = i;
		}
	}
	return idx;
}

void UVR_SLAM::PlaneEstimator::reversePlaneSign(cv::Mat& param) {
	if (param.at<float>(3, 0) < 0.0) {
		param *= -1.0;
	}
}
//평면 추정 관련 함수들
//////////////////////////////////////////////////////////////////////

//플루커 라인 프로젝션 관련 함수
cv::Mat UVR_SLAM::PlaneInformation::FlukerLineProjection(cv::Mat P1, cv::Mat P2, cv::Mat R, cv::Mat t, cv::Mat K, float& m) {
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
	cv::Mat res = K*Nc;
	if (res.at<float>(0) < 0)
		res *= -1;
	if (res.at<float>(0) != 0)
		m = res.at<float>(1) / res.at<float>(0);
	else
		m = 9999.0;
	return res.clone();
}
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
		/*if (pMP->GetRecentLocalMapID()>=nID) {
			nTotal++;
		}*/
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
	return abs(normal.dot(p->normal) / (d1*d2));
}

float UVR_SLAM::PlaneInformation::CalcCosineSimilarity(cv::Mat P) {

	float d1 = this->norm;
	cv::Mat tempNormal = P.rowRange(0, 3);

	float d2 = sqrt(tempNormal.dot(tempNormal));
	
	if (CheckZero(d1) || CheckZero(d2))
		return 0.0;
	return abs(normal.dot(tempNormal) / (d1*d2));
}


float UVR_SLAM::PlaneInformation::CalcPlaneDistance(PlaneInformation* p) {
	return abs((distance) - (p->distance));
	//return abs(abs(distance) - abs(p->distance));
}

float UVR_SLAM::PlaneInformation::CalcPlaneDistance(cv::Mat X) {
	return X.dot(this->normal) + distance;
}

float UVR_SLAM::PlaneInformation::CalcCosineSimilarity(cv::Mat P1, cv::Mat P2){
	cv::Mat normal1 = P1.rowRange(0, 3);
	cv::Mat normal2 = P2.rowRange(0, 3);
	float d1 = sqrt(normal1.dot(normal1));
	float d2 = sqrt(normal2.dot(normal2));
	if (CheckZero(d1) || CheckZero(d2))
		return 0.0;
	return abs(normal1.dot(normal2)) / (d1*d2);
}

//두 평면의 거리를 비교
float UVR_SLAM::PlaneInformation::CalcPlaneDistance(cv::Mat X1, cv::Mat X2){
	float d1 = (X1.at<float>(3));
	float d2 = (X2.at<float>(3));

	return abs(d1 - d2);

}
bool UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(cv::Point2f pt, cv::Mat invP, cv::Mat invT, cv::Mat invK, cv::Mat& X3D) {
	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	temp = invK*temp;
	cv::Mat matDepth = -invP.at<float>(3) / (invP.rowRange(0, 3).t()*temp);
	float depth = matDepth.at<float>(0);
	if (depth < 0) {
		//depth *= -1.0;
		return false;
	}
	temp *= depth;
	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	cv::Mat estimated = invT*temp;
	X3D = estimated.rowRange(0, 3);
	return true;
}

bool UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(cv::Point2f pt, cv::Mat invPw, cv::Mat invT, cv::Mat invK, cv::Mat fNormal, float fDist, cv::Mat& X3D) {
	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	temp = invK*temp;
	cv::Mat matDepth = -invPw.at<float>(3) / (invPw.rowRange(0, 3).t()*temp);
	float depth = matDepth.at<float>(0);
	if (depth < 0) {
		//depth *= -1.0;
		return false;
	}
	temp *= depth;
	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	cv::Mat estimated = invT*temp;
	X3D = estimated.rowRange(0, 3);
	float val = X3D.dot(fNormal) + fDist;
	if (val < 0.0)
		return false;
	return true;
}

bool UVR_SLAM::PlaneEstimator::ConnectedComponentLabeling(cv::Mat img, cv::Mat& dst, cv::Mat& stat) {
	dst = img.clone();
	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(img, img_labels, stats, centroids, 8, CV_32S);

	if (numOfLables == 0)
		return false;

	int maxArea = 0;
	int maxIdx = 0;
	//라벨링 된 이미지에 각각 직사각형으로 둘러싸기 
	for (int j = 1; j < numOfLables; j++) {
		int area = stats.at<int>(j, CC_STAT_AREA);
		if (area > maxArea) {
			maxArea = area;
			maxIdx = j;
		}
	}
	/*int left = stats.at<int>(maxIdx, CC_STAT_LEFT);
	int top = stats.at<int>(maxIdx, CC_STAT_TOP);
	int width = stats.at<int>(maxIdx, CC_STAT_WIDTH);
	int height = stats.at<int>(maxIdx, CC_STAT_HEIGHT);*/

	for (int j = 1; j < numOfLables; j++) {
		if (j == maxIdx)
			continue;
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);

		rectangle(dst, Point(left, top), Point(left + width, top + height), Scalar(0, 0, 0), -1);
	}
	stat = stats.row(maxIdx).clone();
	return true;
}

cv::Mat UVR_SLAM::PlaneInformation::CalcPlaneRotationMatrix(cv::Mat P) {
	//euler zxy
	cv::Mat Nidealfloor = cv::Mat::zeros(3, 1, CV_32FC1);
	cv::Mat normal = P.rowRange(0, 3);

	Nidealfloor.at<float>(1) = -1.0;
	float nx = P.at<float>(0);
	float ny = P.at<float>(1);
	float nz = P.at<float>(2);

	float d1 = atan2(nx, -ny);
	float d2 = atan2(-nz, sqrt(nx*nx + ny*ny));
	cv::Mat R = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngles(d1, d2, 0.0, "ZXY");


	///////////한번 더 돌리는거 테스트
	/*cv::Mat R1 = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngle(d1, 'z');
	cv::Mat R2 = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngle(d2, 'x');
	cv::Mat Nnew = R2.t()*R1.t()*normal;
	float d3 = atan2(Nnew.at<float>(0), Nnew.at<float>(2));
	cv::Mat Rfinal = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngles(d1, d2, d3, "ZXY");*/
	///////////한번 더 돌리는거 테스트

	/*cv::Mat test1 = R*Nidealfloor;
	cv::Mat test3 = R.t()*normal;
	std::cout << "ATEST::" << P.t() << test1.t() << test3.t()<< std::endl;*/


	/*
	cv::Mat test2 = Rfinal*Nidealfloor;
	cv::Mat test4 = Rfinal.t()*normal;
	std::cout << d1 << ", " << d2 << ", " << d3 << std::endl;
	std::cout << "ATEST::" << P.t() << test1.t() << test2.t() << test3.t() << test4.t() << std::endl;*/

	return R;
}

////////////////////////////////////////////////////////////////////////
//////////코드 백업
void UVR_SLAM::PlaneEstimator::CreatePlanarMapPoints(std::vector<UVR_SLAM::MapPoint*> mvpMPs, std::vector<UVR_SLAM::ObjectType> mvpOPs, UVR_SLAM::PlaneInformation* pPlane, cv::Mat invT) {
	
	//int nTargetID = mpTargetFrame->GetFrameID();
	//
	//cv::Mat invP1 = invT.t()*pPlane->GetParam();

	//float minDepth = FLT_MAX;
	//float maxDepth = 0.0f;
	////create new mp in current frame
	//for (int j = 0; j < mvpMPs.size(); j++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//	auto oType = mvpOPs[j];

	//	if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR)
	//		continue;
	//	cv::Point2f pt = mpTargetFrame->mvKeyPoints[j].pt;
	//	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	//	temp = mK.inv()*temp;
	//	cv::Mat matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
	//	float depth = matDepth.at<float>(0);
	//	if (depth < 0){
	//		//depth *= -1.0;
	//		continue;
	//	}
	//	temp *= depth;
	//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	//	/*if (maxDepth < depth)
	//		maxDepth = depth;
	//	if (minDepth > depth)
	//		minDepth = depth;*/

	//	cv::Mat estimated = invT*temp;
	//	if (pMP) {
	//		pMP->SetWorldPos(estimated.rowRange(0, 3));
	//	}
	//	else {
	//		UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), mpTargetFrame->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
	//		pNewMP->SetPlaneID(pPlane->mnPlaneID);
	//		pNewMP->SetObjectType(pPlane->mnPlaneType);
	//		pNewMP->AddFrame(mpTargetFrame, j);
	//		pNewMP->UpdateNormalAndDepth();
	//		pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
	//		mpSystem->mlpNewMPs.push_back(pNewMP);
	//		//pPlane->mvpMPs.push_back(pNewMP);
	//		pPlane->tmpMPs.push_back(pNewMP);
	//		//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
	//	}
	//}

	///*cv::Mat R, t;
	//mpTargetFrame->GetPose(R, t);

	//for (int j = 0; j < mvpMPs.size(); j++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//	auto oType = mvpOPs[j];
	//	if (!pMP)
	//		continue;
	//	if (pMP->isDeleted())
	//		continue;
	//	if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR) {
	//		cv::Mat X3D = pMP->GetWorldPos();
	//		cv::Mat Xcam = R*X3D + t;
	//		float depth = Xcam.at<float>(2);
	//		if (depth < 0.0 || depth > maxDepth)
	//			pMP->SetDelete(true);
	//	}
	//}

	//mpTargetFrame->SetDepthRange(minDepth, maxDepth);*/
}

void UVR_SLAM::PlaneEstimator::CreateWallMapPoints(std::vector<UVR_SLAM::MapPoint*> mvpMPs, std::vector<UVR_SLAM::ObjectType> mvpOPs, UVR_SLAM::PlaneInformation* pPlane, cv::Mat invT, int wtype,int MinX, int MaxX, bool b1, bool b2) {

	//int nTargetID = mpTargetFrame->GetFrameID();

	//cv::Mat invP1 = invT.t()*pPlane->GetParam();
	//	
	////create new mp in current frame
	//for (int j = 0; j < mvpMPs.size(); j++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//	auto oType = mvpOPs[j];

	//	if (oType != UVR_SLAM::ObjectType::OBJECT_WALL)
	//		continue;
	//	cv::Point2f pt = mpTargetFrame->mvKeyPoints[j].pt;
	//	int x = pt.x;
	//	if (wtype == 1) {
	//		if (x > MinX)
	//			continue;
	//	}
	//	else if (wtype == 2) {
	//		if (x < MinX || x > MaxX)
	//			continue;
	//	}
	//	else {
	//		if (x < MaxX)
	//			continue;
	//	}
	//	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	//	temp = mK.inv()*temp;
	//	cv::Mat matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
	//	float depth = matDepth.at<float>(0);
	//	if (depth < 0.0){
	//		//depth *= -1.0;
	//		continue;
	//	}
	//	temp *= depth;
	//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	//	cv::Mat estimated = invT*temp;
	//	if (pMP) {
	//		pMP->SetWorldPos(estimated.rowRange(0, 3));
	//	}
	//	else {
	//		UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), mpTargetFrame->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
	//		pNewMP->SetPlaneID(pPlane->mnPlaneID);
	//		pNewMP->SetObjectType(pPlane->mnPlaneType);
	//		pNewMP->AddFrame(mpTargetFrame, j);
	//		pNewMP->UpdateNormalAndDepth();
	//		pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
	//		mpSystem->mlpNewMPs.push_back(pNewMP);
	//		pPlane->mvpMPs.push_back(pNewMP);
	//		//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
	//	}
	//}

}

void CheckWallNormal(cv::Mat& normal) {
	float a = (normal.at<float>(0));
	float c = (normal.at<float>(2));
	int idx;
	if (abs(a) > abs(c)) {
		if (a < 0)
			normal *= -1.0;
	}
	else {
		if (c < 0)
			normal *= -1.0;
	}
}

cv::Mat UVR_SLAM::PlaneInformation::PlaneWallEstimator(cv::Mat s, cv::Mat e, cv::Mat normal1, cv::Mat invP, cv::Mat invT, cv::Mat invK) {
	cv::Mat normal2 = s - e;
	normal2 = normal2.rowRange(0, 3);
	float norm2 = sqrt(normal2.dot(normal2));
	normal2 /= norm2;
	auto normal3 = normal1.cross(normal2);
	float norm3 = sqrt(normal3.dot(normal3));
	normal3 /= norm3;

	cv::Mat matDist = normal3.t()*s.rowRange(0, 3);

	normal3.push_back(-matDist);
	CheckWallNormal(normal3);
	return normal3;
}

cv::Mat UVR_SLAM::PlaneInformation::PlaneWallEstimator(UVR_SLAM::Line* line, cv::Mat normal1, cv::Mat invP, cv::Mat invT, cv::Mat invK) {

	Point2f from = line->from;
	Point2f to = line->to;
	cv::Mat s, e;
	bool bs = CreatePlanarMapPoint(s, from, invP, invT, invK);
	bool be = CreatePlanarMapPoint(e, to, invP, invT, invK);

	cv::Mat normal2 = s - e;
	normal2 = normal2.rowRange(0, 3);
	float norm2 = sqrt(normal2.dot(normal2));
	normal2 /= norm2;
	auto normal3 = normal1.cross(normal2);
	float norm3 = sqrt(normal3.dot(normal3));
	normal3 /= norm3;

	cv::Mat matDist = normal3.t()*s.rowRange(0, 3);

	normal3.push_back(-matDist);
	CheckWallNormal(normal3);
	return normal3;
}

cv::Mat UVR_SLAM::PlaneInformation::PlaneWallEstimator(cv::Vec4i line, cv::Mat normal1, cv::Mat invP, cv::Mat invT, cv::Mat invK) {
	
	Point2f from(line[0], line[1]);
	Point2f to(line[2], line[3]);
	cv::Mat s, e;
	bool bs = CreatePlanarMapPoint(s, from, invP, invT, invK);
	bool be = CreatePlanarMapPoint(e, to, invP, invT, invK);

	cv::Mat normal2 = s - e;
	normal2 = normal2.rowRange(0, 3);
	float norm2 = sqrt(normal2.dot(normal2));
	normal2 /= norm2;
	auto normal3 = normal1.cross(normal2);
	float norm3 = sqrt(normal3.dot(normal3));
	normal3 /= norm3;

	cv::Mat matDist = normal3.t()*s.rowRange(0, 3);

	normal3.push_back(-matDist);
	CheckWallNormal(normal3);
	return normal3;  
}
//3차원값? 4차원으로?ㄴㄴ
 bool UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(cv::Mat& res, cv::Point2f pt, cv::Mat invP, cv::Mat invT, cv::Mat invK){
	cv::Mat temp1 = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	temp1 = invK*temp1;
	float depth = -invP.at<float>(3)/temp1.dot(invP.rowRange(0, 3));
	//cv::Mat matDepth1 = -invP.at<float>(3) / (invP.rowRange(0, 3).t()*temp1);
	//float depth = matDepth1.at<float>(0);
	
	if (depth <= 0.0){
		std::cout << "aaa depth : " << depth << std::endl;
		//return false;
	}
	temp1 *= depth;
	temp1.push_back(cv::Mat::ones(1, 1, CV_32FC1));
	cv::Mat estimated = invT*temp1;
	res = estimated.rowRange(0, 3);
	return true;
}

bool UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(cv::Mat& res, cv::Point2f pt, cv::Mat invP, cv::Mat invK, cv::Mat Rinv, cv::Mat Tinv, float max_thresh) {
	 cv::Mat temp1 = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	 temp1 = invK*temp1;
	 float depth = -invP.at<float>(3) / temp1.dot(invP.rowRange(0, 3));

	 temp1 *= depth;
	 res = Rinv*temp1 + Tinv;

	 if (depth <= 0.0 || depth >= max_thresh) {
		 return false;
	 }
	 return true;
 }

void UVR_SLAM::PlaneInformation::CreatePlanarMapPoints(Frame* pF, System* pSystem) {
	
	//int nTargetID = pF->GetFrameID();
	//cv::Mat invT, invK, invP;
	//pF->mpPlaneInformation->Calculate();
	//pF->mpPlaneInformation->GetInformation(invP, invT, invK);
	//auto pPlane = pF->mpPlaneInformation->GetFloorPlane();
	//
	//auto mvpMPs = pF->GetMapPoints();
	//auto mvpOPs = pF->GetObjectVector();
	//
	//float minDepth = FLT_MAX;
	//float maxDepth = 0.0f;
	////create new mp in current frame
	//for (int j = 0; j < mvpMPs.size(); j++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//	auto oType = mvpOPs[j];

	//	if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR)
	//		continue;
	//	cv::Point2f pt = pF->mvKeyPoints[j].pt;
	//	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	//	temp = invK*temp;
	//	cv::Mat matDepth = -invP.at<float>(3) / (invP.rowRange(0, 3).t()*temp);
	//	float depth = matDepth.at<float>(0);
	//	if (depth < 0) {
	//		//depth *= -1.0;
	//		continue;
	//	}
	//	temp *= depth;
	//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	//	/*if (maxDepth < depth)
	//	maxDepth = depth;
	//	if (minDepth > depth)
	//	minDepth = depth;*/

	//	cv::Mat estimated = invT*temp;
	//	if (pMP) {
	//		pMP->SetWorldPos(estimated.rowRange(0, 3));
	//	}
	//	else {
	//		UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(pF, estimated.rowRange(0, 3), pF->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
	//		pNewMP->SetPlaneID(pPlane->mnPlaneID);
	//		pNewMP->SetObjectType(pPlane->mnPlaneType);
	//		pNewMP->AddFrame(pF, j);
	//		pNewMP->UpdateNormalAndDepth();
	//		pNewMP->mnFirstKeyFrameID = pF->GetKeyFrameID();
	//		pSystem->mlpNewMPs.push_back(pNewMP);
	//		//pPlane->mvpMPs.push_back(pNewMP);
	//		pPlane->tmpMPs.push_back(pNewMP);
	//		//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
	//	}
	//}
	//
	///*cv::Mat R, t;
	//mpTargetFrame->GetPose(R, t);

	//for (int j = 0; j < mvpMPs.size(); j++) {
	//UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//auto oType = mvpOPs[j];
	//if (!pMP)
	//continue;
	//if (pMP->isDeleted())
	//continue;
	//if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR) {
	//cv::Mat X3D = pMP->GetWorldPos();
	//cv::Mat Xcam = R*X3D + t;
	//float depth = Xcam.at<float>(2);
	//if (depth < 0.0 || depth > maxDepth)
	//pMP->SetDelete(true);
	//}
	//}

	//mpTargetFrame->SetDepthRange(minDepth, maxDepth);*/
}
void UVR_SLAM::PlaneInformation::CreateDensePlanarMapPoint(std::vector<cv::Mat>& vX3Ds, cv::Mat label_map, Frame* pF, int nPatchSize) {
	//dense_map = cv::Mat::zeros(pF->mnMaxX, pF->mnMaxY, CV_32FC3);
	//cv::resize(label_map, label_map, pF->GetOriginalImage().size());
	int nTargetID = pF->mnFrameID;
	cv::Mat invT, invPfloor, invK;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);

	auto pFloor = pF->mpPlaneInformation->GetPlane(1);
	cv::Mat matFloorParam = pFloor->GetParam();
	cv::Mat normal;
	float dist;
	pFloor->GetParam(normal, dist);

	int inc = nPatchSize / 2;
	int mX = pF->mnMaxX - inc;
	int mY = pF->mnMaxY - inc;
	for (int x = inc; x < mX; x += nPatchSize) {
		for (int y = inc; y < mY; y += nPatchSize) {
			cv::Point2f pt(x, y);
			int label = label_map.at<uchar>(y, x);
			if (label == 150) {
				cv::Mat X3D;
				bool bRes = PlaneInformation::CreatePlanarMapPoint(pt, invPfloor, invT, invK, X3D);
				if (bRes)
				{
					vX3Ds.push_back(X3D);
					//dense_map.at<Vec3f>(y, x) = cv::Vec3f(X3D.at<float>(0), X3D.at<float>(1), X3D.at<float>(2));
				}
			}
		}
	}
}
void UVR_SLAM::PlaneInformation::CreateDenseWallPlanarMapPoint(std::vector<cv::Mat>& vX3Ds, cv::Mat label_map, Frame* pF, WallPlane* pWall, Line* pLine, int nPatchSize) {

	//dense_map = cv::Mat::zeros(pF->mnMaxX, pF->mnMaxY, CV_32FC3);
	//cv::resize(label_map, label_map, pF->GetOriginalImage().size());
	int nTargetID = pF->mnFrameID;
	cv::Mat invT, invPfloor, invK;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);
	cv::Mat invPwawll = invT.t()*pWall->GetParam();

	auto pFloor = pF->mpPlaneInformation->GetPlane(1);
	cv::Mat matFloorParam = pFloor->GetParam();
	cv::Mat normal;
	float dist;
	pFloor->GetParam(normal, dist);

	int inc = nPatchSize / 2;
	int mX = pF->mnMaxX - inc;
	int mY = pF->mnMaxY - inc;

	int sX = (pLine->to.x / nPatchSize)*nPatchSize+nPatchSize;
	int eX = (pLine->from.x / nPatchSize)*nPatchSize;

	for (int x = sX; x <= eX; x += nPatchSize) {
		for (int y = inc; y < mY; y += nPatchSize) {
			cv::Point2f pt(x, y);
			int label = label_map.at<uchar>(y, x);
			if (label == 255) {
				/*if (x < pLine->to.x || x > pLine->from.x)
					continue;*/
				cv::Mat temp = (cv::Mat_<float>(3, 1) << x, y, 1);
				temp = invK*temp;
				cv::Mat matDepth = -invPwawll.at<float>(3) / (invPwawll.rowRange(0, 3).t()*temp);
				float depth = matDepth.at<float>(0);
				if (depth < 0.0) {
					//depth *= -1.0;
					continue;
				}
				temp *= depth;
				temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

				cv::Mat estimated = invT*temp;
				estimated = estimated.rowRange(0, 3);
				//check on floor
				float val = estimated.dot(normal) + dist;

				if (val < 0.0)
					continue;
				vX3Ds.push_back(estimated);
				//dense_map.at<Vec3f>(y, x) = cv::Vec3f(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
			}
		}
	}
}
void UVR_SLAM::PlaneInformation::CreateDensePlanarMapPoint(cv::Mat& dense_map, cv::Mat label_map, Frame* pF, WallPlane* pWall, Line* pLine, int nPatchSize) {
	
	//dense_map = cv::Mat::zeros(pF->mnMaxX, pF->mnMaxY, CV_32FC3);
	//cv::resize(label_map, label_map, pF->GetOriginalImage().size());
	int nTargetID = pF->mnFrameID;
	cv::Mat invT, invPfloor, invK;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);
	cv::Mat invPwawll = invT.t()*pWall->GetParam();

	auto pFloor = pF->mpPlaneInformation->GetPlane(1);
	cv::Mat matFloorParam = pFloor->GetParam();
	cv::Mat normal;
	float dist;
	pFloor->GetParam(normal, dist);

	int inc = nPatchSize / 2;
	int mX = pF->mnMaxX - inc;
	int mY = pF->mnMaxY - inc;
	for (int x = inc; x < mX; x += nPatchSize) {
		for (int y = inc; y < mY; y+= nPatchSize) {
			cv::Point2f pt(x, y);
			int label = label_map.at<uchar>(y,x);
			if (label == 255) {
				if (x < pLine->to.x || x > pLine->from.x)
					continue;
				cv::Mat temp = (cv::Mat_<float>(3, 1) << x, y, 1);
				temp = invK*temp;
				cv::Mat matDepth = -invPwawll.at<float>(3) / (invPwawll.rowRange(0, 3).t()*temp);
				float depth = matDepth.at<float>(0);
				if (depth < 0.0) {
					//depth *= -1.0;
					continue;
				}
				temp *= depth;
				temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

				cv::Mat estimated = invT*temp;
				estimated = estimated.rowRange(0, 3);
				//check on floor
				float val = estimated.dot(normal) + dist;

				if (val < 0.0)
					continue;

				dense_map.at<Vec3f>(y, x) = cv::Vec3f(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
			}
		}
	}
}

void UVR_SLAM::PlaneInformation::CreateWallMapPoints(Frame* pF, WallPlane* pWall, Line* pLine, std::vector<cv::Mat>& vPlanarMaps, System* pSystem) {

	//int nTargetID = pF->GetFrameID();
	//cv::Mat invT, invPfloor, invK;
	//pF->mpPlaneInformation->Calculate();
	//pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);
	//cv::Mat invPwawll = invT.t()*pWall->GetParam();

	//auto pFloor = pF->mpPlaneInformation->GetFloorPlane();
	//cv::Mat matFloorParam = pFloor->GetParam();
	//cv::Mat normal;
	//float dist;
	//pFloor->GetParam(normal, dist);

	//auto mvpMPs = pF->GetMapPoints();
	//auto mvpOPs = pF->GetObjectVector();
	//int count = 0;
	////create new mp in current frame
	//for (int j = 0; j < mvpMPs.size(); j++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//	if (pMP)
	//		continue;
	//	auto oType = mvpOPs[j];
	//	if (oType != UVR_SLAM::ObjectType::OBJECT_WALL)
	//		continue;
	//	cv::Point2f pt = pF->mvKeyPoints[j].pt;
	//	int x = pt.x;
	//	if (x < pLine->to.x || x > pLine->from.x)
	//		continue;
	//	
	//	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	//	temp = invK*temp;
	//	cv::Mat matDepth = -invPwawll.at<float>(3) / (invPwawll.rowRange(0, 3).t()*temp);
	//	float depth = matDepth.at<float>(0);
	//	if (depth < 0.0) {
	//		//depth *= -1.0;
	//		continue;
	//	}
	//	temp *= depth;
	//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	//	cv::Mat estimated = invT*temp;
	//	estimated = estimated.rowRange(0, 3);
	//	//check on floor
	//	float val = estimated.dot(normal) + dist;
	//	
	//	if (val < 0.0)
	//		continue;
	//	count++;
	//	std::cout <<"planar::"<< val << estimated.t()<< std::endl;
	//	vPlanarMaps[j] = estimated.clone();

	//	//if (pMP) {
	//	//	pMP->SetWorldPos(estimated.rowRange(0, 3));
	//	//}
	//	//else {
	//	//	UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(pF, estimated.rowRange(0, 3), pF->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
	//	//	pNewMP->SetPlaneID(0);
	//	//	pNewMP->SetObjectType(UVR_SLAM::ObjectType::OBJECT_WALL);
	//	//	pNewMP->AddFrame(pF, j);
	//	//	pNewMP->UpdateNormalAndDepth();
	//	//	pNewMP->mnFirstKeyFrameID = pF->GetKeyFrameID();
	//	//	pSystem->mlpNewMPs.push_back(pNewMP);
	//	//	//pPlane->mvpMPs.push_back(pNewMP);
	//	//	//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
	//	//}
	//}
	//std::cout << "wall::cre::" << count << std::endl;

}

//vector size = keypoint size
void UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(Frame* pTargetF, PlaneInformation* pFloor, std::vector<cv::Mat>& vPlanarMaps) {
	
	//cv::Mat invP, invT, invK;
	//pTargetF->mpPlaneInformation->Calculate();
	//pTargetF->mpPlaneInformation->GetInformation(invP, invT, invK);

	//auto mvpMPs = pTargetF->GetMapPoints();
	//auto mvpOPs = pTargetF->GetObjectVector();
	//
	////매칭 확인하기
	//for (int i = 0; i < pTargetF->mvKeyPoints.size(); i++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[i];
	//	if (mvpOPs[i] != ObjectType::OBJECT_FLOOR)
	//		continue;
	//	/*if (pMP && pMP->isDeleted()) {
	//		vPlanarMaps[i] = pMP->GetWorldPos();
	//		continue;
	//	}*/
	//	cv::Mat X3D;
	//	bool bRes = PlaneInformation::CreatePlanarMapPoint(pTargetF->mvKeyPoints[i].pt, invP, invT, invK, X3D);
	//	if (bRes)
	//		vPlanarMaps[i] = X3D.clone();
	//}
}
//////////코드 백업
////////////////////////////////////////////////////////////////////////


void UVR_SLAM::PlaneEstimator::SetPlaneParam(PlaneInformation* pParam){
	std::unique_lock<std::mutex> lockTemp(mMutexFloorPlaneParam);
	mpFloorPlaneInformation->mbInit = pParam->mbInit;
	mpFloorPlaneInformation->SetParam(pParam->GetParam());
}
UVR_SLAM::PlaneInformation* UVR_SLAM::PlaneEstimator::GetPlaneParam(){
	std::unique_lock<std::mutex> lockTemp(mMutexFloorPlaneParam);
	return mpFloorPlaneInformation;
}

void UVR_SLAM::PlaneEstimator::SetTempPTs(Frame* pKF, std::vector<UVR_SLAM::FrameGrid*> vGrids, std::vector<cv::Mat> vPts){
	std::unique_lock<std::mutex> lockTemp(mMutexVisPt);
	mpTempFrame = pKF;
	mvTempPTs = std::vector<cv::Mat>(vPts.begin(), vPts.end());
	mvTempGrids = std::vector<UVR_SLAM::FrameGrid*>(vGrids.begin(), vGrids.end());
}

void UVR_SLAM::PlaneEstimator::GetTempPTs(Frame*& pKF, std::vector<UVR_SLAM::FrameGrid*>& vGrids, std::vector<cv::Mat>& vPts){
	std::unique_lock<std::mutex> lockTemp(mMutexVisPt);
	pKF = mpTempFrame;
	vPts = std::vector<cv::Mat>(mvTempPTs.begin(), mvTempPTs.end());
	vGrids = std::vector<UVR_SLAM::FrameGrid*>(mvTempGrids.begin(), mvTempGrids.end());
}

void UVR_SLAM::PlaneEstimator::GetTempPTs(std::vector<cv::Mat>& vPts) {
	std::unique_lock<std::mutex> lockTemp(mMutexVisPt);
	vPts = std::vector<cv::Mat>(mvTempPTs.begin(), mvTempPTs.end());
}