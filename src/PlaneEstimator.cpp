#include <PlaneEstimator.h>
#include <random>
#include <System.h>
#include <Map.h>
#include <Plane.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <SegmentationData.h>
#include <MapPoint.h>
#include <Matcher.h>
#include <Initializer.h>
#include <MatrixOperator.h>

static int nPlaneID = 0;

UVR_SLAM::PlaneEstimator::PlaneEstimator() :mbDoingProcess(false), mnProcessType(0), mpLayoutFrame(nullptr){
}
UVR_SLAM::PlaneEstimator::PlaneEstimator(Map* pMap,std::string strPath,cv::Mat K, cv::Mat K2, int w, int h) : mK(K), mK2(K2),mbDoingProcess(false), mnWidth(w), mnHeight(h), mnProcessType(0), mpLayoutFrame(nullptr),
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

	mpMap = pMap;
}
void UVR_SLAM::PlaneInformation::SetParam(cv::Mat n, float d){
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
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
void UVR_SLAM::PlaneEstimator::SetMatcher(UVR_SLAM::Matcher* pMatcher) {
	mpMatcher = pMatcher;
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
	mpPPrevFrame = mpPrevFrame;
	mpPrevFrame = mpTargetFrame;
	mpTargetFrame = mKFQueue.front();
	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_LAYOUT_FRAME);
	mpSystem->SetPlaneFrameID(mpTargetFrame->GetKeyFrameID());

	/*if (mpMap->isFloorPlaneInitialized()) {
		mpTargetFrame->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpTargetFrame, mpMap->mpFloorPlane);
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

	std::string mStrPath;

	std::vector<UVR_SLAM::PlaneInformation*> mvpPlanes;

	UVR_SLAM::Frame* pTestF = nullptr;

	while (1) {
		if (CheckNewKeyFrames()) {
			
			//저장 디렉토리 명 획득
			SetBoolDoingProcess(true,0);
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			ProcessNewKeyFrame();
			std::cout << "pe::start::"<<mpTargetFrame->GetKeyFrameID()<< std::endl;
			if (!mpPrevFrame)
				std::cout << "null::prev" << std::endl;
			if (!mpPPrevFrame)
				std::cout << "null::pprev" << std::endl;
			//////평면 포인트 생성
			std::vector<cv::Mat> vPlanarMaps;
			vPlanarMaps = std::vector<cv::Mat>(mpTargetFrame->mvKeyPoints.size(), cv::Mat::zeros(0, 0, CV_8UC1));
			
			//초기화 후에 진행되기 때문에 이제는 초기화가 안되어있으면 그냥 리턴해도 됨.
			if (!mpMap->isFloorPlaneInitialized()) {
				////unlock & notify
				std::unique_lock<std::mutex> lockPlanar(mpSystem->mMutexUsePlaneEstimation);
				mpSystem->mbPlaneEstimationEnd = true;
				lockPlanar.unlock();
				mpSystem->cvUsePlaneEstimation.notify_all();
				std::cout << "pe::end::s::" << mpTargetFrame->GetKeyFrameID() << std::endl;
				continue;
			}

			mStrPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());
			//현재 레이아웃 추정하는 키프레임 설정
			int nTargetID = mpTargetFrame->GetFrameID();
			mpFrameWindow->SetLastLayoutFrameID(nTargetID);
			cv::Mat vImg = mpTargetFrame->GetOriginalImage();
			//matching test

			//////////////////////////////////////////////////////////////////////////////
			//////////라인 검출
			//바닥 및 벽 정보와 추후 결합. 
			
			//////////라인 검출
			//////////////////////////////////////////////////////////////////////////////
			std::chrono::high_resolution_clock::time_point line_start = std::chrono::high_resolution_clock::now();
			/*cv::Mat img = mpTargetFrame->GetOriginalImage();
			cv::Mat gray, filtered, edge;
			std::vector<cv::Vec4i> vecLines;
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			GaussianBlur(gray, filtered, cv::Size(5, 5), 0.0);
			Canny(filtered, edge, 100, 200);
			cv::HoughLinesP(edge, vecLines, 1, CV_PI / 180.0, 30, 30, 3);*/
			
			//imshow("line", img);
			std::chrono::high_resolution_clock::time_point line_end = std::chrono::high_resolution_clock::now();
			auto lineduration = std::chrono::duration_cast<std::chrono::milliseconds>(line_end - line_start).count();
			float line_time = lineduration / 1000.0;

			///////////////////////////////////////////
			////이전 키프레임에서 추정한 맵포인트 업데이트
			//int nUpdateT
			int nPrevTest = 0;
			bool bInitFloorPlane = mpMap->isFloorPlaneInitialized();//true;
			auto pFloor = mpMap->mpFloorPlane;
			int nPrevTest2 = 0;
			cv::Mat pmat;
			if (bInitFloorPlane) {

				////update dense map
				auto mvpDenseMPs = mpPrevFrame->GetDenseVectors();
				cv::Mat debugging;
				std::vector<std::pair<int, cv::Point2f>> vPairs;
				std::vector<bool> vbInliers;
				int mnDenseMatching = mpMatcher->DenseMatchingWithEpiPolarGeometry(mpTargetFrame, mpPrevFrame, mvpDenseMPs, vPairs, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugging);
				int nInc = 1;
				if (vPairs.size() > 1000) {
					nInc = 20;
				}
				for (int i = 0; i < vPairs.size(); i+= nInc) {
					auto idx = vPairs[i].first;
					auto pt = vPairs[i].second;
					mpTargetFrame->AddDenseMP(mvpDenseMPs[idx], pt);
				}
				////update dense map

				////이전 플라나 포인트로 생성된 포인트에 대해서 트래킹에 성공한 포인트에 한해서 현재 평면 벡터에 포함시킴.
				for (int i = 0; i < pFloor->tmpMPs.size(); i++) {
					UVR_SLAM::MapPoint* pMP = pFloor->tmpMPs[i];
					if (!pMP)
						continue;
					if (pMP->isDeleted())
						continue;
					if (pMP->GetNumConnectedFrames() > 1){
						pFloor->mvpMPs.push_back(pMP);
						nPrevTest2++;
					}
				}
				pFloor->tmpMPs.clear();
				UpdatePlane(pFloor, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
				pmat = pFloor->GetParam().t();
			}else
				pmat = cv::Mat::zeros(1, 4, CV_32FC1);
			////이전 키프레임에서 추정한 맵포인트 업데이트
			///////////////////////////////////////////
			
			//////////////////////////////////////////////////////////////
			////lock
			//local 맵에서 키포인트 생성 전에 막기 위함
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUsePlaneEstimation);
			mpSystem->mbPlaneEstimationEnd = false;

			////labeling 끝날 때까지 대기
			{
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseSegmentation);
				while (!mpSystem->mbSegmentationEnd) {
					mpSystem->cvUseSegmentation.wait(lock);
				}
			}
			////labeling 끝날 때까지 대기

			std::cout << "pe::2" << std::endl;
			////////////////////////////////////////////////////////////////////////
			///////////////Conneted Component Labeling & object labeling
			//image
			cv::Mat segmented = mpTargetFrame->matLabeled.clone();
			std::cout <<"seg::size::"<< segmented.size() << std::endl;
			std::chrono::high_resolution_clock::time_point saa = std::chrono::high_resolution_clock::now();
			cv::Mat imgStructure = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgWall = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgFloor = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgCeil = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgObject = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat seg_color = cv::Mat::zeros(segmented.size(), CV_8UC3);
			int minY = imgWall.rows;
			int maxY = 0;
			int minFloorX = imgWall.cols;
			int maxFloorX = 0;
			int minCeilX = imgWall.cols;
			int maxCeilX = 0;

			bool bMinX = false;
			bool bMaxX = false;
			bool bMinCeilY = false;
			bool bMaxFloorY = false;

			//////////////////////////////////////////////////////////////////////////////////
			////전체 레이블링

			for (int i = 0; i < segmented.rows; i++) {
				for (int j = 0; j < segmented.cols; j++) {
					seg_color.at<Vec3b>(i, j) = UVR_SLAM::ObjectColors::mvObjectLabelColors[segmented.at<uchar>(i, j)];
					int val = segmented.at<uchar>(i, j);
					switch (val) {
					case 255:
						imgWall.at<uchar>(i, j) = 255;
						imgStructure.at<uchar>(i, j) = 255;
						break;
					case 150:
						imgFloor.at<uchar>(i, j) = 255;
						imgStructure.at<uchar>(i, j) = 150;
						break;
					case 100:
						imgStructure.at<uchar>(i, j) = 100;
						imgCeil.at<uchar>(i, j) = 255;
						break;
					case 20:
						break;
					case 50:
						imgObject.at<uchar>(i, j) = 255;
						imgStructure.at<uchar>(i, j) = 50;
						break;

					}
				}
			}
			////
			//////////////////////////////////////////////////////////////////////////////////
			
			////////////////////////////////
			//Dense Map 관련 설정
			mpTargetFrame->mDenseMap = cv::Mat::zeros(mpTargetFrame->GetOriginalImage().size(), CV_32FC3);
			cv::resize(imgStructure, imgStructure, mpTargetFrame->mDenseMap.size());

			////기존의 칼라이미지와 세그멘테이션 결과를 합치는 부분
			////여기는 시각화로 보낼 수 있으면 보내는게 좋을 듯.
			/*cv::resize(seg_color, seg_color, colorimg.size());
			cv::addWeighted(seg_color, 0.4, colorimg, 0.6, 0.0, colorimg);
			cv::imshow("object : ", imgObject);*/
			cv::imshow("Output::Segmentation", seg_color);
			////기존의 칼라이미지와 세그멘테이션 결과를 합치는 부분

			std::cout << "pe::3" << std::endl;
			////////////////////바닥과 벽을 나누는 함수.
			////https://webnautes.tistory.com/823
			cv::Mat floorCCL, ceilCCL, sumCCL;
			cv::Mat floorStat, ceilStat;
			ConnectedComponentLabeling(imgFloor, floorCCL, floorStat);
			ConnectedComponentLabeling(imgCeil, ceilCCL, ceilStat);
			sumCCL = floorCCL + ceilCCL;
			maxY = floorStat.at<int>(CC_STAT_TOP);
			minY = ceilStat.at<int>(CC_STAT_TOP) + ceilStat.at<int>(CC_STAT_HEIGHT);
			if (minY != segmented.rows)
				bMinCeilY = true;
			if (maxY != 0)
				bMaxFloorY = true;
			////////////////////바닥과 벽을 나누는 함수.
			///////////////Conneted Component Labeling & object labeling
			////////////////////////////////////////////////////////////////////////
			

			std::cout << "pe::4" << std::endl;
			////////////////////////////////////////////////////////////////////////
			//////바닥, 벽이 만나는 지점을 바탕으로 라인을 추정
			////추정된 라인은 가상의 포인트를 만들고 테스트 하는데 이용
			PlaneInformation* pTestWall = new PlaneInformation();
			std::vector<cv::Vec4i> tlines;
			std::vector<Line*> lines;
			{
				
				cv::Ptr<cv::LineSegmentDetector> pLSD = createLineSegmentDetector();
				pLSD->detect(floorCCL, tlines);//pLSD->detect(tempFloor, tlines);
				//graph.convertTo(graph, CV_8UC3);
				//cv::cvtColor(graph, graph, CV_GRAY2BGR);
				for (int i = 0; i < tlines.size(); i++) {
					Vec4i v = tlines[i];
					Point2f from(v[0]*2, v[1] * 2);
					Point2f to(v[2] * 2, v[3] * 2);
					Point2f diff = from - to;
					float dist = sqrt(diff.dot(diff));
					/*if (tempWallLines.at<uchar>(from) == 0 || tempWallLines.at<uchar>(to) == 0)
						continue;*/
					if (dist < 25*2)
						continue;
					if (to.y < maxY+3 || from.y < maxY + 3)
						continue;
					float slope = abs(LineProcessor::CalcSlope(from, to));
					if (slope > 3.0)
						continue;
					Line* line = new Line(mpTargetFrame, from, to);
					lines.push_back(line);
					//cv::line(fLineImg, from, to, cv::Scalar(255,0,0));
					//else
					//cv::line(graph, from, to, cv::Scalar(0, 0, 255));
					cv::line(vImg, line->from, line->to, cv::Scalar(0,255,255),2);
				}
				//해당 라인은 타겟 프레임에 설정함.
				mpTargetFrame->SetLines(lines);
				
			}
			
			std::cout << "pe::5" << std::endl;
			//////////////////////////////////////////////////////////////
			//////LINE 추가 과정.
			//WallLineTest
			if (bInitFloorPlane) {
				//바닥 평면 맵 생성
				UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(mpTargetFrame, pFloor, vPlanarMaps);
				//바닥 덴스 평면 생성
				//UVR_SLAM::PlaneInformation::CreateDensePlanarMapPoint(mpTargetFrame->mDenseMap, imgStructure, mpTargetFrame, mpSystem->mnPatchSize);
				UVR_SLAM::PlaneInformation::CreateDensePlanarMapPoint(mpTargetFrame->mvX3Ds, imgStructure, mpTargetFrame, mpSystem->mnPatchSize);

				bool bInitWallPlane = mpMap->isWallPlaneInitialized();
				auto mvFrames = mpTargetFrame->GetConnectedKFs(10);
				auto mvWallPlanes = mpMap->GetWallPlanes();

				cv::Mat normal1;
				float dist1;
				pFloor->GetParam(normal1, dist1);
				cv::Mat planeParam = pFloor->GetParam().clone();
				
				///연결된 프레임에 대해서 정보 업데이트
				for (int i = 0; i < mvFrames.size(); i++) {
					/*if (!mvFrames[i]->mpPlaneInformation)
						mvFrames[i]->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mvFrames[i], plane);*/
					if (!mvFrames[i]->mpPlaneInformation){
						std::cout << "error!!!!"<<mpTargetFrame->GetKeyFrameID()<<", "<< mvFrames[i]->GetKeyFrameID() << std::endl;

					}
					mvFrames[i]->mpPlaneInformation->Calculate();
				}

				//현재 프레임의 라인에 대해서 기존 벽 평면과 비교
				if (bInitWallPlane) {
					////update wall plane parameters
					for (int i = 0; i < mvWallPlanes.size(); i++) {

						auto lines = mvWallPlanes[i]->GetLines();
						for (int j = 0; j < lines.size(); j++) {
							lines[j]->SetLinePts();
						}
						cv::Mat parama = UVR_SLAM::PlaneInformation::PlaneLineEstimator(mvWallPlanes[i], pFloor);
						//이걸 이제 벽으로 변환해야 함.

						/*
						cv::Mat R, t;
						line->mpFrame->GetPose(R, t);

						auto plane = line->mpFrame->mvpPlanes[0];
						cv::Mat normal1;
						float dist1;
						plane->GetParam(normal1, dist1);
						cv::Mat planeParam = plane->matPlaneParam.clone();
						
						Point2f from = line->from;
						Point2f to = line->to;
						cv::Mat s = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(from, invP, invT, invK);
						cv::Mat e = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(to, invP, invT, invK);
						cv::Mat param = UVR_SLAM::PlaneInformation::PlaneWallEstimator(s,e,normal1, invP, invT, invK);*/
						
						/*Line* line = mvWallPlanes[i]->GetLines()[0];
						cv::Mat invT, invK, invP;
						line->mpFrame->mpPlaneInformation->GetInformation(invP, invT, invK);
						cv::Mat param = UVR_SLAM::PlaneInformation::PlaneWallEstimator(line, normal1, invP, invT, invK);*/
						mvWallPlanes[i]->SetParam(parama);
					}
					////update wall plane parameters
				}

				//현재 라인으로 임시 벽을 생성함.
				//기존의 벽과 비교.
				{
					auto mvpCurrLines = mpTargetFrame->Getlines();
					//std::cout << "wall::line::" << mvpCurrLines.size() << std::endl;
					
					cv::Mat invCurrP, invCurrPlane, invK;
					mpTargetFrame->mpPlaneInformation->Calculate();
					mpTargetFrame->mpPlaneInformation->GetInformation(invCurrPlane, invCurrP, invK);

					//현재 프레임으로부터 정보 생성
					for (int li = 0; li < mvpCurrLines.size(); li++) {
						cv::Mat param = UVR_SLAM::PlaneInformation::PlaneWallEstimator(mvpCurrLines[li], normal1, invCurrPlane, invCurrP, invK);

						if (mvWallPlanes.size() == 0) {
							UVR_SLAM::WallPlane* tempWallPlane = new UVR_SLAM::WallPlane();
							tempWallPlane->SetParam(param);
							tempWallPlane->AddLine(mvpCurrLines[li]);
							tempWallPlane->AddFrame(mpTargetFrame);
							tempWallPlane->SetRecentKeyFrameID(mpTargetFrame->GetKeyFrameID());
							mvWallPlanes.push_back(tempWallPlane);
						}
						else {
							bool bInsert = true;
							bool bConnect = false;
							float minDist = FLT_MAX;
							int   minIdx = 0;
							for (int j = 0; j < mvWallPlanes.size(); j++) {
								cv::Mat wParam = mvWallPlanes[j]->GetParam();
								float normal = PlaneInformation::CalcCosineSimilarity(wParam, param);
								float dist = PlaneInformation::CalcPlaneDistance(wParam, param);
								/*if(mvWallPlanes[j]->mnPlaneID>0)
									std::cout << "1::"<< mvWallPlanes[j]->mnPlaneID <<"::"<< normal << ", " << dist << std::endl;
								else
									std::cout << "2::" << normal <<", " << dist << std::endl;*/

									////아예 노말이 다르다.
									////노말이
								if (normal > 0.97) {
									//겹치는 것/

									if (dist < 0.02 && normal > 0.9995) {
										bInsert = false;
										bConnect = true;
										if (dist < minDist) {
											minIdx = j;
											minDist = dist;
										}
										
										//std::cout << "connect!!" << std::endl;
										//break;
									}
									else if (dist < 0.1 || normal < 0.9995) {
										bInsert = false;
										//break;
									}

									/*bInsert = false;
									mvWallPlanes[j]->AddLine(mvpCurrLines[li]);
									mvWallPlanes[j]->AddFrame(mpTargetFrame);
									mvWallPlanes[j]->SetRecentKeyFrameID(mpTargetFrame->GetKeyFrameID());
									break;*/
								}
								//if (normal > 0.97 && dist < 0.01) {
								//	//겹치는 것/
								//	bInsert = false;
								//	mvWallPlanes[j]->AddLine(mvpCurrLines[li]);
								//	mvWallPlanes[j]->AddFrame(mpTargetFrame);
								//	mvWallPlanes[j]->SetRecentKeyFrameID(mpTargetFrame->GetKeyFrameID());
								//	break;
								//}
								//else if (normal > 0.97 && (dist <0.1 && dist >= 0.01)) {
								//	//노이즈를 줄이기 위한 것.
								//	//얘는 추가x
								//	bInsert = false;
								//}
							}
							if (bInsert) {
								//std::cout << "insert!!!!" << std::endl;
								UVR_SLAM::WallPlane* tempWallPlane = new UVR_SLAM::WallPlane();
								tempWallPlane->SetParam(param);
								tempWallPlane->AddLine(mvpCurrLines[li]);
								tempWallPlane->AddFrame(mpTargetFrame);
								tempWallPlane->SetRecentKeyFrameID(mpTargetFrame->GetKeyFrameID());
								mvWallPlanes.push_back(tempWallPlane);
							}
							else if(bConnect){
								mvWallPlanes[minIdx]->AddLine(mvpCurrLines[li]);
								mvWallPlanes[minIdx]->AddFrame(mpTargetFrame);
								mvWallPlanes[minIdx]->SetRecentKeyFrameID(mpTargetFrame->GetKeyFrameID());
								UVR_SLAM::PlaneInformation::CreateWallMapPoints(mpTargetFrame, mvWallPlanes[minIdx], mvpCurrLines[li], vPlanarMaps, mpSystem);
								cv::line(vImg, mvpCurrLines[li]->from, mvpCurrLines[li]->to, ObjectColors::mvObjectLabelColors[mvWallPlanes[minIdx]->mnPlaneID + 10], 2);

								float m;
								cv::Mat mLine = UVR_SLAM::PlaneInformation::FlukerLineProjection(mvWallPlanes[minIdx]->GetParam(), planeParam, mpTargetFrame->GetRotation(), mpTargetFrame->GetTranslation(), mK2, m);
								cv::Point2f sPt, ePt;
								UVR_SLAM::PlaneInformation::CalcFlukerLinePoints(sPt, ePt, 0.0, mnHeight, mLine);
								cv::line(vImg, sPt, ePt, ObjectColors::mvObjectLabelColors[mvWallPlanes[minIdx]->mnPlaneID + 10], 1);

								UVR_SLAM::PlaneInformation::CreateDenseWallPlanarMapPoint(mpTargetFrame->mvX3Ds, imgStructure, mpTargetFrame, mvWallPlanes[minIdx], mvpCurrLines[li], mpSystem->mnPatchSize);

								//if (!pTestF) {
								//	pTestF = mpTargetFrame;
								//	UVR_SLAM::PlaneInformation::CreateDensePlanarMapPoint(pTestF->mDenseMap, imgStructure, pTestF, mvWallPlanes[minIdx], mvpCurrLines[li], mpSystem->mnPatchSize);
								//	std::string base = mpSystem->GetDirPath(0);
								//	std::stringstream ss;
								//	ss << base << "/dense";
								//	_mkdir(ss.str().c_str());
								//}
								//else {
								//	cv::Mat debugging;
								//	mpMatcher->DenseMatchingWithEpiPolarGeometry(pTestF->mDenseMap, pTestF, mpTargetFrame, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugging);
								//	std::string base = mpSystem->GetDirPath(0);
								//	std::stringstream ss;
								//	ss << base << "/dense/dense_"<<pTestF->GetKeyFrameID() << "_" << mpTargetFrame->GetKeyFrameID() << ".jpg";
								//	imwrite(ss.str(), debugging);
								//}//if

							}//if connect
						}
					}
				}

				std::cout << "pe::6" << std::endl;
				///////새로운 평면 검출
				////현재 프레임은 평면 추가 및 연결
				//이전 프레임들은 연결만 수행하기 
				std::vector<cv::Mat> wallParams;
				std::vector<cv::Mat> twallParams;
				//mvFrames.push_back(mpTargetFrame);
				for (int k = 0; k < mvFrames.size(); k++) {
					
					cv::Mat invP, invK, invT;
					mvFrames[k]->mpPlaneInformation->GetInformation(invP, invT, invK);

					auto mvpLines = mvFrames[k]->Getlines();

					bool bConnect = false;
					int minIdx = 0;
					float minDist = FLT_MAX;
					for (int i = 0; i < mvpLines.size(); i++) {

						if (mvpLines[i]->mnPlaneID > 0)
							continue;
						cv::Mat param = UVR_SLAM::PlaneInformation::PlaneWallEstimator(mvpLines[i], normal1, invP, invT, invK);
						
						cv::Mat s = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(mvpLines[i]->from, invP, invT, invK);
						s.push_back(cv::Mat::ones(1, 1, CV_32FC1));
						cv::Mat e = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(mvpLines[i]->to, invP, invT, invK);
						e.push_back(cv::Mat::ones(1, 1, CV_32FC1));

						for (int j = 0; j < mvWallPlanes.size(); j++) {
							cv::Mat wParam = mvWallPlanes[j]->GetParam();
							float normal = PlaneInformation::CalcCosineSimilarity(wParam, param);
							float dist = PlaneInformation::CalcPlaneDistance(wParam, param);

							/*if (mvWallPlanes[j]->mnPlaneID>0)
								std::cout << "3::" << mvWallPlanes[j]->mnPlaneID << "::" << normal << ", " << dist << std::endl;
							else
								std::cout << "4::" << normal << ", " << dist << std::endl;*/

							//////아예 노말이 다르다.
							//////노말이
							if (normal > 0.9995 && dist < 0.02) {
							//if (normal > 0.99) {
								bConnect = true;
								if(dist < minDist){
									minDist = dist;
									minIdx = j;
								}
								
							}
							//std::cout << "test::" << normal << ", " << dist << std::endl;
							//std::cout << k << "::"<<i<<"=="<<wParam.dot(s) << ", " << wParam.dot(e) << std::endl;
							//std::cout << k << "::" << i << "==" << planeParam.dot(s) << ", " << planeParam.dot(e) << std::endl;

							/*if (abs(planeParam.dot(s)) < 0.01 && abs(planeParam.dot(e)) < 0.01) {
								mvWallPlanes[j]->AddLine(mvpLines[i]);
								mvWallPlanes[j]->AddFrame(mvFrames[k]);
							}*/
							
						}

						if (bConnect) {
							mvWallPlanes[minIdx]->AddLine(mvpLines[i]);
							mvWallPlanes[minIdx]->AddFrame(mvFrames[k]);
						}
					}//for
				}
				///////새로운 평면 검출

				for (int i = 0; i < mvWallPlanes.size(); i++) {
					if (mvWallPlanes[i]->mnPlaneID > 0) {
						std::cout << mvWallPlanes[i]->mnPlaneID << "::" <<mvWallPlanes[i]->GetNumFrames()<<", "<< mvWallPlanes[i]->GetSize() <<"::"<<mvWallPlanes[i]->GetParam().t()<< std::endl;
					}
					/*
					else {
						std::cout << mvWallPlanes[i]->mnPlaneID << "::" << mvWallPlanes[i]->GetNumFrames() << ", " << mvWallPlanes[i]->GetSize() << "::" << mvWallPlanes[i]->GetParam().t() << std::endl;
					}*/
					if (mvWallPlanes[i]->GetSize()> 3 && mvWallPlanes[i]->mnPlaneID < 0) {
						mvWallPlanes[i]->CreateWall();
						mpMap->AddWallPlane(mvWallPlanes[i]);
						if (!mpMap->isWallPlaneInitialized()) {
							mpMap->SetWallPlaneInitialization(true);
						}
					}
				}
				//mpTargetFrame->SetWallParams(twallParams);
			}
			
			//WallLineTest
			//////////////////////////////////////////////////////////////
			
			std::cout << "pe::7" << std::endl;
			////////매칭 테스트
			if (bInitFloorPlane) {

				////////////////////////////////
				//최종 맵포인트 생성 돤계
				///////////////////////////////
				//앞의 프레임이 이전 프레임
				//뒤의 프렝미이 현재 프레임으로 현재프레임에서 포인트를 생성함.

				cv::Mat debugImg;
				
				std::vector<UVR_SLAM::Frame*> mvpKFs;
				mvpKFs.push_back(mpPrevFrame);
				mvpKFs.push_back(mpPPrevFrame);
				for (int ki = 0; ki < mvpKFs.size(); ki++) {
					///////dense
					//cv::Mat debugging;
					////mpMatcher->DenseMatchingWithEpiPolarGeometry(mpTargetFrame, mvpKFs[ki], mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugging);
					//mpMatcher->DenseMatchingWithEpiPolarGeometry(mvpKFs[ki], mpTargetFrame,mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugging);
					//std::string base = mpSystem->GetDirPath(0);
					//std::stringstream ssss;
					//ssss << base << "/dense/dense_"<< mpTargetFrame->GetKeyFrameID() << "_" << mvpKFs[ki]->GetKeyFrameID() << ".jpg";
					//imwrite(ssss.str(), debugging);
					///////dense


					std::vector<cv::DMatch> vMatches;
					std::vector<bool> vbInliers;
					UVR_SLAM::Frame* pKFi = mvpKFs[ki];
					std::cout << "pe::matching::0" << std::endl;
					std::vector<std::pair<int, cv::Point2f>> vPairs;
					mpMatcher->MatchingWithEpiPolarGeometry(pKFi, mpTargetFrame, vPlanarMaps, vbInliers,vMatches, vPairs, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugImg);
					//mpMatcher->DenseMatchingWithEpiPolarGeometry(pKFi, mpTargetFrame, vPlanarMaps, vPairs, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugImg);
					std::cout << "pe::matching::1" << std::endl;
					std::stringstream ss;
					std::cout << "pe::" << mStrPath << std::endl;
					std::cout << "pe::kf::" << pKFi->GetKeyFrameID() << std::endl;
					ss << mpSystem->GetDirPath(0) << "/kfmatching/" << mpTargetFrame->GetKeyFrameID() << "_" << pKFi->GetKeyFrameID() << ".jpg";
					imwrite(ss.str(), debugImg);
					std::cout << "pe::matching::2" << std::endl;

					for (int i = 0; i < vPairs.size(); i++) {
						//기존 평면인지 확인이 어려움.
						auto idx = vPairs[i].first;
						auto pt = vPairs[i].second;

						UVR_SLAM::MapPoint* pMP1 = pKFi->GetDenseMP(pt);
						UVR_SLAM::MapPoint* pMP2 = mpTargetFrame->GetDenseMP(mpTargetFrame->mvKeyPoints[idx].pt);

						bool b1 = pMP1 && !pMP1->isDeleted();
						bool b2 = pMP2 && !pMP2->isDeleted();

						if (!b1 && !b2) {
							UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, vPlanarMaps[idx], mpTargetFrame->matDescriptor.row(idx), UVR_SLAM::PLANE_DENSE_MP);
							pNewMP->SetPlaneID(pFloor->mnPlaneID);
							pNewMP->SetObjectType(pFloor->mnPlaneType);
							pNewMP->AddDenseFrame(pKFi, pt);
							pNewMP->AddDenseFrame(mpTargetFrame, mpTargetFrame->mvKeyPoints[idx].pt);
							pNewMP->UpdateNormalAndDepth();
							pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
							mpSystem->mlpNewMPs.push_back(pNewMP);
							pFloor->tmpMPs.push_back(pNewMP); //이것도 바닥과 벽을 나눌 필요가 있음.
						}
						else if (b1 && b2) {
							if (pMP1->mnMapPointID != pMP2->mnMapPointID) {
								pMP1->Fuse(pMP2);
								pMP2->SetWorldPos(vPlanarMaps[idx]);
							}
						}
						else if (b1) {
							pMP1->AddDenseFrame(mpTargetFrame, mpTargetFrame->mvKeyPoints[idx].pt);
							pMP1->SetWorldPos(vPlanarMaps[idx]);
						}
						else if (b2) {
							pMP2->AddDenseFrame(pKFi, pt);
							pMP2->SetWorldPos(vPlanarMaps[idx]);
						}
						
					}

					for (int i = 0; i < vbInliers.size(); i++) {
						if (vbInliers[i]) {
							int idx1 = vMatches[i].trainIdx;
							int idx2 = vMatches[i].queryIdx;

							UVR_SLAM::MapPoint* pMP1 = pKFi->mvpMPs[idx1];
							UVR_SLAM::MapPoint* pMP2 = mpTargetFrame->mvpMPs[idx2];

							bool b1 = pMP1 && !pMP1->isDeleted();
							bool b2 = pMP2 && !pMP2->isDeleted();

							if (!b1 && !b2)
							{
								UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, vPlanarMaps[idx2], mpTargetFrame->matDescriptor.row(idx2), UVR_SLAM::PLANE_MP);
								pNewMP->SetPlaneID(pFloor->mnPlaneID);
								pNewMP->SetObjectType(pFloor->mnPlaneType);
								pNewMP->AddFrame(pKFi, idx1);
								pNewMP->AddFrame(mpTargetFrame, idx2);
								pNewMP->UpdateNormalAndDepth();
								pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
								mpSystem->mlpNewMPs.push_back(pNewMP);
								pFloor->tmpMPs.push_back(pNewMP);
							}
							else if (b1 && b2) {
								if(pMP1->mnMapPointID != pMP2->mnMapPointID){
									pMP1->Fuse(pMP2);
									pMP2->SetWorldPos(vPlanarMaps[idx2]);
								}
							}
							else if (b1) {
								pMP1->AddFrame(mpTargetFrame, idx2);
								pMP1->SetWorldPos(vPlanarMaps[idx2]);
							}
							else if (b2) {
								pMP2->AddFrame(pKFi, idx1);
								pMP2->SetWorldPos(vPlanarMaps[idx2]);
							}
						}
						else {
							if (vPlanarMaps[i].rows == 0)
								continue;
							/*UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, vPlanarMaps[i], mpTargetFrame->matDescriptor.row(i), UVR_SLAM::PLANE_MP);
							pNewMP->SetPlaneID(pFloor->mnPlaneID);
							pNewMP->SetObjectType(pFloor->mnPlaneType);
							pNewMP->AddFrame(mpTargetFrame, i);
							pNewMP->UpdateNormalAndDepth();
							pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
							mpSystem->mlpNewMPs.push_back(pNewMP);
							pFloor->tmpMPs.push_back(pNewMP);*/
						}
						
					}
					std::cout << "pe::matching::3" << std::endl;
				}
				mpMap->SetCurrFrame(mpTargetFrame);
				//////////////////////////////////////////////////////////////////////////////////


				//std::vector<cv::DMatch> vMatches;
				//////매칭 결과만 획득. 여기에서는 두 프레임 사이의 매칭 결과만 저장. 맵포인트 존재 유무 판단x
				//mpMatcher->MatchingWithEpiPolarGeometry(mpPrevFrame, mpTargetFrame, pFloor, vPlanarMaps, vMatches, debugImg);
				//////fuse와 동일한 과정을 수행함.
				//for (int i = 0; i < vMatches.size(); i++) {
				//	int idx1 = vMatches[i].trainIdx;
				//	int idx2 = vMatches[i].queryIdx;

				//	UVR_SLAM::MapPoint* pMP1 = mpPrevFrame->mvpMPs[idx1];
				//	UVR_SLAM::MapPoint* pMP2 = mpTargetFrame->mvpMPs[idx2];

				//	bool b1 = pMP1 && !pMP1->isDeleted();
				//	bool b2 = pMP2 && !pMP2->isDeleted();

				//	if (!b1 && !b2)
				//	{
				//		UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, vPlanarMaps[idx2], mpTargetFrame->matDescriptor.row(idx2), UVR_SLAM::PLANE_MP);
				//		pNewMP->SetPlaneID(pFloor->mnPlaneID);
				//		pNewMP->SetObjectType(pFloor->mnPlaneType);
				//		pNewMP->AddFrame(mpPrevFrame, idx1);
				//		pNewMP->AddFrame(mpTargetFrame, idx2);
				//		pNewMP->UpdateNormalAndDepth();
				//		pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
				//		mpSystem->mlpNewMPs.push_back(pNewMP);
				//		pFloor->tmpMPs.push_back(pNewMP);
				//	}
				//	else if (b1 && b2) {
				//		if (pMP1->GetConnedtedFrames() > pMP2->GetConnedtedFrames()) {
				//			pMP2->Fuse(pMP1);
				//		}
				//		else
				//			pMP1->Fuse(pMP2);

				//		/*if (pMPinKF->GetConnedtedFrames()>pMP->GetConnedtedFrames())
				//		pMP->Fuse(pMPinKF);
				//		else
				//		pMPinKF->Fuse(pMP);*/
				//	}
				//	else if (b1) {
				//		pMP1->AddFrame(mpTargetFrame, idx2);
				//	}
				//	else if (b2) {
				//		pMP2->AddFrame(mpPrevFrame, idx1);
				//	}
				//}
			}
			////////매칭 테스트
			
			std::cout << "pe::8" << std::endl;
			//////바닥, 벽이 만나는 지점을 바탕으로 라인을 추정
			////추정된 라인은 가상의 포인트를 만들고 테스트 하는데 이용
			////unlock & notify
			mpSystem->mbPlaneEstimationEnd = true;
			lock.unlock();
			mpSystem->cvUsePlaneEstimation.notify_all();
			////////////////////////////////////////////////////////////////////////
						
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto leduration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float letime = leduration / 1000.0;
			std::stringstream ss;
			ss <<std::fixed<< std::setprecision(3) << "Layout:"<<mpTargetFrame->GetKeyFrameID()<<" t=" << letime <<":"<<pmat.at<float>(0)<<" "<<pmat.at<float>(1)<<" "<< pmat.at<float>(2) <<" "<< pmat.at<float>(3) <<"::"<<line_time<<"::"<<lines.size()<<"::"<< nPrevTest<<", "<< nPrevTest2 <<"::"<< mpTargetFrame->mspWallMPs.size() << ", " << mpTargetFrame->mspFloorMPs.size();
			mpSystem->SetPlaneString(ss.str());
			
			//////test
			//////save txt
			/*std::ofstream f;
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
			f.close();*/
			//////save txt
			
			////////////////image
			
			if(bMaxFloorY)
				cv::line(vImg, cv::Point2f(0, maxY * 2), cv::Point2f(vImg.cols, maxY * 2), cv::Scalar(255, 0, 0), 2);
			cv::Mat R, t;
			mpTargetFrame->GetPose(R, t);
			/*for (int j = 0; j < mvpDummys.size(); j++) {
				auto pMP = mvpDummys[j];
				cv::Mat temp = R*pMP->GetWorldPos() + t;
				temp = mK*temp;
				cv::Point2f pt(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
				cv::circle(vImg, pt, 3, cv::Scalar(0, 255, 255), -1);
			}*/
			
			//for (int j = 0; j < mvpMPs.size(); j++) {
			//	auto type = mvpOPs[j];
			//	int wtype;
			//	int x;
			//	switch (type) {
			//	case ObjectType::OBJECT_WALL:
			//		x = mpTargetFrame->mvKeyPoints[j].pt.x;
			//		wtype = 2;
			//		if (bMinX && x < minFloorX) {
			//			wtype = 1;
			//		}
			//		if (bMaxX && x > maxFloorX) {
			//			wtype = 3;
			//		}
			//		if (wtype == 1) {
			//			cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 0, 255), -1);
			//		}
			//		else if (wtype == 3) {
			//			cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 0), -1);
			//		}
			//		else {
			//			cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(255,0, 0), -1);
			//		}
			//		break;
			//	case ObjectType::OBJECT_CEILING:
			//		//cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 0), -1);
			//		break;
			//	case ObjectType::OBJECT_NONE:
			//		//cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 0, 0), -1);
			//		break;
			//	case ObjectType::OBJECT_FLOOR:
			//		cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 255), -1);
			//		break;
			//	}
			//}
			
			/*sss.str("");
			sss << mStrPath.c_str() << "/plane.jpg";
			cv::imwrite(sss.str(), vImg);*/
			imshow("Output::PlaneEstimation", vImg); cv::waitKey(1);
			std::cout << "pe::end::" << mpTargetFrame->GetKeyFrameID() << std::endl;
			SetBoolDoingProcess(false, 1);
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
bool UVR_SLAM::PlaneInformation::PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, std::vector<UVR_SLAM::MapPoint*> vpMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
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
			if (checkIdx == 0)
				continue;
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
	return abs(abs(distance) - abs(p->distance));
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

void UVR_SLAM::PlaneEstimator::CreatePlanarMapPoints(std::vector<UVR_SLAM::MapPoint*> mvpMPs, std::vector<UVR_SLAM::ObjectType> mvpOPs, UVR_SLAM::PlaneInformation* pPlane, cv::Mat invT) {
	
	int nTargetID = mpTargetFrame->GetFrameID();
	
	cv::Mat invP1 = invT.t()*pPlane->GetParam();

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
		if (depth < 0){
			//depth *= -1.0;
			continue;
		}
		temp *= depth;
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

		/*if (maxDepth < depth)
			maxDepth = depth;
		if (minDepth > depth)
			minDepth = depth;*/

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
			//pPlane->mvpMPs.push_back(pNewMP);
			pPlane->tmpMPs.push_back(pNewMP);
			//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
		}
	}

	/*cv::Mat R, t;
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

	mpTargetFrame->SetDepthRange(minDepth, maxDepth);*/
}

void UVR_SLAM::PlaneEstimator::CreateWallMapPoints(std::vector<UVR_SLAM::MapPoint*> mvpMPs, std::vector<UVR_SLAM::ObjectType> mvpOPs, UVR_SLAM::PlaneInformation* pPlane, cv::Mat invT, int wtype,int MinX, int MaxX, bool b1, bool b2) {

	int nTargetID = mpTargetFrame->GetFrameID();

	cv::Mat invP1 = invT.t()*pPlane->GetParam();
		
	//create new mp in current frame
	for (int j = 0; j < mvpMPs.size(); j++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[j];
		auto oType = mvpOPs[j];

		if (oType != UVR_SLAM::ObjectType::OBJECT_WALL)
			continue;
		cv::Point2f pt = mpTargetFrame->mvKeyPoints[j].pt;
		int x = pt.x;
		if (wtype == 1) {
			if (x > MinX)
				continue;
		}
		else if (wtype == 2) {
			if (x < MinX || x > MaxX)
				continue;
		}
		else {
			if (x < MaxX)
				continue;
		}
		cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
		temp = mK.inv()*temp;
		cv::Mat matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
		float depth = matDepth.at<float>(0);
		if (depth < 0.0){
			//depth *= -1.0;
			continue;
		}
		temp *= depth;
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

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
			pPlane->mvpMPs.push_back(pNewMP);
			//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
		}
	}

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

	cv::Mat s = CreatePlanarMapPoint(from, invP, invT, invK);
	cv::Mat e = CreatePlanarMapPoint(to, invP, invT, invK);

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

	cv::Mat s = CreatePlanarMapPoint(from, invP, invT, invK);
	cv::Mat e = CreatePlanarMapPoint(to, invP, invT, invK);

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
cv::Mat UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(cv::Point2f pt, cv::Mat invP, cv::Mat invT, cv::Mat invK){
	cv::Mat temp1 = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	temp1 = invK*temp1;
	cv::Mat matDepth1 = -invP.at<float>(3) / (invP.rowRange(0, 3).t()*temp1);
	float depth = matDepth1.at<float>(0);
	if (depth < 0)
		depth *= -1.0;
	temp1 *= depth;
	temp1.push_back(cv::Mat::ones(1, 1, CV_32FC1));
	cv::Mat estimated = invT*temp1;
	return estimated.rowRange(0,3);
}

void UVR_SLAM::PlaneInformation::CreatePlanarMapPoints(Frame* pF, System* pSystem) {
	
	int nTargetID = pF->GetFrameID();
	cv::Mat invT, invK, invP;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invP, invT, invK);
	auto pPlane = pF->mpPlaneInformation->GetFloorPlane();
	
	auto mvpMPs = pF->GetMapPoints();
	auto mvpOPs = pF->GetObjectVector();
	
	float minDepth = FLT_MAX;
	float maxDepth = 0.0f;
	//create new mp in current frame
	for (int j = 0; j < mvpMPs.size(); j++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[j];
		auto oType = mvpOPs[j];

		if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR)
			continue;
		cv::Point2f pt = pF->mvKeyPoints[j].pt;
		cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
		temp = invK*temp;
		cv::Mat matDepth = -invP.at<float>(3) / (invP.rowRange(0, 3).t()*temp);
		float depth = matDepth.at<float>(0);
		if (depth < 0) {
			//depth *= -1.0;
			continue;
		}
		temp *= depth;
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

		/*if (maxDepth < depth)
		maxDepth = depth;
		if (minDepth > depth)
		minDepth = depth;*/

		cv::Mat estimated = invT*temp;
		if (pMP) {
			pMP->SetWorldPos(estimated.rowRange(0, 3));
		}
		else {
			UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(pF, estimated.rowRange(0, 3), pF->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
			pNewMP->SetPlaneID(pPlane->mnPlaneID);
			pNewMP->SetObjectType(pPlane->mnPlaneType);
			pNewMP->AddFrame(pF, j);
			pNewMP->UpdateNormalAndDepth();
			pNewMP->mnFirstKeyFrameID = pF->GetKeyFrameID();
			pSystem->mlpNewMPs.push_back(pNewMP);
			//pPlane->mvpMPs.push_back(pNewMP);
			pPlane->tmpMPs.push_back(pNewMP);
			//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
		}
	}
	
	/*cv::Mat R, t;
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

	mpTargetFrame->SetDepthRange(minDepth, maxDepth);*/
}
void UVR_SLAM::PlaneInformation::CreateDensePlanarMapPoint(std::vector<cv::Mat>& vX3Ds, cv::Mat label_map, Frame* pF, int nPatchSize) {
	//dense_map = cv::Mat::zeros(pF->mnMaxX, pF->mnMaxY, CV_32FC3);
	//cv::resize(label_map, label_map, pF->GetOriginalImage().size());
	int nTargetID = pF->GetFrameID();
	cv::Mat invT, invPfloor, invK;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);

	auto pFloor = pF->mpPlaneInformation->GetFloorPlane();
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
	int nTargetID = pF->GetFrameID();
	cv::Mat invT, invPfloor, invK;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);
	cv::Mat invPwawll = invT.t()*pWall->GetParam();

	auto pFloor = pF->mpPlaneInformation->GetFloorPlane();
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
	int nTargetID = pF->GetFrameID();
	cv::Mat invT, invPfloor, invK;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);
	cv::Mat invPwawll = invT.t()*pWall->GetParam();

	auto pFloor = pF->mpPlaneInformation->GetFloorPlane();
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

	int nTargetID = pF->GetFrameID();
	cv::Mat invT, invPfloor, invK;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);
	cv::Mat invPwawll = invT.t()*pWall->GetParam();

	auto pFloor = pF->mpPlaneInformation->GetFloorPlane();
	cv::Mat matFloorParam = pFloor->GetParam();
	cv::Mat normal;
	float dist;
	pFloor->GetParam(normal, dist);

	auto mvpMPs = pF->GetMapPoints();
	auto mvpOPs = pF->GetObjectVector();
	int count = 0;
	//create new mp in current frame
	for (int j = 0; j < mvpMPs.size(); j++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[j];
		if (pMP)
			continue;
		auto oType = mvpOPs[j];
		if (oType != UVR_SLAM::ObjectType::OBJECT_WALL)
			continue;
		cv::Point2f pt = pF->mvKeyPoints[j].pt;
		int x = pt.x;
		if (x < pLine->to.x || x > pLine->from.x)
			continue;
		
		cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
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
		count++;
		std::cout <<"planar::"<< val << estimated.t()<< std::endl;
		vPlanarMaps[j] = estimated.clone();

		//if (pMP) {
		//	pMP->SetWorldPos(estimated.rowRange(0, 3));
		//}
		//else {
		//	UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(pF, estimated.rowRange(0, 3), pF->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
		//	pNewMP->SetPlaneID(0);
		//	pNewMP->SetObjectType(UVR_SLAM::ObjectType::OBJECT_WALL);
		//	pNewMP->AddFrame(pF, j);
		//	pNewMP->UpdateNormalAndDepth();
		//	pNewMP->mnFirstKeyFrameID = pF->GetKeyFrameID();
		//	pSystem->mlpNewMPs.push_back(pNewMP);
		//	//pPlane->mvpMPs.push_back(pNewMP);
		//	//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
		//}
	}
	std::cout << "wall::cre::" << count << std::endl;

}

//vector size = keypoint size
void UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(Frame* pTargetF, PlaneInformation* pFloor, std::vector<cv::Mat>& vPlanarMaps) {
	
	cv::Mat invP, invT, invK;
	pTargetF->mpPlaneInformation->Calculate();
	pTargetF->mpPlaneInformation->GetInformation(invP, invT, invK);

	auto mvpMPs = pTargetF->GetMapPoints();
	auto mvpOPs = pTargetF->GetObjectVector();
	
	//매칭 확인하기
	for (int i = 0; i < pTargetF->mvKeyPoints.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		if (mvpOPs[i] != ObjectType::OBJECT_FLOOR)
			continue;
		/*if (pMP && pMP->isDeleted()) {
			vPlanarMaps[i] = pMP->GetWorldPos();
			continue;
		}*/
		cv::Mat X3D;
		bool bRes = PlaneInformation::CreatePlanarMapPoint(pTargetF->mvKeyPoints[i].pt, invP, invT, invK, X3D);
		if (bRes)
			vPlanarMaps[i] = X3D.clone();
	}
}