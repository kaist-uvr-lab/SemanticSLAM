#include <PlaneEstimator.h>
#include <random>
#include <System.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <SegmentationData.h>
#include <MapPoint.h>
#include <Matcher.h>
#include <Initializer.h>
#include <MatrixOperator.h>

static int nPlaneID = 0;

UVR_SLAM::PlaneEstimator::PlaneEstimator() :mbDoingProcess(false), mnProcessType(0), mpLayoutFrame(nullptr), mbInitFloorPlane(false){
}
UVR_SLAM::PlaneEstimator::PlaneEstimator(std::string strPath,cv::Mat K, cv::Mat K2, int w, int h) : mK(K), mK2(K2),mbDoingProcess(false), mnWidth(w), mnHeight(h), mnProcessType(0), mpLayoutFrame(nullptr),
mpPrevFrame(nullptr), mpTargetFrame(nullptr), mbInitFloorPlane(false)
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
void UVR_SLAM::PlaneInformation::SetParam(cv::Mat n, float d){
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
	normal = n.clone();
	distance = d;
	norm = this->normal.dot(this->normal);
}
void UVR_SLAM::PlaneInformation::GetParam(cv::Mat& n, float& d){
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
	n = normal.clone();
	d = distance;
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
	mpPrevFrame = mpTargetFrame;
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
			bool bInitFloorPlane = isFloorPlaneInitialized();//true;
			int nPrevTest2 = 0;
			cv::Mat pmat;
			if (bInitFloorPlane) {
				int nPrevID = mpPrevFrame->GetFrameID();
				//이전 플라나 포인트로 생성된 포인트에 대해서 트래킹에 성공한 포인트에 한해서 현재 평면 벡터에 포함시킴.
				for (int i = 0; i < mpPrevFrame->mvpPlanes[0]->tmpMPs.size(); i++) {
					UVR_SLAM::MapPoint* pMP = mpPrevFrame->mvpPlanes[0]->tmpMPs[i];
					if (!pMP)
						continue;
					if (pMP->isDeleted())
						continue;
					if (pMP->GetNumConnectedFrames() > 1){
						mpPrevFrame->mvpPlanes[0]->mvpMPs.push_back(pMP);
						nPrevTest2++;
					}
				}
				mpPrevFrame->mvpPlanes[0]->tmpMPs.clear();
				
				UpdatePlane(mpPrevFrame->mvpPlanes[0], nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
				
				////이전 프레임의 포인트 중에서 얼마나 트래킹이 되는지 확인
				auto mvpPrevMPs = mpPrevFrame->GetMapPoints();
				int nPrevPlaneID = mpPrevFrame->mvpPlanes[0]->mnPlaneID;
				for (int i = 0; i < mvpPrevMPs.size(); i++) {
					UVR_SLAM::MapPoint* pMP = mvpPrevMPs[i];
					if (!pMP)
						continue;
					if (pMP->isDeleted())
						continue;
					if (pMP->GetPlaneID() == nPrevPlaneID)
						nPrevTest++;
				}
				////이전 프레임의 포인트 중에서 얼마나 트래킹이 되는지 확인
				pmat = mpPrevFrame->mvpPlanes[0]->matPlaneParam.t();
			}else
				pmat = cv::Mat::zeros(1, 4, CV_32FC1);
			////이전 키프레임에서 추정한 맵포인트 업데이트
			///////////////////////////////////////////
			
			//////////////////////////////////////////////////////////////
			////lock
			//local 맵에서 키포인트 생성 전에 막기 위함
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUsePlaneEstimation);
			mpSystem->mbPlaneEstimationEnd = false;

			//lock
			//labeling 끝날 때까지 대기
			{
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseSegmentation);
				while (!mpSystem->mbSegmentationEnd) {
					mpSystem->cvUseSegmentation.wait(lock);
				}
			}	
			////////////////////////////////////////////////////////////////////////
			///////////////Conneted Component Labeling & object labeling
			//image
			cv::Mat segmented = mpTargetFrame->matSegmented.clone();
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
					case 1://벽
					case 9://유리창
					case 11://캐비넷
					case 15://문
					case 23: //그림
					case 36://옷장
							//case 43://기둥
					case 44://갚난
							//case 94://막대기
					case 101://포스터
						imgWall.at<uchar>(i, j) = 255;
						imgStructure.at<uchar>(i, j) = 255;
						break;
					case 4:
						imgFloor.at<uchar>(i, j) = 255;
						imgStructure.at<uchar>(i, j) = 100;
						/*if (i < minY) {
						minY = i;
						}*/
						break;
					case 6:
						imgStructure.at<uchar>(i, j) = 150;
						imgCeil.at<uchar>(i, j) = 255;
						/*if (i > maxY) {
						maxY = i;
						}*/
						break;
					default:
						imgObject.at<uchar>(i, j) = 255;
						break;
					}
				}
			}
			////
			//////////////////////////////////////////////////////////////////////////////////
			
			////기존의 칼라이미지와 세그멘테이션 결과를 합치는 부분
			////여기는 시각화로 보낼 수 있으면 보내는게 좋을 듯.
			/*cv::resize(seg_color, seg_color, colorimg.size());
			cv::addWeighted(seg_color, 0.4, colorimg, 0.6, 0.0, colorimg);
			cv::imshow("object : ", imgObject);*/
			cv::imshow("Output::Segmentation", seg_color);
			////기존의 칼라이미지와 세그멘테이션 결과를 합치는 부분

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
			
			///////////////////////////////////////////////////////////////////
			//////////max depth & dummy points test
			////lock
			/*std::unique_lock<std::mutex> lockDepth(mpSystem->mMutexUsePlaneEstimation);
			mpSystem->mbPlaneEstimationEnd = false;*/
			std::vector<UVR_SLAM::MapPoint*> mvpDummys;
			int mnTempMaxY = maxY + 10;
			{
				if (bInitFloorPlane && bMaxFloorY) {

					cv::Mat R, t;
					mpTargetFrame->GetPose(R, t);
					cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
					R.copyTo(T.rowRange(0, 3).colRange(0, 3));
					t.copyTo(T.col(3).rowRange(0, 3));
					cv::Mat invT = T.inv();

					float maxDepth = 0.0;
					cv::Mat K = mK.clone();
					K.at<float>(0, 0) /= 2.0;
					K.at<float>(1, 1) /= 2.0;
					K.at<float>(0, 2) /= 2.0;
					K.at<float>(1, 2) /= 2.0;

					cv::Mat invP1 = invT.t()*mpPrevFrame->mvpPlanes[0]->matPlaneParam.clone();

					//for (int y = maxY; y < maxY + 3; y++) {
					int y = maxY + 3;
					for (int x = 0; x < segmented.cols; x++) {
						cv::Point2f pt = cv::Point2f(x, y);
						int val = imgStructure.at<uchar>(pt);
						if (val != 100)
							continue;
						cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
						temp = K.inv()*temp;
						cv::Mat matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
						float depth = matDepth.at<float>(0);
						if (depth < 0){
							//continue;
							depth *= -1.0;
						}
						if (depth > maxDepth){
							maxDepth = depth;
							//mnTempMaxY = pt.y;
						}
						temp *= depth;
						temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

						cv::Mat estimated = invT*temp;
						UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), cv::Mat(), UVR_SLAM::PLANE_MP);
						//mvpDummys.push_back(pNewMP);
					}
					//}

					//mpFrameWindow->ClearDummyMPs();
					//mpFrameWindow->SetDummyPoints(mvpDummys);
					mpTargetFrame->SetDepthRange(0.0, maxDepth);
				}

			}
			
			
			//////////max depth & dummy points test
			///////////////////////////////////////////////////////////////////
			
			/////lock
			//planar mp
			std::unique_lock<std::mutex> lockPlanar(mpSystem->mMutexUsePlanarMP);
			mpSystem->mbPlanarMPEnd = false;
			//planar mp
			/////lock
			
			//max y 기준으로 라인 그어버리기
			//minx, maxx 계산하기
			int mnLineSize = 10;
			cv::Mat tempWall = imgStructure.clone();
			//cv::line(tempWall, cv::Point2f(0, minY), cv::Point2f(imgWall.cols, minY), cv::Scalar(0, 0, 0), mnLineSize);
			//cv::line(imgFloor, cv::Point2f(0, minY), cv::Point2f(imgWall.cols, minY), cv::Scalar(0, 0, 0), mnLineSize);

			////check floor & ceil
			//cv::rectangle(tempWall, cv::Point2f(0, 0), cv::Point2f(tempWall.cols, minY), cv::Scalar(0, 0, 0), -1);
			//cv::rectangle(tempWall, cv::Point2f(0, minY), cv::Point2f(tempWall.cols, minY), cv::Scalar(0, 0, 0), -1);//mnLineSize
			//cv::rectangle(tempWall, cv::Point2f(0, maxY), cv::Point2f(tempWall.cols, maxY), cv::Scalar(0, 0, 0), -1);
			////check floor & ceil

			//////////////////////////////////////////////////////////////////////////
			//////세그멘테이션 결과를 바탕으로 바닥과 벽, 벽과 천장이 만나는 지점을 계산하는 과정
			//1 = 바닥과 옆벽
			//2 = 바닥과 앞벽
			//3 = 천장과 옆벽
			//4 = 천장과 앞벽
			cv::Mat tempColor;
			cv::Mat tempWallLines = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat tempFloor = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat tempCeil = cv::Mat::zeros(segmented.size(), CV_8UC1);
			tempWall.convertTo(tempColor, CV_8UC3);
			cv::cvtColor(tempColor, tempColor, CV_GRAY2BGR);
			for (int y = 1; y < tempColor.rows - 1; y++) {
				for (int x = 1; x < tempColor.cols - 1; x++) {
					int val = tempWall.at<uchar>(y, x);
					int val_l = tempWall.at<uchar>(y, x - 1);
					int val_r = tempWall.at<uchar>(y, x + 1);
					int val_u = tempWall.at<uchar>(y + 1, x);
					int val_d = tempWall.at<uchar>(y - 1, x);
					/*int count = 0;
					if (val_l == 255)
					count++;
					if (val_r == 255)
					count++;
					if (val_u == 255)
					count++;
					if (val_u == 255)
					count++;
					if (count > 3 || count < 1)
					continue;*/
					cv::Point2f pt(x, y);
					if (sumCCL.at<uchar>(pt) != 255)
						continue;
					if (val == 100) {
						if (val_l == 255 || val_r == 255) {
							tempFloor.at<uchar>(pt) = 255;
							tempWallLines.at<uchar>(pt) = 1;
							cv::circle(tempColor, pt, 1, cv::Scalar(0, 0, 255), -1);
						}
						else if (val_u == 255 || val_d == 255) {
							tempFloor.at<uchar>(pt) = 255;
							cv::circle(tempColor, pt, 1, cv::Scalar(0, 255, 255), -1);
							tempWallLines.at<uchar>(pt) = 2;
						}
					}
					else if (val == 150) {
						if (val_l == 255 || val_r == 255) {
							tempCeil.at<uchar>(pt) = 255;
							cv::circle(tempColor, pt, 1, cv::Scalar(255, 0, 0), -1);
							tempWallLines.at<uchar>(pt) = 3;
						}
						else if (val_u == 255 || val_d == 255) {
							tempCeil.at<uchar>(pt) = 255;
							cv::circle(tempColor, pt, 1, cv::Scalar(255, 255, 0), -1);
							tempWallLines.at<uchar>(pt) = 4;
						}
					}
				}
			}
			//////////houghlinesp test
			/*for (int i = 0; i<vecLines.size(); i++)
			{
				Vec4i L = vecLines[i];
				line(tempColor, Point2f(L[0]/2.0, L[1] / 2.0), Point2f(L[2] / 2.0, L[3] / 2.0),
					Scalar(0, 0, 255), 1, LINE_AA);
			}*/
			//////////houghlinesp test

			//cv::line(tempColor, cv::Point2f(0, minY), cv::Point2f(imgWall.cols, minY), cv::Scalar(0, 255, 0), 1);
			//cv::line(tempColor, cv::Point2f(0, maxY), cv::Point2f(imgWall.cols, maxY), cv::Scalar(0, 255, 0), 1);
			//////세그멘테이션 결과를 바탕으로 바닥과 벽, 벽과 천장이 만나는 지점을 계산하는 과정
			//////////////////////////////////////////////////////////////////////////
			
			////////////////////////////////////////////////////////////////////////
			//////바닥, 벽이 만나는 지점을 바탕으로 라인을 추정
			////추정된 라인은 가상의 포인트를 만들고 테스트 하는데 이용
			PlaneInformation* pTestWall = new PlaneInformation();
			std::vector<cv::Vec4i> tlines, lines;
			{
				
				cv::Ptr<cv::LineSegmentDetector> pLSD = createLineSegmentDetector();
				pLSD->detect(floorCCL, tlines);//pLSD->detect(tempFloor, tlines);
				//graph.convertTo(graph, CV_8UC3);
				//cv::cvtColor(graph, graph, CV_GRAY2BGR);
				for (int i = 0; i < tlines.size(); i++) {
					Vec4i v = tlines[i];
					Point2f from(v[0], v[1]);
					Point2f to(v[2], v[3]);
					Point2f diff = from - to;
					float dist = sqrt(diff.dot(diff));
					/*if (tempWallLines.at<uchar>(from) == 0 || tempWallLines.at<uchar>(to) == 0)
						continue;*/
					if (dist < 25)
						continue;
					if (to.y < maxY+3 || from.y < maxY + 3)
						continue;
					float slope = abs(LineProcessor::CalcSlope(from, to));
					if (slope > 3.0)
						continue;

					lines.push_back(tlines[i]);
					//cv::line(fLineImg, from, to, cv::Scalar(255,0,0));
					//else
					//cv::line(graph, from, to, cv::Scalar(0, 0, 255));
				}
				//해당 라인은 타겟 프레임에 설정함.
				mpTargetFrame->SetLines(lines);
			}
			
			//////////////////////////////////////////////////////////////
			//WallLineTest
			if (bInitFloorPlane && mpPrevFrame) {
				auto mvFrames = mpTargetFrame->GetConnectedKFs(10);
				//mvFrames.push_back(mpTargetFrame);
				std::vector<cv::Mat> wallParams;
				std::vector<cv::Mat> twallParams;
				std::map<int, int> mapWallParams;
				std::map<int, std::vector<UVR_SLAM::Frame*>> mapWallParamFrames;
				cv::Mat K = mK.clone();
				K.at<float>(0, 0) /= 2.0;
				K.at<float>(1, 1) /= 2.0;
				K.at<float>(0, 2) /= 2.0;
				K.at<float>(1, 2) /= 2.0;
				for (int k = 0; k < mvFrames.size(); k++) {
					auto plane = mvFrames[k]->mvpPlanes[0];
					cv::Mat normal1;
					float dist1;
					plane->GetParam(normal1, dist1);
					cv::Mat planeParam = plane->matPlaneParam.clone();

					cv::Mat R, t;
					mvFrames[k]->GetPose(R, t);
					cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
					R.copyTo(T.rowRange(0, 3).colRange(0, 3));
					t.copyTo(T.col(3).rowRange(0, 3));

					cv::Mat invT = T.inv();
					cv::Mat invP = invT.t()*planeParam;
					cv::Mat invK = K.inv();

					for (int i = 0; i < lines.size(); i++) {
						cv::Mat param = UVR_SLAM::PlaneInformation::PlaneWallEstimator(lines[i], normal1, invP, invT, invK);

						//비교 후 추가
						if (mapWallParams.empty()) {
							mapWallParams.insert(std::make_pair(wallParams.size(), 0));
							mapWallParamFrames.insert(std::make_pair(wallParams.size(), std::vector<UVR_SLAM::Frame*>()));
							mapWallParamFrames[0].push_back(mvFrames[k]);
							wallParams.push_back(param);
						}
						else {
							bool bInsert = true;
							for (int j = 0; j < wallParams.size(); j++) {
								float normal = PlaneInformation::CalcCosineSimilarity(wallParams[j], param);
								float dist = PlaneInformation::CalcPlaneDistance(wallParams[j], param);
								/*if (normal < 0.98 || dist > 0.01) {
									continue;
								}*/
								////아예 노말이 다르다.
								////노말이
								if (normal > 0.97 && dist < 0.01) {
									//겹치는 것/
									bInsert = false;
									mapWallParams[j]++;
									mapWallParamFrames[j].push_back(mvFrames[k]);
								}
								else if (normal > 0.97 && (dist <0.1 && dist >= 0.01)) {
									//얘는 추가x
									bInsert = false;
								}
							}
							if (bInsert) {
								mapWallParams.insert(std::make_pair(wallParams.size(), 1));
								mapWallParamFrames.insert(std::make_pair(wallParams.size(), std::vector<UVR_SLAM::Frame*>()));
								mapWallParamFrames[wallParams.size()].push_back(mvFrames[k]);
								wallParams.push_back(param);
							}
						}
						
					}//for
				}
				for (auto iter = mapWallParams.begin(); iter != mapWallParams.end(); iter++) {
					int idx = iter->first;
					int count = iter->second;
					if (count > 3){
						twallParams.push_back(wallParams[idx]);
						std::cout << "param : " << wallParams[idx].t() << ", " << count << std::endl;
						for (int aa = 0; aa < count; aa++) {
							std::cout << "frame test : " << mapWallParamFrames[idx][aa]->GetFrameID() << std::endl;
						}
					}
				}
				mpTargetFrame->SetWallParams(twallParams);
			}
			
			//WallLineTest
			//////////////////////////////////////////////////////////////



			//////라인 생성 및 테스트
			{
				if (mpPrevFrame && bMaxFloorY && mpPrevFrame->mvpPlanes.size() > 0) {

					cv::Mat matPlane = cv::Mat::zeros(0, 4, CV_32FC1);
					cv::Mat R, t;
					mpTargetFrame->GetPose(R, t);
					cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
					R.copyTo(T.rowRange(0, 3).colRange(0, 3));
					t.copyTo(T.col(3).rowRange(0, 3));
					cv::Mat invT = T.inv();

					float maxDepth = 0.0;
					cv::Mat K = mK.clone();
					K.at<float>(0, 0) /= 2.0;
					K.at<float>(1, 1) /= 2.0;
					K.at<float>(0, 2) /= 2.0;
					K.at<float>(1, 2) /= 2.0;

					cv::Mat normal1;
					float dist1;
					mpPrevFrame->mvpPlanes[0]->GetParam(normal1, dist1);
					cv::Mat invP1 = invT.t()*mpPrevFrame->mvpPlanes[0]->matPlaneParam.clone();

					///////////////////////////////////////////
					////마스킹 일단 칼라로
					cv::Mat maskImg = cv::Mat::zeros(segmented.size(), CV_8UC3);
					cv::Mat pmask = cv::Mat::zeros(segmented.size(), CV_8UC1);
					////마스킹
					///////////////////////////////////////////

					int nPts = 15;
					for (int i = 0; i < lines.size(); i++) {
						Vec4i v = lines[i];
						Point2f from(v[0], v[1]);
						Point2f to(v[2], v[3]);

						cv::Point2f diff = to - from;

						std::vector<UVR_SLAM::MapPoint*> tempvec;
						for (int j = 0; j < nPts; j++) {

							cv::Point2f pt(from.x + diff.x / nPts * j, from.y + diff.y / nPts * j);
							cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
							temp = K.inv()*temp;
							cv::Mat matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
							float depth = matDepth.at<float>(0);
							if (depth < 0)
								depth *= -1.0;
							temp *= depth;
							temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

							cv::Mat estimated = invT*temp;
							UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), cv::Mat(), UVR_SLAM::PLANE_MP);
							mvpDummys.push_back(pNewMP);
							tempvec.push_back(pNewMP);
						}
						{
							///////////////////////////////////////////
							////마스킹
							cv::Point2f pt1(from.x, 0);
							cv::Point2f pt2(to.x, segmented.rows);
							cv::Scalar scalar((i + 1)*20, (i + 1) * 20, (i + 1) * 20);
							cv::rectangle(maskImg, pt1, pt2, scalar, -1);
							cv::line(maskImg, from, to, cv::Scalar(255, 255, 255));
							cv::Scalar scalar2((i + 1), (i + 1), (i + 1));
							cv::rectangle(pmask, pt1, pt2, scalar2, -1);
							////마스킹
							///////////////////////////////////////////
						}
						{
							///////////////////////////////////////////
							////평면 벽 파라메터 추정
							cv::Mat normal2 = tempvec[0]->GetWorldPos() - tempvec[nPts - 1]->GetWorldPos();
							float norm2 = sqrt(normal2.dot(normal2));
							normal2 /= norm2;
							
							auto normal3 = normal1.cross(normal2);
							float norm3 = sqrt(normal3.dot(normal3));
							normal3 /= norm3;
							cv::Mat matDist = normal3.t()*tempvec[0]->GetWorldPos();
							
							normal3.push_back(-matDist);
							matPlane.push_back(normal3.t());
							if (i == 0)
								pTestWall->matPlaneParam = normal3.clone();
							////평면 벽 파라메터 추정
							///////////////////////////////////////////
						}

						/*cv::Mat temp = (cv::Mat_<float>(3, 1) << from.x, from.y, 1);
						temp = K.inv()*temp;
						cv::Mat matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
						float depth = matDepth.at<float>(0);
						if (depth < 0)
							depth *= -1.0;
						temp *= depth;
						temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

						cv::Mat estimated = invT*temp;
						UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), cv::Mat(), UVR_SLAM::PLANE_MP);
						mvpDummys.push_back(pNewMP);

						temp = (cv::Mat_<float>(3, 1) << to.x, to.y, 1);
						temp = K.inv()*temp;
						matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
						depth = matDepth.at<float>(0);
						if (depth < 0)
							depth *= -1.0;
						temp *= depth;
						temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

						estimated = invT*temp;
						pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), cv::Mat(), UVR_SLAM::PLANE_MP);
						mvpDummys.push_back(pNewMP);*/

					}
					/////////////////////////////////////////////
					////////create wall planar mp
					//auto mvpOPs = mpTargetFrame->GetObjectVector();
					//auto mvpMPs = mpTargetFrame->GetMapPoints();
					//for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
					//	/*if (mvpMPs[i])
					//		continue;*/
					//	if (mvpMPs[i]) {
					//		if (mvpMPs[i]->GetMapPointType() == MapPointType::PLANE_MP)
					//			continue;
					//	}
					//	auto type = mvpOPs[i];
					//	auto pt = mpTargetFrame->mvKeyPoints[i].pt;
					//	if (type != ObjectType::OBJECT_WALL)
					//		continue;
					//	int pid = pmask.at<uchar>(pt.y/2, pt.x/2);
					//	if (pid == 0)
					//		continue;
					//	cv::Mat plane = matPlane.row(pid - 1).t();
					//	cv::Mat invP = invT.t()*plane;
					//	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
					//	temp = mK.inv()*temp;
					//	cv::Mat matDepth = -invP.at<float>(3) / (invP.rowRange(0, 3).t()*temp);
					//	float depth = matDepth.at<float>(0);
					//	if (depth < 0){
					//		//depth *= -1.0;
					//		continue;
					//	}
					//	temp *= depth;
					//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

					//	cv::Mat estimated = invT*temp;
					//	UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), mpTargetFrame->matDescriptor.row(i), UVR_SLAM::PLANE_MP);
					//	pNewMP->SetObjectType(ObjectType::OBJECT_WALL);
					//	pNewMP->AddFrame(mpTargetFrame, i);
					//	pNewMP->UpdateNormalAndDepth();
					//	pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
					//	mpSystem->mlpNewMPs.push_back(pNewMP);
					//	mvpDummys.push_back(pNewMP);
					//}
					////////create wall planar mp
					mpFrameWindow->SetDummyPoints(mvpDummys);
					////마스킹
					/*maskImg.convertTo(maskImg, CV_8UC3);
					cv::cvtColor(maskImg, maskImg, CV_GRAY2BGR);*/
					cv::imshow("line mask", maskImg);
					////마스킹
					///////////////////////////////////////////
				}
			}
			//////바닥, 벽이 만나는 지점을 바탕으로 라인을 추정
			////추정된 라인은 가상의 포인트를 만들고 테스트 하는데 이용
			////unlock & notify
			mpSystem->mbPlaneEstimationEnd = true;
			lock.unlock();
			mpSystem->cvUsePlaneEstimation.notify_all();
			////////////////////////////////////////////////////////////////////////
			
			/////////////////////////////////////////////////////////////////////////////
			////바닥의 좌우 값을 추정하기 위한 것
			for (int x = 0; x < tempWallLines.cols; x++) {
				int ay1 = max(minY - 3, 0);
				int ay2 = min(maxY + 3, segmented.rows - 1);
				int val1 = tempWallLines.at<uchar>(ay1, x);
				int val2 = tempWallLines.at<uchar>(ay2, x);
				if (val1 == 3 && bMinCeilY) {

					//minCeilX
					if (x < minFloorX)
						minFloorX = x;
					if (x > maxFloorX)
						maxFloorX = x;
				}
				if (val2 == 1 && bMaxFloorY) {

					if (x < minFloorX)
						minFloorX = x;
					if (x > maxFloorX)
						maxFloorX = x;
				}
			}
			
			if (minFloorX != segmented.cols)
			{
				minFloorX *= 2;
				bMinX = true;
			}
			if (maxFloorX != 0) {
				maxFloorX *= 2;
				bMaxX = true;
			}
			//cv::line(tempColor, cv::Point2f(minFloorX, minY), cv::Point2f(minFloorX, maxY), cv::Scalar(0, 255, 0), 1);
			//cv::line(tempColor, cv::Point2f(maxFloorX, minY), cv::Point2f(maxFloorX, maxY), cv::Scalar(0, 255, 0), 1);
			cv::imshow("temptemp wall2", tempColor);
			////바닥의 좌우 값을 추정하기 위한 것
			/////////////////////////////////////////////////////////////////////////////
			
			//////////////////////////////////////////////////////
			//////현재 키프레임에서 바닥 평면 추정
			UVR_SLAM::PlaneInformation* pPlane2;
			pPlane2 = new UVR_SLAM::PlaneInformation();
			bool bLocalFloor = false;

			if (!bInitFloorPlane) {
				auto mvpFrameFloorMPs = std::vector<UVR_SLAM::MapPoint*>(mpTargetFrame->mspFloorMPs.begin(), mpTargetFrame->mspFloorMPs.end());
				if (mpTargetFrame->mspFloorMPs.size() > 10) {
					UVR_SLAM::PlaneInformation* pTemp = new UVR_SLAM::PlaneInformation();
					bLocalFloor = PlaneInitialization(pPlane2, mvpFrameFloorMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
					//평면을 찾은 경우 현재 평면을 교체하던가 수정을 해야 함.
					if (bLocalFloor) {
						pPlane2->mnFrameID = nTargetID;
						pPlane2->mnPlaneType = ObjectType::OBJECT_FLOOR;
						pPlane2->mnCount = 1;
						//CreatePlanarMapPoints(mvpMPs, mvpOPs, pPlane2, invT);
						
						std::unique_lock<std::mutex> lock(mpSystem->mMutexUseLocalMap);
						while (!mpSystem->mbTrackingEnd) {
							mpSystem->cvUseLocalMap.wait(lock);
						}
						mpSystem->mbLocalMapUpdateEnd = false;


						auto frames = mpFrameWindow->GetLocalMapFrames();
						auto mps = mpFrameWindow->GetLocalMap();


						/*cv::Mat Rcw = CalcPlaneRotationMatrix(pPlane2->matPlaneParam);
						cv::Mat tempP = Rcw.t()*pPlane2->normal;
						std::cout << "Normal : " << tempP.t() << " " << pPlane2->normal.t() << pPlane2->distance << std::endl;

						for (int j = 0; j < frames.size(); j++) {
							cv::Mat R, t;
							frames[j]->GetPose(R, t);
							frames[j]->SetPose(R*Rcw, t);
							std::cout << "test::a::" << frames[j]->GetFrameID() << std::endl;
						}

						for (int j = 0; j < mps.size(); j++) {
							UVR_SLAM::MapPoint* pMP = mps[j];
							if (!pMP)
								continue;
							if (pMP->isDeleted())
								continue;
							cv::Mat tempX = Rcw.t()*pMP->GetWorldPos();
							pMP->SetWorldPos(tempX);

						}*/
						
						/*pPlane2->normal = tempP.clone();
						tempP.copyTo(pPlane2->matPlaneParam.rowRange(0, 3));
						

						cv::Mat R, t;
						mpFrameWindow->GetPose(R, t);
						mpFrameWindow->SetPose(R*Rcw, t);*/

						mpTargetFrame->mvpPlanes.push_back(pPlane2);
						SetFloorPlaneInitialization(true);

						mpSystem->mbLocalMapUpdateEnd = true;
						lock.unlock();
						mpSystem->cvUseLocalMap.notify_one();

						
					}
				}
			}
			else {
				bLocalFloor = true;
				mpTargetFrame->mvpPlanes.push_back(mpPrevFrame->mvpPlanes[0]);
				if (mpPrevFrame->mvpPlanes.size() == 1) {
					mpTargetFrame->mvpPlanes.push_back(pTestWall);
				}
				else {
					mpTargetFrame->mvpPlanes.push_back(mpPrevFrame->mvpPlanes[1]);
				}
				
			}
			
			//////////////////////////////////////////////////////////////////////////////
			////Planar MP 생성 완료
			auto mvpMPs = mpTargetFrame->GetMapPoints();
			auto mvpOPs = mpTargetFrame->GetObjectVector();
			
			if (bLocalFloor) {
				cv::Mat R, t;
				mpTargetFrame->GetPose(R, t);
				cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
				R.copyTo(T.rowRange(0, 3).colRange(0, 3));
				t.copyTo(T.col(3).rowRange(0, 3));
				cv::Mat invT = T.inv();
				CreatePlanarMapPoints(mvpMPs, mvpOPs, mpTargetFrame->mvpPlanes[0], invT);
				//mpFrameWindow->SetLocalMap(nTargetID);
			}
			
			////unlock & notify
			mpSystem->mbPlanarMPEnd = true;
			lockPlanar.unlock();
			mpSystem->cvUsePlanarMP.notify_all();
			////Planar MP 생성 완료

			////projection test
			//if(bInitFloorPlane)
			//{
			//	auto mvpKFs = mpTargetFrame->GetConnectedKFs(10);
			//	for (int i = 0; i < mvpKFs.size(); i++) {
			//		cv::Mat img1 = mpTargetFrame->GetOriginalImage();
			//		cv::Mat img2 = mvpKFs[i]->GetOriginalImage();

			//		cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

			//		cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
			//		cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);

			//		cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
			//		img1.copyTo(debugging(mergeRect1));
			//		img2.copyTo(debugging(mergeRect2));

			//		auto mvpMPs = mpTargetFrame->mvpPlanes[0]->mvpMPs;

			//		//cv::Mat R, t;
			//		//mpPrevFrame->GetPose(R, t);

			//		for (int j = 0; j < mvpMPs.size(); j++) {
			//			UVR_SLAM::MapPoint* pMP = mvpMPs[j];
			//			if (!pMP)
			//				continue;
			//			if (pMP->isDeleted())
			//				continue;
			//			cv::Point2f pt1 = mpTargetFrame->Projection(pMP->GetWorldPos());
			//			cv::Point2f pt2 = mvpKFs[i]->Projection(pMP->GetWorldPos());
			//			cv::line(debugging, pt1, pt2 + ptBottom, cv::Scalar(255), 1);
			//		}
			//		std::stringstream ss;
			//		ss << mStrPath.c_str() << "/floor_" << mpTargetFrame->GetFrameID() <<"_"<< mvpKFs[i]->GetFrameID()<< ".jpg";
			//		imwrite(ss.str(), debugging);
			//	}
			//	
			//}
			//////////////////////////////////////////////////////////////////////////////

			//////현재 키프레임에서 바닥 평면 추정
			//////////////////////////////////////////////////////
			
			//////////////////////////////////////////////////////
			////////벽 레이블링 쪼개기 및 테스트
			//{
			//	std::vector<UVR_SLAM::MapPoint*> mvpWallMPs1, mvpWallMPs2, mvpWallMPs3;
			//	auto mvpMPs = mpTargetFrame->GetMapPoints();
			//	auto mvpOPs = mpTargetFrame->GetObjectVector();
			//	for (int i = 0; i < mvpMPs.size(); i++) {
			//		UVR_SLAM::MapPoint* pMP = mvpMPs[i];
			//		if (!pMP)
			//			continue;
			//		if (pMP->isDeleted())
			//			continue;
			//		auto type = mvpOPs[i];
			//		if (type != ObjectType::OBJECT_WALL)
			//			continue;
			//		int x = mpTargetFrame->mvKeyPoints[i].pt.x;
			//		if (bMinX && x < minFloorX) {
			//			mvpWallMPs1.push_back(pMP);
			//			continue;
			//		}
			//		if (bMaxX && x > maxFloorX) {
			//			mvpWallMPs3.push_back(pMP);
			//			continue;
			//		}
			//		mvpWallMPs2.push_back(pMP);
			//	}
			//	if (mvpWallMPs1.size() > 10) {
			//		UVR_SLAM::PlaneInformation* pPlane;
			//		pPlane = new UVR_SLAM::PlaneInformation();
			//		bool b = PlaneInitialization(pPlane, mvpWallMPs1, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
			//		if (b) {
			//			cv::Mat R, t;
			//			mpTargetFrame->GetPose(R, t);
			//			cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
			//			R.copyTo(T.rowRange(0, 3).colRange(0, 3));
			//			t.copyTo(T.col(3).rowRange(0, 3));
			//			cv::Mat invT = T.inv();
			//			CreateWallMapPoints(mvpMPs, mvpOPs, pPlane, invT, 1, minFloorX, maxFloorX, bMinX, bMaxX);

			//			if (mpPrevFrame && mpPrevFrame->mvpPlanes.size() > 0) {
			//				float m;
			//				cv::Mat mLine = pPlane->FlukerLineProjection(mpPrevFrame->mvpPlanes[0], mpTargetFrame->GetRotation(), mpTargetFrame->GetTranslation(), mK2, m);
			//				cv::Point2f sPt, ePt;
			//				mvpPlanes[0]->CalcFlukerLinePoints(sPt, ePt, 0.0, mnHeight, mLine);
			//				cv::line(vImg, sPt, ePt, cv::Scalar(0, 255, 0), 3);
			//			}
			//		}
			//	}
			//}
			////////벽 레이블링 쪼개기 및 테스트
			//////////////////////////////////////////////////////

			/*if (mpTargetFrame->GetKeyFrameID() > 2) {
				auto mvpKFs = mpTargetFrame->GetConnectedKFs(10);
				for (int i = 0; i < mvpKFs.size(); i++) {
					std::vector<cv::DMatch> matches;
					std::vector<int> temaap;
					if (mvpKFs[i]->mWallDescriptor.rows == 0)
						continue;
					std::cout << mvpKFs[i]->mWallDescriptor.rows << ", " << mpTargetFrame->mWallDescriptor.rows << std::endl;
					mpMatcher->KeyFrameFeatureMatching(mvpKFs[i], mpTargetFrame, mvpKFs[i]->mWallDescriptor, mpTargetFrame->mWallDescriptor, mvpKFs[i]->mWallIdxs, mpTargetFrame->mWallIdxs, matches);

					cv::Mat img1 = mpTargetFrame->GetOriginalImage();
					cv::Mat img2 = mvpKFs[i]->GetOriginalImage();

					cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

					cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
					cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);

					cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
					img1.copyTo(debugging(mergeRect1));
					img2.copyTo(debugging(mergeRect2));

					for (int j = 0; j < matches.size(); j++) {
						cv::line(debugging, mpTargetFrame->mvKeyPoints[matches[j].trainIdx].pt, mvpKFs[i]->mvKeyPoints[matches[j].queryIdx].pt + ptBottom, cv::Scalar(255), 1);
					}
					std::stringstream ss;
					ss << mStrPath.c_str() << "/wall_" << mpTargetFrame->GetFrameID() << "_" << mvpKFs[i]->GetFrameID() << ".jpg";
					imwrite(ss.str(), debugging);
				}
			}*/

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

			/*UVR_SLAM::PlaneInformation *pPlane1, *pPlane3;
			int tempWallID = 0;
			pPlane1 = new UVR_SLAM::PlaneInformation();
			auto mvpFrameWallMPs = std::vector<UVR_SLAM::MapPoint*>(mpTargetFrame->mspWallMPs.begin(), mpTargetFrame->mspWallMPs.end());*/

			//평면 RANSAC 초기화
			
			/*if (mspLocalFloorMPs.size() > 10)
				bLocalFloor = PlaneInitialization(pPlane2, mspLocalFloorMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
			*/


			

			//////로컬 맵에서 벽 평면
			//pPlane3 = new UVR_SLAM::PlaneInformation();
			//auto mvpLoclaMapWallMPs = std::vector<UVR_SLAM::MapPoint*>(mpFrameWindow->mspWallMPs.begin(), mpFrameWindow->mspWallMPs.end());
			//bool bLocalMapWall = false;
			//if (mpFrameWindow->mspWallMPs.size() > 10 && mvpPlanes.size() > 0) {
			//	UVR_SLAM::PlaneInformation* pTemp = new UVR_SLAM::PlaneInformation();
			//	bLocalMapWall = PlaneInitialization(pPlane3, mvpLoclaMapWallMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
			//	//bLocalMapWall = PlaneInitialization(pPlane3, pPlane2, 1, mvpLoclaMapWallMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
			//}
			//////로컬 맵에서 벽 평면

			

			///////벽을 찾는 과정
			//벽은 현재 프레임에서 바닥을 찾았거나 바닥을 찾은 상태일 때 수행.
			//bool bLocalWall = false;
			//if(mpTargetFrame->mspWallMPs.size() > 10 && (bLocalFloor || mvpPlanes.size() > 0 )){
			//	//bLocalWall = PlaneInitialization(pPlane1, mspLocalWallMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);

			//	UVR_SLAM::PlaneInformation* groundPlane = nullptr;
			//	/*if (mvpPlanes.size() > 0)
			//		groundPlane = mvpPlanes[0];
			//	else*/
			//	groundPlane = pPlane2;
			//	bLocalWall = PlaneInitialization(pPlane1, groundPlane, 1, mvpFrameWallMPs, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
			//}
			////라인 프로젝션
			//if (bLocalWall) {
			//	tempWallID = pPlane1->mnPlaneID;
			//	pPlane1->mnFrameID = nTargetID;
			//	pPlane1->mnPlaneType = ObjectType::OBJECT_WALL;
			//	pPlane1->mnCount = 1;

			//}
			///////벽을 찾는 과정
			
			///이제 굳이 필요가 없는???
			bool bFailCase = false;
			bool bFailCase2 = false;
			//compare & association
			//평면을 교체하게 되면 잘못되는 경우가 생김.

			//if (mpPrevFrame && mpPrevFrame->mvpPlanes.size() > 0) {
			//	UVR_SLAM::PlaneInformation* p1 = mpPrevFrame->mvpPlanes[0];
			//	UVR_SLAM::PlaneInformation* p2 = mpTargetFrame->mvpPlanes[0];
			//	//ratio = pPlane2->CalcOverlapMPs(p, nTargetID);
			//	if (p1->CalcCosineSimilarity(p2) < 0.98) {
			//		bFailCase = true;
			//		//mvpPlanes[i] = pPlane2;
			//	}
			//	if (p1->CalcPlaneDistance(p2) >= 0.015)
			//		bFailCase2 = true;
			//}
			////////////////////////////////////////////////////////////////////////////////////////////
			
			/*if (mvpPlanes.size() == 0 && bLocalFloor) {
				mvpPlanes.push_back(pPlane2);
			}
			else {*/
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
			//}
			//update local map
			//mpFrameWindow->SetLocalMap(mpTargetFrame->GetFrameID());

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
			for (int j = 0; j < mvpDummys.size(); j++) {
				auto pMP = mvpDummys[j];
				cv::Mat temp = R*pMP->GetWorldPos() + t;
				temp = mK*temp;
				cv::Point2f pt(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
				cv::circle(vImg, pt, 3, cv::Scalar(0, 255, 255), -1);
			}
			for (int j = 0; j < mvpMPs.size(); j++) {
				auto type = mvpOPs[j];
				int wtype;
				int x;
				switch (type) {
				case ObjectType::OBJECT_WALL:
					x = mpTargetFrame->mvKeyPoints[j].pt.x;
					wtype = 2;
					if (bMinX && x < minFloorX) {
						wtype = 1;
					}
					if (bMaxX && x > maxFloorX) {
						wtype = 3;
					}
					if (wtype == 1) {
						cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 0, 255), -1);
					}
					else if (wtype == 3) {
						cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 0), -1);
					}
					else {
						cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(255,0, 0), -1);
					}
					break;
				case ObjectType::OBJECT_CEILING:
					//cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 0), -1);
					break;
				case ObjectType::OBJECT_NONE:
					//cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 0, 0), -1);
					break;
				case ObjectType::OBJECT_FLOOR:
					cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 255), -1);
					break;
				}
				
				/*UVR_SLAM::MapPoint* pMP = mvpMPs[j];

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
				else if (bLocalMapWall && pMP->GetPlaneID() == pPlane3->mnPlaneID) {
					cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 3, cv::Scalar(0, 255, 255));
				}
				if (!bLocalFloor && mvpPlanes.size() > 0 && pMP->GetPlaneID() == mvpPlanes[0]->mnPlaneID ) {
					cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 3, cv::Scalar(255, 0, 255));
				}*/
			}
			
			//////////////////////////////////////////
			////test wall line visualization
			////가끔 죽음. 여기서 왜인지는 모름.
			/*if (mpTargetFrame->mvpPlanes.size() > 1) {
				float m;
				cv::Mat mLine = mpTargetFrame->mvpPlanes[0]->FlukerLineProjection(mpTargetFrame->mvpPlanes[1], mpTargetFrame->GetRotation(), mpTargetFrame->GetTranslation(), mK2, m);
				cv::Point2f sPt, ePt;
				mpTargetFrame->mvpPlanes[1]->CalcFlukerLinePoints(sPt, ePt, 0.0, mnHeight, mLine);
				cv::line(vImg, sPt, ePt, cv::Scalar(0, 255, 0), 3);
			}*/
			////test wall line visualization
			//////////////////////////////////////////
			
			if (bFailCase) {
				cv::Mat temp = cv::Mat::zeros(50,50, CV_8UC3);
				cv::Point2f pt1 = cv::Point(0, vImg.rows-temp.rows);
				cv::Point2f pt2 = cv::Point(temp.cols, vImg.rows);
				cv::Rect rect = cv::Rect(pt1, pt2);
				rectangle(temp, cv::Point2f(0,0), cv::Point2f(temp.cols, temp.rows), cv::Scalar(0, 255, 0), -1);
				cv::Mat a = vImg(rect);
				cv::addWeighted(a, 0.7, temp, 0.3, 0.0, a);
			}

			if (bFailCase2) {
				cv::Mat temp = cv::Mat::zeros(50, 50, CV_8UC3);
				cv::Point2f pt1 = cv::Point(50, vImg.rows - temp.rows);
				cv::Point2f pt2 = cv::Point(pt1.x+temp.cols, vImg.rows);
				cv::Rect rect = cv::Rect(pt1, pt2);
				rectangle(temp, cv::Point2f(0, 0), cv::Point2f(temp.cols, temp.rows), cv::Scalar(255, 0, 0), -1);
				cv::Mat a = vImg(rect);
				cv::addWeighted(a, 0.7, temp, 0.3, 0.0, vImg(rect));	
			}

			if (nPrevTest < 20) {
				cv::Mat temp = cv::Mat::zeros(50, 50, CV_8UC3);
				cv::Point2f pt1 = cv::Point(50*2, vImg.rows - temp.rows);
				cv::Point2f pt2 = cv::Point(50*3, vImg.rows);
				cv::Rect rect = cv::Rect(pt1, pt2);
				rectangle(temp, cv::Point2f(0, 0), cv::Point2f(temp.cols, temp.rows), cv::Scalar(0, 0, 255), -1);
				cv::Mat a = vImg(rect);
				cv::addWeighted(a, 0.7, temp, 0.3, 0.0, vImg(rect));
			}
			
			/*sss.str("");
			sss << mStrPath.c_str() << "/plane.jpg";
			cv::imwrite(sss.str(), vImg);*/
			imshow("Output::PlaneEstimation", vImg); cv::waitKey(1);

			SetBoolDoingProcess(false, 1);
		}
	}
}

//////////////////////////////////////////////////////////////////////
//평면 추정 관련 함수들
bool UVR_SLAM::PlaneEstimator::PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, std::vector<UVR_SLAM::MapPoint*> vpMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
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
		calcUnitNormalVector(X);
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
		calcUnitNormalVector(X);
		if (X.at<float>(1) > 0.0)
			X *= -1.0;

		//std::cout << "Update::" << pPlane->matPlaneParam.t() << ", " << X.t() <<", "<<pPlane->mvpMPs.size()<<", "<<nReject<< std::endl;

		pPlane->matPlaneParam = X.clone();
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
	float d1 = abs(X1.at<float>(3));
	float d2 = abs(X2.at<float>(3));

	return abs(d1 - d2);

}

void UVR_SLAM::PlaneEstimator::CreatePlanarMapPoints(std::vector<UVR_SLAM::MapPoint*> mvpMPs, std::vector<UVR_SLAM::ObjectType> mvpOPs, UVR_SLAM::PlaneInformation* pPlane, cv::Mat invT) {
	
	int nTargetID = mpTargetFrame->GetFrameID();
	
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

	cv::Mat invP1 = invT.t()*pPlane->matPlaneParam.clone();
		
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

cv::Mat UVR_SLAM::PlaneEstimator::CalcPlaneRotationMatrix(cv::Mat P) {
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
	
	
	/*cv::Mat R1 = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngle(d1, 'z');
	cv::Mat R2 = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngle(d2, 'x');

	cv::Mat Nnew = R2.t()*R1.t()*normal;
	float d3 = atan2(Nnew.at<float>(0), Nnew.at<float>(2));
	cv::Mat Rfinal = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngles(d1, d2, d3, "ZXY");*/
	
	cv::Mat test1 = R*Nidealfloor;
	cv::Mat test3 = R.t()*normal;
	std::cout << "ATEST::" << P.t() << test1.t() << test3.t()<< std::endl;
	
	
	/*
	cv::Mat test2 = Rfinal*Nidealfloor;
	cv::Mat test4 = Rfinal.t()*normal;
	std::cout << d1 << ", " << d2 << ", " << d3 << std::endl;
	std::cout << "ATEST::" << P.t() << test1.t() << test2.t() << test3.t() << test4.t() << std::endl;*/

	return R;
}

bool UVR_SLAM::PlaneEstimator::isFloorPlaneInitialized() {
	std::unique_lock<std::mutex> lockTemp(mMutexInitFloorPlane);
	return mbInitFloorPlane;
}
void UVR_SLAM::PlaneEstimator::SetFloorPlaneInitialization(bool b){
	std::unique_lock<std::mutex> lockTemp(mMutexInitFloorPlane);
	mbInitFloorPlane = b;
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
	return normal3;  
}
//3차원값? 4차원으로?ㄴ
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