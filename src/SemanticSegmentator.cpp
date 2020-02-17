#include <SemanticSegmentator.h>
#include <System.h>
#include <SegmentationData.h>
#include <FrameWindow.h>
#include <PlaneEstimator.h>

std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;

UVR_SLAM::SemanticSegmentator::SemanticSegmentator():mbDoingProcess(false){
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(std::string _ip, int _port, int w, int h): ip(_ip), port(_port),mbDoingProcess(false), mnWidth(w), mnHeight(h), mpPrevTargetFrame(nullptr){
	UVR_SLAM::ObjectColors::Init();
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(const std::string & strSettingPath) :mbDoingProcess(false), mpPrevTargetFrame(nullptr) {
	UVR_SLAM::ObjectColors::Init();
	cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
	mbOn = static_cast<int>(fSettings["Segmentator.on"]) != 0;
	ip = fSettings["Segmentator.ip"];
	port = fSettings["Segmentator.port"];
	mnWidth = fSettings["Image.width"];
	mnHeight = fSettings["Image.height"];
	cx = fSettings["Camera.cx"];
	cy = fSettings["Camera.cy"];
	fSettings.release();
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}

UVR_SLAM::SemanticSegmentator::~SemanticSegmentator(){}


void UVR_SLAM::SemanticSegmentator::InsertKeyFrame(UVR_SLAM::Frame *pKF)
{
	mpSystem->SetDirPath(pKF->GetKeyFrameID());
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	std::cout << "Segmentator::" << mKFQueue.size() << std::endl;
	mKFQueue.push(pKF);
}

bool UVR_SLAM::SemanticSegmentator::CheckNewKeyFrames()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	return(!mKFQueue.empty());
}

void UVR_SLAM::SemanticSegmentator::ProcessNewKeyFrame()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mpTargetFrame = mKFQueue.front();
	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_SEGMENTED_FRAME);
	mpSystem->SetSegFrameID(mpTargetFrame->GetKeyFrameID());
	mKFQueue.pop();
}

void UVR_SLAM::SemanticSegmentator::Run() {

	JSONConverter::Init();

	while (1) {
		std::string mStrDirPath;
		if (CheckNewKeyFrames()) {
			SetBoolDoingProcess(true);
			ProcessNewKeyFrame();
			
			mStrDirPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());

			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			cv::Mat colorimg, resized_color, segmented;
			cv::cvtColor(mpTargetFrame->GetOriginalImage(), colorimg, CV_RGBA2BGR);
			colorimg.convertTo(colorimg, CV_8UC3);
			cv::resize(colorimg, resized_color, cv::Size(320, 180));
			//cv::resize(colorimg, resized_color, cv::Size(160, 90));

			//lock
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUseSegmentation);
			mpSystem->mbSegmentationEnd = false;
			//insert kyeframe to plane estimator
			mpPlaneEstimator->InsertKeyFrame(mpTargetFrame);
			//request post
			//리사이즈 안하면 칼라이미지로
			JSONConverter::RequestPOST(ip, port, resized_color, segmented, mpTargetFrame->GetFrameID());
			//cv::resize(segmented, segmented, colorimg.size());
			
			int n1 = mpTargetFrame->mPlaneDescriptor.rows;
			int n2 = mpTargetFrame->mWallDescriptor.rows;

			int nRatio = colorimg.rows / segmented.rows;
			//ratio 버전이 아닌 다르게
			ObjectLabeling(segmented, nRatio);

			//unlock & notify
			mpSystem->mbSegmentationEnd = true;
			lock.unlock();
			mpSystem->cvUseSegmentation.notify_all();
			
			//시간체크
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float tttt = duration / 1000.0;

			///////////////Conneted Component Labeling
			//image
			std::chrono::high_resolution_clock::time_point saa = std::chrono::high_resolution_clock::now();
			cv::Mat imgStructure = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgWall = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgFloor = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgCeil = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat seg_color = cv::Mat::zeros(segmented.size(), CV_8UC3);
			int minY = imgWall.rows;
			int maxY = 0;
			int minFloorX = imgWall.cols;
			int maxFloorX = 0;
			int minCeilX = imgWall.cols;
			int maxCeilX = 0;

			bool bMinCeilY = false;
			bool bMaxFloorY = false;

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
					}
				}
			}

			////기존의 칼라이미지와 세그멘테이션 결과를 합치는 부분
			////여기는 시각화로 보낼 수 있으면 보내는게 좋을 듯.
			cv::resize(seg_color, seg_color, colorimg.size());
			cv::addWeighted(seg_color, 0.4, colorimg, 0.6, 0.0, colorimg);
			cv::imshow("Output::Segmentation", colorimg);
			////기존의 칼라이미지와 세그멘테이션 결과를 합치는 부분

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

			//////tempwallline 값
			//1 = 바닥과 옆벽
			//2 = 바닥과 앞벽
			//3 = 천장과 옆벽
			//4 = 천장과 앞벽
			cv::Mat tempColor;
			cv::Mat tempWallLines = cv::Mat::zeros(colorimg.size(), CV_8UC1);
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
					if (val == 100){
						if (val_l == 255 || val_r == 255) {
							tempWallLines.at<uchar>(pt) = 1;
							cv::circle(tempColor, pt, 1, cv::Scalar(0, 0, 255), -1);
						}else if (val_u == 255 || val_d == 255) {
							cv::circle(tempColor, pt, 1, cv::Scalar(0, 255, 255), -1);
							tempWallLines.at<uchar>(pt) = 2;
						}
					}else if (val == 150) {
						if (val_l == 255 || val_r == 255) {
							cv::circle(tempColor, pt, 1, cv::Scalar(255, 0, 0), -1);
							tempWallLines.at<uchar>(pt) = 3;
						}
						else if (val_u == 255 || val_d == 255) {
							cv::circle(tempColor, pt, 1, cv::Scalar(255, 255, 0), -1);
							tempWallLines.at<uchar>(pt) = 4;
						}
					}
				}
			}

			////
			cv::Point2f tmpPt(cx / 2, cy / 2);
			cv::Point2f tmpPt2(160,90);
			cv::circle(tempColor, tmpPt, 3, cv::Scalar(255, 0, 255), -1);
			cv::circle(tempColor, tmpPt2, 3, cv::Scalar(0, 255, 0), -1);

			////앞의 벽과 만나는 라인을 구하기 위함.
			int minDistVerticalLine = colorimg.rows;
			for (int x = 0; x < tempWallLines.cols; x++) {
				int ceilY = 0;
				int floorY = tempWall.cols;
				for (int y = 0; y < tempWallLines.rows; y++) {
					int val = tempWallLines.at<uchar>(y, x);
					if (val == 2) {
						floorY = y;
					}
					if (val == 4)
						ceilY = y;
				}
				int dist = floorY - ceilY;
				if (dist < minDistVerticalLine) {
					minDistVerticalLine = dist;
					minY = ceilY;
					maxY = floorY;
				}
			}

			if (minY != colorimg.rows)
				bMinCeilY = true;
			if (maxY != 0)
				bMaxFloorY = true;

			cv::line(tempColor, cv::Point2f(0, minY), cv::Point2f(imgWall.cols, minY), cv::Scalar(0, 255, 0), 1);
			cv::line(tempColor, cv::Point2f(0, maxY), cv::Point2f(imgWall.cols, maxY), cv::Scalar(0, 255, 0), 1);

			for (int x = 0; x < tempWallLines.cols; x++) {
				int ay1 = max(minY - 5, 0);
				int ay2 = min(maxY + 5, colorimg.rows - 1);
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
			cv::line(tempColor, cv::Point2f(minFloorX, minY), cv::Point2f(minFloorX, maxY), cv::Scalar(0, 255, 0), 1);
			cv::line(tempColor, cv::Point2f(maxFloorX, minY), cv::Point2f(maxFloorX, maxY), cv::Scalar(0, 255, 0), 1);

			/*int minX = imgFloor.cols;
			int maxX = 0;
			
			for(int y = minY-1; y < minY+1; y++){
				for (int x = 0; x < imgFloor.cols; x++) {
					int val = imgFloor.at<uchar>(y, x);
					if (val == 100) {
						continue;
					}
					if (x < minX)
						minX = x;
					if (x > maxX)
						maxX = x;
				}
			}
			cv::line(tempWall, cv::Point2f(minX, 0), cv::Point2f(minX, imgWall.rows), cv::Scalar(0, 0, 0), mnLineSize);
			cv::line(tempWall, cv::Point2f(maxX, 0), cv::Point2f(maxX, imgWall.rows), cv::Scalar(0, 0, 0), mnLineSize);*/
			cv::imshow("temptemp wall2", tempColor);
			cv::imshow("temptemp wall", tempWall);

			////test save image
			std::stringstream ssas;
			ssas << mStrDirPath.c_str() << "/test_labeling2.jpg";
			cv::imwrite(ssas.str(), tempColor);
			////test save image

			//
			//
			////나누어진 이미지를 바탕으로 레이블링 수행

			cv::Mat floorCCL, ceilCCL;
			cv::Mat floorStat, ceilStat;
			ConnectedComponentLabeling(imgFloor, floorCCL, floorStat);
			ConnectedComponentLabeling(imgCeil,  ceilCCL,  ceilStat);

			
			imshow("floor", floorCCL);
			imshow("ceil", ceilCCL);



			std::chrono::high_resolution_clock::time_point see = std::chrono::high_resolution_clock::now();
			auto durationa = std::chrono::duration_cast<std::chrono::milliseconds>(see - saa).count();
			float tttt5 = durationa / 1000.0;
			///////////////Conneted Component Labeling


			//////디버깅 값 전달
			std::stringstream ssa;
			ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() << " : " << tttt << ", " << tttt5 << "||" << n1 << ", " << n2;
			//ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() << " : " << tttt << ", " << tttt5 << "||" << n1 << ", " << n2 << "::" << numOfLables;
			mpSystem->SetSegmentationString(ssa.str());
			//////디버깅 값 전달


			//cv::resize(seg_color, seg_color, colorimg.size());

			//imshow("segsegseg", seg_color);
			////PlaneEstimator
			//if (!mpPlaneEstimator->isDoingProcess()) {
			//	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_LAYOUT_FRAME);
			//	mpPlaneEstimator->SetTargetFrame(mpTargetFrame);
			//	mpPlaneEstimator->SetBoolDoingProcess(true, 3);
			//}

			//피쳐 레이블링 결과 확인
			/*cv::Mat test = mpTargetFrame->undistorted.clone();
			for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
				UVR_SLAM::ObjectType type = mpTargetFrame->GetObjectType(i);
				if(type != OBJECT_NONE)
				circle(test, mpTargetFrame->mvKeyPoints[i].pt, 3, ObjectColors::mvObjectLabelColors[type], 1);
				UVR_SLAM::MapPoint* pMP = mpTargetFrame->GetMapPoint(i);
				if (pMP) {
				if (pMP->isDeleted())
				continue;
				UVR_SLAM::ObjectType type2 = pMP->GetObjectType();
				if (type2 != OBJECT_NONE)
				circle(test, mpTargetFrame->mvKeyPoints[i].pt, 1, ObjectColors::mvObjectLabelColors[type2], -1);
				}
			}*/
			//imshow("segmendted feature", test);

			/*std::stringstream ss;
			ss << "../../bin/segmentation/res_keypoints_" << mpTargetFrame->GetFrameID() << ".jpg";
			imwrite(ss.str(), test);*/
			//cv::imwrite("../../bin/segmentation/res/label_kp.jpg", test);
			
			std::stringstream ss;
			ss << mStrDirPath.c_str() << "/segmentation.jpg";
			cv::imwrite(ss.str(), segmented);
			ss.str("");
			ss << mStrDirPath.c_str() << "/segmentation_color.jpg";
			cv::imwrite(ss.str(), colorimg);
			cv::waitKey(1);

			SetBoolDoingProcess(false);
		}
	}
}
void UVR_SLAM::SemanticSegmentator::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}
void UVR_SLAM::SemanticSegmentator::SetFrameWindow(UVR_SLAM::FrameWindow* pWindow) {
	mpFrameWindow = pWindow;
}
void UVR_SLAM::SemanticSegmentator::SetPlaneEstimator(UVR_SLAM::PlaneEstimator* pEstimator) {
	mpPlaneEstimator = pEstimator;
}
void UVR_SLAM::SemanticSegmentator::SetTargetFrame(Frame* pFrame) {
	mpTargetFrame = pFrame;
}
void UVR_SLAM::SemanticSegmentator::SetBoolDoingProcess(bool b) {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	mbDoingProcess = b;
}
bool UVR_SLAM::SemanticSegmentator::isDoingProcess() {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	return mbDoingProcess;
}
bool UVR_SLAM::SemanticSegmentator::isRun() {
	return mbOn;
}

/////////////////////////////////////////////////////////////////////////////////////////
//모든 특징점과 트래킹되고 있는 맵포인트를 레이블링함.
void UVR_SLAM::SemanticSegmentator::ObjectLabeling(cv::Mat masked, int ratio) {

	//레이블링을 매칭에 이용하기 위한 디스크립터 설정.
	/*bool bMatch = false;
	if (mpTargetFrame->mPlaneDescriptor.rows == 0)
		bMatch = true;*/
	//초기화 하지 말고 중복 체크하면서 추가하도록 해야 함.
	bool bMatch = true;
	/*mpTargetFrame->mPlaneDescriptor = cv::Mat::zeros(0, mpTargetFrame->matDescriptor.cols, mpTargetFrame->matDescriptor.type());
	mpTargetFrame->mPlaneIdxs.clear();
	mpTargetFrame->mWallDescriptor = cv::Mat::zeros(0, mpTargetFrame->matDescriptor.cols, mpTargetFrame->matDescriptor.type());
	mpTargetFrame->mWallIdxs.clear();*/

	//오브젝트 및 맵포인트 정보 획득
	std::vector<UVR_SLAM::ObjectType> vObjTypes(mpTargetFrame->mvKeyPoints.size(), UVR_SLAM::OBJECT_NONE);
	auto mvpMPs = mpTargetFrame->GetMapPoints();
	auto mvMapObject = mpTargetFrame->mvMapObjects;
	//다른 프레임들과의 매칭 결과를 반영해야 함.

	for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {

		//get object type
		cv::Point2f pt = mpTargetFrame->mvKeyPoints[i].pt;
		pt.x /= ratio;
		pt.y /= ratio;
		int val = masked.at<uchar>(pt);
		UVR_SLAM::ObjectType type = vObjTypes[i];
		
		bool bMP = false;
		UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		if (pMP) {
			if (!pMP->isDeleted()) {
				if (mpTargetFrame->isInFrustum(pMP, 0.5)) {
					bMP = true;
				}
			}
		}

		////find object type in map
		//auto val2 = static_cast<ObjectType>(val);
		//auto iter = mpTargetFrame->mvMapObjects[i].find(val2);
		//if (iter != mpTargetFrame->mvMapObjects[i].end()) {
		//	iter->second++;
		//}
		//else {
		//	mpTargetFrame->mvMapObjects[i].insert(std::make_pair(val2, 1));
		//}
		//
		//int maxVal = 0;
		//
		//for (std::multimap<ObjectType, int, std::greater<int>>::iterator iter = mpTargetFrame->mvMapObjects[i].begin(); iter != mpTargetFrame->mvMapObjects[i].end(); iter++) {
		//	//std::cout << "begin ::" << iter->first << ", " << iter->second << std::endl;
		//	if (maxVal < iter->second) {
		//		val2 = iter->first;
		//		maxVal = iter->second;
		//	}
		//}

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
			type = ObjectType::OBJECT_WALL;
			if (bMP){
				mpTargetFrame->mspWallMPs.insert(pMP);
				mpFrameWindow->mspWallMPs.insert(pMP);
			}
			if (mpTargetFrame->mLabelStatus.at<uchar>(i) == 0) {
				mpTargetFrame->mLabelStatus.at<uchar>(i) = val;
				mpTargetFrame->mWallDescriptor.push_back(mpTargetFrame->matDescriptor.row(i));
				mpTargetFrame->mWallIdxs.push_back(i);
			}
			break;
		case 4:
			type = ObjectType::OBJECT_FLOOR;
			if (bMP){
				mpTargetFrame->mspFloorMPs.insert(pMP);
				mpFrameWindow->mspFloorMPs.insert(pMP);
			}
			if (mpTargetFrame->mLabelStatus.at<uchar>(i) == 0) {
				mpTargetFrame->mLabelStatus.at<uchar>(i) = val;
				mpTargetFrame->mPlaneDescriptor.push_back(mpTargetFrame->matDescriptor.row(i));
				mpTargetFrame->mPlaneIdxs.push_back(i);
			}
			break;
		case 6:
			type = ObjectType::OBJECT_CEILING;
			if (bMP){
				mpTargetFrame->mspCeilMPs.insert(pMP);
				mpFrameWindow->mspCeilMPs.insert(pMP);
			}
			break;
		default:
			break;
		}
		vObjTypes[i] = type;
		if (bMP) {
			pMP->SetObjectType(type);
		}
	}
	mpTargetFrame->SetObjectVector(vObjTypes);

}
void UVR_SLAM::SemanticSegmentator::ObjectLabeling() {

	std::vector<UVR_SLAM::ObjectType> vObjTypes(mpTargetFrame->mvKeyPoints.size(), UVR_SLAM::OBJECT_NONE);

	for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
		cv::Point2f pt = mpTargetFrame->mvKeyPoints[i].pt;
		for (int j = 0; j < mVecLabelMasks.size(); j++) {
			if (mVecLabelMasks[j].at<uchar>(pt) == 255) {
				UVR_SLAM::ObjectType type = static_cast<ObjectType>(j);
				//mpTargetFrame->SetObjectType(type, i);
				vObjTypes[i] = type;
				if (mpTargetFrame->mvbMPInliers[i]) {
					UVR_SLAM::MapPoint* pMP = mpTargetFrame->mvpMPs[i];
					if (pMP) {
						if (pMP->isDeleted())
							continue;
						cv::Point2f p2D;
						cv::Mat pcam;
						if (pMP->Projection(p2D, pcam, mpTargetFrame->GetRotation(), mpTargetFrame->GetTranslation(), mpTargetFrame->mK, mnWidth, mnHeight)) {
							pMP->SetObjectType(type);
						}//projection
					}//pMP
				}//inlier
			}
		}
	}
	mpTargetFrame->SetObjectVector(vObjTypes);

}

//참조하고 있는 오브젝트 클래스 별로 마스크 이미지를 생성하고 픽셀 단위로 마스킹함.
void UVR_SLAM::SemanticSegmentator::SetSegmentationMask(cv::Mat segmented) {
	mVecLabelMasks.clear();
	for (int i = 0; i < ObjectColors::mvObjectLabelColors.size()-1; i++) {
		cv::Mat temp = cv::Mat::zeros(segmented.size(), CV_8UC1);
		mVecLabelMasks.push_back(temp.clone());
	}
	for (int y = 0; y < segmented.rows; y++) {
		for (int x = 0; x < segmented.cols; x++) {
			cv::Vec3b tempColor = segmented.at<Vec3b>(y, x);
			for (int i = 0; i < ObjectColors::mvObjectLabelColors.size()-1; i++) {
				if (tempColor == ObjectColors::mvObjectLabelColors[i]) {
					mVecLabelMasks[i].at<uchar>(y, x) = 255;
				}
			}
		}
	}
}

bool UVR_SLAM::SemanticSegmentator::ConnectedComponentLabeling(cv::Mat img, cv::Mat& dst, cv::Mat& stat) {
	dst = img.clone();
	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(img, img_labels, stats, centroids, 8, CV_32S);

	if (numOfLables == 0)
		return false;

	cv::Mat img_color = img.clone();
	img_color.convertTo(img_color, CV_8UC3);
	cv::cvtColor(img_color, img_color, CV_GRAY2BGR);

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
	int left = stats.at<int>(maxIdx, CC_STAT_LEFT);
	int top = stats.at<int>(maxIdx, CC_STAT_TOP);
	int width = stats.at<int>(maxIdx, CC_STAT_WIDTH);
	int height = stats.at<int>(maxIdx, CC_STAT_HEIGHT);
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


