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
			//�������� ���ϸ� Į���̹�����
			JSONConverter::RequestPOST(ip, port, resized_color, segmented, mpTargetFrame->GetFrameID());
			//cv::resize(segmented, segmented, colorimg.size());
			
			int n1 = mpTargetFrame->mPlaneDescriptor.rows;
			int n2 = mpTargetFrame->mWallDescriptor.rows;

			int nRatio = colorimg.rows / segmented.rows;
			//ratio ������ �ƴ� �ٸ���
			ObjectLabeling(segmented, nRatio);
			mpTargetFrame->matSegmented = segmented.clone();
			//unlock & notify
			mpSystem->mbSegmentationEnd = true;
			lock.unlock();
			mpSystem->cvUseSegmentation.notify_all();
			
			//�ð�üũ
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float tttt = duration / 1000.0;

			

			/*

			//////test save image
			//std::stringstream ssas;
			//ssas << mStrDirPath.c_str() << "/test_labeling2.jpg";
			//cv::imwrite(ssas.str(), tempColor);
			//////test save image

			////
			////
			//////�������� �̹����� �������� ���̺� ����
			//

			////�ٴڰ� ���� ������ ǥ���� ��.
			
			//
			//////���� ������ ���� �׽�Ʈ.
			////cv::Mat dists = cv::Mat::zeros(1, floorCCL.cols, CV_32SC1);
			////cv::Mat graph = cv::Mat::zeros(400, floorCCL.cols, CV_8UC1);
			////for (int x = 0; x < sumCCL.cols; x++) {
			////	int count = countNonZero(sumCCL.col(x));
			////	dists.at<int>(x) = count;
			////	
			////	int val = 400 - count;
			////	cv::line(graph, cv::Point2f(x, 400), cv::Point2f(x, val), cv::Scalar(255));
			////}

			////cv::Ptr<cv::LineSegmentDetector> pLSD = createLineSegmentDetector();
			////std::vector<cv::Vec4i> lines;
			////pLSD->detect(graph, lines);
			////graph.convertTo(graph, CV_8UC3);
			////cv::cvtColor(graph, graph, CV_GRAY2BGR);
			////for (int i = 0; i < lines.size(); i++) {
			////	Vec4i v = lines[i];
			////	Point from(v[0], v[1]);
			////	Point to(v[2], v[3]);
			////	Point2f diff = from - to;
			////	float dist = sqrt(diff.dot(diff));
			////	//if(dist < 20)
			////	//cv::line(fLineImg, from, to, cv::Scalar(255,0,0));
			////	//else
			////	cv::line(graph, from, to, cv::Scalar(0, 0, 255));
			////}
			////cv::imshow("graph", graph);

			///*i
			//imgFloor*/
			//
			////std::cout << floorY << ", " << ceilY << std::endl;
			//cv::Mat fLineImg = floorCCL + ceilCCL;
			//
			//
			//

			////cv::line(floorCCL, cv::Point2f(0, floorY), cv::Point2f(imgFloor.cols, floorY), cv::Scalar(255, 255, 255));
			////cv::line(ceilCCL, cv::Point2f(0, ceilY), cv::Point2f(imgCeil.cols, ceilY), cv::Scalar(255, 255, 255));
			//imshow("floor", floorCCL+ ceilCCL);
			//imshow("fline", fLineImg);
			////imshow("ceil", ceilCCL);


			//std::chrono::high_resolution_clock::time_point see = std::chrono::high_resolution_clock::now();
			//auto durationa = std::chrono::duration_cast<std::chrono::milliseconds>(see - saa).count();
			//float tttt5 = durationa / 1000.0;
			/////////////////Conneted Component Labeling


			//////����� �� ����
			std::stringstream ssa;
			ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() << " : " << tttt << "||" << n1 << ", " << n2;
			//ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() << " : " << tttt << ", " << tttt5 << "||" << n1 << ", " << n2 << "::" << numOfLables;
			mpSystem->SetSegmentationString(ssa.str());
			//////����� �� ����


			//cv::resize(seg_color, seg_color, colorimg.size());

			//imshow("segsegseg", seg_color);
			////PlaneEstimator
			//if (!mpPlaneEstimator->isDoingProcess()) {
			//	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_LAYOUT_FRAME);
			//	mpPlaneEstimator->SetTargetFrame(mpTargetFrame);
			//	mpPlaneEstimator->SetBoolDoingProcess(true, 3);
			//}

			//���� ���̺� ��� Ȯ��
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
//��� Ư¡���� Ʈ��ŷ�ǰ� �ִ� ������Ʈ�� ���̺���.
void UVR_SLAM::SemanticSegmentator::ObjectLabeling(cv::Mat masked, int ratio) {

	//���̺��� ��Ī�� �̿��ϱ� ���� ��ũ���� ����.
	/*bool bMatch = false;
	if (mpTargetFrame->mPlaneDescriptor.rows == 0)
		bMatch = true;*/
	//�ʱ�ȭ ���� ���� �ߺ� üũ�ϸ鼭 �߰��ϵ��� �ؾ� ��.
	bool bMatch = true;
	/*mpTargetFrame->mPlaneDescriptor = cv::Mat::zeros(0, mpTargetFrame->matDescriptor.cols, mpTargetFrame->matDescriptor.type());
	mpTargetFrame->mPlaneIdxs.clear();
	mpTargetFrame->mWallDescriptor = cv::Mat::zeros(0, mpTargetFrame->matDescriptor.cols, mpTargetFrame->matDescriptor.type());
	mpTargetFrame->mWallIdxs.clear();*/

	//������Ʈ �� ������Ʈ ���� ȹ��
	std::vector<UVR_SLAM::ObjectType> vObjTypes(mpTargetFrame->mvKeyPoints.size(), UVR_SLAM::OBJECT_NONE);
	auto mvpMPs = mpTargetFrame->GetMapPoints();
	auto mvMapObject = mpTargetFrame->mvMapObjects;
	//�ٸ� �����ӵ���� ��Ī ����� �ݿ��ؾ� ��.

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
		case 1://��
		case 9://����â
		case 11://ĳ���
		case 15://��
		case 23: //�׸�
		case 36://����
		//case 43://���
		case 44://����
		//case 94://�����
		case 101://������
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

//�����ϰ� �ִ� ������Ʈ Ŭ���� ���� ����ũ �̹����� �����ϰ� �ȼ� ������ ����ŷ��.
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




