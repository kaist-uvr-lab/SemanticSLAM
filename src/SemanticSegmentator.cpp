#include <SemanticSegmentator.h>
#include <System.h>
#include <Frame.h>
#include <SegmentationData.h>
#include <FrameWindow.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <LocalMapper.h>
#include <Visualizer.h>
#include <CandidatePoint.h>
#include <MapPoint.h>
#include <FrameGrid.h>
#include <Map.h>
#include <LocalBinaryPatternProcessor.h>
#include <Database.h>
#include <future>
#include <FeatureMatchingWebAPI.h>
#include <WebAPI.h>
//#include "lbplibrary.hpp"

std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;

UVR_SLAM::SemanticSegmentator::SemanticSegmentator():mbDoingProcess(false){
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(std::string _ip, int _port, int w, int h): ip(_ip), port(_port),mbDoingProcess(false), mnWidth(w), mnHeight(h), mpPrevFrame(nullptr){
	UVR_SLAM::ObjectColors::Init();
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(System* pSys, const std::string & strSettingPath) :mbDoingProcess(false), mpPrevFrame(nullptr) {
	UVR_SLAM::ObjectColors::Init();
	cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
	mbOn = static_cast<int>(fSettings["Segmentator.on"]) != 0;
	ip = fSettings["Segmentator.ip"];
	port = fSettings["Segmentator.port"];
	fSettings.release();
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
	
	mpSystem = pSys;
}

UVR_SLAM::SemanticSegmentator::~SemanticSegmentator(){}


void UVR_SLAM::SemanticSegmentator::InsertKeyFrame(UVR_SLAM::Frame *pKF)
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	//std::cout << "Segmentator::" << mKFQueue.size() << std::endl;
	std::queue<Frame*> emptyQueue;
	std::swap(mKFQueue, emptyQueue);
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
	mpPrevFrame = mpTargetFrame;
	mpTargetFrame = mKFQueue.front();
	//mKFQueue.pop();
}

void UVR_SLAM::SemanticSegmentator::Init() {
	mpMap = mpSystem->mpMap;
	mpLocalMapper = mpSystem->mpLocalMapper;
	mpPlaneEstimator = mpSystem->mpPlaneEstimator;
	mpVisualizer = mpSystem->mpVisualizer;
	mpLBPProcessor = mpSystem->mpLBPProcessor;
	mpDatabase = mpSystem->mpDatabase;
	mnWidth = mpSystem->mnWidth;
	mnHeight = mpSystem->mnHeight;
}

unsigned long long ConvertID(cv::Mat hist, int numPatterns, int numIDs) {
	unsigned long long id = 0;
	unsigned long long base = 1;
	for (size_t i = 0, iend = hist.cols; i < iend; i++) {
		//std::cout << "curr base id::" << base << std::endl;
		/*unsigned long long tempID = base-1;*/
		int val = hist.at<int>(i);
		if (val == numIDs)
			val--;
		/*if(val > 0)
			id += (tempID + val);*/
		id += (base*val);
		base *= numIDs;
		//std::cout << "next base id::" <<base<< std::endl;
	}
	//std::cout << id << "::" << hist << std::endl;
	return id;
}

void UVR_SLAM::SemanticSegmentator::Run() {

	//Base64Encoder::Init();
	
	////LBP param
	lbplibrary::LBP* lbp = new lbplibrary::SCSLBP(2, 4);
	mpSystem->mPlaneHist = cv::Mat::zeros(1, lbp->numPatterns, CV_32SC1);
	cv::Mat mWallHist = cv::Mat::zeros(1, lbp->numPatterns, CV_32SC1);
	int mnLabel_floor = 4;
	int mnLabel_wall = 1;
	int mnLabel_ceil = 6;
	cv::Point minLoc, maxLoc;
	maxLoc = cv::Point(-1, -1);
	cv::Point minLoc1, maxLoc1;
	maxLoc1 = cv::Point(-1, -1);
	cv::Point minLoc2, maxLoc2;
	maxLoc2 = cv::Point(-1, -1);
	int minIdx, maxIdx;
	maxIdx = -1;
	unsigned char maxOverlapCode, maxPlaneCode, maxWallCode;
	cv::Mat matCandidatePlaneCode, matCandidateWallCode;

	int numPatterns = lbp->numPatterns; //2^N
	int nHalfPatchSize = 5;
	int nPatchSize = nHalfPatchSize * 2;
	int numIDs = nPatchSize;// nHalfPatchSize;// nHalfPatchSize;	   //patch size / n
	int nDicrete = nPatchSize*nPatchSize / numIDs;
	std::map<unsigned long long, cv::Mat> mmObjLBP;
	////LBP param

	//unsigned long long nMaxID = (unsigned long long)pow(10, numPatterns);
	//unsigned long nMaxID2 = (unsigned long)pow(numPatterns, numIDs);
	//unsigned int nMaxID3 = (unsigned int)pow(numPatterns, numIDs);
	//unsigned long long val = 1LL;
	//for (int i = 0; i < numPatterns; i++) {
	//	val *= 10;
	//	std::cout << val <<"::"<<i+1<< std::endl;
	//}
	//std::cout << "maxID::" << LLONG_MAX<<", "<<INT_MAX<<", "<< nDicrete <<"::"<<nMaxID<<", "<<nMaxID2<<", "<< (LLONG_MAX > nMaxID) << std::endl;

	//cv::Mat a = cv::Mat::zeros(1, numPatterns, CV_32SC1);
	//a.at<int>(1) = 3;
	////std::vector<cv::Mat> mvLBPDesc(nMaxID,a);
	////std::cout << "make vector::" <<mvLBPDesc.size()<< std::endl;
	//ConvertID(a, numPatterns, numIDs);

	//cv::Mat b = cv::Mat::zeros(1, numPatterns, CV_32SC1);
	//b.at<int>(13) = 4;
	//ConvertID(b, numPatterns, numIDs);

	//UVR_SLAM::ObjectColors::mvObjectLabelColors

	/*std::map<cv::Mat, Frame*> mmpTest;
	cv::Mat A = cv::Mat::zeros(1, 10, CV_8UC1);
	cv::Mat B = cv::Mat::zeros(1, 10, CV_8UC1);
	cv::Mat C = cv::Mat::ones(1, 10, CV_8UC1);
	cv::Mat D = cv::Mat::ones(1, 10, CV_8UC1)*10;
	auto a = A > B;
	
	mmpTest.insert(std::make_pair(A, mpTargetFrame));
	mmpTest.insert(std::make_pair(B, mpTargetFrame));
	mmpTest.insert(std::make_pair(C, mpTargetFrame));
	mmpTest.insert(std::make_pair(D, mpTargetFrame));
	std::cout << "asdfasdfasdf::" << mmpTest.size() << std::endl;*/
	////LBP

	while (1) {
		std::string mStrDirPath;

		auto lambda_api_match = [](std::string ip, int port, int id1, int id2, int n) {
			std::vector<int> matches;
			cv::Mat res;
			WebAPI* api = new WebAPI(ip, port);
			std::cout << "lambda = " << id1 << ", " << id2 << "= start" << std::endl;
			WebAPIDataConverter::ConvertStringToMatches(api->Send("match", WebAPIDataConverter::ConvertNumberToString(id1, id2)).c_str(), n, matches);
			std::cout << "lambda = " << id1 << ", " << id2 << "= end = " << matches.size() << std::endl;

			////디버깅
			/*cv::Mat aimg = f1->GetOriginalImage().clone();
			cv::Mat bimg = f2->GetOriginalImage().clone();
			cv::Point2f ptBottom = cv::Point2f(aimg.cols, 0);
			cv::Rect mergeRect1 = cv::Rect(0, 0, aimg.cols, aimg.rows);
			cv::Rect mergeRect2 = cv::Rect(aimg.cols, 0, aimg.cols, aimg.rows);
			res = cv::Mat::zeros(aimg.rows, aimg.cols * 2, aimg.type());
			aimg.copyTo(res(mergeRect1));
			bimg.copyTo(res(mergeRect2));*/
			////디버깅

			//for (size_t i = 0, iend = matches.size(); i < iend ; i++) {
			//	int idx = matches[i];
			//	if (idx < 0)
			//		continue;
			//	auto prevPt = f2->mvPts[idx];
			//	auto currPt = f1->mvPts[i];
			//	/*cv::circle(res, currPt, 3, cv::Scalar(0, 255, 0), -1);
			//	cv::circle(res, prevPt + ptBottom, 3, cv::Scalar(0, 255, 0), -1);
			//	cv::line(res, currPt, prevPt + ptBottom, cv::Scalar(255, 255, 0), 1);*/
			//}

			return res;
		};

		if (CheckNewKeyFrames()) {
			
			SetBoolDoingProcess(true);
			ProcessNewKeyFrame();
			if (mpSystem->mbInitialized)
			{

				//////thread로 api 전송 테스트 = 성공
				/*std::vector<int> testMatches1, testMatches2;
				auto f1 = std::async([](std::string ip, int port, int id1, int id2, std::vector<int>& matches) {
				WebAPI* api = new WebAPI(ip, port);
				WebAPIDataConverter::ConvertStringToLabels(api->Send("match", WebAPIDataConverter::ConvertNumberToString(id1, id2)).c_str(), matches);
				}, ip, port, mpTargetFrame->mnFrameID, mpPrevKeyFrame->mnFrameID, testMatches1);
				if (mpPPrevKeyFrame)
				auto f1 = std::async([](std::string ip, int port, int id1, int id2, std::vector<int>& matches) {
				WebAPI* api = new WebAPI(ip, port);
				WebAPIDataConverter::ConvertStringToLabels(api->Send("match", WebAPIDataConverter::ConvertNumberToString(id1, id2)).c_str(), matches);
				}, ip, port, mpTargetFrame->mnFrameID, mpPPrevKeyFrame->mnFrameID, testMatches2);*/
				//////thread로 api 전송 테스트 = 성공

				
				//std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
				//auto vpKFs = mpTargetFrame->GetConnectedKFs(8);
				//int nTestSize = vpKFs.size();
				//std::vector<cv::Mat> vRes;
				////for (size_t i = 0, iend = vpKFs.size(); i < iend; i+=2)
				////{
				////	auto pKF = vpKFs[i];
				////	/*std::vector<int> matches;
				////	auto f1 = std::async([](std::string ip, int port, int id1, int id2, std::vector<int>& matches) {
				////		WebAPI* api = new WebAPI(ip, port);
				////		WebAPIDataConverter::ConvertStringToLabels(api->Send("match", WebAPIDataConverter::ConvertNumberToString(id1, id2)).c_str(), matches);
				////	}, ip, port, mpTargetFrame->mnFrameID, pKF->mnFrameID, matches);
				////	f1.get();*/
				////	auto f = std::async(std::launch::async, lambda_api_match, ip, port, mpTargetFrame->mnFrameID, pKF->mnFrameID);
				////	//vRes.push_back(f.get());
				////	//std::cout <<i<< "= match num = " << f.get().size() << std::endl;
				////}

				///*if (vRes.size() > 1) {
				//	cv::imshow("test1::", vRes[0]);
				//	cv::imshow("test2::", vRes[vRes.size()-1]);
				//	cv::waitKey(1);
				//}*/
				//

				//std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
				//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
				//float tttt = duration / 1000.0;
				//std::cout << "match time = " << tttt << std::endl;

				//////////////Segmentation
				//auto fseg = std::async([](std::string ip, int port, int id) {
				//	WebAPI* api = new WebAPI(ip, port);
				//	//api->Send("receiveimage", WebAPIDataConverter::ConvertImageToString(pF->GetOriginalImage(), pF->mnFrameID));
				//	api->Send("segment", WebAPIDataConverter::ConvertNumberToString(id));
				//	//WebAPIDataConverter::ConvertStringToDepthImage(api->Send("depthestimate", WebAPIDataConverter::ConvertNumberToString(pF->mnFrameID)).c_str(), depthImg);
				//}, "143.248.96.81", port, mpTargetFrame->mnFrameID);
				//////////////Segmentation

				//////////////Depth 추정
				cv::Mat depthImg;
				auto f1 = std::async([](std::string ip, int port, Frame* pF, cv::Mat& depthImg, cv::Mat invK, Map* map) {
					WebAPI* api = new WebAPI(ip, port);
					api->Send("receiveimage", WebAPIDataConverter::ConvertImageToString(pF->GetOriginalImage(), pF->mnFrameID));
					WebAPIDataConverter::ConvertStringToDepthImage(api->Send("depthestimate", WebAPIDataConverter::ConvertNumberToString(pF->mnFrameID)).c_str(), depthImg);
					std::cout << "depth estimation = " << pF->mnFrameID << std::endl;

					std::vector<std::tuple<cv::Point2f, float, int>> vecTuples;

					cv::Mat R, t;
					pF->GetPose(R, t);

					////depth 정보 저장 및 포인트와 웨이트 정보를 튜플로 저장
					cv::Mat Rcw2 = R.row(2);
					Rcw2 = Rcw2.t();
					float zcw = t.at<float>(2);
					auto vpMPs = pF->GetMapPoints();
					for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
						auto pMPi = vpMPs[i];
						if (!pMPi || pMPi->isDeleted())
							continue;
						auto pt = pF->mvPts[i];
						cv::Mat x3Dw = pMPi->GetWorldPos();
						float z = (float)Rcw2.dot(x3Dw) + zcw;
						std::tuple<cv::Point2f, float, int> data = std::make_tuple(std::move(pt), 1.0 / z, pMPi->GetNumConnectedFrames());//cv::Point2f(pt.x / 2, pt.y / 2)
						vecTuples.push_back(data);
					}

					////웨이트와 포인트 정보로 정렬
					std::sort(vecTuples.begin(), vecTuples.end(),
						[](std::tuple<cv::Point2f, float, int> const &t1, std::tuple<cv::Point2f, float, int> const &t2) {
						if (std::get<2>(t1) == std::get<2>(t2)) {
							return std::get<0>(t1).x != std::get<0>(t2).x ? std::get<0>(t1).x > std::get<0>(t2).x : std::get<0>(t1).y > std::get<0>(t2).y;
						}
						else {
							return std::get<2>(t1) > std::get<2>(t2);
						}
					}
					);

					////파라메터 검색 및 뎁스 정보 복원
					int nTotal = 20;
					if (vecTuples.size() > nTotal) {
						int nData = nTotal;
						cv::Mat A = cv::Mat::ones(nData, 2, CV_32FC1);
						cv::Mat B = cv::Mat::zeros(nData, 1, CV_32FC1);
						
						for (size_t i = 0; i < nData; i++) {
							auto data = vecTuples[i];
							auto pt = std::get<0>(data);
							auto invdepth = std::get<1>(data);
							auto nConnected = std::get<2>(data);

							float p = depthImg.at<float>(pt);
							A.at<float>(i, 0) = invdepth;
							B.at<float>(i) = p;
						}

						cv::Mat X = A.inv(cv::DECOMP_QR)*B;
						float a = X.at<float>(0);
						float b = X.at<float>(1);

						depthImg = (depthImg - b) / a;
						for (int x = 0, cols = depthImg.cols; x < cols; x++) {
							for (int y = 0, rows = depthImg.rows; y < rows; y++) {
								float val = 1.0 / depthImg.at<float>(y, x);
								/*if (val < 0.0001)
								val = 0.5;*/
								depthImg.at<float>(y, x) = val;
							}
						}
						//복원 확인
						cv::Mat Rinv, Tinv;
						pF->GetInversePose(Rinv, Tinv);
						map->ClearTempMPs();
						int inc = 10;
						for (size_t row = inc, rows = depthImg.rows; row < rows; row+= inc) {
							for (size_t col = inc, cols = depthImg.cols; col < cols; col+= inc) {
								cv::Point2f pt(col, row);
								float depth = depthImg.at<float>(pt);
								if (depth < 0.0001)
									continue;
								cv::Mat a = Rinv*(invK*(cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.0)*depth) + Tinv;
								map->AddTempMP(a);
							}
						}
						/*for (size_t i = 0, iend = vecTuples.size(); i < iend; i++) {
							auto data = vecTuples[i];
							auto pt = std::get<0>(data);
							float depth = depthImg.at<float>(pt);
							if (depth < 0.0001)
								continue;
							cv::Mat a = Rinv*(invK*(cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.0)*depth) + Tinv;
							map->AddTempMP(a);
						}*/
						/*imshow("depth test::", depthImg);
						cv::waitKey(1);*/
					}
					return depthImg;
				}, "143.248.95.112", port, mpTargetFrame, depthImg, mpSystem->mInvK, mpMap);
				depthImg = f1.get();
				cv::normalize(depthImg, depthImg, 0,255, cv::NORM_MINMAX, CV_8UC1);
				cv::resize(depthImg, depthImg, cv::Size(depthImg.cols / 2, depthImg.rows / 2));
				cv::cvtColor(depthImg, depthImg, CV_GRAY2BGR);
				//depthImg.convertTo(depthImg, CV_8UC3);
				mpVisualizer->SetOutputImage(depthImg, 2);
				//////////////Depth 추정
			}
			
			SetBoolDoingProcess(false);
			continue;

			//std::cout << "segmentation::start::" << std::endl;
			
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			cv::Mat colorimg, resized_color, segmented;
			/*cv::cvtColor(mpTargetFrame->GetOriginalImage(), colorimg, CV_RGBA2BGR);
			colorimg.convertTo(colorimg, CV_8UC3);*/
			cv::resize(mpTargetFrame->GetOriginalImage().clone(), resized_color, cv::Size(mnWidth/2, mnHeight/2));
			//cv::resize(colorimg, resized_color, cv::Size(160, 90));
			
			//request post
			//리사이즈 안하면 칼라이미지로
			int status = 0;

			JSONConverter::RequestPOST(ip, port, resized_color, segmented, mpTargetFrame->mnFrameID, status);

			int nRatio = colorimg.rows / segmented.rows;
			//ratio 버전이 아닌 다르게
			/////////미리 지정된 오브젝트만 레이블링 하는 코드
			//ImageLabeling(segmented, mpTargetFrame->matLabeled);
			mpTargetFrame->matLabeled = segmented.clone();
			mpTargetFrame->matSegmented = segmented.clone();
			
			cv::Mat testImg = mpTargetFrame->GetOriginalImage().clone();
			////그리드 사이즈
			int nGridSize = mpSystem->mnRadius * 2;
			
			////////////LBP process
			////LBP용 가우시안 블러
			/*cv::Mat blurred, lbpImg, lbpHist;
			cv::GaussianBlur(mpTargetFrame->matFrame, blurred, cv::Size(7, 7), 5, 3, cv::BORDER_CONSTANT);
			lbpImg = mpLBPProcessor->ConvertDescriptor(blurred);
			cv::Mat resized_lbp;
			cv::resize(lbpImg, resized_lbp, cv::Size(lbpImg.cols / 2, lbpImg.rows / 2));
			cv::normalize(resized_lbp, resized_lbp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			cv::cvtColor(resized_lbp, resized_lbp, CV_GRAY2BGR);
			mpVisualizer->SetOutputImage(resized_lbp, 2);*/
			////////////LBP process

			////세그멘테이션 칼라 전달
			int nMaxLabel;
			int nMaxLabelCount = 0;
			cv::Mat seg_color = cv::Mat::zeros(segmented.size(), CV_8UC3);
			for (int y = 0; y < seg_color.rows; y++) {
				for (int x = 0; x < seg_color.cols; x++) {
					int label = segmented.at<uchar>(y, x);
					seg_color.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
					
					cv::Point2f pt(x * 2, y * 2);
					
					//////LBP code update
					//unsigned char code;
					//cv::Point2f ptLeft1(pt.x - 10, pt.y - 10);
					//cv::Point2f ptRight1(pt.x + 10, pt.y + 10);
					//bool bLeft = ptLeft1.x >= 0 && ptLeft1.y >= 0;
					//bool bRight = ptRight1.x < mnWidth && ptRight1.y < mnHeight;
					//bool bCode = bLeft && bRight;
					//cv::Mat hist;
					//unsigned long long id;
					//if (bCode) {
					//	cv::Mat desc, hist;
					//	cv::Rect rect1(ptLeft1, ptRight1);
					//	//lbp->run(mpTargetFrame->matFrame(rect1), desc);
					//	
					//	/*desc = lbpImg(rect1);
					//	hist = lbplibrary::histogram(desc, lbp->numPatterns);
					//	hist /= nDicrete;
					//	id = ConvertID(hist, numPatterns, numIDs);*/
					//	/*if (mmObjLBP.count(id) == 0) {
					//		mmObjLBP[id] = cv::Mat::zeros(1, ObjectColors::mvObjectLabelColors.size(), CV_32SC1);
					//	}
					//	mmObjLBP[id].at<int>(label)++;*/

					//	hist = mpLBPProcessor->ConvertHistogram(lbpImg, rect1);
					//	id = mpLBPProcessor->GetID(hist);
					//	mpDatabase->AddData(id, label);
					//	//std::cout <<"pt::"<<x<<", "<<y<<"::code::"<< id << "::label=" << label <<"::label2="<< maxLoc2.x<<"::"<<max_val<<", "<<mmObjLBP[id].at<int>(label)<< std::endl;
					//}
					//////LBP code update

					////그리드 처리
					auto gridBasePt = mpTargetFrame->GetGridBasePt(pt, nGridSize);
					if (mpTargetFrame->mmpFrameGrids.count(gridBasePt)) {
						auto pGrid = mpTargetFrame->mmpFrameGrids[gridBasePt];
						
						if (pGrid) {
							pGrid->mObjCount.at<int>(label)++;
							//cv::circle(resized_color, cv::Point2f(pGrid->basePt.x/2, pGrid->basePt.y/2), 3, cv::Scalar(0, 0, 255));

							////LBP texture classification
							//if (bCode) {
							//	/*double max_val = 0;
							//	minMaxLoc(mmObjLBP[id], 0, &max_val, &minLoc2, &maxLoc2);
							//	int label2 = maxLoc2.x;*/
							//	int label2 = mpDatabase->GetData(id);
							//	cv::circle(testImg, pGrid->pt, 4, ObjectColors::mvObjectLabelColors[label], -1);
							//	cv::circle(testImg, pGrid->pt, 2, ObjectColors::mvObjectLabelColors[label2], -1);
							//}
							////LBP texture classification

							//if (label == mnLabel_floor) {
							//	hist = mpSystem->mPlaneHist;
							//	if (bCode) {
							//		code = lbp->run(mpTargetFrame->matFrame, pGrid->pt);
							//		mpSystem->mPlaneHist.at<int>(code)++;
							//	}
							//	/*if (bCode) {
							//		code = lbp->run(mpTargetFrame->matFrame, pGrid->pt);
							//		mpSystem->mPlaneHist.at<int>(code)++;

							//		if (maxLoc1.x != -1 && maxLoc1.y != -1 && matCandidatePlaneCode.at<uchar>(code) == 255) {
							//			cv::circle(testImg, pGrid->pt, 4, cv::Scalar(255, 0, 0), -1);
							//		}
							//	}*/
							//}
							//else if (label == mnLabel_wall) {
							//	if (bCode) {
							//		code = lbp->run(mpTargetFrame->matFrame, pGrid->pt);
							//		mWallHist.at<int>(code)++;
							//	}
							//	hist = mWallHist;
							//	/*if (bCode) {
							//		code = lbp->run(mpTargetFrame->matFrame, pGrid->pt);
							//		mWallHist.at<int>(code)++;
							//		if (maxLoc2.x != -1 && maxLoc2.y != -1 && code == maxLoc2.x) {
							//			cv::circle(testImg, pGrid->pt, 4, cv::Scalar(0, 255, 0), -1);
							//		}
							//	}*/
							//}
							//if (bCode) {
							//	
							//	if (maxLoc1.x != -1 && maxLoc1.y != -1 && matCandidatePlaneCode.at<uchar>(code) == 255) {
							//		cv::circle(testImg, pGrid->pt, 4, cv::Scalar(255, 0, 0), -1);
							//	}
							//	if (maxLoc1.x != -1 && maxLoc1.y != -1 && matCandidateWallCode.at<uchar>(code) == 255) {
							//		cv::circle(testImg, pGrid->pt, 3, cv::Scalar(0, 255, 0), -1);
							//	}
							//}

							/*if (bCode) {
								auto code = lbp->run(mpTargetFrame->matFrame, pGrid->pt);
								hist.at<int>(code)++;
							}*/
							//if (bCode && maxLoc.x != -1 && maxLoc.y != -1 && matCandidateWallCode.at<uchar>(code) == 255) {
							//	//시각화
							//	cv::circle(testImg, pGrid->pt, 2, cv::Scalar(255, 0, 255), -1);
							//	//cv::imshow("overlap::max", mpTargetFrame->GetOriginalImage()(pGrid->rect)); cv::waitKey(1);
							//}
							////LBP code update
						}
						/*else {
						std::cout << "seg::error::grid::nullptr" << std::endl;
						}*/

					}
					////그리드 처리


					//////프레임에 대해서도 레이블 정보 누적 
					////여기는 레이블 벡터가 되어도 될 거 같음.
					//if (!mpTargetFrame->mpMatchInfo->mmLabelMasks.count(label)) {
					//	mpTargetFrame->mpMatchInfo->mmLabelMasks[label] = cv::Mat::zeros(segmented.size(), CV_8UC1);
					//}
					//else {
					//	mpTargetFrame->mpMatchInfo->mmLabelAcc[label]++;
					//	if (mpTargetFrame->mpMatchInfo->mmLabelAcc[label] > nMaxLabelCount)
					//	{
					//		nMaxLabel = label;
					//		nMaxLabelCount = mpTargetFrame->mpMatchInfo->mmLabelAcc[label];
					//	}
					//	mpTargetFrame->mpMatchInfo->mmLabelMasks[label].at<uchar>(y, x) = 255;
					//}
					//////프레임에 대해서도 레이블 정보 누적 
					////여기는 레이블 벡터가 되어도 될 거 같음.
				}
			}//for seg_color image

			////그리드 정보 획득
			auto vpGrids = mpTargetFrame->mmpFrameGrids;
			////그리드 정보 획득

			////Grid area calculation
			////그리드 오브젝트 타입
			cv::Mat testImgType= mpTargetFrame->GetOriginalImage().clone();
			for (auto iter = vpGrids.begin(), iend = vpGrids.end(); iter != iend; iter++) {
				auto pGrid = iter->second;
				auto pt = iter->first;
				if (!pGrid)
					continue;
				/*if (!pGrid->mpCP)
					continue;*/
				auto rect = pGrid->rect;
				float area = (float)rect.area();
				pGrid->mObjCount.convertTo(pGrid->mObjArea, CV_32FC1);
				pGrid->mObjArea = pGrid->mObjArea / area;

				/*int nCountFloor = pGrid->mObjCount.at<int>(mnLabel_floor);
				int nCountWall  = pGrid->mObjCount.at<int>(mnLabel_wall);
				int nCountCeil = pGrid->mObjCount.at<int>(mnLabel_ceil);*/

				float fAreaFloor = pGrid->mObjArea.at<float>(mnLabel_floor);
				float fAreaWall  = pGrid->mObjArea.at<float>(mnLabel_wall);
				float fAreaCeil  = pGrid->mObjArea.at<float>(mnLabel_ceil);

				if (fAreaWall > 0.01) {
					if (fAreaFloor > 0.01) {
						cv::circle(testImgType, pt, 3, cv::Scalar(255, 0, 0), -1);
					}
					else if (fAreaCeil > 0.01) {
						cv::circle(testImgType, pt, 3, cv::Scalar(0, 0, 255), -1);
					}
				}
				/*for (auto oiter = pGrid->mmObjCounts.begin(), oiend = pGrid->mmObjCounts.end(); oiter != oiend; oiter++) {
					int label = oiter->first;
					int count = oiter->second;
					float orea = count / area;
					pGrid->mmObjAreas[label] = orea;
				}*/
			}
			cv::Mat resized_test;
			cv::resize(testImgType, resized_test, cv::Size(testImgType.cols / 2, testImgType.rows / 2));
			mpVisualizer->SetOutputImage(resized_test, 2);
			////Grid area calculation
				
			//////평면 정보로 복원
			{
				cv::Mat R, t;
				mpTargetFrame->GetPose(R, t);
				cv::Mat Rcw2 = R.row(2).t();
				float zcw = t.at<float>(2);
				float mfMaxDepth = mpTargetFrame->mfMedianDepth + mpTargetFrame->mfRange;
				//cv::Mat testImg = mpTargetFrame->GetOriginalImage().clone();
				cv::Mat invP, invT, invK;
				cv::Mat Rinv, Tinv;
				invK = mpSystem->mInvK;
				mpTargetFrame->GetInversePose(Rinv, Tinv);
				auto pFloorParam = mpPlaneEstimator->GetPlaneParam();
				if (pFloorParam->mbInit) {
					mpTargetFrame->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpTargetFrame, pFloorParam);
					mpTargetFrame->mpPlaneInformation->Calculate(pFloorParam);
					mpTargetFrame->mpPlaneInformation->GetInformation(invP, invT, invK);
					std::vector<cv::Mat> vTempVisPts;
					std::vector<FrameGrid*> vTempGrids;
					for (auto iter = vpGrids.begin(), iend = vpGrids.end(); iter != iend; iter++) {
						auto pGrid = iter->second;
						if (pGrid->mvpCPs.size())
							continue;
						/*if (!pGrid->mpCP)
							continue;
						auto pMP = pGrid->mpCP->GetMP();
						if (pMP && !pMP->isDeleted())
							continue;*/
						auto pt = pGrid->basePt;
						int nCountFloor = pGrid->mObjCount.at<int>(mnLabel_floor);//pGrid->mmObjCounts.count(mnLabel_floor);
						float fWallArea = pGrid->mObjArea.at<float>(mnLabel_wall);
						float fFloorArea = pGrid->mObjArea.at<float>(mnLabel_floor);
						bool bFloor = nCountFloor > 0 && fFloorArea > fWallArea*5.0;
						if (!bFloor){
							//cv::circle(testImg, pGrid->basePt, 3, cv::Scalar(0, 255, 0));
							continue;
						}
						cv::Mat s;
						//bool b = PlaneInformation::CreatePlanarMapPoint(s, pt, invP, invT, invK);
						bool b = PlaneInformation::CreatePlanarMapPoint(s, pt, invP, invK, Rinv, Tinv, mfMaxDepth);
						if (b){
							vTempVisPts.push_back(s);
							vTempGrids.push_back(pGrid);
						}
						/*else {
							cv::circle(testImg, pGrid->basePt, 3, cv::Scalar(0, 255, 255));
						}*/
					}
					mpPlaneEstimator->SetTempPTs(mpTargetFrame, vTempGrids,vTempVisPts);
				}
			}
			//////평면 정보로 복원

			//////LBP hist visualization
			//int bins = lbp->numPatterns;
			//int hist_height = 256;
			//{
			//	double max_val = 0;
			//	cv::Mat hist_plane = mpSystem->mPlaneHist.clone();
			//	cv::Mat3b hist_image_plane = cv::Mat3b::zeros(hist_height, bins);
			//	minMaxLoc(hist_plane, 0, &max_val, &minLoc1, &maxLoc1);
			//	for (int b = 0; b < bins; b++) {
			//		float const binVal = (float)hist_plane.at<int>(b);
			//		int   const height = cvRound(binVal*hist_height / max_val);
			//		cv::line
			//		(hist_image_plane
			//			, cv::Point(b, hist_height - height), cv::Point(b, hist_height)
			//			, cv::Scalar::all(255)
			//		);
			//	}
			//	matCandidatePlaneCode = hist_plane > max_val * 0.6;
			//	cv::imshow("PlaneHist", hist_image_plane);
			//}
			//{
			//	double max_val = 0;
			//	cv::Mat hist_wall = mWallHist.clone();
			//	cv::Mat3b hist_image_wall = cv::Mat3b::zeros(hist_height, bins);
			//	minMaxLoc(hist_wall, 0, &max_val, &minLoc2, &maxLoc2);
			//	for (int b = 0; b < bins; b++) {
			//		float const binVal = (float)hist_wall.at<int>(b);
			//		int   const height = cvRound(binVal*hist_height / max_val);
			//		cv::line
			//		(hist_image_wall
			//			, cv::Point(b, hist_height - height), cv::Point(b, hist_height)
			//			, cv::Scalar::all(255)
			//		);
			//	}
			//	matCandidateWallCode = hist_wall > max_val * 0.6;
			//	cv::imshow("WallHist", hist_image_wall);
			//}
			//{
			//	double max_val = 0;
			//	cv::Mat hist = mWallHist & mpSystem->mPlaneHist;
			//	cv::Mat3b hist_image_plane = cv::Mat3b::zeros(hist_height, bins);
			//	minMaxLoc(hist, 0, &max_val, &minLoc, &maxLoc);
			//	std::cout << "overlap::" << max_val <<", "<<maxLoc<< std::endl;
			//	for (int b = 0; b < bins; b++) {
			//		float const binVal = (float)hist.at<int>(b);
			//		int   const height = cvRound(binVal*hist_height / max_val);
			//		cv::line
			//		(hist_image_plane
			//			, cv::Point(b, hist_height - height), cv::Point(b, hist_height)
			//			, cv::Scalar::all(255)
			//		);
			//	}
			//	cv::imshow("Overlap", hist_image_plane);
			//	cv::imshow("overlap::max", testImg); cv::waitKey(1);
			//}
			//cv::imshow("overlap::max", testImg); cv::waitKey(1);
			//////LBP hist visualization


			////다음 그리드에 전파 
			//float nTotalArea = nGridSize*nGridSize;
			//for (auto iter = mpTargetFrame->mmpFrameGrids.begin(), iend = mpTargetFrame->mmpFrameGrids.end(); iter != iend; iter++) {
			//	auto pGrid = iter->second;
			//	if (!pGrid)
			//		continue;
			//	FrameGrid* nextPtr = pGrid->mpNext;
			//	while (nextPtr) {
			//		//std::copy(pGrid->mmObjCounts.begin(), pGrid->mmObjCounts.end(), nextPtr->mmObjCounts.begin());
			//		nextPtr->mObjArea = pGrid->mObjArea.clone();
			//		nextPtr->mObjCount = pGrid->mObjCount.clone();
			//		/*for (auto oiter = pGrid->mmObjCounts.begin(), oiend = pGrid->mmObjCounts.end(); oiter != oiend; oiter++) {
			//			auto label = oiter->first;
			//			auto count = oiter->second;
			//			auto area = pGrid->mmObjAreas[label];
			//			nextPtr->mmObjCounts[label] = count;
			//			nextPtr->mmObjAreas[label] = area;
			//		}*/
			//		nextPtr = nextPtr->mpNext;
			//		//break;
			//	}
			//}
			////다음 그리드에 전파

			//////////////////////////////////////////////////
			////////디버깅을 위한 이미지 저장
			/*mStrDirPath = mpSystem->GetDirPath(0);
			std::stringstream ss;
			ss << mStrDirPath.c_str() << "/seg/ori_" << mpTargetFrame->mnFrameID << ".jpg";
			cv::imwrite(ss.str(), mpTargetFrame->GetOriginalImage());
			ss.str("");
			ss << mStrDirPath.c_str() << "/seg/segmentation_color_"<<mpTargetFrame->mnFrameID <<".jpg";
			cv::imwrite(ss.str(), seg_color);*/
			////////디버깅을 위한 이미지 저장
			//////////////////////////////////////////////////

			///////////CCL TEST
			cv::addWeighted(seg_color, 0.5, resized_color, 0.5, 0.0, resized_color);
			for (auto iter = mpTargetFrame->mpMatchInfo->mmLabelMasks.begin(), eiter = mpTargetFrame->mpMatchInfo->mmLabelMasks.end(); iter != eiter; iter++) {
				cv::Mat img_labels, stats, centroids;
				int label = iter->first;
				cv::Mat mask = iter->second;
				
				int numOfLables = connectedComponentsWithStats(mask, img_labels, stats, centroids, 8, CV_32S);
				if (numOfLables > 0) {
					int maxArea = 0;
					int maxIdx = 0;
					
					////라벨링 된 이미지에 각각 직사각형으로 둘러싸기 
					//for (int j = 0; j < numOfLables; j++) {
					//	int area = stats.at<int>(j, CC_STAT_AREA);
					//	
					//	//std::cout << area << std::endl;
					//	if (area > maxArea) {
					//		maxArea = area;
					//		maxIdx = j;
					//	}
					//}

					for (int j = 0; j < numOfLables; j++) {
						/*if (j == maxIdx)
						continue;*/
						int area = stats.at<int>(j, cv::CC_STAT_AREA);
						if (area < 300)
							continue;
						int left = stats.at<int>(j, cv::CC_STAT_LEFT);
						int top = stats.at<int>(j, cv::CC_STAT_TOP);
						int width = stats.at<int>(j, cv::CC_STAT_WIDTH);
						int height = stats.at<int>(j, cv::CC_STAT_HEIGHT);
						cv::Rect rect(cv::Point(left, top), cv::Point(left + width, top + height));
						/*mpTargetFrame->mpMatchInfo->mmLabelRects.insert(std::make_pair(label, rect));
						mpTargetFrame->mpMatchInfo->mmLabelCPs.insert(std::make_pair(label, std::list<CandidatePoint*>()));*/
						auto info = std::make_pair(rect, std::list<CandidatePoint*>());
						mpTargetFrame->mpMatchInfo->mmLabelRectCPs.insert(std::make_pair(label, info));
						rectangle(resized_color, rect, cv::Scalar(255, 255, 255), 2);
						//rectangle(ccl_res, Point(left, top), Point(left + width, top + height), Scalar(255, 255, 255), 2);
					}
				}
			}

			mpTargetFrame->SetBoolSegmented(true);
			mpTargetFrame->mpMatchInfo->SetLabel();

			for (auto iter = mpTargetFrame->mpMatchInfo->mmLabelRectCPs.begin(), eiter = mpTargetFrame->mpMatchInfo->mmLabelRectCPs.end(); iter != eiter; iter++) {
				auto label = iter->first;
				auto rect = iter->second.first;
				auto lpCPs = iter->second.second;
				for (auto liter = lpCPs.begin(), leiter = lpCPs.end(); liter != leiter; liter++) {
					auto pCPi = *liter;
					int idx = pCPi->GetPointIndexInFrame(mpTargetFrame->mpMatchInfo);
					if (idx > -1) {
						auto pt = mpTargetFrame->mpMatchInfo->mvMatchingPts[idx] / 2;
						cv::circle(resized_color, pt, 2, UVR_SLAM::ObjectColors::mvObjectLabelColors[label], -1);
					}
				}
				//std::cout << "Object Info = " << label << ", " << lpCPs.size() << std::endl;
			}

			mpVisualizer->SetOutputImage(resized_color, 1);
			////시간체크
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float tttt = duration / 1000.0;
			////시간체크

			//if (mmLabelMasks.size() > 0) {
			//	Mat img_labels, stats, centroids;
			//	cv::Mat maskA = mmLabelMasks[nMaxLabel];
			//	cv::Mat ccl_res = seg_color.clone();
			//	int numOfLables = connectedComponentsWithStats(maskA, img_labels, stats, centroids, 8, CV_32S);
			//	if (numOfLables > 0) {
			//		int maxArea = 0;
			//		int maxIdx = 0;
			//		//라벨링 된 이미지에 각각 직사각형으로 둘러싸기 
			//		for (int j = 0; j < numOfLables; j++) {
			//			int area = stats.at<int>(j, CC_STAT_AREA);
			//			if (area > maxArea) {
			//				maxArea = area;
			//				maxIdx = j;
			//			}
			//		}
			//		for (int j = 0; j < numOfLables; j++) {
			//			/*if (j == maxIdx)
			//			continue;*/
			//			int area = stats.at<int>(j, CC_STAT_AREA);
			//			int left = stats.at<int>(j, CC_STAT_LEFT);
			//			int top = stats.at<int>(j, CC_STAT_TOP);
			//			int width = stats.at<int>(j, CC_STAT_WIDTH);
			//			int height = stats.at<int>(j, CC_STAT_HEIGHT);

			//			rectangle(ccl_res, Point(left, top), Point(left + width, top + height), Scalar(255, 255, 255), 2);
			//		}
			//		imshow("seg mask", maskA);
			//		imshow("aaabbb", ccl_res); cv::waitKey(1);
			//	}
			//}
			///////////CCL TEST
		
			//////////////////////////////////////////////
			//////디버깅 값 전달
			int nFrameSize = 0;
			{
				std::unique_lock<std::mutex> lock(mMutexNewKFs);
				nFrameSize = mKFQueue.size();
			}
			std::stringstream ssa;
			ssa << "Segmentation : " << mpTargetFrame->mnKeyFrameID << " : " << tttt << " ::" << nFrameSize;
			mpSystem->SetSegmentationString(ssa.str());
			//////디버깅 값 전달
			//////////////////////////////////////////////

			////////평면 추정 쓰레드 전달
			mpPlaneEstimator->InsertKeyFrame(mpTargetFrame);
			
			SetBoolDoingProcess(false);
		}
	}
}
void UVR_SLAM::SemanticSegmentator::SetInitialSegFrame(UVR_SLAM::Frame* pKF1) {
	mpPrevFrame = pKF1;
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
void UVR_SLAM::SemanticSegmentator::ImageLabeling(cv::Mat segmented, cv::Mat& labeld) {

	labeld = cv::Mat::zeros(segmented.size(), CV_8UC1);

	for (int i = 0; i < segmented.rows; i++) {
		for (int j = 0; j < segmented.cols; j++) {
			int val = segmented.at<uchar>(i, j);
			switch (val) {
			case 1://벽
			case 9://유리창
			case 11://캐비넷
			case 15://문
			
			case 36://옷장
					//case 43://기둥
			case 44://갚난
					//case 94://막대기
			case 23: //그림
			case 101://포스터
				labeld.at<uchar>(i, j) = 255;
				break;
			case 4:  //floor
			case 29: //rug
				labeld.at<uchar>(i, j) = 150;
				break;
			case 6:
				labeld.at<uchar>(i, j) = 100;
				break;
			case 13: //person, moving object
				labeld.at<uchar>(i, j) = 20;
				break;
			default:
				labeld.at<uchar>(i, j) = 50;
				break;
			}
		}
	}
}

