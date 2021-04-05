#include "LoopCloser.h"
#include "Frame.h"
#include <MapPoint.h>
#include "System.h"
#include <Converter.h>
#include <Matcher.h>
#include <KeyframeDatabase.h>
#include <Sim3Solver.h>
#include <Optimization.h>
#include <Visualizer.h>
#include "Map.h"
#include <User.h>
#include <ServerMap.h>
#include <WebAPI.h>
#include <future>
#include <random>
#include <PlaneEstimator.h>
#include <Subspace.h>

/*
�������� seterase, setnoterase, setbadflag ���õ� ó���� ���� �ȵǾ�����.
matcher�� SearchBySim3 ����
CorrectLoop ���� ���� �ʿ�
SearchAndFuse ���� �ʿ�
RunGlobalBundleAdjustment ���� �ʿ�
matcher�� fuse ���� �ʿ�
������Ʈ Replace ���� �ʿ�
Ű������ updateconnection ���� �ʿ�
Optimization�� OptimizeEssentialGraph ���� ����
*/

namespace UVR_SLAM {
	
	LoopCloser::LoopCloser() {}
	LoopCloser::LoopCloser(System* pSys):mpSystem(pSys), mbLoadData(false), mbSaveData(false), mbFixScale(false), mbProcessing(false), mnCovisibilityConsistencyTh(3){
	}
	LoopCloser::~LoopCloser() {}
	void LoopCloser::Init() {
		mpMap = mpSystem->mpMap;
		mpKeyFrameDatabase = mpSystem->mpKeyframeDatabase;
		mpVoc = mpSystem->mpDBoWVoc;
		mpMatcher = mpSystem->mpMatcher;
	}
	void LoopCloser::LoadMapData(std::string map) {
		mbLoadData = true;
		mapName = map;
	}
	void LoopCloser::SaveMapData(std::string map) {
		mbSaveData = true;
		mapName = map;
	}
	void LoopCloser::ConstructBowDB(std::vector<Frame*> vpFrames) {
		for (size_t i = 0, iend = vpFrames.size(); i < iend; i++) {
			mpKeyFrameDatabase->Add(vpFrames[i]);
		}
	}

	bool LoopCloser::Relocalization(Frame* pF) {
		auto vpCandidateKFs = mpKeyFrameDatabase->DetectPlaceCandidates(pF);
		//std::cout << "candidate : "<<vpCandidateKFs.size() << std::endl;
		int nMax = 0;
		cv::Mat maxMatch;
		Frame* maxFrame = nullptr;
		for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++) {
			auto pCandidate = vpCandidateKFs[i];
			cv::Mat matches = mpMatcher->KnnMatching(pF, pCandidate);
			auto vpMPs = pCandidate->GetMapPoints();
			int nMatch = 0;
			int nTempMatch = 0;
			for (int j = 0, jend = matches.rows; j < jend; j++) {
				int idx1 = j;
				int idx2 = matches.at<int>(idx1);
				if (idx2 == 10000)
					continue;
				nTempMatch++;
				auto pMP = vpMPs[idx2];
				if (!pMP || pMP->isDeleted())
					continue;
				nMatch++;
			}
			if (nMatch > nMax) {
				nMax = nMatch;
				maxMatch = matches.clone();
				maxFrame = pCandidate;
			}
			//std::cout << i << "=" << nTempMatch <<"::"<< nMatch << std::endl;
		}
		if (nMax < 30 || !maxFrame)
			return false;

		//Pose Optimization Test
		cv::Mat R, t;
		maxFrame->GetPose(R, t);

		std::vector<cv::Point2f> vPTs;
		std::vector<cv::Point3f> vXws;
		std::vector<MapPoint*> vMPs;
		std::vector <bool> vInliers;
		std::vector<int> vnIDXs;
		auto vpMPs = maxFrame->GetMapPoints();
		for (int j = 0, jend = maxMatch.rows; j < jend; j++) {
			int idx1 = j;
			int idx2 = maxMatch.at<int>(idx1);
			if (idx2 == 10000)
				continue;
			auto pMP = vpMPs[idx2];
			if (!pMP || pMP->isDeleted())
				continue;
			vPTs.push_back(pF->mvPts[j]);
			vMPs.push_back(pMP);
			vInliers.push_back(true);
			vnIDXs.push_back(j);
		}

		pF->SetPose(R, t);
		int nPoseRecovery = Optimization::PoseOptimization(mpMap, pF, vMPs, vPTs, vInliers);
		if (nPoseRecovery < 20)
			return false;

		for (size_t i = 0; i < vInliers.size(); i++)
		{
			if (vInliers[i]) {
				int idx = vnIDXs[i];
				pF->AddMapPoint(vMPs[i], idx);
			}
		}
		return true;
	}
	bool calcTempUnitNormalVector(cv::Mat& X) {

		float sum = sqrt(X.at<float>(0)*X.at<float>(0) + X.at<float>(1)*X.at<float>(1) + X.at<float>(2)*X.at<float>(2));
		if (sum != 0) {
			X.at<float>(0, 0) = X.at<float>(0, 0) / sum;
			X.at<float>(1, 0) = X.at<float>(1, 0) / sum;
			X.at<float>(2, 0) = X.at<float>(2, 0) / sum;
			X.at<float>(3, 0) = X.at<float>(3, 0) / sum;
			return true;
		}
		return false;
	}
	int checkTempNormalType(cv::Mat X) {
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

	void LoopCloser::InsertData(std::pair<Frame*, std::string> pairInfo){
		std::unique_lock<std::mutex> lock(mMutexDataQueue);
		mDataQueue.push(pairInfo);
	}
	bool LoopCloser::CheckNewDatas(){
		std::unique_lock<std::mutex> lock(mMutexDataQueue);
		return(!mDataQueue.empty());
	}
	void LoopCloser::ProcessData(){
		std::unique_lock<std::mutex> lock(mMutexDataQueue);
		mPairFrameInfo = mDataQueue.front();
		mDataQueue.pop();

		std::string user = mPairFrameInfo.second;
		mpTargetFrame = mPairFrameInfo.first;
		mpTargetUser = mpSystem->GetUser(user);
		if (mpTargetUser)
			mpTargetMap = mpSystem->GetMap(mpTargetUser->mapName);

	}

	auto lambda_plane_init = [](std::vector<cv::Mat> src, PlaneInformation* pPlane, int ransac_trial, float thresh_distance, float thresh_ratio) {
		
		//N
		//N-1
		
		//RANSAC
		int max_num_inlier = 0;
		cv::Mat best_plane_param;
		cv::Mat inlier;

		cv::Mat param, paramStatus;

		//�ʱ� ��Ʈ���� ����
		cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
		std::vector<int> vIdxs;
		for (int i = 0; i < src.size(); i++) {
			cv::Mat temp = src[i].clone();
			temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
			mMat.push_back(temp.t());
			vIdxs.push_back(i);
		}
		if (mMat.rows < 30)
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

			calcTempUnitNormalVector(X);
			cv::Mat checkResidual = abs(mMat*X) < thresh_distance;
			//checkResidual = checkResidual / 255;
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
			
			cv::Mat normal = param.rowRange(0, 3);
			float dist = param.at<float>(3);
			pPlane->SetParam(normal, dist);
			
			for (int i = 0; i < mMat.rows; i++) {
				int checkIdx = paramStatus.at<uchar>(i);
				
				cv::Mat temp = src[vIdxs[i]];
				if (checkIdx == 0) {
					pPlane->mvOutliers.push_back(temp);
				}
				else {
					pPlane->mvInliers.push_back(temp);
					tempMat.push_back(mMat.row(i));
				}
			}

			float ratio2 = ((float)pPlane->mvOutliers.size()) / mMat.rows;
			if (ratio2 > 0.3f)
				pPlane->mbParallel = true;

			//std::cout << "plane::3::" << planeParam.t() << ", " << inliers.size() <<", "<<outliers.size()<< std::endl;
			//��� ���� ����.
			/*
			cv::Mat X;
			cv::Mat w, u, vt;
			cv::SVD::compute(tempMat, w, u, vt, cv::SVD::FULL_UV);
			std::cout << "plane::3-1" << std::endl;
			X = vt.row(3).clone();
			cv::transpose(X, X);
			calcTempUnitNormalVector(X);
			int idx = checkTempNormalType(X);
			if (X.at<float>(idx) > 0.0)
				X *= -1.0;
			planeParam = X.clone();
			std::cout << "plane::4" << std::endl;*/
			return true;
		}
		else
		{
			//std::cout << "failed" << std::endl;
			return false;
		}
	};

	void LoopCloser::RunWithMappingServer() {
		std::cout << "MappingServer::LoopCloser::Start" << std::endl;

		int nPrevSegFrame = -1;
		int nCurrSegFrame;
		int nPrevDepthFrame = -1;
		int nCurrDepthFrame;

		const int mnLabel_floor = 4;
		const int mnLabel_wall = 1;
		const int mnLabel_ceil = 6;

		cv::FileStorage fSettings(mpSystem->mstrFilePath, cv::FileStorage::READ);
		int mnRansacTrial = fSettings["Layout.trial"];
		float mfThreshPlaneDistance = fSettings["Layout.dist"];
		float mfThreshPlaneRatio = fSettings["Layout.ratio"];
		float mfThreshNormal = fSettings["Layout.normal"];
		fSettings.release();
		
		std::set<MapPoint*>    sTempAllFloorMPs;
		std::vector<cv::Mat> vTempAllFloorMPs;
		std::map<std::pair<Frame*, int>, std::vector<MapPoint*>> mapFrameObjectInfos;
		std::vector<cv::Mat> vecWallParams;
		

		while (true) {

			if (mbLoadData) {
				std::cout << "Load Data" << std::endl;
				auto pTargetMap = mpSystem->GetMap(mapName);
				if (pTargetMap) {
					pTargetMap->LoadMapDataFromServer(mapName, mpSystem->ip, mpSystem->port);
					////��� �����ϰ� �Ǹ� ���� �ٸ� �ʵ����Ͱ� ������ �� ����.��
					ConstructBowDB(pTargetMap->GetFrames());
				}
				//mpMap->LoadMapDataFromServer(mapName, mpMap->mvpMapFrames);
				
				mbLoadData = false;
			}
			if (mbSaveData) {
				auto pTargetMap = mpSystem->GetMap(mapName);
				if (pTargetMap) {
					pTargetMap->SaveMapDataToServer(mapName, mpSystem->ip, mpSystem->port);
				}
				mbSaveData = false;
			}

			if (CheckNewDatas()) {
				std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
				ProcessData();

				if (!mpTargetUser || !mpTargetMap)
				{
					continue;
				}

#ifdef DEBUG_LOOP_CLOSING_LEVEL_1
				std::cout << "MappingServer::LoopClosing::" << mpTargetFrame->mnFrameID << "::Start" << std::endl;
#endif
				mpKeyFrameDatabase->Add(mpTargetFrame);

				if (mpTargetFrame->bSeg){
					//mask
					cv::Mat mask_total = cv::Mat::zeros(mpTargetFrame->mnHeight, mpTargetFrame->mnWidth, CV_8UC1);
					cv::Mat mask_floor = cv::Mat::zeros(mpTargetFrame->mnHeight, mpTargetFrame->mnWidth, CV_8UC1);
					cv::Mat mask_wall = cv::Mat::zeros(mpTargetFrame->mnHeight, mpTargetFrame->mnWidth, CV_8UC1);
					cv::Mat mask_ceil = cv::Mat::zeros(mpTargetFrame->mnHeight, mpTargetFrame->mnWidth, CV_8UC1);

					////segmentation image
					mpTargetFrame->mSegImage = cv::Mat::zeros(mpTargetFrame->mSegLabel.size(), CV_8UC3);
					for (int y = 0; y < mpTargetFrame->mSegLabel.rows; y++) {
						for (int x = 0; x < mpTargetFrame->mSegLabel.cols; x++) {
							int label = mpTargetFrame->mSegLabel.at<uchar>(y, x) + 1;

							switch (label) {
							case mnLabel_floor:
								mpTargetFrame->mSegImage.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
								mask_floor.at<uchar>(y, x) = 255;
								break;
							case mnLabel_wall:
								mpTargetFrame->mSegImage.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
								mask_wall.at<uchar>(y, x) = 255;
								break;
							case mnLabel_ceil:
								mpTargetFrame->mSegImage.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
								mask_ceil.at<uchar>(y, x) = 255;
								break;
							}
						}
					}
					std::vector<cv::Mat> vTempFloorMPs, vTempWallMPs, vTempCeilMPs;
					auto vpMPs = mpTargetFrame->GetMapPoints();
					for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
						auto pMPi = vpMPs[i];
						if (!pMPi || pMPi->isDeleted())
							continue;
						cv::Mat a = pMPi->GetWorldPos();
						auto pt = mpTargetFrame->mvPts[i];
						int label = mpTargetFrame->mSegLabel.at<uchar>(pt.y, pt.x) + 1;
						switch (label) {
						case mnLabel_floor:
							vTempFloorMPs.push_back(a);
							if (!sTempAllFloorMPs.count(pMPi)) {
								sTempAllFloorMPs.insert(pMPi);
								vTempAllFloorMPs.push_back(a);
							}
							break;
						case mnLabel_wall:
							vTempWallMPs.push_back(a);
							break;
						case mnLabel_ceil:
							vTempCeilMPs.push_back(a);
							break;
						}//switch
						auto key = std::make_pair(mpTargetFrame, label);
						mapFrameObjectInfos[key].push_back(pMPi);
					}

					auto tempFloor = new UVR_SLAM::PlaneInformation();
					auto tempWall1 = new UVR_SLAM::PlaneInformation();
					auto tempWall2 = new UVR_SLAM::PlaneInformation();

					auto ffloorplane = std::async(std::launch::async, UVR_SLAM::lambda_plane_init, vTempAllFloorMPs, tempFloor, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
					auto fwallplane1 = std::async(std::launch::async, UVR_SLAM::lambda_plane_init, vTempWallMPs, tempWall1, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);

					mpTargetMap->ClearPlanarTest();
					if (ffloorplane.get()) {

						////plane param
						std::async(std::launch::async, [](std::string ip, int port, std::string mapName, cv::Mat param) {
							WebAPI* mpAPI = new WebAPI(ip, port);
							////xz, 5, 5
							float x = 20.0;
							float z = 20.0;
							float y1 = -( x*param.at<float>(0) + z*param.at<float>(2) + param.at<float>(3)) / param.at<float>(1);
							float y2 = -( x*param.at<float>(0) - z*param.at<float>(2) + param.at<float>(3)) / param.at<float>(1);
							float y3 = -(-x*param.at<float>(0) - z*param.at<float>(2) + param.at<float>(3)) / param.at<float>(1);
							float y4 = -(-x*param.at<float>(0) + z*param.at<float>(2) + param.at<float>(3)) / param.at<float>(1);
							cv::Mat data = cv::Mat::zeros(12, 1, CV_32FC1);
							int idx = 0;
							data.at<float>(idx++) = x;
							data.at<float>(idx++) = y1;
							data.at<float>(idx++) = z;
							data.at<float>(idx++) = x;
							data.at<float>(idx++) = y2;
							data.at<float>(idx++) = -z;
							data.at<float>(idx++) = -x;
							data.at<float>(idx++) = y3;
							data.at<float>(idx++) = -z;
							data.at<float>(idx++) = -x;
							data.at<float>(idx++) = y4;
							data.at<float>(idx++) = z;

							std::stringstream ss;
							ss << "/ReceiveData?map=" << mapName << "&attr=Models&id=1&key=bregion";
							mpAPI->Send(ss.str(), data.data, data.rows * sizeof(float));
						}, mpSystem->ip, mpSystem->port, mpTargetUser->mapName, tempFloor->GetParam());
						////plane param

						std::async(std::launch::async, [](std::string ip, int port, std::string mapName, cv::Mat data) {
							WebAPI* mpAPI = new WebAPI(ip, port);
							std::stringstream ss;
							ss << "/ReceiveData?map=" << mapName <<"&attr=Models&id=1&key=bparam";
							mpAPI->Send(ss.str(), data.data, data.rows * sizeof(float));
						},mpSystem->ip, mpSystem->port, mpTargetUser->mapName, tempFloor->GetParam());
						

						std::vector<cv::Mat> vfIns = tempFloor->mvInliers;
						std::vector<cv::Mat> vfOuts = tempFloor->mvOutliers;
						
						std::cout << "Plane::Floor::Success::" << tempFloor->GetParam().t()<<"::"<< vfIns.size() << " " << vfOuts.size() << "::" << vTempFloorMPs.size() << std::endl;
						for (size_t i = 0, iend = vfIns.size(); i < iend; i++) {
							mpTargetMap->AddPlanarTemp(vfIns[i], 1);
						}
						/*for (size_t i = 0, iend = vfOuts.size(); i < iend; i++) {
							mpTargetMap->AddPlanarTemp(vfOuts[i], 2);
						}*/

						

					}
					else
						std::cout << "Plane::Floor::Fail::" << vTempFloorMPs.size() << std::endl;
					if (fwallplane1.get()) {
						std::future<bool> fwallplane2;
						/*if (tempWall1->mbParallel) {
							std::cout << "wall2 start" << std::endl;
							fwallplane2 = std::async(std::launch::async, UVR_SLAM::lambda_plane_init, tempWall1->mvOutliers, tempWall2, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
						}*/

						std::vector<cv::Mat> vWIns = tempWall1->mvInliers;
						std::vector<cv::Mat> vWOuts = tempWall1->mvOutliers;
						//std::cout << "Wall Test = " << vWIns.size() << " " << vWOuts.size() << ":" << vTempFloorMPs.size() << ", " << vTempWallMPs.size() << std::endl;
						for (size_t i = 0, iend = vWIns.size(); i < iend; i++) {
							//mpSystem->mpMap->AddTempMP(vWIns[i]);
							mpTargetMap->AddPlanarTemp(vWIns[i], 3);
						}
						/*for (size_t i = 0, iend = vWOuts.size(); i < iend; i++) {
							mpTargetMap->AddPlanarTemp(vWOuts[i], 4);
						}*/

						//if (tempWall1->mbParallel && fwallplane2.get()) {
						//	std::cout << "wall2 test::2" << std::endl;
						//	std::vector<cv::Mat> vWIns2 = tempWall2->mvInliers;
						//	std::vector<cv::Mat> vWOuts2 = tempWall2->mvOutliers;
						//	for (size_t i = 0, iend = vWIns2.size(); i < iend; i++) {
						//		mpTargetMap->AddPlanarTemp(vWIns2[i], 5);
						//	}
						//	/*for (size_t i = 0, iend = vWOuts2.size(); i < iend; i++) {
						//		mpTargetMap->AddPlanarTemp(vWOuts2[i], 6);
						//	}*/
						//}
					}

					{
						cv::Mat vis = cv::Mat::zeros(1000, 1000, CV_8UC3);
						////acquire subspace
						Subspace* pSub = nullptr;
						if (mpTargetMap->mvpSubspaces.size() > 0) {
							pSub = mpTargetMap->mvpSubspaces[mpTargetMap->mvpSubspaces.size() - 1];
							
						}
						else if (mpTargetMap->mvpSubspaces.size() == 0)
						{
							pSub = new Subspace();
							pSub->mpStartFrame = mpTargetFrame;
							mpTargetMap->mvpSubspaces.push_back(pSub);
						}
						pSub->mvpSubspaceFrames.push_back(mpTargetFrame);
						/*else {
							pSub = mpTargetMap->mvpSubspaces[mpTargetMap->mvpSubspaces.size() - 1];
						}*/

						//add current frame data
						auto key = std::make_pair(mpTargetFrame, mnLabel_wall);
						auto vec = mapFrameObjectInfos[key];
						for (size_t i = 0, iend = vec.size(); i < iend; i++) {
							auto pMPi = vec[i];
							if (!pMPi || pMPi->isDeleted())
								continue;
							if (pSub->mspSubspaceMapPoints.count(pMPi)) {
								continue;
							}
							pSub->mspSubspaceMapPoints.insert(pMPi);
						}
						//add current frame data
						
						
						cv::Mat data = cv::Mat::zeros(0, 3, CV_32FC1);
						////new parameter estimation
						auto vpMPs = pSub->mspSubspaceMapPoints;
						if (pSub->mspSubspaceMapPoints.size() > 20) {
							cv::Mat normal;
							float d;
							tempFloor->GetParam(normal, d);

							//rotate with floor 
							cv::Mat Rcw = UVR_SLAM::PlaneInformation::CalcPlaneRotationMatrix(normal).clone();
							cv::Mat Pnew = Rcw.t()*normal;
							if (Pnew.at<float>(0) < 0.00001)
								Pnew.at<float>(0) = 0.0;
							if (Pnew.at<float>(2) < 0.00001)
								Pnew.at<float>(2) = 0.0;

							cv::Point2f mVisMidPt(500, 500);
							auto vpSubspaces = mpTargetMap->mvpSubspaces;
							for (size_t i = 0, iend = vpSubspaces.size(); i < iend; i++) {
								auto tempSub = vpSubspaces[i];
								auto spMPs = vpSubspaces[i]->mspSubspaceMapPoints;
								int nID = vpSubspaces[i]->mnID;
								bool bSub = pSub == tempSub;
								for (auto it = spMPs.begin(), itend = spMPs.end(); it != itend; it++)
								{
									auto pMPi = *it;
									if (!pMPi || pMPi->isDeleted())
										continue;
									cv::Mat a = pMPi->GetWorldPos();
									cv::Mat Xnew = Rcw.t()*a;
									cv::Mat Xproj = Xnew.clone();

									cv::Mat temp = cv::Mat::ones(1, 3, CV_32FC1);
									temp.at<float>(0) = Xproj.at<float>(0);
									temp.at<float>(1) = Xproj.at<float>(2);
									if(bSub)
										data.push_back(temp);
									cv::Point2f tpt = cv::Point2f(Xnew.at<float>(0) * 20, Xnew.at<float>(2) * 20);
									tpt += mVisMidPt;
									cv::circle(vis, tpt, 2, ObjectColors::mvObjectLabelColors[nID], -1);
								}
							}
							//for (auto it = pSub->mspSubspaceMapPoints.begin(), iend = pSub->mspSubspaceMapPoints.end(); it != iend; it++) {
							//	auto pMPi = *it;
							//	cv::Mat a = pMPi->GetWorldPos();
							//	cv::Mat Xnew = Rcw.t()*a;
							//	cv::Mat Xproj = Xnew.clone();

							//	cv::Mat temp = cv::Mat::ones(1, 3, CV_32FC1);
							//	temp.at<float>(0) = Xproj.at<float>(0);
							//	temp.at<float>(1) = Xproj.at<float>(2);
							//	data.push_back(temp);

							//	/*Xproj.at<float>(1) = 0.0f;
							//	cv::Point2f tpt = cv::Point2f(Xnew.at<float>(0) * 20, Xnew.at<float>(2) * 20);
							//	tpt += mVisMidPt;
							//	cv::circle(vis, tpt, 2, cv::Scalar(255, 0, 255), -1);*/
							//}

							//add data in new frame
							cv::Mat data2 = cv::Mat::zeros(0, 3, CV_32FC1);
							for (size_t i = 0, iend = vec.size(); i < iend; i++) {
								auto pMPi = vec[i];
								if (!pMPi || pMPi->isDeleted())
									continue;
								
								cv::Mat a = pMPi->GetWorldPos();
								cv::Mat Xnew = Rcw.t()*a;
								cv::Mat Xproj = Xnew.clone();

								cv::Mat temp = cv::Mat::ones(1, 3, CV_32FC1);
								temp.at<float>(0) = Xproj.at<float>(0);
								temp.at<float>(1) = Xproj.at<float>(2);
								data2.push_back(temp);

								Xproj.at<float>(1) = 0.0f;
								cv::Point2f tpt = cv::Point2f(Xnew.at<float>(0) * 20, Xnew.at<float>(2) * 20);
								tpt += mVisMidPt;
								cv::circle(vis, tpt, 3, cv::Scalar(255, 255, 255), -1);
							}

							////check need new subspace


							////plane estimation
							cv::Mat param;
							cv::Mat inliers = cv::Mat::zeros(0, 3, CV_32FC1);
							cv::Mat outliers = cv::Mat::zeros(0, 3, CV_32FC1);
							bool bWall = PlaneInformation::Ransac_fitting(data, param, inliers, outliers, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
							
							if (bWall) {

								////update avg normal vector and store parameter
								if (pSub->mvParams.size() == 0) {
									pSub->avgParam = param.clone();
									pSub->avgParam.at<float>(2) = 0.0f;
								}
								else {
									int nSize = pSub->mvParams.size();
									pSub->avgParam = (pSub->avgParam*nSize + param)/(nSize+1);

									//check need new subspace
									cv::Mat prevParam = pSub->mvParams[nSize - 1];
									cv::Mat paramStat = (abs(data2*prevParam) < mfThreshPlaneDistance) / 255;
									std::cout << "Prev Param : " << cv::sum(paramStat).val[0] << ", " << data2.rows << std::endl;
								}
								pSub->mvParams.push_back(param);
								////update avg normal vector and store parameter

								auto val = PlaneInformation::CalcCosineSimilarity(param.rowRange(0, 2), pSub->avgParam.rowRange(0, 2));
								std::cout << "normal : " << val << "::" << param << ", " << pSub->avgParam << std::endl;

								cv::Scalar color = ObjectColors::mvObjectLabelColors[pSub->mnID];
								int cSize = 2;

								if (val > 0.95) {
									color.val[0] = 0;
									color.val[1] = 255;
									color.val[2] = 255;
								}
								else if (val < 0.3) {
									color.val[0] = 0;
									color.val[1] = 255;
									color.val[2] = 0;
									cSize = 4;
								}

								for (size_t i = 0, iend = inliers.rows; i < iend; i++) {
									cv::Mat temp = inliers.row(i);
									cv::Point2f tpt = cv::Point2f(temp.at<float>(0) * 20, temp.at<float>(1) * 20);
									tpt += mVisMidPt;
									cv::circle(vis, tpt, cSize, color, -1);
								}
							}

							//if (bWall) {
							//	//compare with previous wall geomtric parameter
							//	/*for (size_t i = 0, iend = vecWallParams.size(); i < iend; i++) {

							//	}*/
							//	cv::Scalar color(255, 255, 0);
							//	int cSize = 2;
							//	if (vecWallParams.size() > 0) {
							//		auto lastParam = vecWallParams[vecWallParams.size() - 1];
							//		auto val = PlaneInformation::CalcCosineSimilarity(param.rowRange(0, 2), lastParam.rowRange(0, 2));
							//		std::cout << "normal : " << val << "::" << param << ", " << lastParam << std::endl;
							//		if (val > 0.95) {
							//			color.val[0] = 0;
							//			color.val[1] = 255;
							//			color.val[2] = 255;
							//		}
							//		else if (val < 0.3) {
							//			color.val[0] = 0;
							//			color.val[1] = 255;
							//			color.val[2] = 0;
							//			cSize = 4;
							//		}
							//	}
							//	vecWallParams.push_back(param);
							//	for (size_t i = 0, iend = inliers.rows; i < iend; i++) {
							//		cv::Mat temp = inliers.row(i);
							//		cv::Point2f tpt = cv::Point2f(temp.at<float>(0) * 20, temp.at<float>(1) * 20);
							//		tpt += mVisMidPt;
							//		cv::circle(vis, tpt, cSize, color, -1);
							//	}
							//}

							////check to need new subspace
							bool bNeedNewSubspace = true;
							if (pSub) {
								bNeedNewSubspace = pSub->CheckNeedNewSubspace(vec);
							}
							if (bNeedNewSubspace) {
								if (pSub) {
									pSub->mpEndFrame = pSub->mvpSubspaceFrames[pSub->mvpSubspaceFrames.size() - 1];
								}
								pSub = new Subspace();
								pSub->mpStartFrame = mpTargetFrame;
								mpTargetMap->mvpSubspaces.push_back(pSub);
							}
							
							////check to need new subspace

							imshow("floor test", vis); cv::waitKey(1);
						}

						//if (vec.size() > 20) {
						//	cv::Mat normal;
						//	float d;
						//	tempFloor->GetParam(normal, d);

						//	cv::Mat Rcw = UVR_SLAM::PlaneInformation::CalcPlaneRotationMatrix(normal).clone();
						//	cv::Mat Pnew = Rcw.t()*normal;
						//	if (Pnew.at<float>(0) < 0.00001)
						//		Pnew.at<float>(0) = 0.0;
						//	if (Pnew.at<float>(2) < 0.00001)
						//		Pnew.at<float>(2) = 0.0;
						//								
						//	cv::Point2f mVisMidPt(500, 500);
						//	for (size_t i = 0, iend = vec.size(); i < iend; i++) {
						//		auto pMPi = vec[i];
						//		if (!pMPi || pMPi->isDeleted())
						//			continue;
						//		cv::Mat a = pMPi->GetWorldPos();
						//		cv::Mat Xnew = Rcw.t()*a;
						//		cv::Mat Xproj = Xnew.clone();

						//		cv::Mat temp = cv::Mat::ones(1, 3, CV_32FC1);
						//		temp.at<float>(0) = Xproj.at<float>(0);
						//		temp.at<float>(1) = Xproj.at<float>(2);
						//		data.push_back(temp);

						//		Xproj.at<float>(1) = 0.0f;
						//		cv::Point2f tpt = cv::Point2f(Xnew.at<float>(0) * 20, Xnew.at<float>(2) * 20);
						//		tpt += mVisMidPt;
						//		cv::circle(vis, tpt, 2, cv::Scalar(255,0,255), -1);
						//	}
						//	bool bWall = PlaneInformation::Ransac_fitting(data, param, inliers, outliers, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
						//	if (bWall) {
						//		//compare with previous wall geomtric parameter
						//		/*for (size_t i = 0, iend = vecWallParams.size(); i < iend; i++) {

						//		}*/
						//		cv::Scalar color(255, 255, 0);
						//		int cSize = 2;
						//		if (vecWallParams.size() > 0) {
						//			auto lastParam = vecWallParams[vecWallParams.size() - 1];
						//			auto val = PlaneInformation::CalcCosineSimilarity(param.rowRange(0,2), lastParam.rowRange(0,2));
						//			std::cout << "normal : " << val <<"::"<<param<<", "<<lastParam<< std::endl;
						//			if (val > 0.95) {
						//				color.val[0] = 0;
						//				color.val[1] = 255;
						//				color.val[2] = 255;
						//			}else if (val < 0.3) {
						//				color.val[0] = 0;
						//				color.val[1] = 255;
						//				color.val[2] = 0;
						//				cSize = 4;
						//			}
						//		}
						//		vecWallParams.push_back(param);
						//		for (size_t i = 0, iend = inliers.rows; i < iend; i++) {
						//			cv::Mat temp = inliers.row(i);
						//			cv::Point2f tpt = cv::Point2f(temp.at<float>(0) * 20, temp.at<float>(1) * 20);
						//			tpt += mVisMidPt;
						//			cv::circle(vis, tpt, cSize, color, -1);
						//		}
						//	}
						//	imshow("floor test", vis); cv::waitKey(1);
						//}
						
					}

					cv::Mat seg;
					cv::resize(mpTargetFrame->mSegImage, seg, mpSystem->mpVisualizer->mSizeOutputImg);
					mpSystem->mpVisualizer->SetOutputImage(seg, 2);

				}

				if (mpTargetFrame->bSeg && mpTargetFrame->bDepth) {
					
					//mask
					cv::Mat mask_total = cv::Mat::zeros(mpTargetFrame->mnHeight, mpTargetFrame->mnWidth, CV_8UC1);
					cv::Mat mask_floor = cv::Mat::zeros(mpTargetFrame->mnHeight, mpTargetFrame->mnWidth, CV_8UC1);
					cv::Mat mask_wall = cv::Mat::zeros(mpTargetFrame->mnHeight, mpTargetFrame->mnWidth, CV_8UC1);
					cv::Mat mask_ceil = cv::Mat::zeros(mpTargetFrame->mnHeight, mpTargetFrame->mnWidth, CV_8UC1);

					////divide static structure label

					////relative inverse depth

					////depth image
					cv::Mat depth;
					cv::normalize(mpTargetFrame->mRawDepth, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
					cv::cvtColor(depth, depth, CV_GRAY2BGR);
					cv::resize(depth, depth, mpSystem->mpVisualizer->mSizeOutputImg);
					mpSystem->mpVisualizer->SetOutputImage(depth, 1);
					////depth image

					////segmentation image
					mpTargetFrame->mSegImage = cv::Mat::zeros(mpTargetFrame->mSegLabel.size(), CV_8UC3);
					for (int y = 0; y < mpTargetFrame->mSegLabel.rows; y++) {
						for (int x = 0; x < mpTargetFrame->mSegLabel.cols; x++) {
							int label = mpTargetFrame->mSegLabel.at<uchar>(y, x)+1;
							switch (label) {
								case mnLabel_floor:
									mpTargetFrame->mSegImage.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
									mask_floor.at<uchar>(y, x) = 255;
									break;
								case mnLabel_wall:
									mpTargetFrame->mSegImage.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
									mask_wall.at<uchar>(y, x) = 255;
									break;
								case mnLabel_ceil:
									mpTargetFrame->mSegImage.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
									mask_ceil.at<uchar>(y, x) = 255;
									break;
							}
						}
					}
					////////new depth module test
					//////depth reconstruction
					std::vector<cv::Mat> vTempFloorMPs, vTempWallMPs, vTempCeilMPs;
					cv::Mat Rinv, Tinv;
					mpTargetFrame->GetInversePose(Rinv, Tinv);
					//mpSystem->mpMap->ClearTempMPs();
					int inc = 3;
					for (size_t row = inc, rows = mpTargetFrame->mRawDepth.rows; row < rows; row += inc) {
						for (size_t col = inc, cols = mpTargetFrame->mRawDepth.cols; col < cols; col += inc) {
							cv::Point2f pt(col, row);
							float depth = mpTargetFrame->mRawDepth.at<float>(pt);
							if (depth < 0.0001)
								continue;
							cv::Mat a = Rinv*(mpTargetFrame->mInvK*(cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.0)*depth) + Tinv;
							//mpSystem->mpMap->AddTempMP(a);

							int label = mpTargetFrame->mSegLabel.at<uchar>(row, col) + 1;
							switch (label) {
							case mnLabel_floor:
								vTempFloorMPs.push_back(a);
								break;
							case mnLabel_wall:
								vTempWallMPs.push_back(a);
								break;
							case mnLabel_ceil:
								vTempCeilMPs.push_back(a);
								break;
							}
						}
					}

					auto tempFloor = new UVR_SLAM::PlaneInformation();
					auto tempWall1 = new UVR_SLAM::PlaneInformation();
					auto tempWall2 = new UVR_SLAM::PlaneInformation();

					auto ffloorplane = std::async(std::launch::async, UVR_SLAM::lambda_plane_init, vTempFloorMPs, tempFloor, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
					auto fwallplane1 = std::async(std::launch::async, UVR_SLAM::lambda_plane_init, vTempWallMPs,  tempWall1, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
						
					//Wall
					//mpSystem->mpMap->ClearTempMPs();
					mpTargetMap->ClearPlanarTest();
					if (ffloorplane.get()) {

						std::vector<cv::Mat> vfIns = tempFloor->mvInliers;
						std::vector<cv::Mat> vfOuts = tempFloor->mvOutliers;
						for (size_t i = 0, iend = vfIns.size(); i < iend; i++) {
							//mpSystem->mpMap->AddTempMP(vWIns[i]);
							mpTargetMap->AddPlanarTemp(vfIns[i], 1);
						}
						for (size_t i = 0, iend = vfOuts.size(); i < iend; i++) {
							mpTargetMap->AddPlanarTemp(vfOuts[i], 2);
						}


					}
					if (fwallplane1.get()) {
						std::future<bool> fwallplane2;
						/*if (tempWall1->mbParallel) {
							fwallplane2 = std::async(std::launch::async, UVR_SLAM::lambda_plane_init, tempWall1->mvOutliers, tempWall2, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
						}*/

						std::vector<cv::Mat> vWIns = tempWall1->mvInliers;
						std::vector<cv::Mat> vWOuts = tempWall1->mvOutliers;

						std::cout << "Wall Test = " << vWIns.size() << " " << vWOuts.size() << ":" << vTempFloorMPs.size() << ", " << vTempWallMPs.size() << std::endl;

						for (size_t i = 0, iend = vWIns.size(); i < iend; i++) {
							//mpSystem->mpMap->AddTempMP(vWIns[i]);
							mpTargetMap->AddPlanarTemp(vWIns[i], 3);
						}
						for (size_t i = 0, iend = vWOuts.size(); i < iend; i++) {
							mpTargetMap->AddPlanarTemp(vWOuts[i], 4);
						}
						/*if (fwallplane2.get()) {
							std::vector<cv::Mat> vWIns = tempWall2->mvInliers;
							for (size_t i = 0, iend = vWIns.size(); i < iend; i++) {
								mpSystem->mpMap->AddTempMP(vWIns[i]);
							}
						}*/
					}

					////////structure map point
					//std::vector<std::tuple<cv::Point2f, float, int>> vecTuples;
					//cv::Mat R, t;
					//mpTargetFrame->GetPose(R, t);
					//cv::Mat Rcw2 = R.row(2);
					//Rcw2 = Rcw2.t();
					//float zcw = t.at<float>(2);
					//mask_total = mask_floor + mask_ceil + mask_wall;
					//auto mvpMPs = mpTargetFrame->GetMapPoints();
					//
					//for (size_t i = 0, iend = mvpMPs.size(); i < iend; i++) {
					//	auto pMPi = mvpMPs[i];
					//	if (!pMPi || pMPi->isDeleted() || pMPi->isNewMP()) {
					//		continue;
					//	}
					//	auto pt = mpTargetFrame->mvPts[i];
					//	if (mask_total.at<uchar>(pt)) {
					//		cv::Mat x3Dw = pMPi->GetWorldPos();
					//		float z = (float)Rcw2.dot(x3Dw) + zcw;
					//		std::tuple<cv::Point2f, float, int> data = std::make_tuple(std::move(pt), 1.0 / z, pMPi->GetNumObservations());//cv::Point2f(pt.x / 2, pt.y / 2)
					//		vecTuples.push_back(data);
					//	}
					//}
					////////depth estimation
					////////depth ���� ���� �� ����Ʈ�� ����Ʈ ������ Ʃ�÷� ����

					////////����Ʈ�� ����Ʈ ������ ����
					/////*std::sort(vecTuples.begin(), vecTuples.end(),
					////	[](std::tuple<cv::Point2f, float, int> const &t1, std::tuple<cv::Point2f, float, int> const &t2) {
					////		if (std::get<2>(t1) == std::get<2>(t2)) {
					////			return std::get<0>(t1).x != std::get<0>(t2).x ? std::get<0>(t1).x > std::get<0>(t2).x : std::get<0>(t1).y > std::get<0>(t2).y;
					////		}
					////		else {
					////			return std::get<2>(t1) > std::get<2>(t2);
					////		}
					////	}
					////);*/

					////////�Ķ���� �˻� �� ���� ���� ����
					//int nTotal = 20;
					//if (vecTuples.size() > nTotal) {
					//	int nData = nTotal;
					//	cv::Mat A = cv::Mat::ones(nData, 2, CV_32FC1);
					//	cv::Mat B = cv::Mat::zeros(nData, 1, CV_32FC1);

					//	for (size_t i = 0; i < nData; i++) {
					//		auto data = vecTuples[i];
					//		auto pt = std::get<0>(data);
					//		auto invdepth = std::get<1>(data);
					//		auto nConnected = std::get<2>(data);

					//		float p = mpTargetFrame->mRawDepth.at<float>(pt);
					//		A.at<float>(i, 0) = p;//invdepth;
					//		B.at<float>(i) = invdepth;//p;
					//	}

					//	//cv::Mat X = A.inv(cv::DECOMP_QR)*B;
					//	cv::Mat S = A.t()*A;
					//	cv::Mat X = S.inv()*A.t()*B;
					//	float a = X.at<float>(0);
					//	float b = X.at<float>(1);

					//	/*cv::Mat C = A*X - B;
					//	std::cout << "depth val = " << cv::sum(C) / C.rows << std::endl;*/

					//	//depth = a*depth + b; //(depth - b) / a;

					//	mpTargetFrame->mDepthImage = cv::Mat::zeros(mpTargetFrame->mnHeight, mpTargetFrame->mnWidth, CV_32FC1);
					//	for (int x = 0, cols = mpTargetFrame->mRawDepth.cols; x < cols; x++) {
					//		for (int y = 0, rows = mpTargetFrame->mRawDepth.rows; y < rows; y++) {
					//			float val = a*mpTargetFrame->mRawDepth.at<float>(y, x) + b;//1.0 / depth.at<float>(y, x);
					//			/*if (val < 0.0001)
					//			val = 0.5;*/
					//			mpTargetFrame->mDepthImage.at<float>(y, x) = 1.0/val;
					//		}
					//	}
						////////depth reconstruction
						////////�ð�ȭ �׽�Ʈ
						//std::vector<cv::Mat> vTempFloorMPs, vTempWallMPs, vTempCeilMPs;
						//cv::Mat Rinv, Tinv;
						//mpTargetFrame->GetInversePose(Rinv, Tinv);
						////mpSystem->mpMap->ClearTempMPs();
						//int inc = 3;
						//for (size_t row = inc, rows = mpTargetFrame->mDepthImage.rows; row < rows; row += inc) {
						//	for (size_t col = inc, cols = mpTargetFrame->mDepthImage.cols; col < cols; col += inc) {
						//		cv::Point2f pt(col, row);
						//		float depth = mpTargetFrame->mDepthImage.at<float>(pt);
						//		if (depth < 0.0001)
						//			continue;
						//		cv::Mat a = Rinv*(mpTargetFrame->mInvK*(cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.0)*depth) + Tinv;
						//		//mpSystem->mpMap->AddTempMP(a);

						//		int label = mpTargetFrame->mSegLabel.at<uchar>(row, col) + 1;
						//		switch (label) {
						//		case mnLabel_floor:
						//			vTempFloorMPs.push_back(a);
						//			break;
						//		case mnLabel_wall:
						//			vTempWallMPs.push_back(a);
						//			break;
						//		case mnLabel_ceil:
						//			vTempCeilMPs.push_back(a);
						//			break;
						//		}
						//	}
						//}
					//	////Plane Estimation
					//	////Correlation
					//	/*cv::Mat fParam, wParam, cParam;
					//	std::vector<cv::Mat> vFIns, vFOuts;
					//	std::vector<cv::Mat> vWIns, vWOuts;
					//	std::vector<cv::Mat> vCIns, vCOuts;*/

					//	auto tempFloor = new UVR_SLAM::PlaneInformation();
					//	auto tempWall1 = new UVR_SLAM::PlaneInformation();
					//	auto tempWall2 = new UVR_SLAM::PlaneInformation();

					//	auto ffloorplane = std::async(std::launch::async, UVR_SLAM::lambda_plane_init, vTempFloorMPs, tempFloor, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
					//	auto fwallplane1 = std::async(std::launch::async, UVR_SLAM::lambda_plane_init, vTempWallMPs,  tempWall1, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
					//	
					//	//Wall
					//	//mpSystem->mpMap->ClearTempMPs();
					//	mpTargetMap->ClearPlanarTest();
					//	if (ffloorplane.get()) {
					//		std::vector<cv::Mat> vfIns = tempFloor->mvInliers;
					//		std::vector<cv::Mat> vfOuts = tempFloor->mvOutliers;
					//		for (size_t i = 0, iend = vfIns.size(); i < iend; i++) {
					//			//mpSystem->mpMap->AddTempMP(vWIns[i]);
					//			mpTargetMap->AddPlanarTemp(vfIns[i], 1);
					//		}
					//		for (size_t i = 0, iend = vfOuts.size(); i < iend; i++) {
					//			mpTargetMap->AddPlanarTemp(vfOuts[i], 2);
					//		}
					//	}
					//	if (fwallplane1.get()) {
					//		std::future<bool> fwallplane2;
					//		/*if (tempWall1->mbParallel) {
					//			fwallplane2 = std::async(std::launch::async, UVR_SLAM::lambda_plane_init, tempWall1->mvOutliers, tempWall2, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
					//		}*/

					//		std::vector<cv::Mat> vWIns = tempWall1->mvInliers;
					//		std::vector<cv::Mat> vWOuts = tempWall1->mvOutliers;

					//		std::cout << "Wall Test = " << vWIns.size() << " " << vWOuts.size() << ":" << vTempFloorMPs.size() << ", " << vTempWallMPs.size() << std::endl;

					//		for (size_t i = 0, iend = vWIns.size(); i < iend; i++) {
					//			//mpSystem->mpMap->AddTempMP(vWIns[i]);
					//			mpTargetMap->AddPlanarTemp(vWIns[i], 3);
					//		}
					//		for (size_t i = 0, iend = vWOuts.size(); i < iend; i++) {
					//			mpTargetMap->AddPlanarTemp(vWOuts[i], 4);
					//		}
					//		/*if (fwallplane2.get()) {
					//			std::vector<cv::Mat> vWIns = tempWall2->mvInliers;
					//			for (size_t i = 0, iend = vWIns.size(); i < iend; i++) {
					//				mpSystem->mpMap->AddTempMP(vWIns[i]);
					//			}
					//		}*/
					//	}
					//	

					//	////Estimated Depth image
					//	/*cv::normalize(mpTargetFrame->mDepthImage, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
					//	cv::cvtColor(depth, depth, CV_GRAY2BGR);
					//	cv::resize(depth, depth, mpSystem->mpVisualizer->mSizeOutputImg);
					//	mpSystem->mpVisualizer->SetOutputImage(depth, 1);*/
					//}

					cv::Mat seg;
					cv::resize(mpTargetFrame->mSegImage, seg, mpSystem->mpVisualizer->mSizeOutputImg);
					mpSystem->mpVisualizer->SetOutputImage(seg,2);
					////segmentation image
				}
				continue;

				////////relocaalization
				//auto vpCandidateKFs = mpKeyFrameDatabase->DetectPlaceCandidates(mpTargetFrame);
				//std::cout << "Place Recognizer = " << vpCandidateKFs.size() << std::endl;
				//int nMax = 0; 
				//cv::Mat maxMatch;
				//Frame* maxFrame = nullptr;
				//for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++) {
				//	auto pCandidate = vpCandidateKFs[i];
				//	cv::Mat matches = lambda_api_kf_match_loop_closing(mpSystem->ip, mpSystem->port, mpTargetFrame->mstrMapName, mpTargetFrame->mnFrameID, pCandidate->mnFrameID, mpTargetFrame->mvPts.size());
				//	auto vpMPs = pCandidate->GetMapPoints();
				//	int nMatch = 0;
				//	int nTempMatch = 0;
				//	for (int j = 0, jend = matches.cols; j < jend; j++) {
				//		int idx1 = j;
				//		int idx2 = matches.at<int>(idx1);
				//		if (idx2 == 10000)
				//			continue;
				//		nTempMatch++;
				//		auto pMP = vpMPs[idx2];
				//		if (!pMP || pMP->isDeleted())
				//			continue;
				//		nMatch++;
				//	}
				//	if (nMatch > nMax) {
				//		nMax = nMatch;
				//		maxMatch = matches.clone();
				//		maxFrame = pCandidate;
				//	}
				//	//std::cout <<"ID = "<<pCandidate->mnFrameID<<", "<< nTempMatch << std::endl;
				//	//cv::imshow("candidate", pCandidate->GetOriginalImage()); cv::waitKey(1);
				//}
				//if (maxFrame)
				//{
				//	//cv::imshow("candidate", maxFrame->GetOriginalImage()); cv::waitKey(10);
				//	//Pose Optimization Test
				//	cv::Mat R, t;
				//	maxFrame->GetPose(R, t);
				//	mpTargetFrame->SetPose(R, t);
				//	cv::Mat targetImg = [](std::string ip, int port, int id, std::string map, int w, int h) {
				//		WebAPI* mpAPI = new WebAPI(ip, port);
				//		std::stringstream ss;
				//		ss << "/SendData?map=" << map << "&id=" << id << "&key=bimage";
				//		auto res = mpAPI->Send(ss.str(), "");
				//		int n = res.size();
				//		cv::Mat temp = cv::Mat::zeros(n, 1, CV_8UC1);
				//		std::memcpy(temp.data, res.data(), n * sizeof(uchar));
				//		cv::Mat img = cv::imdecode(temp, cv::IMREAD_COLOR);
				//		return img;
				//	}(mpSystem->ip, mpSystem->port, mpTargetFrame->mnFrameID, mpTargetFrame->mstrMapName, mpTargetFrame->mnWidth, mpTargetFrame->mnHeight);

				//	//cv::Mat targetImg = mpTargetFrame->GetOriginalImage().clone();
				//	cv::Mat resimg = maxFrame->GetOriginalImage().clone();
				//	
				//	
				//	std::vector<cv::Point2f> vPTs;
				//	std::vector<cv::Point3f> vXws;
				//	std::vector<MapPoint*> vMPs;
				//	std::vector <bool> vInliers;
				//	auto vpMPs = maxFrame->GetMapPoints();
				//	for (int j = 0, jend = maxMatch.cols; j < jend; j++) {
				//		int idx1 = j;
				//		int idx2 = maxMatch.at<int>(idx1);
				//		if (idx2 == 10000)
				//			continue;
				//		auto pMP = vpMPs[idx2];
				//		if (!pMP || pMP->isDeleted())
				//			continue;
				//		vPTs.push_back(mpTargetFrame->mvPts[j]);
				//		vMPs.push_back(pMP);
				//		vInliers.push_back(true);
				//		cv::Mat Xw = pMP->GetWorldPos();
				//		
				//		vXws.push_back(cv::Point3f(Xw.at<float>(0), Xw.at<float>(1), Xw.at<float>(2)));

				//		cv::Mat proj = mpTargetFrame->mK*(R*Xw + t);
				//		cv::Point2f projPt(proj.at<float>(0) / proj.at<float>(2), proj.at<float>(1) / proj.at<float>(2));
				//		cv::circle(targetImg, mpTargetFrame->mvPts[idx1], 3, cv::Scalar(255, 255, 0), -1);
				//		cv::circle(resimg, maxFrame->mvPts[idx2], 3, cv::Scalar(255, 0, 255), -1);
				//		cv::circle(resimg, projPt, 3, cv::Scalar(0, 255, 255), -1);
				//	}
				//	cv::resize(targetImg, targetImg, cv::Size(mpTargetFrame->mnWidth / 2, mpTargetFrame->mnHeight / 2));
				//	cv::resize(resimg, resimg, cv::Size(mpTargetFrame->mnWidth / 2, mpTargetFrame->mnHeight / 2));
				//	mpSystem->mpVisualizer->SetOutputImage(targetImg, 0);
				//	mpSystem->mpVisualizer->SetOutputImage(resimg, 1);
				//	/*cv::Mat rvec, tvec;
				//	cv::solvePnPRansac(vXws, vPTs, mpSystem->mK, cv::Mat(), rvec, tvec);
				//	cv::Rodrigues(rvec, R);*/
				//	mpTargetFrame->SetPose(R, t);
				//	int nPoseRecovery = Optimization::PoseOptimization(mpMap, mpTargetFrame, vMPs, vPTs, vInliers);
				//	mpMap->SetUserPosition(mpTargetFrame->GetCameraCenter());
				//	std::cout << "Place Recognizer::res::" << "::"<< nMax << std::endl;
				//}
				////////relocaalization

				//if (mpTargetFrame->GetConnectedKFs().size() < 3) {
				//	for (size_t i = 0, iend = vpNeighKFs.size(); i < iend; i++) {
				//		auto pKF = vpNeighKFs[i];
				//		/*auto ftest = std::async(std::launch::async, UVR_SLAM::lambda_api_kf_match, "143.248.96.81", 35005, mpTargetFrame->mnFrameID, pKF->mnFrameID, mpTargetFrame->mvPts.size());
				//		cv::Mat temp = ftest.get();*/
				//		cv::Mat temp = lambda_api_kf_match_loop_closing(mpSystem->ip, mpSystem->port, mpTargetFrame->mnFrameID, pKF->mnFrameID, mpTargetFrame->mvPts.size());
				//		if (mpTargetFrame->mvPts.size() != temp.cols) {
				//			std::cout << "Error::Matching::Invalid Matching Size::" << temp.cols << ", " << mpTargetFrame->mvPts.size() << std::endl;
				//		}
				//		std::vector<bool> vecBoolOverlap(pKF->mvPts.size(), false);
				//		for (size_t j = 0, jend = temp.cols; j < jend; j++) {
				//			int idx1 = j;
				//			int idx2 = temp.at<int>(idx1);
				//			if (idx2 == -1)
				//				continue;
				//			if (idx2 >= pKF->mvPts.size() || idx2 < -1) {
				//				temp.at<int>(idx1) = -1;
				//				std::cout << "Error::Matching::Invalid Frame2 Indexs = " << idx2 << ", " << pKF->mvPts.size() << "::" << j << std::endl;
				//				continue;
				//			}
				//			if (vecBoolOverlap[idx2])
				//			{
				//				temp.at<int>(idx1) = -1;
				//				continue;
				//			}
				//			vecBoolOverlap[idx2] = true;
				//		}
				//	}
				//	//mMatches.push_back(temp);
				//}


				{
					//segmentation test
					WebAPI* mpAPI = new WebAPI(mpSystem->ip, mpSystem->port);
					std::stringstream ss;
					ss << "/GetLastFrameID?map="<<mpTargetFrame->mstrMapName<<"&key=bsegmentation";
					WebAPIDataConverter::ConvertStringToNumber(mpAPI->Send(ss.str(), "").c_str(), nCurrSegFrame);
					if (nCurrSegFrame >= 0 && nCurrSegFrame != nPrevSegFrame) {
						ss.str("");
						ss << "/SendData?map=" << mpTargetFrame->mstrMapName << "&id="<< nCurrSegFrame << "&key=bsegmentation";
						auto f = std::async(std::launch::async, [](WebAPI* wapi,std::string method, int w, int h) {
							auto resdata = wapi->Send(method, "");
							////ó�� �� ����
							cv::Mat seg = cv::Mat::zeros(h, w, CV_8UC1);
							std::memcpy(seg.data, resdata.data(), w*h*sizeof(uchar));
							cv::Mat seg_color = cv::Mat::zeros(seg.size(), CV_8UC3);
							for (int y = 0; y < seg_color.rows; y++) {
								for (int x = 0; x < seg_color.cols; x++) {
									int label = seg.at<uchar>(y, x);
									seg_color.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
								}
							}


							return seg_color;
							//imshow("segmentation", seg_color); cv::waitKey(1);
						}, mpAPI, ss.str(), mpTargetFrame->mnWidth/2, mpTargetFrame->mnHeight/2);
						nPrevSegFrame = nCurrSegFrame;
						auto res = f.get();

						//edge
						cv::Mat filtered,edge;
						GaussianBlur(res, filtered, cv::Size(5, 5), 0.0);
						cv::Canny(filtered, edge, 50, 200);//150
						imshow("edge", edge); cv::waitKey(1);
						cv::resize(res, res, mpSystem->mpVisualizer->mSizeOutputImg);
						mpSystem->mpVisualizer->SetOutputImage(res, 2);
					}
				}

				{
					//segmentation test
					WebAPI* mpAPI = new WebAPI(mpSystem->ip, mpSystem->port);
					std::stringstream ss;
					ss << "/GetLastFrameID?map=" << mpTargetFrame->mstrMapName << "&key=bdepth";
					WebAPIDataConverter::ConvertStringToNumber(mpAPI->Send(ss.str(), "").c_str(), nCurrDepthFrame);
					if (nCurrDepthFrame >= 0 && nCurrDepthFrame != nPrevDepthFrame) {
						ss.str("");
						ss << "/SendData?map=" << mpTargetFrame->mstrMapName << "&id=" << nCurrDepthFrame << "&key=bdepth";
						auto f = std::async(std::launch::async, [](WebAPI* wapi, std::string method, int w, int h, Frame* pF) {
							auto resdata = wapi->Send(method, "");
							
							////ó�� �� ����
							cv::Mat depth = cv::Mat::zeros(h, w, CV_32FC1);
							std::memcpy(depth.data, resdata.data(), w*h * sizeof(float));

							cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
							cv::cvtColor(depth, depth, CV_GRAY2BGR);
							//cv::resize(depth, depth, depth.size() / 2);

							///////�� �Ʒ��� ���� ���� ����
							//std::vector<std::tuple<cv::Point2f, float, int>> vecTuples;
							//cv::Mat R, t;
							//pF->GetPose(R, t);

							//////depth ���� ���� �� ����Ʈ�� ����Ʈ ������ Ʃ�÷� ����
							//cv::Mat Rcw2 = R.row(2);
							//Rcw2 = Rcw2.t();
							//float zcw = t.at<float>(2);
							//auto vpMPs = pF->GetMapPoints();
							//for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
							//	auto pMPi = vpMPs[i];
							//	if (!pMPi || pMPi->isDeleted())
							//		continue;
							//	auto pt = pF->mvPts[i];
							//	cv::Mat x3Dw = pMPi->GetWorldPos();
							//	float z = (float)Rcw2.dot(x3Dw) + zcw;
							//	std::tuple<cv::Point2f, float, int> data = std::make_tuple(std::move(pt), 1.0 / z, pMPi->GetNumObservations());//cv::Point2f(pt.x / 2, pt.y / 2)
							//	vecTuples.push_back(data);
							//}

							//////����Ʈ�� ����Ʈ ������ ����
							//std::sort(vecTuples.begin(), vecTuples.end(),
							//	[](std::tuple<cv::Point2f, float, int> const &t1, std::tuple<cv::Point2f, float, int> const &t2) {
							//		if (std::get<2>(t1) == std::get<2>(t2)) {
							//			return std::get<0>(t1).x != std::get<0>(t2).x ? std::get<0>(t1).x > std::get<0>(t2).x : std::get<0>(t1).y > std::get<0>(t2).y;
							//		}
							//		else {
							//			return std::get<2>(t1) > std::get<2>(t2);
							//		}
							//	}
							//);

							//////�Ķ���� �˻� �� ���� ���� ����
							//int nTotal = 20;
							//if (vecTuples.size() > nTotal) {
							//	int nData = nTotal;
							//	cv::Mat A = cv::Mat::ones(nData, 2, CV_32FC1);
							//	cv::Mat B = cv::Mat::zeros(nData, 1, CV_32FC1);

							//	for (size_t i = 0; i < nData; i++) {
							//		auto data = vecTuples[i];
							//		auto pt = std::get<0>(data);
							//		auto invdepth = std::get<1>(data);
							//		auto nConnected = std::get<2>(data);

							//		float p = depth.at<float>(pt);
							//		A.at<float>(i, 0) = p;//invdepth;
							//		B.at<float>(i) = invdepth;//p;
							//	}

							//	//cv::Mat X = A.inv(cv::DECOMP_QR)*B;
							//	cv::Mat S = A.t()*A;
							//	cv::Mat X = S.inv()*A.t()*B;
							//	float a = X.at<float>(0);
							//	float b = X.at<float>(1);

							//	/*cv::Mat C = A*X - B;
							//	std::cout << "depth val = " << cv::sum(C) / C.rows << std::endl;*/

							//	//depth = a*depth + b; //(depth - b) / a;
							//	for (int x = 0, cols = depth.cols; x < cols; x++) {
							//		for (int y = 0, rows = depth.rows; y < rows; y++) {
							//			float val = a*depth.at<float>(y, x) + b;//1.0 / depth.at<float>(y, x);
							//			/*if (val < 0.0001)
							//			val = 0.5;*/
							//			depth.at<float>(y, x) = 1.0/val;
							//		}
							//	}
							//	
							//}

							return depth;
						}, mpAPI, ss.str(), mpTargetFrame->mnWidth, mpTargetFrame->mnHeight, mpTargetFrame);
						nPrevDepthFrame = nCurrDepthFrame;
						auto res = f.get();
						cv::Mat tempDepth = res.clone();

						/*cv::normalize(res, res, 0, 255, cv::NORM_MINMAX, CV_8UC1);
						cv::cvtColor(res, res, CV_GRAY2BGR);
						cv::resize(res, res, res.size() / 2);*/
						cv::resize(res, res, mpSystem->mpVisualizer->mSizeOutputImg);
						mpSystem->mpVisualizer->SetOutputImage(res, 3);

						////�ð�ȭ �׽�Ʈ
						/*cv::Mat Rinv, Tinv;
						mpTargetFrame->GetInversePose(Rinv, Tinv);
						mpSystem->mpMap->ClearTempMPs();
						int inc = 10;
						for (size_t row = inc, rows = tempDepth.rows; row < rows; row += inc) {
							for (size_t col = inc, cols = tempDepth.cols; col < cols; col += inc) {
								cv::Point2f pt(col, row);
								float depth = tempDepth.at<float>(pt);
								if (depth < 0.0001)
									continue;
								cv::Mat a = Rinv*(mpTargetFrame->mInvK*(cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.0)*depth) + Tinv;
								mpSystem->mpMap->AddTempMP(a);
							}
						}*/

						////ó�� �� ���� ��� ����
						/*ss.str("");
						ss << "/ReceiveData?map=" << mpTargetFrame->mstrMapName << "&id=" << nCurrDepthFrame << "&key=rdepth";
						mpAPI->Send(ss.str(), tempDepth.data, tempDepth.rows*tempDepth.cols*sizeof(float));*/
					}
				}

				std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
				auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				float t_test1 = du_test1 / 1000.0;
#ifdef DEBUG_LOOP_CLOSING_LEVEL_1
				std::cout << "MappingServer::LoopClosing::" << mpTargetFrame->mnFrameID << "::"<< vpNeighKFs.size()<<"::"<<t_test1 << "::End" << std::endl;
#endif
			}
		}
	}
	void LoopCloser::Run() {

		while (true) {
			if (CheckNewKeyFrames()) {
				SetBoolProcessing(true);
				ProcessNewKeyFrame();

				if (DetectLoop()) {
					std::cout << "Loop Frame Detection!!" << std::endl;
					if (ComputeSim3())
					{
						std::cout << "Loop Closing!!" << std::endl;
						CorrectLoop();
					}
				}

				///////////////VoW ��Ī
				//auto vpGrahWindows = mpMap->GetGraphFrames();
				//for (int i = 0; i < vpGrahWindows.size(); i++) {
				//	auto pKFi = vpGrahWindows[i];
				//	auto score = mpTargetFrame->Score(pKFi);
				//	
				//	if (score > 0.01) {
				//		std::cout << "Loop::Score::" << score << std::endl;
				//		imshow("Loop::1", mpTargetFrame->GetOriginalImage());
				//		imshow("Loop::2", pKFi->GetOriginalImage());
				//		cv::waitKey(500);
				//	}
				//}
				///////////////VoW ��Ī
				mpKeyFrameDatabase->Add(mpTargetFrame);
				SetBoolProcessing(false);
			}//visualize
		}
	}
	void LoopCloser::InsertKeyFrame(UVR_SLAM::Frame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		mKFQueue.push(pKF);
	}

	bool LoopCloser::CheckNewKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		return(!mKFQueue.empty());
	}

	void LoopCloser::ProcessNewKeyFrame()
	{
		{
			std::unique_lock<std::mutex> lock(mMutexNewKFs);
			mpTargetFrame = mKFQueue.front();
			mKFQueue.pop();
		}
		/*[](std::string ip, int port, Frame* pF) {
			WebAPI* mpAPI = new WebAPI(ip, port);
			std::stringstream ss;
			ss << "/SendData?map=" << pF->mstrMapName << "&id=" << pF->mnFrameID << "&key=bdesc";
			WebAPIDataConverter::ConvertBytesToDesc(mpAPI->Send(ss.str(), "").c_str(), pF->mvPts.size(), pF->matDescriptor);
			pF->ComputeBoW();
		}(mpSystem->ip, mpSystem->port, mpTargetFrame);*/
	}


	bool LoopCloser::isProcessing() {
		std::unique_lock<std::mutex> lock(mMutexProcessing);
		return mbProcessing;
	}
	void LoopCloser::SetBoolProcessing(bool b) {
		std::unique_lock<std::mutex> lock(mMutexProcessing);
		mbProcessing = b;
	}

	bool LoopCloser::DetectLoop() {

		std::vector<Frame*> vpKFs = mpTargetFrame->GetConnectedKFs();
		float minScore = 1;
		const DBoW3::BowVector &CurrentBowVec = mpTargetFrame->mBowVec;
		for (auto iter = vpKFs.begin(), iter2 = vpKFs.end(); iter != iter2; iter++) {
			auto pKF = *iter;
			if (mpTargetFrame == pKF)
				continue;
			const DBoW3::BowVector &BowVec = pKF->mBowVec;
			float score = mpVoc->score(CurrentBowVec, BowVec);
			if (score < minScore)
				minScore = score;
		}

		auto vpCandidateKFs = mpKeyFrameDatabase->DetectLoopCandidates(mpTargetFrame, minScore);
		if(vpCandidateKFs.empty())
		{
			
			mvConsistentGroups.clear();
			//mpKeyFrameDB->add(mpCurrentKF);
			//mpCurrentKF->SetErase();
			return false;
		}
		mvpEnoughConsistentCandidates.clear();

		//�ϳ��� ConsistentGroup�� frame�� ���հ� int(�ش� �׷�� ����� �׷��� ���� �ǹ�)�� ����. ConsistentGroup�� ����
		//mvConsistentGroup ���� �� �׷���� ���� ������� ����.
		//���ڱ� �ѹ� ���� ��Ī�� �������� �ƴ� ���������� ��Ī�� �Ǿ�� ���� �������� ����. �ٸ�, ������ ������ �������� �ƴ� �ٸ� �����Ӱ� ����� ��쿡�� ������.
		std::vector<ConsistentGroup> vCurrentConsistentGroups;
		std::vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);
		for (size_t i = 0, iend = vpCandidateKFs.size(); i<iend; i++)
		{
			Frame* pCandidateKF = vpCandidateKFs[i];
			
			//bow vector�� �̿��ؼ� ȹ���� �ĺ� Ű������ ���� ����� �������� ���� �����ؼ� ĵ����Ʈ �׷����� ����
			//�̰� ���� Ű�����ӿ� ���� ConsistentGroup,�̸� ������ ����
			std::set<Frame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrameSet();
			spCandidateGroup.insert(pCandidateKF);

			bool bEnoughConsistent = false;
			bool bConsistentForSomeGroup = false;
			//���� ����� ��� ConsistentGroup�� üũ��.
			for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG<iendG; iG++)
			{
				//���� �˻��ϰ��� �ϴ� ConsistentGroup�� ������ ����
				std::set<Frame*> sPreviousGroup = mvConsistentGroups[iG].first;

				bool bConsistent = false;
				for (std::set<Frame*>::iterator sit = spCandidateGroup.begin(), send = spCandidateGroup.end(); sit != send; sit++)
				{
					//���� ConsistentGroup �׷� �߿� ���� Ű�������� ConsistentGroup�� ��ġ�°� �ִ��� ã�� ������ �ٷ� ��ž
					if (sPreviousGroup.count(*sit))
					{
						bConsistent = true;
						bConsistentForSomeGroup = true;
						break;
					}
				}
				//ConsistentGroup�� ã�� ���
				if (bConsistent)
				{
					//���� Ű�������� ConsistentGroup ���Ϳ� ���� ������ ������ �� �߰�. 
					//�ĺ� �������� ���� ������ ã���� ��� ã�� �׷쿡 �ƿ� �߰�. �� ������ �ĺ� �����ӵ鸶�� �� �ѹ��� ���� ��
					//3���� �����Ӱ� ����Ǹ� ��.
					int nPreviousConsistency = mvConsistentGroups[iG].second;
					int nCurrentConsistency = nPreviousConsistency + 1;
					if (!vbConsistentGroup[iG])
					{
						ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
						vCurrentConsistentGroups.push_back(cg);
						vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
					}
					if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
					{
						mvpEnoughConsistentCandidates.push_back(pCandidateKF);
						bEnoughConsistent = true; //this avoid to insert the same candidate more than once
					}
				}
			}

			//���� �˻����� �ĺ� �����Ӱ� ����Ǵ� ConsistentGroup�� ������ 0���� ����
			// If the group is not consistent with any previous group insert with consistency counter set to zero
			if (!bConsistentForSomeGroup)
			{
				ConsistentGroup cg = make_pair(spCandidateGroup, 0);
				vCurrentConsistentGroups.push_back(cg);
			}
		}

		// Update Covisibility Consistent Groups
		mvConsistentGroups = vCurrentConsistentGroups;


		// Add Current Keyframe to database
		if (mvpEnoughConsistentCandidates.empty())
		{
			//mpCurrentKF->SetErase();
			return false;
		}
		else
		{
			return true;
		}

		//mpCurrentKF->SetErase();
		return false;
	}
	bool LoopCloser::ComputeSim3() {
		
		////�ĺ� ���� �������� Sim3 ���
		const int nInitialCandidates = mvpEnoughConsistentCandidates.size();
		std::vector<Sim3Solver*> vpSim3Solvers;
		vpSim3Solvers.resize(nInitialCandidates);

		//map point�� 2���� ���ͷ� ���� Ÿ�� �����Ӱ� �ĺ� ���� ������ ������ ��Ī ������ ������.
		//���� Ÿ�� �������� ����Ʈ ������ x �ĺ� ���� ������ ���� ũ�Ⱑ ��
		std::vector<std::vector<MapPoint*> > vvpMapPointMatches;
		vvpMapPointMatches.resize(nInitialCandidates);
		
		std::vector<bool> vbDiscarded;
		vbDiscarded.resize(nInitialCandidates);

		int nCandidates = 0; //candidates with enough matches

		for (int i = 0; i<nInitialCandidates; i++)
		{
			Frame* pKF = mvpEnoughConsistentCandidates[i];

			// avoid that local mapping erase it while it is being processed in this thread
			//pKF->SetNotErase();

			if (pKF->isDeleted())
			{
				vbDiscarded[i] = true;
				continue;
			}

			////bow ��� ��Ī
			int nmatches = mpMatcher->BagOfWordsMatching(mpTargetFrame, pKF, vvpMapPointMatches[i]);
			if (nmatches<20)
			{
				vbDiscarded[i] = true;
				continue;
			}
			else
			{
				//�� �������� �ڼ��� ��Ī ������ RANSAC�� �̿��ؼ� Sim3Solver ����
				Sim3Solver* pSolver = new Sim3Solver(mpTargetFrame, pKF, vvpMapPointMatches[i], mbFixScale);
				pSolver->SetRansacParameters(0.99, 20, 300);
				vpSim3Solvers[i] = pSolver;
			}

			nCandidates++;
		}
		bool bMatch = false;

		// Perform alternatively RANSAC iterations for each candidate
		// until one is succesful or all fail
		while (nCandidates>0 && !bMatch)
		{
			for (int i = 0; i<nInitialCandidates; i++)
			{
				if (vbDiscarded[i])
					continue;

				Frame* pKF = mvpEnoughConsistentCandidates[i];

				// Perform 5 Ransac Iterations
				std::vector<bool> vbInliers;
				int nInliers;
				bool bNoMore;

				Sim3Solver* pSolver = vpSim3Solvers[i];
				cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

				// If Ransac reachs max. iterations discard keyframe
				if (bNoMore)
				{
					vbDiscarded[i] = true;
					nCandidates--;
				}

				// If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
				if (!Scm.empty())
				{
					std::vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
					for (size_t j = 0, jend = vbInliers.size(); j<jend; j++)
					{
						if (vbInliers[j])
							vpMapPointMatches[j] = vvpMapPointMatches[i][j];
					}

					cv::Mat R = pSolver->GetEstimatedRotation();
					cv::Mat t = pSolver->GetEstimatedTranslation();
					const float s = pSolver->GetEstimatedScale();
					
					//matcher.SearchBySim3(mpTargetFrame, pKF, vpMapPointMatches, s, R, t, 7.5);

					g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);
					const int nInliers = Optimization::OptimizeSim3(mpTargetFrame, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

					// If optimization is succesful stop ransacs and continue
					if (nInliers >= 20)
					{
						bMatch = true;
						mpMatchedKF = pKF;
						g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()), Converter::toVector3d(pKF->GetTranslation()), 1.0);
						mg2oScw = gScm*gSmw;
						mScw = Converter::toCvMat(mg2oScw);

						mvpCurrentMatchedPoints = vpMapPointMatches;
						break;
					}
				}
			}
		}

		////Erase ���� 
		if (!bMatch)
		{
			/*for (int i = 0; i<nInitialCandidates; i++)
				mvpEnoughConsistentCandidates[i]->SetErase();
			mpCurrentKF->SetErase();*/
			return false;
		}

		// Retrieve MapPoints seen in Loop Keyframe and neighbors
		std::vector<Frame*> vpLoopConnectedKFs = mpMatchedKF->GetConnectedKFs();
		vpLoopConnectedKFs.push_back(mpMatchedKF);
		mvpLoopMapPoints.clear();
		for (std::vector<Frame*>::iterator vit = vpLoopConnectedKFs.begin(); vit != vpLoopConnectedKFs.end(); vit++)
		{
			Frame* pKF = *vit;
			std::vector<MapPoint*> vpMapPoints = pKF->GetMapPoints();
			for (size_t i = 0, iend = vpMapPoints.size(); i<iend; i++)
			{
				MapPoint* pMP = vpMapPoints[i];
				if (pMP)
				{
					if (!pMP->isDeleted() && pMP->mnLoopPointForKF != mpTargetFrame->mnKeyFrameID)
					{
						mvpLoopMapPoints.push_back(pMP);
						pMP->mnLoopPointForKF = mpTargetFrame->mnKeyFrameID;
					}
				}
			}
		}

		//// Find more matches projecting with the computed Sim3
		//matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints, 10);

		//// If enough matches accept Loop
		int nTotalMatches = 0;
		for (size_t i = 0; i<mvpCurrentMatchedPoints.size(); i++)
		{
			if (mvpCurrentMatchedPoints[i])
				nTotalMatches++;
		}

		if (nTotalMatches >= 40)
		{
			/*for (int i = 0; i<nInitialCandidates; i++)
				if (mvpEnoughConsistentCandidates[i] != mpMatchedKF)
					mvpEnoughConsistentCandidates[i]->SetErase();*/
			return true;
		}
		else
		{
			/*for (int i = 0; i<nInitialCandidates; i++)
				mvpEnoughConsistentCandidates[i]->SetErase();
			mpCurrentKF->SetErase();*/
			return false;
		}
	}
	void LoopCloser::CorrectLoop() {
		
		////���� ���ο� ��ž ��û

		////GBA ���� ���߱� && �����ϱ�

		////KF update connection ���� �� ���� �ʿ�
		//mpTargetFrame->UpdateConnections();

		// Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
		mvpCurrentConnectedKFs = mpTargetFrame->GetConnectedKFs();
		mvpCurrentConnectedKFs.push_back(mpTargetFrame);

		KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
		CorrectedSim3[mpTargetFrame] = mg2oScw;
		cv::Mat Rwc, twc;
		cv::Mat Twc;
		mpTargetFrame->GetInversePose(Rwc, twc);
		cv::hconcat(Rwc, twc, Twc);


		{
			// Get Map Mutex
			std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

			for (std::vector<Frame*>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
			{
				Frame* pKFi = *vit;

				cv::Mat Riw, tiw;
				cv::Mat Tiw;
				pKFi->GetPose(Riw, tiw);
				cv::hconcat(Riw, tiw, Tiw);

				if (pKFi != mpTargetFrame)
				{
					cv::Mat Tic = Tiw*Twc;
					cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
					cv::Mat tic = Tic.rowRange(0, 3).col(3);
					g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
					g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
					//Pose corrected with the Sim3 of the loop closure
					CorrectedSim3[pKFi] = g2oCorrectedSiw;
				}

				g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
				//Pose without correction
				NonCorrectedSim3[pKFi] = g2oSiw;
			}

			// Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
			for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; mit++)
			{
				Frame* pKFi = mit->first;
				g2o::Sim3 g2oCorrectedSiw = mit->second;
				g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

				g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

				std::vector<MapPoint*> vpMPsi = pKFi->GetMapPoints();
				for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
				{
					MapPoint* pMPi = vpMPsi[iMP];
					if (!pMPi || pMPi->isDeleted() || pMPi->mnCorrectedByKF == mpTargetFrame->mnKeyFrameID)
						continue;

					// Project with non-corrected pose and project back with corrected pose
					cv::Mat P3Dw = pMPi->GetWorldPos();
					Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
					Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

					cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
					pMPi->SetWorldPos(cvCorrectedP3Dw);
					pMPi->mnCorrectedByKF = mpTargetFrame->mnKeyFrameID;
					pMPi->mnCorrectedReference = pKFi->mnKeyFrameID;
					//pMPi->UpdateNormalAndDepth();
				}

				// Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
				Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
				Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
				double s = g2oCorrectedSiw.scale();

				eigt *= (1. / s); //[R t/s;0 1]

				cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

				pKFi->SetPose(correctedTiw.colRange(0,3).rowRange(0,3), correctedTiw.col(3).rowRange(0,3));

				// Make sure connections are updated
				//pKFi->UpdateConnections();
			}

			// Start Loop Fusion
			// Update matched map points and replace if duplicated
			for (size_t i = 0; i<mvpCurrentMatchedPoints.size(); i++)
			{
				if (mvpCurrentMatchedPoints[i])
				{
					MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
					MapPoint* pCurMP = mpTargetFrame->GetMapPoint(i);
					if (pCurMP)
					{
						//pCurMP->Replace(pLoopMP);
					}
					else
					{
						mpTargetFrame->AddMapPoint(pLoopMP, i);
						pLoopMP->AddObservation(mpTargetFrame, i);
						//pLoopMP->ComputeDistinctiveDescriptors();
					}
				}
			}

		}

		// Project MapPoints observed in the neighborhood of the loop keyframe
		// into the current keyframe and neighbors using corrected poses.
		// Fuse duplications.
		SearchAndFuse(CorrectedSim3);


		// After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
		std::map<Frame*, std::set<Frame*> > LoopConnections;

		for (std::vector<Frame*>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
		{
			Frame* pKFi = *vit;
			std::vector<Frame*> vpPreviousNeighbors = pKFi->GetConnectedKFs();

			// Update connections. Detect new links.
			//pKFi->UpdateConnections();
			LoopConnections[pKFi] = pKFi->GetConnectedKeyFrameSet();
			for (std::vector<Frame*>::iterator vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end(); vit_prev != vend_prev; vit_prev++)
			{
				LoopConnections[pKFi].erase(*vit_prev);
			}
			for (std::vector<Frame*>::iterator vit2 = mvpCurrentConnectedKFs.begin(), vend2 = mvpCurrentConnectedKFs.end(); vit2 != vend2; vit2++)
			{
				LoopConnections[pKFi].erase(*vit2);
			}
		}

		// Optimize graph
		Optimization::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpTargetFrame, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

		//mpMap->InformNewBigChange();

		//// Add loop edge
		//mpMatchedKF->AddLoopEdge(mpCurrentKF);
		//mpCurrentKF->AddLoopEdge(mpMatchedKF);

		//// Launch a new thread to perform Global Bundle Adjustment
		//mbRunningGBA = true;
		//mbFinishedGBA = false;
		//mbStopGBA = false;
		//mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);

		//// Loop closed. Release Local Mapping.
		//mpLocalMapper->Release();

		mLastLoopKFid = mpTargetFrame->mnKeyFrameID;

	}
	void LoopCloser::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap) {

	}
}

/////////////////////////////��� ������ �̿��� alignment
////cv::Mat Rcw = UVR_SLAM::PlaneInformation::CalcPlaneRotationMatrix(param).clone();
////cv::Mat normal;
////float dist;
////pFloor->GetParam(normal, dist);
////cv::Mat tempP = Rcw.t()*normal;
////if (tempP.at<float>(0) < 0.00001)
////	tempP.at<float>(0) = 0.0;
////if (tempP.at<float>(2) < 0.00001)
////	tempP.at<float>(2) = 0.0;

//////��ü ������Ʈ ��ȯ
////for (int i = 0; i < tempMPs.size(); i++) {
////	UVR_SLAM::MapPoint* pMP = tempMPs[i];
////	/*if (!pMP)
////		continue;
////	if (pMP->isDeleted())
////		continue;*/
////	cv::Mat tempX = Rcw.t()*pMP->GetWorldPos();
////	pMP->SetWorldPos(tempX);
////}
//////��� �Ķ���� ��ȯ
////pFloor->SetParam(tempP, dist);

////
/////////////////////////////��� ������ �̿��� alignment