
#include <Optimization.h>
#include <Converter.h>
#include <map>
#include <FrameWindow.h>
#include <opencv2/core/eigen.hpp>
//#include <Edge.h>
//#include <Optimizer.h>
//#include <PlaneBastedOptimization.h>
#include <PoseGraphOptimization.h>
#include <MatrixOperator.h>
#include <PlaneBA.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <Map.h>
#include <MapOptimizer.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/types/types_seven_dof_expmap.h"

//int UVR_SLAM::Optimization::PoseOptimization(UVR_SLAM::Frame* pF, std::vector<std::pair<int, bool>>& mvMatches, bool bStatus, int trial1, int trial2){
//	cv::Mat mK;
//	pF->mK.convertTo(mK, CV_64FC1);
//	double fx = mK.at<double>(0, 0);
//	double fy = mK.at<double>(1, 1);
//
//	int nCurrFrameID = pF->GetFrameID();
//	int nPoseJacobianSize = 6;
//	int nResidualSize = 2;
//
//	GraphOptimizer::Optimizer* mpOptimizer = new GraphOptimizer::GNOptimizer();
//
//	cv::Mat Rinit, Tinit;
//	Rinit = pF->GetRotation();
//	Tinit = pF->GetTranslation();
//
//	FrameVertex* mpVertex1 = new FrameVertex(Rinit, Tinit, nPoseJacobianSize);
//	mpOptimizer->AddVertex(mpVertex1);
//
//	//Add Edge
//	const double deltaMono = sqrt(5.991);
//	const double chiMono = 5.991;
//
//	std::vector<PoseOptimizationEdge*> mvpEdges;
//	std::vector<int> mvIndexes;
//
//	for (int i = 0; i < mvMatches.size(); i++) {
//
//		std::pair<int, bool> matchInfos = mvMatches[i];
//		int idx = matchInfos.first;
//		bool b = matchInfos.second;
//
//		UVR_SLAM::MapPoint* pMP = pF->mvpMPs[idx];
//		if (!pMP)
//			continue;
//		if (pMP->isDeleted())
//			continue;
//		if (!b)
//			continue;
//		mvIndexes.push_back(i);
//		UVR_SLAM::PoseOptimizationEdge* pEdge1 = new UVR_SLAM::PoseOptimizationEdge(pMP->GetWorldPos(), nResidualSize);
//		Eigen::Vector2d temp = Eigen::Vector2d();
//		cv::Point2f pt1 = pF->mvKeyPoints[idx].pt;
//		temp(0) = pt1.x;
//		temp(1) = pt1.y;
//		pEdge1->SetMeasurement(temp);
//		cv::cv2eigen(mK, pEdge1->K);
//		pEdge1->AddVertex(mpVertex1);
//		double info = (double)pF->mvInvLevelSigma2[pF->mvKeyPoints[idx].octave];
//		//std::cout << "information::" << info2 << std::endl;
//		pEdge1->SetInformation(cv::Mat::eye(nResidualSize, nResidualSize, CV_64FC1)*info);
//		pEdge1->fx = fx;
//		pEdge1->fy = fy;
//		GraphOptimizer::RobustKernel* huber = new GraphOptimizer::HuberKernel();
//		huber->SetDelta(deltaMono);
//		pEdge1->mpRobustKernel = huber;
//		mpOptimizer->AddEdge(pEdge1);
//		mvpEdges.push_back(pEdge1);
//	}
//
//	int nInlier = 0;
//	for (int trial = 0; trial < trial1; trial++) {
//		mpOptimizer->Optimize(trial2, 0, bStatus);
//		nInlier = 0;
//		for (int edgeIdx = 0; edgeIdx < mvpEdges.size(); edgeIdx++) {
//			std::pair<int, bool> matchInfo = mvMatches[mvIndexes[edgeIdx]];
//			int idx = matchInfo.first;
//			bool b = matchInfo.second;
//			
//			if (!pF->mvbMPInliers[idx]) {
//				mvpEdges[edgeIdx]->CalcError();
//			}
//			if (mvpEdges[edgeIdx]->GetError() > chiMono || !mvpEdges[edgeIdx]->GetDepth()) {
//				mvpEdges[edgeIdx]->SetLevel(1);
//				pF->mvbMPInliers[idx] = false;
//				pF->mvpMPs[idx]->SetRecentTrackingFrameID(-1);
//				mvMatches[mvIndexes[edgeIdx]].second = false;
//			}
//			else
//			{
//				mvpEdges[edgeIdx]->SetLevel(0);
//				pF->mvbMPInliers[idx] = true;
//				pF->mvpMPs[idx]->SetRecentTrackingFrameID(nCurrFrameID);
//				mvMatches[mvIndexes[edgeIdx]].second = true;
//				nInlier++;
//			}
//		}
//	}
//
//	//mp inlier 
//	//std::cout << "PoseOptimization::inlier=" << nInlier << std::endl;
//	mpVertex1->RestoreData();
//	pF->SetPose(mpVertex1->Rmat, mpVertex1->Tmat);
//	if (bStatus)
//		std::cout << "PoseOptimization::End" << std::endl;
//	return nInlier;
//}
//
//int UVR_SLAM::Optimization::PoseOptimization(UVR_SLAM::FrameWindow* pWindow, UVR_SLAM::Frame* pF, std::vector<MapPoint*> mvpLocalMPs, std::vector<bool>& mvbLocalMapInliers, bool bStatus, int trial1, int trial2) {
//	if(bStatus)
//		std::cout << "PoseOptimization::Start" << std::endl;
//	cv::Mat mK;
//	pF->mK.convertTo(mK, CV_64FC1);
//	double fx = mK.at<double>(0, 0);
//	double fy = mK.at<double>(1, 1);
//
//	int nPoseJacobianSize = 6;
//	int nResidualSize = 2;
//
//	GraphOptimizer::Optimizer* mpOptimizer = new GraphOptimizer::LMOptimizer();
//	FrameVertex* mpVertex1 = new FrameVertex(pWindow->GetRotation(), pWindow->GetTranslation(), nPoseJacobianSize);
//	mpOptimizer->AddVertex(mpVertex1);
//	
//	//Add Edge
//	const double deltaMono = sqrt(5.991);
//	const double chiMono = 5.991;
//
//	std::vector<PoseOptimizationEdge*> mvpEdges;
//	std::vector<int> mvIndexes;
//
//	auto mvpMPs = pF->GetMapPoints();
//
//	//for (int i = 0; i < pWindow->mvMatchInfos.size(); i++) {
//	//	
//	//	cv::DMatch match = pWindow->mvMatchInfos[i];
//	//	if (!mvbLocalMapInliers[i])
//	//		continue;
//	//	
//	//	int idx1 = match.queryIdx; //framewindow
//	//	int idx2 = match.trainIdx; //frame
//
//	//	UVR_SLAM::MapPoint* pMP = mvpMPs[idx2];
//	//	if (!pMP)
//	//		continue;
//	//	if (pMP->isDeleted())
//	//		continue;
//	//	mvIndexes.push_back(i);
//
//	//	UVR_SLAM::PoseOptimizationEdge* pEdge1 = new UVR_SLAM::PoseOptimizationEdge(pMP->GetWorldPos(),nResidualSize);
//	//	Eigen::Vector2d temp = Eigen::Vector2d();
//	//	cv::Point2f pt1 = pF->mvKeyPoints[idx2].pt;
//	//	temp(0) = pt1.x;
//	//	temp(1) = pt1.y;
//	//	pEdge1->SetMeasurement(temp);
//	//	cv::cv2eigen(mK, pEdge1->K);
//	//	pEdge1->AddVertex(mpVertex1);
//	//	double info = (double)pF->mvInvLevelSigma2[pF->mvKeyPoints[idx2].octave];
//	//	//std::cout << "information::" << info2 << std::endl;
//	//	pEdge1->SetInformation(cv::Mat::eye(nResidualSize, nResidualSize, CV_64FC1)*info);
//	//	pEdge1->fx = fx;
//	//	pEdge1->fy = fy;
//	//	GraphOptimizer::RobustKernel* huber = new GraphOptimizer::HuberKernel();
//	//	huber->SetDelta(deltaMono);
//	//	pEdge1->mpRobustKernel = huber;
//	//	mpOptimizer->AddEdge(pEdge1);
//	//	mvpEdges.push_back(pEdge1);
//	//}
//
//	//int nInlier = 0;
//	//for (int trial = 0; trial < trial1; trial++) {
//	//	mpOptimizer->Optimize(trial2, 0, bStatus);
//	//	nInlier = 0;
//	//	for (int edgeIdx = 0; edgeIdx < mvpEdges.size(); edgeIdx++) {
//	//		cv::DMatch match = pWindow->mvMatchInfos[mvIndexes[edgeIdx]];
//	//		
//	//		int idx1 = match.queryIdx; //framewindow
//	//		int idx2 = match.trainIdx; //frame
//	//		if (!pF->mvbMPInliers[idx2]) {
//	//			mvpEdges[edgeIdx]->CalcError();
//	//		}
//	//		if (mvpEdges[edgeIdx]->GetError() > chiMono || !mvpEdges[edgeIdx]->GetDepth()) {
//	//			mvpEdges[edgeIdx]->SetLevel(1);
//	//			pF->mvbMPInliers[idx2] = false;
//	//			mvbLocalMapInliers[idx1] = false;
//	//		}
//	//		else
//	//		{
//	//			mvpEdges[edgeIdx]->SetLevel(0);
//	//			pF->mvbMPInliers[idx2] = true;
//	//			mvbLocalMapInliers[idx1] = true;
//	//			nInlier++;
//	//		}
//	//	}
//	//}
//
//	//mp inlier 
//	//std::cout << "PoseOptimization::inlier=" << nInlier << std::endl;
//	mpVertex1->RestoreData();
//	pF->SetPose(mpVertex1->Rmat, mpVertex1->Tmat);
//	pWindow->SetPose(mpVertex1->Rmat, mpVertex1->Tmat);
//	if(bStatus)
//		std::cout << "PoseOptimization::End" << std::endl;
//	return 0;
//}
//int UVR_SLAM::Optimization::InitOptimization(UVR_SLAM::InitialData* data, std::vector<cv::DMatch> Matches, UVR_SLAM::Frame* pInitFrame1, UVR_SLAM::Frame* pInitFrame2, cv::Mat K, bool& bInit, int trial1, int trial2) {
//	cv::Mat mK;
//	K.convertTo(mK, CV_64FC1);
//	double fx = mK.at<double>(0, 0);
//	double fy = mK.at<double>(1, 1);
//
//	int nPoseJacobianSize = 6;
//	int nMapJacobianSize = 3;
//	int nResidualSize = 2;
//	GraphOptimizer::Optimizer* mpOptimizer = new GraphOptimizer::LMOptimizer();
//	mpOptimizer->bSchur = true;
//	FrameVertex* mpVertex1 = new FrameVertex(data->R0, data->t0, nPoseJacobianSize);
//	mpVertex1->SetFixed(true);
//	FrameVertex* mpVertex2 = new FrameVertex(data->R, data->t, nPoseJacobianSize);
//	mpOptimizer->AddVertex(mpVertex1, true);
//	mpOptimizer->AddVertex(mpVertex2, true);
//	
//	const double deltaMono = sqrt(5.991);
//	const double chiMono = 5.991;
//
//	//Add MapPoints
//	std::vector<MapPointVertex*> mvpMPVertices(0);
//	for (int i = 0; i < data->mvX3Ds.size(); i++) {
//		if (!data->vbTriangulated[i])
//			continue;
//		MapPointVertex* mpVertex = new MapPointVertex(data->mvX3Ds[i], i, nMapJacobianSize);
//		//std::cout << mpVertex->GetParam().rows() << std::endl;
//		mpOptimizer->AddVertex(mpVertex, false);
//		mvpMPVertices.push_back(mpVertex);
//	}
//
//	//Add Edge
//	std::vector<EdgePoseNMap*> mvpEdges;
//	for (int i = 0; i < mvpMPVertices.size(); i++) {
//		int idx = mvpMPVertices[i]->GetIndex();
//		
//		UVR_SLAM::EdgePoseNMap* pEdge1 = new UVR_SLAM::EdgePoseNMap(nResidualSize);
//		Eigen::Vector2d temp = Eigen::Vector2d();
//		cv::Point2f pt1 = pInitFrame1->mvKeyPoints[Matches[idx].queryIdx].pt;
//		temp(0) = pt1.x;
//		temp(1) = pt1.y;
//		pEdge1->SetMeasurement(temp);
//		cv::cv2eigen(mK, pEdge1->K);
//		
//		pEdge1->AddVertex(mpVertex1);
//		pEdge1->AddVertex(mvpMPVertices[i]);
//		//pEdge1->SetSubHessian();
//		double info1 = (double)pInitFrame1->mvInvLevelSigma2[pInitFrame1->mvKeyPoints[Matches[idx].queryIdx].octave];
//		pEdge1->SetInformation(cv::Mat::eye(nResidualSize, nResidualSize, CV_64FC1)*info1);
//		
//		pEdge1->fx = fx;
//		pEdge1->fy = fy;
//		
//		GraphOptimizer::RobustKernel* huber = new GraphOptimizer::HuberKernel();
//		huber->SetDelta(deltaMono);
//		pEdge1->mpRobustKernel = huber;
//	
//		mpOptimizer->AddEdge(pEdge1);
//		mvpEdges.push_back(pEdge1);
//
//		UVR_SLAM::EdgePoseNMap* pEdge2 = new UVR_SLAM::EdgePoseNMap(nResidualSize);
//		Eigen::Vector2d temp2 = Eigen::Vector2d();
//		cv::Point2f pt2 = pInitFrame2->mvKeyPoints[Matches[idx].trainIdx].pt;
//		temp2(0) = pt2.x;
//		temp2(1) = pt2.y;
//		pEdge2->SetMeasurement(temp2);
//		cv::cv2eigen(mK, pEdge2->K);
//
//		pEdge2->AddVertex(mpVertex2);
//		pEdge2->AddVertex(mvpMPVertices[i]);
//		//pEdge2->SetSubHessian();
//		double info2 = (double)pInitFrame2->mvInvLevelSigma2[pInitFrame2->mvKeyPoints[Matches[idx].trainIdx].octave];
//		pEdge2->SetInformation(cv::Mat::eye(nResidualSize, nResidualSize, CV_64FC1)*info2);
//
//		pEdge2->fx = fx;
//		pEdge2->fy = fy;
//
//		GraphOptimizer::RobustKernel* huber2 = new GraphOptimizer::HuberKernel();
//		huber2->SetDelta(deltaMono);
//		pEdge2->mpRobustKernel = huber2;
//
//		mpOptimizer->AddEdge(pEdge2);
//		mvpEdges.push_back(pEdge2);
//	}
//
//	for (int trial = 0; trial < trial1; trial++) {
//		mpOptimizer->Optimize(trial2, 0, true);
//
//		for (int i = 0; i < mvpMPVertices.size(); i++) {
//			int eidx1 = i * 2;
//			int eidx2 = eidx1 + 1;
//
//			int idx = mvpMPVertices[i]->GetIndex();
//			if (!data->vbTriangulated[idx]){
//				mvpEdges[eidx1]->CalcError();
//				mvpEdges[eidx2]->CalcError();
//			}
//			double err1 = mvpEdges[eidx1]->GetError();
//			double err2 = mvpEdges[eidx2]->GetError();
//			if (err1 > chiMono || err2 > chiMono || !mvpEdges[eidx1]->GetDepth() || !mvpEdges[eidx2]->GetDepth()) {
//				data->vbTriangulated[idx] = false;
//				mvpEdges[eidx1]->SetLevel(1);
//				mvpEdges[eidx2]->SetLevel(1);
//			}
//			else {
//				data->vbTriangulated[idx] = true;
//				mvpEdges[eidx1]->SetLevel(0);
//				mvpEdges[eidx2]->SetLevel(0);
//			}
//		}
//	}
//	
//	//update
//	mpVertex1->RestoreData();
//	mpVertex2->RestoreData();
//	int nRes = 0;
//	data->SetRt(mpVertex2->Rmat, mpVertex2->Tmat);
//	for (int i = 0; i < mvpMPVertices.size(); i++) {
//		int idx = mvpMPVertices[i]->GetIndex();
//		if (data->vbTriangulated[idx]) {
//			nRes++;
//			mvpMPVertices[i]->RestoreData();
//			data->mvX3Ds[idx] = mvpMPVertices[i]->Xw.clone();
//		}
//	}
//	if (nRes > 79) {
//		bInit = true;
//	}
//	else {
//		bInit = false;
//	}
//
//	std::cout << mpVertex1->Rmat << mpVertex1->Tmat << std::endl;
//	std::cout << mpVertex2->Rmat << mpVertex2->Tmat << std::endl;
//
//	return nRes;
//}
//
//void UVR_SLAM::Optimization::LocalBundleAdjustment(UVR_SLAM::FrameWindow* pWindow, int nTargetID,bool& bStopBA, int trial1, int trial2, bool bShowStatus) {
//	//fixed frame
//	//connected kf = 3
//	//check newmp
//	//KF
//	if(bShowStatus)
//		std::cout << "LocalBA::Start"<< std::endl;
//	
//	int nPoseJacobianSize = 6;
//	int nMapJacobianSize = 3;
//	int nResidualSize = 2;
//
//	GraphOptimizer::Optimizer* mpOptimizer = new GraphOptimizer::LMOptimizer();
//	mpOptimizer->bSchur = true;
//
//	//Add vertices in window
//	//check initial keyframe
//	std::vector<FrameVertex*> mvpFrameVertices;
//	std::vector<MapPointVertex*> mvpMapPointVertices;
//	std::map<UVR_SLAM::Frame*, int> mmpFrames; int nFrame = 0; int nFixedFrames = 0;
//	//std::vector<FrameVertex*> mvpFixedVertices;
//	std::set<UVR_SLAM::Frame*> mspFixedFrames;
//
//	std::vector<UVR_SLAM::Frame*> mvpFrames;
//	std::vector<UVR_SLAM::MapPoint*> mvpMPs;
//	std::vector<int> mvLocalMPIndex;
//
//	auto mvpKFs = pWindow->GetLocalMapFrames();
//	for (auto iter = mvpKFs.begin(); iter != mvpKFs.end(); iter++) {
//		Frame* pF = *iter;
//		if (pF->mnLocalBAID == nTargetID)
//			continue;
//		pF->mnLocalBAID = nTargetID;
//		FrameVertex* mpVertex = new FrameVertex(pF->GetRotation(), pF->GetTranslation(), nPoseJacobianSize);
//		mpOptimizer->AddVertex(mpVertex);
//		mvpFrameVertices.push_back(mpVertex);
//		mmpFrames.insert(std::make_pair(pF, nFrame++));
//		mvpFrames.push_back(pF);
//	}
//	//K
//	cv::Mat mK = mvpKFs[0]->mK.clone();
//	mK.convertTo(mK, CV_64FC1);
//	double fx = mK.at<double>(0, 0);
//	double fy = mK.at<double>(1, 1);
//	
//	auto mvpLocalMPs = pWindow->GetLocalMap();
//	//Add map point
//	for (int i = 0; i < mvpLocalMPs.size(); i++) {
//
//		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
//		if (!pMP)
//			continue;
//		if (pMP->isDeleted()){
//			continue;
//		}
//		/*MapPointVertex* mpVertex = new MapPointVertex(pMP->GetWorldPos(), i, nMapJacobianSize);
//		mpOptimizer->AddVertex(mpVertex, false);
//		mvpMapPointVertices.push_back(mpVertex);*/
//
//		//fixed keyframe
//		auto mmpConnectedFrames = pMP->GetConnedtedFrames();
//		for (auto iter = mmpConnectedFrames.begin(); iter != mmpConnectedFrames.end(); iter++) {
//			UVR_SLAM::Frame* pF = iter->first;
//
//			if (pF->mnLocalBAID == nTargetID)
//				continue;
//			pF->mnLocalBAID = nTargetID;
//			mmpFrames.insert(std::make_pair(pF, nFrame++));
//			mspFixedFrames.insert(pF);
//			nFixedFrames++;
//
//			//auto findres = mmpFrames.find(pF);
//			//if (findres == mmpFrames.end()) {
//			//	mmpFrames.insert(std::make_pair(pF, nFrame++));
//			//	mspFixedFrames.insert(pF);
//			//	nFixedFrames++;
//				
//				
//				/*auto findres2 = mspFixedVertices.find(pF);
//				if (findres2 == mspFixedVertices.end()) {
//				}
//				FrameVertex* mpFVertex = new FrameVertex(pF->GetRotation(), pF->GetTranslation(), nPoseJacobianSize);
//				mpFVertex->SetFixed(true);
//				mpOptimizer->AddVertex(mpFVertex);*/
//				
//			//}
//		}
//	}
//	
//	for (auto iter = mspFixedFrames.begin(); iter != mspFixedFrames.end(); iter++) {
//		UVR_SLAM::Frame* pF = *iter;
//		FrameVertex* mpFVertex = new FrameVertex(pF->GetRotation(), pF->GetTranslation(), nPoseJacobianSize);
//		mpFVertex->SetFixed(true);
//		mvpFrameVertices.push_back(mpFVertex);
//		mpOptimizer->AddVertex(mpFVertex);
//	}
//
//	for (int i = 0; i < mvpLocalMPs.size(); i++) {
//
//		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
//		if (!pMP)
//			continue;
//		if (pMP->isDeleted())
//			continue;
//		MapPointVertex* mpVertex = new MapPointVertex(pMP->GetWorldPos(), i, nMapJacobianSize);
//		mpOptimizer->AddVertex(mpVertex, false);
//		mvpMapPointVertices.push_back(mpVertex);
//		mvpMPs.push_back(pMP);
//		mvLocalMPIndex.push_back(i);
//	}
//	if (bShowStatus)
//		std::cout << "LocalBA::FixedFrame::" << nFixedFrames << std::endl;
//	
//	//Add Edge
//	const double deltaMono = sqrt(5.991);
//	const double chiMono = 5.991;
//
//	std::vector<EdgePoseNMap*> mvpEdges;
//	typedef std::tuple<UVR_SLAM::MapPoint*, UVR_SLAM::Frame*, int> LocalBAEdgeData;
//	std::vector<LocalBAEdgeData> mvTupleData;
//	//std::vector<std::pair<UVR_SLAM::MapPoint*, int>> mvPairEdges;
//	std::vector<bool> mvbEdges;
//	
//	//Add Edge
//	for (int i = 0; i < mvpMapPointVertices.size(); i++) {
//		int idx = mvpMapPointVertices[i]->GetIndex();
//		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[idx];
//		auto mmpConnectedFrames = pMP->GetConnedtedFrames();
//		for (auto iter = mmpConnectedFrames.begin(); iter != mmpConnectedFrames.end(); iter++) {
//			UVR_SLAM::Frame* pF = iter->first;
//			int feature_idx = iter->second;
//			int frame_idx = mmpFrames[pF];
//
//			UVR_SLAM::EdgePoseNMap* pEdge = new UVR_SLAM::EdgePoseNMap(nResidualSize);
//			Eigen::Vector2d temp = Eigen::Vector2d();
//			cv::Point2f pt = pF->mvKeyPoints[feature_idx].pt;
//			temp(0) = pt.x;
//			temp(1) = pt.y;
//			pEdge->SetMeasurement(temp);
//			cv::cv2eigen(mK, pEdge->K);
//
//			pEdge->AddVertex(mvpFrameVertices[frame_idx]);
//			pEdge->AddVertex(mvpMapPointVertices[i]);
//
//			double info1 = (double)pF->mvInvLevelSigma2[pF->mvKeyPoints[feature_idx].octave];
//			pEdge->SetInformation(cv::Mat::eye(nResidualSize, nResidualSize, CV_64FC1)*info1);
//
//			pEdge->fx = fx;
//			pEdge->fy = fy;
//
//			GraphOptimizer::RobustKernel* huber = new GraphOptimizer::HuberKernel();
//			huber->SetDelta(deltaMono);
//			pEdge->mpRobustKernel = huber;
//
//			mpOptimizer->AddEdge(pEdge);
//			mvpEdges.push_back(pEdge);
//			mvbEdges.push_back(true);
//			LocalBAEdgeData tempData = std::make_tuple(pMP, pF, feature_idx);
//			mvTupleData.push_back(tempData);
//
//		}
//
//	}
//	//Optimize
//	for (int trial = 0; trial < trial1; trial++) {
//		if (bStopBA)
//			break;
//		mpOptimizer->Optimize(trial2, 0, bShowStatus);
//		for (int i = 0; i < mvpEdges.size(); i++) {
//			EdgePoseNMap* pEdge = mvpEdges[i];
//			if (!mvbEdges[i]) {
//				mvpEdges[i]->CalcError();
//			}
//			double err = mvpEdges[i]->GetError();
//			if (err > chiMono || !mvpEdges[i]->GetDepth()) {
//				mvbEdges[i] = false;
//				mvpEdges[i]->SetLevel(1);
//			}
//			else {
//				mvbEdges[i] = true;
//				mvpEdges[i]->SetLevel(0);
//			}
//		}
//	}
//	if (bShowStatus)
//		std::cout << "Update Parameter::Start" << std::endl;
//	//update parameter
//	//std::vector<
//	for (int i = 0; i < mvpEdges.size(); i++) {
//		
//		if (!mvbEdges[i]) {
//			LocalBAEdgeData tempData = mvTupleData[i];
//			UVR_SLAM::MapPoint* pMP = std::get<0>(tempData);
//			UVR_SLAM::Frame* pF = std::get<1>(tempData);
//			int feature_idx = std::get<2>(tempData);
//			pMP->RemoveFrame(pF);
//		}
//	}
//	for (int i = 0; i < mvpFrames.size(); i++) {
//		mvpFrameVertices[i]->RestoreData();
//		mvpFrames[i]->SetPose(mvpFrameVertices[i]->Rmat, mvpFrameVertices[i]->Tmat);
//	}
//	for (int i = 0; i <  mvpMPs.size(); i++) {
//		int nConnectedThresh = 3;
//		if (mvpMPs[i]->GetMapPointType() == UVR_SLAM::MapPointType::PLANE_MP)
//			nConnectedThresh = 1;
//		else if (mvpMPs[i]->isNewMP())
//			nConnectedThresh = 2;
//		if (mvpMPs[i]->GetNumConnectedFrames() < nConnectedThresh) {
//
//			int idx = mvLocalMPIndex[i];
//			UVR_SLAM::MapPoint* pMP = mvpLocalMPs[idx];
//			pMP->SetDelete(true);
//			pMP->Delete();
//			//pWindow->SetMapPoint(nullptr, idx);
//			//pWindow->SetBoolInlier(false, idx);
//			continue;
//		}
//		
//		mvpMapPointVertices[i]->RestoreData();
//		mvpMPs[i]->SetWorldPos(mvpMapPointVertices[i]->Xw);
//		//std::cout <<"Connected MPs = "<< mvpMPs[i]->GetNumConnectedFrames() << std::endl;
//	}
//	if (bShowStatus)
//		std::cout << "Update Parameter::End" << std::endl;
//}

//double UVR_SLAM::Optimization::CalibrationGridPlaneTest(UVR_SLAM::Pose* pPose, std::vector<cv::Point2f> pts, std::vector<cv::Point3f> wPts, cv::Mat K, int mnWidth, int mnHeight) {
//
//	int jacobianSize = 6;
//	int nMapJacobianSize = 3;
//	int residualSize1 = 2;
//
//	GraphOptimizer::Optimizer* mpOptimizer = new GraphOptimizer::GNOptimizer();
//	FrameVertex* mpVertex = new FrameVertex(pPose, jacobianSize);
//
//	cv::Mat mK;
//	K.convertTo(mK, CV_64FC1);
//	double fx = mK.at<double>(0, 0);
//	double fy = mK.at<double>(1, 1);
//
//	mpOptimizer->AddVertex(mpVertex);
//
//	for (int i = 0; i < pts.size(); i++) {
//		Eigen::Vector3d tempMap = Eigen::Vector3d();
//		tempMap(0) = wPts[i].x;
//		tempMap(1) = wPts[i].y;
//		tempMap(2) = wPts[i].z;
//		
//		Eigen::Vector2d temp = Eigen::Vector2d();
//		temp(0) = pts[i].x;
//		temp(1) = pts[i].y;
//		
//		UVR_SLAM::PoseOptimizationEdge* pEdge = new UVR_SLAM::PoseOptimizationEdge();
//		pEdge->AddVertex(mpVertex);
//		pEdge->SetMeasurement(temp);
//		pEdge->SetInformation(cv::Mat::eye(2, 2, CV_64FC1));
//		//pEdge->K = mK;
//		pEdge->K = Eigen::Matrix3d();
//		cv::cv2eigen(mK, pEdge->K);
//
//		pEdge->fx = fx;
//		pEdge->fy = fy;
//		pEdge->Xw = tempMap;
//
//		//UVR::RobustKernel* huber = new UVR::HuberKernel();
//		//huber->SetDelta(2.0);
//		//pEdge->mpRobustKernel = huber;
//
//		mpOptimizer->AddEdge(pEdge);
//	}
//
//	mpOptimizer->Optimize(10, 0);
//	//waitKey(0);
//	return 0.0;
//}

//double UVR_SLAM::Optimization::PlaneMapDepthEstimationTest(UVR_SLAM::Pose* pPose,
//	std::vector<cv::Point3f> wPts1, std::vector<cv::Point2f> iPts1, std::vector<float>& depths1,
//	std::vector<cv::Point3f> wPts2, std::vector<cv::Point2f> iPts2, std::vector<float>& depths2,
//	cv::Mat K, cv::Mat P1, cv::Mat P2, int mnWidth, int mnHeight) {
//
//	int nPoseJacobianSize = 6;
//	int nDepthJacobianSize = 1;
//	int residualSize1 = 2;
//	int residualSize2 = 1;
//
//	cv::Mat mK;
//	K.convertTo(mK, CV_64FC1);
//	double fx = mK.at<double>(0, 0);
//	double fy = mK.at<double>(1, 1);
//
//	//Add Vertex
//	GraphOptimizer::Optimizer* mpOptimizer = new GraphOptimizer::LMOptimizer();
//	FrameVertex* mpVertex = new FrameVertex(pPose, nPoseJacobianSize);
//
//	//Add Frame Vertex
//	mpOptimizer->AddVertex(mpVertex, true);
//
//	//Add Plane1
//	std::vector<PlaneMapPointVertex*> mvpPlaneVertices1(0);
//	for (int i = 0; i < iPts1.size(); i++) {
//		PlaneMapPointVertex* mpVertex = new PlaneMapPointVertex((double)depths1[i], nDepthJacobianSize);
//		mpOptimizer->AddVertex(mpVertex, false);
//		mvpPlaneVertices1.push_back(mpVertex);
//	}
//	//Add Plane2
//	std::vector<PlaneMapPointVertex*> mvpPlaneVertices2(0);
//	for (int i = 0; i < iPts2.size(); i++) {
//		PlaneMapPointVertex* mpVertex = new PlaneMapPointVertex((double)depths2[i], nDepthJacobianSize);
//		mpOptimizer->AddVertex(mpVertex, false);
//		mvpPlaneVertices2.push_back(mpVertex);
//	}
//
//	std::vector<PlaneEdgeOnlyPoseNMap*> mvpEdges;
//	for (int i = 0; i < iPts1.size(); i++) {
//
//		UVR_SLAM::PlaneEdgeOnlyPoseNMap* pEdge = new UVR_SLAM::PlaneEdgeOnlyPoseNMap(residualSize2);
//
//		cv::cv2eigen(mK, pEdge->K);
//		pEdge->Kinv = pEdge->K.inverse();
//		pEdge->Ximg << iPts1[i].x, iPts1[i].y, 1.0;
//		pEdge->Pw << P1.at<double>(0), P1.at<double>(1), P1.at<double>(2), P1.at<double>(3);
//		
//		pEdge->AddVertex(mpVertex);
//		pEdge->AddVertex(mvpPlaneVertices1[i]);
//
//		pEdge->SetInformation(cv::Mat::eye(1, 1, CV_64FC1));
//		pEdge->SetSubHessian();
//		
//		//UVR::RobustKernel* huber = new UVR::HuberKernel();
//		//huber->SetDelta(2.0);
//		//pEdge->mpRobustKernel = huber;
//
//		mpOptimizer->AddEdge(pEdge);
//		mvpEdges.push_back(pEdge);
//	}
//	for (int i = 0; i < iPts2.size(); i++) {
//
//		UVR_SLAM::PlaneEdgeOnlyPoseNMap* pEdge = new UVR_SLAM::PlaneEdgeOnlyPoseNMap(residualSize2);
//
//		cv::cv2eigen(mK, pEdge->K);
//		pEdge->Kinv = pEdge->K.inverse();
//		pEdge->Ximg << iPts2[i].x, iPts2[i].y, 1.0;
//		pEdge->Pw << P2.at<double>(0), P2.at<double>(1), P2.at<double>(2), P2.at<double>(3);
//
//		pEdge->AddVertex(mpVertex);
//		pEdge->AddVertex(mvpPlaneVertices2[i]);
//		pEdge->SetInformation(cv::Mat::eye(1, 1, CV_64FC1));
//		pEdge->SetSubHessian();
//
//		//UVR::RobustKernel* huber = new UVR::HuberKernel();
//		//huber->SetDelta(2.0);
//		//pEdge->mpRobustKernel = huber;
//
//		mpOptimizer->AddEdge(pEdge);
//		mvpEdges.push_back(pEdge);
//	}
//	
//	mpOptimizer->Optimize(10, 0);
//
//	//update param
//	for (int i = 0; i < mvpPlaneVertices1.size(); i++) {
//		//std::cout << "before depth = " <<i<<"::"<< depths1[i] << std::endl;
//		depths1[i] = (float)mvpPlaneVertices1[i]->GetDepth();
//		//std::cout << "after depth = " << depths1[i] << std::endl;
//	}
//	for (int i = 0; i < mvpPlaneVertices2.size(); i++) {
//		//std::cout << "before depth = " << i + mvpPlaneVertices1 .size()<< "::" << depths2[i] << std::endl;
//		depths2[i] = (float)mvpPlaneVertices2[i]->GetDepth();
//		//std::cout << "after depth = " << depths2[i] << std::endl;
//	}
//
//	return 0.0;
//}


//g2o 버전
void UVR_SLAM::Optimization::LocalBundleAdjustment(UVR_SLAM::MapOptimizer* pMapOptimizer, UVR_SLAM::Frame* pKF, UVR_SLAM::FrameWindow* pWindow) {
	// Local KeyFrames: First Breath Search from Current Keyframe
	std::list<UVR_SLAM::Frame*> lLocalKeyFrames;

	std::cout << "ba::ssssss" << std::endl;

	int nTargetID = pKF->GetFrameID();
	lLocalKeyFrames.push_back(pKF);
	pKF->mnLocalBAID = nTargetID;

	int nn = 15;

	const std::vector<UVR_SLAM::Frame*> vNeighKFs = pKF->GetConnectedKFs(nn);
	for (int i = 0, iend = vNeighKFs.size(); i<iend; i++)
	{
		UVR_SLAM::Frame* pKFi = vNeighKFs[i];
		if (pKFi->mnLocalBAID == nTargetID)
			continue;
		pKFi->mnLocalBAID = nTargetID;
		lLocalKeyFrames.push_back(pKFi);
	}
	std::cout << "ba::aaaa" << std::endl;
	// Local MapPoints seen in Local KeyFrames
	std::list<MapPoint*> lLocalMapPoints;
	for (std::list<UVR_SLAM::Frame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	{
		std::vector<MapPoint*> vpMPs = (*lit)->GetMapPoints();
		for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;
			if (!pMP)
				continue;
			if (pMP->isDeleted()) {
				continue;
			}
			if (pMP->GetNumConnectedFrames() <= 2)
				continue;
			if (pMP->mnLocalBAID == nTargetID)
				continue;
			lLocalMapPoints.push_back(pMP);
			pMP->mnLocalBAID = nTargetID;
		}
	}
	std::cout << "ba::bbbbb" << std::endl;
	////dense map points 추가하기
	int nDenseIdx = lLocalMapPoints.size();
	auto mvpDenseMPs = pKF->GetDenseVectors();
	for (int i = 0; i < mvpDenseMPs.size(); i++)
	{
		lLocalMapPoints.push_back(mvpDenseMPs[i]);
	}
	////dense map points 추가하기
	std::cout << "ba::ccccc" << std::endl;
	// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
	std::list<UVR_SLAM::Frame*> lFixedCameras;
	for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	{
		UVR_SLAM::MapPoint* pMP = *lit;

		if (pMP->GetMapPointType() == UVR_SLAM::PLANE_DENSE_MP) {
			auto observations = pMP->GetConnedtedDenseFrames();
			//map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
			for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				UVR_SLAM::Frame* pKFi = mit->first;

				if (pKFi->mnLocalBAID != nTargetID && pKFi->mnFixedBAID != nTargetID)
				{
					pKFi->mnFixedBAID = nTargetID;
					lFixedCameras.push_back(pKFi);
				}
			}
		}
		else {
			auto observations = pMP->GetConnedtedFrames();
			//map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
			for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				UVR_SLAM::Frame* pKFi = mit->first;

				if (pKFi->mnLocalBAID != nTargetID && pKFi->mnFixedBAID != nTargetID)
				{
					pKFi->mnFixedBAID = nTargetID;
					lFixedCameras.push_back(pKFi);
				}
			}
		}

		
	}
	std::cout << "ba::ddddd" << std::endl;
	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	std::cout << "ba::eeee" << std::endl;

	bool bStopBA = pMapOptimizer->isStopBA();
	if (bStopBA)
		optimizer.setForceStopFlag(&bStopBA);

	unsigned long maxKFid = 0;
	std::cout << "ba::ffff" << std::endl;
	// Set Local KeyFrame vertices
	for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	{
		UVR_SLAM::Frame* pKFi = *lit;
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

		cv::Mat R, t;
		pKFi->GetPose(R, t);
		cv::Mat Tcw = cv::Mat::zeros(4,4, CV_32FC1);
		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
		t.copyTo(Tcw.col(3).rowRange(0, 3));

		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
		vSE3->setId(pKFi->GetKeyFrameID());
		vSE3->setFixed(pKFi->GetKeyFrameID() == 0);
		optimizer.addVertex(vSE3);
		if (pKFi->GetKeyFrameID()>maxKFid)
			maxKFid = pKFi->GetKeyFrameID();
	}
	std::cout << "ba::ggg::"<< maxKFid<< std::endl;
	// Set Fixed KeyFrame vertices
	for (auto lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
	{
		UVR_SLAM::Frame* pKFi = *lit;
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

		cv::Mat R, t;
		pKFi->GetPose(R, t);
		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
		t.copyTo(Tcw.col(3).rowRange(0, 3));

		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
		vSE3->setId(pKFi->GetKeyFrameID());
		vSE3->setFixed(true);
		optimizer.addVertex(vSE3);
		if (pKFi->GetKeyFrameID()>maxKFid)
			maxKFid = pKFi->GetKeyFrameID();
	}
	std::cout << "ba::hhh::" << maxKFid << std::endl;
	// Set MapPoint vertices
	const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size())*lLocalMapPoints.size();

	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	std::vector<MapPoint*> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);
	std::cout << "ba::iii" << std::endl;
	for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	{
		MapPoint* pMP = *lit;
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		int id = pMP->mnMapPointID + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		if (pMP->GetMapPointType() != UVR_SLAM::PLANE_DENSE_MP) {
			const auto observations = pMP->GetConnedtedFrames();
			
			//Set edges
			for (std::map<UVR_SLAM::Frame*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				UVR_SLAM::Frame* pKFi = mit->first;
				if (pKFi->GetKeyFrameID() > maxKFid)
					continue;

				const cv::KeyPoint &kpUn = pKFi->mvKeyPoints[mit->second];

				Eigen::Matrix<double, 2, 1> obs;
				obs << kpUn.pt.x, kpUn.pt.y;

				g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->GetKeyFrameID())));
				e->setMeasurement(obs);
				const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
				e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(thHuberMono);

				e->fx = pKFi->fx;
				e->fy = pKFi->fy;
				e->cx = pKFi->cx;
				e->cy = pKFi->cy;

				optimizer.addEdge(e);
				vpEdgesMono.push_back(e);
				vpEdgeKFMono.push_back(pKFi);
				vpMapPointEdgeMono.push_back(pMP);
			}
		}else {
			
			const auto observations = pMP->GetConnedtedDenseFrames();
			
			//Set edges
			for (std::map<UVR_SLAM::Frame*, cv::Point2f>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				UVR_SLAM::Frame* pKFi = mit->first;
				if (pKFi->GetKeyFrameID() > maxKFid)
					continue;

				auto pt = mit->second;
				Eigen::Matrix<double, 2, 1> obs;
				obs << pt.x, pt.y;

				g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->GetKeyFrameID())));
				e->setMeasurement(obs);
				const float &invSigma2 = 1.0;
				e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(thHuberMono);

				e->fx = pKFi->fx;
				e->fy = pKFi->fy;
				e->cx = pKFi->cx;
				e->cy = pKFi->cy;

				optimizer.addEdge(e);
				vpEdgesMono.push_back(e);
				vpEdgeKFMono.push_back(pKFi);
				vpMapPointEdgeMono.push_back(pMP);
			}
		}//if mappoint type
		
	}

	std::cout << "ba::setting::end" << std::endl;

	bStopBA = pMapOptimizer->isStopBA();
	if (bStopBA)
		return;
	/*if (pbStopFlag)
		if (*pbStopFlag)
			return;*/

	std::cout << "ba::setting::end2" << std::endl;

	optimizer.initializeOptimization();
	optimizer.optimize(5);

	std::cout << "ba::optimize::end" << std::endl;

	bStopBA = pMapOptimizer->isStopBA();

	std::cout << "ba::optimize::end2" << std::endl;

	bool bDoMore = true;
	if (bStopBA)
		bDoMore = false;
	/*if (pbStopFlag)
		if (*pbStopFlag)
			bDoMore = false;*/

	if (bDoMore)
	{

		// Check inlier observations
		for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
		{
			g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
			MapPoint* pMP = vpMapPointEdgeMono[i];

			if (pMP->isDeleted())
				continue;

			if (e->chi2()>5.991 || !e->isDepthPositive())
			{
				e->setLevel(1);
			}

			e->setRobustKernel(0);
		}

		// Optimize again without the outliers

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);

	}

	std::cout << "ba::optimize::end3" << std::endl;

	std::vector<std::pair<UVR_SLAM::Frame*, MapPoint*> > vToErase;
	vToErase.reserve(vpEdgesMono.size());

	// Check inlier observations       
	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	{
		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		MapPoint* pMP = vpMapPointEdgeMono[i];

		if (pMP->isDeleted())
			continue;

		if (e->chi2()>5.991 || !e->isDepthPositive())
		{
			UVR_SLAM::Frame* pKFi = vpEdgeKFMono[i];
			vToErase.push_back(std::make_pair(pKFi, pMP));
		}
	}

	std::cout << "ba::check::inlier::end" << std::endl;

	if (!vToErase.empty())
	{
		for (size_t i = 0; i<vToErase.size(); i++)
		{
			UVR_SLAM::Frame* pKFi = vToErase[i].first;
			MapPoint* pMPi = vToErase[i].second;
			
			if (pMPi->GetMapPointType() == UVR_SLAM::PLANE_DENSE_MP) {
				pMPi->RemoveDenseFrame(pKFi);
			}
			else {
				pMPi->RemoveFrame(pKFi);
			}
			
			//pKFi->EraseMapPointMatch(pMPi);
			//pMPi->EraseObservation(pKFi);

		}
	}

	std::cout << "ba::erase::end" << std::endl;

	// Recover optimized data

	//Keyframes
	for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	{
		UVR_SLAM::Frame* pKF = *lit;
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetKeyFrameID()));
		g2o::SE3Quat SE3quat = vSE3->estimate();

		cv::Mat R, t;
		cv::Mat Tcw = Converter::toCvMat(SE3quat);
		R = Tcw.rowRange(0, 3).colRange(0, 3);
		t = Tcw.rowRange(0, 3).col(3);
		pKF->SetPose(R, t);
	}

	std::cout << "ba::restore::kf::end" << std::endl;

	//Points
	for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	{
		MapPoint* pMP = *lit;
		if (!pMP || pMP->isDeleted())
			continue;

		//remove
		int nConnectedThresh = 2;
		//////////////평면일 경우 쓰레시값 조절
		//if (pMP->GetMapPointType() == UVR_SLAM::MapPointType::PLANE_MP){
		//	//바로 만들어진 포인트는 삭제 금지.
		//	if (pMP->mnFirstKeyFrameID == nTargetID)
		//		continue;
		//	nConnectedThresh = 1;
		//}
		//////////////평면일 경우 쓰레시값 조절
		//else if (pMP->isNewMP())
		//	nConnectedThresh = 1;
		int nConncted = 0;
		if (pMP->GetMapPointType() == UVR_SLAM::PLANE_DENSE_MP) {
			nConncted = pMP->GetNumDensedFrames();
		}
		else {
			nConncted = pMP->GetNumConnectedFrames();
		}
		if (nConncted <= nConnectedThresh) {
			pMP->SetDelete(true);
			pMP->Delete();
			//pWindow->SetMapPoint(nullptr, idx);
			//pWindow->SetBoolInlier(false, idx);
			continue;
		}

		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();
	}
	std::cout << "ba::restore::mp::end" << std::endl;
}

void UVR_SLAM::Optimization::LocalBundleAdjustmentWithPlane(UVR_SLAM::Map* pMap, UVR_SLAM::Frame *pKF, UVR_SLAM::FrameWindow* pWindow, bool* pbStopFlag)
{
	// Local KeyFrames: First Breath Search from Current Keyframe
	std::list<UVR_SLAM::Frame*> lLocalKeyFrames;

	int nTargetID = pKF->GetFrameID();
	lLocalKeyFrames.push_back(pKF);
	pKF->mnLocalBAID = nTargetID;

	int nn = 15;

	const std::vector<UVR_SLAM::Frame*> vNeighKFs = pKF->GetConnectedKFs(nn);
	for (int i = 0, iend = vNeighKFs.size(); i<iend; i++)
	{
		UVR_SLAM::Frame* pKFi = vNeighKFs[i];
		if (pKFi->mnLocalBAID == nTargetID)
			continue;
		pKFi->mnLocalBAID = nTargetID;
		lLocalKeyFrames.push_back(pKFi);
	}

	// Local MapPoints seen in Local KeyFrames
	std::list<MapPoint*> lLocalMapPoints;
	for (std::list<UVR_SLAM::Frame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	{
		std::vector<MapPoint*> vpMPs = (*lit)->GetMapPoints();
		for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;
			if (!pMP)
				continue;
			if (pMP->isDeleted()) {
				continue;
			}
			if (pMP->GetNumConnectedFrames() == 1)
				continue;
			if (pMP->mnLocalBAID == nTargetID)
				continue;
			lLocalMapPoints.push_back(pMP);
			pMP->mnLocalBAID = nTargetID;
		}
	}

	// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
	std::list<UVR_SLAM::Frame*> lFixedCameras;
	for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	{
		UVR_SLAM::MapPoint* pMP = *lit;

		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;

		auto observations = pMP->GetConnedtedFrames();
		//map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
		for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			UVR_SLAM::Frame* pKFi = mit->first;

			if (pKFi->mnLocalBAID != nTargetID && pKFi->mnFixedBAID != nTargetID)
			{
				pKFi->mnFixedBAID = nTargetID;
				lFixedCameras.push_back(pKFi);
			}
		}
	}

	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	if (pbStopFlag)
		optimizer.setForceStopFlag(pbStopFlag);

	unsigned long maxKFid = 0;

	// Set Local KeyFrame vertices
	for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	{
		UVR_SLAM::Frame* pKFi = *lit;
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

		cv::Mat R, t;
		pKFi->GetPose(R, t);
		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
		t.copyTo(Tcw.col(3).rowRange(0, 3));

		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
		vSE3->setId(pKFi->GetKeyFrameID());
		vSE3->setFixed(pKFi->GetKeyFrameID() == 0);
		optimizer.addVertex(vSE3);
		if (pKFi->GetKeyFrameID()>maxKFid)
			maxKFid = pKFi->GetKeyFrameID();
	}

	// Set Fixed KeyFrame vertices
	for (auto lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
	{
		UVR_SLAM::Frame* pKFi = *lit;
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

		cv::Mat R, t;
		pKFi->GetPose(R, t);
		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
		t.copyTo(Tcw.col(3).rowRange(0, 3));

		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
		vSE3->setId(pKFi->GetKeyFrameID());
		vSE3->setFixed(true);
		optimizer.addVertex(vSE3);
		if (pKFi->GetKeyFrameID()>maxKFid)
			maxKFid = pKFi->GetKeyFrameID();
	}

	// Set Local KeyFrame vertices
	/*for (std::list<SemanticSLAM::PlaneInformation*>::iterator lit = lLocalPlanes.begin(), lend = lLocalPlanes.end(); lit != lend; lit++) {
		SemanticSLAM::PlaneInformation* pPlane = *lit;
		g2o::PlaneVertex* vPlane = new g2o::PlaneVertex();
		vPlane->setEstimate(Converter::toVector6d(pPlane->GetPlaneParam()));
		vPlane->setId(maxKFid + pPlane->mnPlaneID);
		vPlane->setFixed(false);
		optimizer.addVertex(vPlane);
		if (pPlane->mnPlaneID > maxPlaneid)
			maxPlaneid = pPlane->mnPlaneID;
	}*/

	// Set MapPoint vertices
	const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size())*lLocalMapPoints.size();

	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	std::vector<MapPoint*> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	std::vector<MapPoint*> vpPlaneEdgeMP;
	std::vector<g2o::PlaneBAEdgeOnlyMapPoint*> vpPlaneEdges;
	vpPlaneEdges.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);

	int nPlaneID = pMap->mpFloorPlane->mnPlaneID;
	cv::Mat normal;
	float dist;
	pMap->mpFloorPlane->GetParam(normal, dist);

	for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	{
		MapPoint* pMP = *lit;

		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		int id = pMP->mnMapPointID + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);
		const auto observations = pMP->GetConnedtedFrames();
		//Set edges
		
		for (std::map<UVR_SLAM::Frame*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			UVR_SLAM::Frame* pKFi = mit->first;
			/*if (pKFi->GetFrameID() > nTargetID)
				continue;*/
			if (pKFi->GetKeyFrameID() > maxKFid)
				continue;
			const cv::KeyPoint &kpUn = pKFi->mvKeyPoints[mit->second];

			Eigen::Matrix<double, 2, 1> obs;
			obs << kpUn.pt.x, kpUn.pt.y;

			g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->GetKeyFrameID())));
			e->setMeasurement(obs);
			const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(thHuberMono);

			e->fx = pKFi->fx;
			e->fy = pKFi->fy;
			e->cx = pKFi->cx;
			e->cy = pKFi->cy;

			optimizer.addEdge(e);
			vpEdgesMono.push_back(e);
			vpEdgeKFMono.push_back(pKFi);
			vpMapPointEdgeMono.push_back(pMP);
		}
		//set plane edge
		if (pMP->GetPlaneID() == nPlaneID) {
			g2o::PlaneBAEdgeOnlyMapPoint* e = new g2o::PlaneBAEdgeOnlyMapPoint();
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));

			e->normal[0] = normal.at<float>(0);
			e->normal[1] = normal.at<float>(1);
			e->normal[2] = normal.at<float>(2);
			e->dist = dist;

			/*cv::Mat Xw = pMP->GetWorldPos();
			e->Xw[0] = Xw.at<float>(0);
			e->Xw[1] = Xw.at<float>(1);
			e->Xw[2] = Xw.at<float>(2);*/

			optimizer.addEdge(e);
			vpPlaneEdgeMP.push_back(pMP);
			vpPlaneEdges.push_back(e);

		}
	}
	
	if (pbStopFlag)
		if (*pbStopFlag)
			return;

	optimizer.initializeOptimization();
	optimizer.optimize(5);

	bool bDoMore = true;

	if (pbStopFlag)
		if (*pbStopFlag)
			bDoMore = false;

	if (bDoMore)
	{

		// Check inlier observations
		for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
		{
			g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
			MapPoint* pMP = vpMapPointEdgeMono[i];
			if (!pMP)
				continue;
			if (pMP->isDeleted())
				continue;

			if (e->chi2()>5.991 || !e->isDepthPositive())
			{
				e->setLevel(1);
			}

			e->setRobustKernel(0);
		}

		// Optimize again without the outliers

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);

	}

	std::vector<std::pair<UVR_SLAM::Frame*, MapPoint*> > vToErase;
	vToErase.reserve(vpEdgesMono.size());

	// Check inlier observations       
	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	{
		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		MapPoint* pMP = vpMapPointEdgeMono[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;

		if (e->chi2()>5.991 || !e->isDepthPositive())
		{
			UVR_SLAM::Frame* pKFi = vpEdgeKFMono[i];
			vToErase.push_back(std::make_pair(pKFi, pMP));
		}
	}

	for (size_t i = 0, iend = vpPlaneEdges.size(); i < iend; i++) {
		g2o::PlaneBAEdgeOnlyMapPoint* e = vpPlaneEdges[i];
		MapPoint* pMP = vpPlaneEdgeMP[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		//if(pMP->GetConnedtedFrames().size() > 1)
		//std::cout << "err:" << pMP->GetConnedtedFrames().size() << std::endl;
		if (e->chi2() > 0.001)
			e->setLevel(1);
		e->setRobustKernel(0);
	}

	if (!vToErase.empty())
	{
		for (size_t i = 0; i<vToErase.size(); i++)
		{
			UVR_SLAM::Frame* pKFi = vToErase[i].first;
			MapPoint* pMPi = vToErase[i].second;
			if (!pMPi)
				continue;
			if (pMPi->isDeleted())
				continue;
			pMPi->RemoveFrame(pKFi);

			//pKFi->EraseMapPointMatch(pMPi);
			//pMPi->EraseObservation(pKFi);

		}
	}

	// Recover optimized data

	//Keyframes
	for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	{
		UVR_SLAM::Frame* pKF = *lit;
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetKeyFrameID()));
		g2o::SE3Quat SE3quat = vSE3->estimate();

		cv::Mat R, t;
		cv::Mat Tcw = Converter::toCvMat(SE3quat);
		R = Tcw.rowRange(0, 3).colRange(0, 3);
		t = Tcw.rowRange(0, 3).col(3);
		pKF->SetPose(R, t);
	}

	//Points
	for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	{
		MapPoint* pMP = *lit;
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		//remove
		int nConnectedThresh = 2;
		/////////평면인경우 쓰레시 홀딩 조절
		//if (pMP->GetMapPointType() == UVR_SLAM::MapPointType::PLANE_MP)
		//	nConnectedThresh = 1;
		/////////평면인경우 쓰레시 홀딩 조절
		//else if (pMP->isNewMP())
		//	nConnectedThresh = 1;
		if (pMP->GetNumConnectedFrames() < nConnectedThresh) {
			pMP->SetDelete(true);
			pMP->Delete();
			//pWindow->SetMapPoint(nullptr, idx);
			//pWindow->SetBoolInlier(false, idx);
			continue;
		}
		
		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + 1));
		
		//cv::Mat tempori = pMP->GetWorldPos();
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();
		
		/*if(pMP->mnFirstKeyFrameID == pKF->GetKeyFrameID() && pMP->GetPlaneID() == nPlaneID){
			cv::Mat temp = pMP->GetWorldPos();
			tempori = normal.t()*tempori;
			temp = normal.t()*temp;
			float val1 = tempori.at<float>(0) + dist;
			float val2 = temp.at<float>(0)+dist;
			std::cout << val1 << ", " << val2 << std::endl;
		}*/
	}
	////////////////////////////////////////////////////////////////////
}


void UVR_SLAM::Optimization::InitBundleAdjustment(const std::vector<UVR_SLAM::Frame*> &vpKFs, const std::vector<UVR_SLAM::MapPoint *> &vpMP, int nIterations)
{
	std::vector<bool> vbNotIncludedMP;
	vbNotIncludedMP.resize(vpMP.size());

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	long unsigned int maxKFid = 0;

	// Set KeyFrame vertices
	for (size_t i = 0; i<vpKFs.size(); i++)
	{
		UVR_SLAM::Frame* pKF = vpKFs[i];
		
		cv::Mat R, t;
		pKF->GetPose(R, t);
		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
		t.copyTo(Tcw.col(3).rowRange(0, 3));
		
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
		vSE3->setId(pKF->GetKeyFrameID());
		vSE3->setFixed(pKF->GetKeyFrameID() == 0);
		optimizer.addVertex(vSE3);
		if (pKF->GetKeyFrameID()>maxKFid)
			maxKFid = pKF->GetKeyFrameID();
	}

	const float thHuber2D = sqrt(5.99);
	const float thHuber3D = sqrt(7.815);

	// Set MapPoint vertices
	for (size_t i = 0; i<vpMP.size(); i++)
	{
		MapPoint* pMP = vpMP[i];
		if (pMP->isDeleted())
			continue;
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		const int id = pMP->mnMapPointID + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		//const std::map<UVR_SLAM::Frame*, int> observations = pMP->GetConnedtedFrames();
		const auto observations = pMP->GetConnedtedDenseFrames();

		int nEdges = 0;
		//SET EDGES
		for (auto mit = observations.begin(); mit != observations.end(); mit++)
		{

			UVR_SLAM::Frame* pKF = mit->first;
			if (pKF->GetKeyFrameID()>maxKFid)
				continue;

			nEdges++;

			/*const cv::KeyPoint &kpUn = pKF->mvKeyPoints[mit->second];

			Eigen::Matrix<double, 2, 1> obs;
			obs << kpUn.pt.x, kpUn.pt.y;

			g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->GetKeyFrameID())));
			e->setMeasurement(obs);
			const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);*/

			const auto pt = mit->second;
			Eigen::Matrix<double, 2, 1> obs;
			obs << pt.x, pt.y;

			g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->GetKeyFrameID())));
			e->setMeasurement(obs);
			const float &invSigma2 = 1.0;
			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(thHuber2D);

			e->fx = pKF->fx;
			e->fy = pKF->fy;
			e->cx = pKF->cx;
			e->cy = pKF->cy;
			optimizer.addEdge(e);
		}

		if (nEdges == 0)
		{
			optimizer.removeVertex(vPoint);
			vbNotIncludedMP[i] = true;
		}
		else
		{
			vbNotIncludedMP[i] = false;
		}
	}

	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(nIterations);

	// Recover optimized data

	//Keyframes
	for (size_t i = 0; i<vpKFs.size(); i++)
	{
		UVR_SLAM::Frame* pKF = vpKFs[i];
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetKeyFrameID()));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		
		cv::Mat R, t;
		cv::Mat Tcw = Converter::toCvMat(SE3quat);
		R = Tcw.rowRange(0, 3).colRange(0, 3);
		t = Tcw.rowRange(0, 3).col(3);
		std::cout<<"ID::Before::"<< pKF->GetKeyFrameID() << pKF->GetRotation() << ", " << pKF->GetTranslation().t() << std::endl;
		pKF->SetPose(R, t);
		//if (i == 0)
		{
			std::cout <<"ID::After::"<<pKF->GetKeyFrameID()<< R << ", " << t.t() << std::endl;
		}
	}

	//Points
	for (size_t i = 0; i<vpMP.size(); i++)
	{
		if (vbNotIncludedMP[i])
			continue;

		MapPoint* pMP = vpMP[i];

		if (pMP->isDeleted())
			continue;
		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
	}

}

int UVR_SLAM::Optimization::PoseOptimization(Frame *pFrame)
{
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	int nInitialCorrespondences = 0;
	int nTargetID = pFrame->GetFrameID();

	// Set Frame vertex
	cv::Mat R, t;
	pFrame->GetPose(R, t);
	cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
	R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
	t.copyTo(Tcw.rowRange(0, 3).col(3));
	g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
	vSE3->setEstimate(Converter::toSE3Quat(Tcw));
	vSE3->setId(0);
	vSE3->setFixed(false);
	optimizer.addVertex(vSE3);

	// Set MapPoint vertices
	const int N = pFrame->mvKeyPoints.size();

	std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
	std::vector<size_t> vnIndexEdgeMono;
	vpEdgesMono.reserve(N);
	vnIndexEdgeMono.reserve(N);

	const float deltaMono = sqrt(5.991);
	const float deltaStereo = sqrt(7.815);

	{

		for (int i = 0; i<N; i++)
		{
			MapPoint* pMP = pFrame->mvpMPs[i];
			if (!pMP)
				continue;
			if (pMP->isDeleted())
				continue;
			if (pMP->GetRecentTrackingFrameID() != nTargetID)
				continue;
			nInitialCorrespondences++;
			//pFrame->mvbMPInliers[i] = true;

			Eigen::Matrix<double, 2, 1> obs;
			const cv::KeyPoint &kpUn = pFrame->mvKeyPoints[i];
			obs << kpUn.pt.x, kpUn.pt.y;

			g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
			e->setMeasurement(obs);
			const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(deltaMono);

			e->fx = pFrame->fx;
			e->fy = pFrame->fy;
			e->cx = pFrame->cx;
			e->cy = pFrame->cy;
			cv::Mat Xw = pMP->GetWorldPos();
			e->Xw[0] = Xw.at<float>(0);
			e->Xw[1] = Xw.at<float>(1);
			e->Xw[2] = Xw.at<float>(2);

			optimizer.addEdge(e);

			vpEdgesMono.push_back(e);
			vnIndexEdgeMono.push_back(i);

		}
	}
	
	if (nInitialCorrespondences<3)
		return 0;

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	const float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };
	const float chi2Stereo[4] = { 7.815,7.815,7.815, 7.815 };
	const int its[4] = { 10,10,10,10 };

	int nBad = 0;
	for (size_t it = 0; it<4; it++)
	{

		vSE3->setEstimate(Converter::toSE3Quat(Tcw)); //이건가??
		optimizer.initializeOptimization(0);
		optimizer.optimize(its[it]);

		nBad = 0;
		for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
		{
			g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];

			if (!pFrame->mvbMPInliers[idx])
			{
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2>chi2Mono[it] || !e->isDepthPositive())
			{
				pFrame->mvbMPInliers[idx] = false;
				pFrame->mvpMPs[idx]->SetRecentTrackingFrameID(-1);
				e->setLevel(1);
				nBad++;
			}
			else
			{
				pFrame->mvbMPInliers[idx] = true;
				pFrame->mvpMPs[idx]->SetRecentTrackingFrameID(nTargetID);
				e->setLevel(0);
			}

			if (it == 2)
				e->setRobustKernel(0);
		}
		
		if (optimizer.edges().size()<10)
			break;
	}

	// Recover optimized pose and return number of inliers
	g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
	g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
	cv::Mat pose = Converter::toCvMat(SE3quat_recov);
	R = pose.rowRange(0, 3).colRange(0, 3);
	t = pose.rowRange(0, 3).col(3);
	pFrame->SetPose(R, t);

	return nInitialCorrespondences - nBad;
}

int UVR_SLAM::Optimization::PoseOptimization(Frame *pFrame, std::vector<UVR_SLAM::MapPoint*> vDenseMPs, std::vector<std::pair<int, cv::Point2f>> vPairs, std::vector<bool>& vbInliers)
{
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	int nInitialCorrespondences = 0;
	int nTargetID = pFrame->GetFrameID();

	// Set Frame vertex
	cv::Mat R, t;
	pFrame->GetPose(R, t);
	cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
	R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
	t.copyTo(Tcw.rowRange(0, 3).col(3));
	g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
	vSE3->setEstimate(Converter::toSE3Quat(Tcw));
	vSE3->setId(0);
	vSE3->setFixed(false);
	optimizer.addVertex(vSE3);

	// Set MapPoint vertices
	const int N = pFrame->mvKeyPoints.size();

	std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
	std::vector<size_t> vnIndexEdgeMono;
	vpEdgesMono.reserve(N);
	vnIndexEdgeMono.reserve(N);

	const float deltaMono = sqrt(5.991);
	const float deltaStereo = sqrt(7.815);

	int nDenseIDX = 0;

	{

		for (int i = 0; i<N; i++)
		{
			MapPoint* pMP = pFrame->mvpMPs[i];
			if (!pMP)
				continue;
			if (pMP->isDeleted())
				continue;
			if (pMP->GetRecentTrackingFrameID() != nTargetID)
				continue;
			nInitialCorrespondences++;
			//pFrame->mvbMPInliers[i] = true;

			Eigen::Matrix<double, 2, 1> obs;
			const cv::KeyPoint &kpUn = pFrame->mvKeyPoints[i];
			obs << kpUn.pt.x, kpUn.pt.y;

			g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
			e->setMeasurement(obs);
			const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(deltaMono);

			e->fx = pFrame->fx;
			e->fy = pFrame->fy;
			e->cx = pFrame->cx;
			e->cy = pFrame->cy;
			cv::Mat Xw = pMP->GetWorldPos();
			e->Xw[0] = Xw.at<float>(0);
			e->Xw[1] = Xw.at<float>(1);
			e->Xw[2] = Xw.at<float>(2);

			optimizer.addEdge(e);

			vpEdgesMono.push_back(e);
			vnIndexEdgeMono.push_back(i);

		}
		nDenseIDX = vpEdgesMono.size();

		for (int i = 0; i < vPairs.size(); i++) {
			int idx = vPairs[i].first;
			cv::Point2f pt = vPairs[i].second;

			MapPoint* pMP = vDenseMPs[idx];
			if (!pMP)
				continue;
			if (pMP->isDeleted())
				continue;
			if (pMP->GetRecentTrackingFrameID() != nTargetID)
				continue;
			Eigen::Matrix<double, 2, 1> obs;
			obs << pt.x, pt.y;

			g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
			e->setMeasurement(obs);
			//const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
			const float invSigma2 = 1.0;
			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(deltaMono);

			e->fx = pFrame->fx;
			e->fy = pFrame->fy;
			e->cx = pFrame->cx;
			e->cy = pFrame->cy;
			cv::Mat Xw = pMP->GetWorldPos();
			e->Xw[0] = Xw.at<float>(0);
			e->Xw[1] = Xw.at<float>(1);
			e->Xw[2] = Xw.at<float>(2);

			optimizer.addEdge(e);

			vpEdgesMono.push_back(e);
			vnIndexEdgeMono.push_back(i);
		}
		
	}

	if (nInitialCorrespondences<3)
		return 0;

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	const float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };
	const float chi2Stereo[4] = { 7.815,7.815,7.815, 7.815 };
	const int its[4] = { 10,10,10,10 };

	int nBad = 0;
	for (size_t it = 0; it<4; it++)
	{

		vSE3->setEstimate(Converter::toSE3Quat(Tcw)); //이건가??
		optimizer.initializeOptimization(0);
		optimizer.optimize(its[it]);

		nBad = 0;
		for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
		{
			g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];

			if (i < nDenseIDX) {
				if (!pFrame->mvbMPInliers[idx])
				{
					e->computeError();
				}

				const float chi2 = e->chi2();

				if (chi2>chi2Mono[it] || !e->isDepthPositive())
				{
					pFrame->mvbMPInliers[idx] = false;
					pFrame->mvpMPs[idx]->SetRecentTrackingFrameID(-1);
					e->setLevel(1);
					nBad++;
				}
				else
				{
					pFrame->mvbMPInliers[idx] = true;
					pFrame->mvpMPs[idx]->SetRecentTrackingFrameID(nTargetID);
					e->setLevel(0);
				}
			}
			else {

				if (!vbInliers[idx])
				{
					e->computeError();
				}
				const float chi2 = e->chi2();

				if (chi2>chi2Mono[it] || !e->isDepthPositive())
				{
					vbInliers[idx] = false;
					vDenseMPs[idx]->SetRecentTrackingFrameID(-1);
					e->setLevel(1);
					nBad++;
				}
				else
				{
					vbInliers[idx] = true;
					vDenseMPs[idx]->SetRecentTrackingFrameID(nTargetID);
					e->setLevel(0);
				}
			}
			

			if (it == 2)
				e->setRobustKernel(0);
		}

		if (optimizer.edges().size()<10)
			break;
	}

	// Recover optimized pose and return number of inliers
	g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
	g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
	cv::Mat pose = Converter::toCvMat(SE3quat_recov);
	R = pose.rowRange(0, 3).colRange(0, 3);
	t = pose.rowRange(0, 3).col(3);
	pFrame->SetPose(R, t);

	return nInitialCorrespondences - nBad;
}

int UVR_SLAM::Optimization::PoseOptimization(Frame *pFrame, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool>& vbInliers, std::vector<int> vnIDXs)
{
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	int nInitialCorrespondences = 0;
	int nTargetID = pFrame->GetFrameID();

	// Set Frame vertex
	cv::Mat R, t;
	pFrame->GetPose(R, t);
	cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
	R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
	t.copyTo(Tcw.rowRange(0, 3).col(3));
	g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
	vSE3->setEstimate(Converter::toSE3Quat(Tcw));
	vSE3->setId(0);
	vSE3->setFixed(false);
	optimizer.addVertex(vSE3);
	
	// Set MapPoint vertices
	const int N = vpMPs.size();

	std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
	std::vector<size_t> vnIndexEdgeMono;
	vpEdgesMono.reserve(N);
	vnIndexEdgeMono.reserve(N);

	std::cout << "opt::" << N << ", " << vnIDXs.size() << std::endl;

	const float deltaMono = sqrt(5.991);
	const float deltaStereo = sqrt(7.815);

	{
		for (int i = 0; i<N; i++)
		{
			MapPoint* pMP = vpMPs[i];
			if (!pMP)
				continue;
			if (pMP->isDeleted())
				continue;
			if (pMP->GetRecentTrackingFrameID() != nTargetID)
				continue;
			nInitialCorrespondences++;
			//pFrame->mvbMPInliers[i] = true;

			Eigen::Matrix<double, 2, 1> obs;
			const cv::Point2f pt = vpPts[vnIDXs[i]];
			obs << pt.x, pt.y;
			//std::cout << pt << std::endl;
			g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
			e->setMeasurement(obs);
			//const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
			e->setInformation(Eigen::Matrix2d::Identity());// *invSigma2);

			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(deltaMono);

			e->fx = pFrame->fx;
			e->fy = pFrame->fy;
			e->cx = pFrame->cx;
			e->cy = pFrame->cy;
			cv::Mat Xw = pMP->GetWorldPos();
			e->Xw[0] = Xw.at<float>(0);
			e->Xw[1] = Xw.at<float>(1);
			e->Xw[2] = Xw.at<float>(2);

			optimizer.addEdge(e);

			vpEdgesMono.push_back(e);
			vnIndexEdgeMono.push_back(i);
		}
	}
	
	if (nInitialCorrespondences<3)
		return 0;

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	const float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };
	const float chi2Stereo[4] = { 7.815,7.815,7.815, 7.815 };
	const int its[4] = { 10,10,10,10 };

	int nBad = 0;
	for (size_t it = 0; it<4; it++)
	{

		vSE3->setEstimate(Converter::toSE3Quat(Tcw)); //이건가??
		optimizer.initializeOptimization(0);
		optimizer.optimize(its[it]);

		nBad = 0;
		for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
		{
			g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];
			int idx2 = vnIDXs[idx];
			if (!vbInliers[idx2])
			{
				e->computeError();
			}

			const float chi2 = e->chi2();

			if (chi2>chi2Mono[it] || !e->isDepthPositive())
			{
				vbInliers[idx2] = false;
				vpMPs[idx]->SetRecentTrackingFrameID(-1);
				//pFrame->mvbMPInliers[idx] = false;
				//pFrame->mvpMPs[idx]->SetRecentTrackingFrameID(-1);
				e->setLevel(1);
				nBad++;
			}
			else
			{
				vbInliers[idx2] = true;
				vpMPs[idx]->SetRecentTrackingFrameID(nTargetID);
				//pFrame->mvbMPInliers[idx] = true;
				//pFrame->mvpMPs[idx]->SetRecentTrackingFrameID(nTargetID);
				e->setLevel(0);
			}

			if (it == 2)
				e->setRobustKernel(0);
		}

		if (optimizer.edges().size()<10)
			break;
	}
	
	// Recover optimized pose and return number of inliers
	g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
	g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
	cv::Mat pose = Converter::toCvMat(SE3quat_recov);
	R = pose.rowRange(0, 3).colRange(0, 3);
	t = pose.rowRange(0, 3).col(3);
	pFrame->SetPose(R, t);
	
	return nInitialCorrespondences - nBad;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////Opticalflow 버전용
void UVR_SLAM::Optimization::OpticalLocalBundleAdjustment(UVR_SLAM::MapOptimizer* pMapOptimizer, UVR_SLAM::Frame* pKF, UVR_SLAM::FrameWindow* pWindow) {
	// Local KeyFrames: First Breath Search from Current Keyframe
	std::list<UVR_SLAM::Frame*> lLocalKeyFrames;

	std::cout << "ba::ssssss" << std::endl;

	int nTargetID = pKF->GetFrameID();
	lLocalKeyFrames.push_back(pKF);
	pKF->mnLocalBAID = nTargetID;

	int nn = 15;

	const std::vector<UVR_SLAM::Frame*> vNeighKFs = pKF->GetConnectedKFs(nn);
	std::cout << "ba::connected kf::" << vNeighKFs.size() << std::endl;
	for (int i = 0, iend = vNeighKFs.size(); i<iend; i++)
	{
		UVR_SLAM::Frame* pKFi = vNeighKFs[i];
		if (pKFi->mnLocalBAID == nTargetID)
			continue;
		pKFi->mnLocalBAID = nTargetID;
		lLocalKeyFrames.push_back(pKFi);
	}
	std::cout << "ba::aaaa" << std::endl;
	// Local MapPoints seen in Local KeyFrames

	//dense mp의 커넥티드 수는 고려하지 않음.
	std::list<MapPoint*> lLocalMapPoints;
	for (std::list<UVR_SLAM::Frame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	{
		std::vector<MapPoint*> vpMPs = (*lit)->GetDenseVectors();
		for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;
			if (!pMP)
				continue;
			if (pMP->isDeleted()) {
				continue;
			}
			/*if (pMP->GetNumConnectedFrames() <= 2)
				continue;*/
			if (pMP->mnLocalBAID == nTargetID)
				continue;
			lLocalMapPoints.push_back(pMP);
			pMP->mnLocalBAID = nTargetID;
		}
	}
	std::cout << "ba::bbbbb" << std::endl;
	
	// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
	std::list<UVR_SLAM::Frame*> lFixedCameras;
	for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	{
		UVR_SLAM::MapPoint* pMP = *lit;
		auto observations = pMP->GetConnedtedDenseFrames();
		//map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
		for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			UVR_SLAM::Frame* pKFi = mit->first;

			if (pKFi->mnLocalBAID != nTargetID && pKFi->mnFixedBAID != nTargetID)
			{
				pKFi->mnFixedBAID = nTargetID;
				lFixedCameras.push_back(pKFi);
			}
		}
	}
	std::cout << "ba::ddddd" << std::endl;
	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	std::cout << "ba::eeee" << std::endl;

	bool bStopBA = pMapOptimizer->isStopBA();
	if (bStopBA)
		optimizer.setForceStopFlag(&bStopBA);

	unsigned long maxKFid = 0;
	std::cout << "ba::ffff" << std::endl;
	// Set Local KeyFrame vertices
	for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	{
		UVR_SLAM::Frame* pKFi = *lit;
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

		cv::Mat R, t;
		pKFi->GetPose(R, t);
		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
		t.copyTo(Tcw.col(3).rowRange(0, 3));

		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
		vSE3->setId(pKFi->GetKeyFrameID());
		vSE3->setFixed(pKFi->GetKeyFrameID() == 0);
		optimizer.addVertex(vSE3);
		if (pKFi->GetKeyFrameID()>maxKFid)
			maxKFid = pKFi->GetKeyFrameID();
	}
	std::cout << "ba::ggg::" << maxKFid << std::endl;
	// Set Fixed KeyFrame vertices
	for (auto lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
	{
		UVR_SLAM::Frame* pKFi = *lit;
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

		cv::Mat R, t;
		pKFi->GetPose(R, t);
		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
		t.copyTo(Tcw.col(3).rowRange(0, 3));

		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
		vSE3->setId(pKFi->GetKeyFrameID());
		vSE3->setFixed(true);
		optimizer.addVertex(vSE3);
		if (pKFi->GetKeyFrameID()>maxKFid)
			maxKFid = pKFi->GetKeyFrameID();
	}
	std::cout << "ba::hhh::" << maxKFid << std::endl;
	// Set MapPoint vertices
	const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size())*lLocalMapPoints.size();

	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	std::vector<MapPoint*> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);
	std::cout << "ba::iii" << std::endl;
	for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	{
		MapPoint* pMP = *lit;
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		int id = pMP->mnMapPointID + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const auto observations = pMP->GetConnedtedDenseFrames();

		//Set edges
		for (std::map<UVR_SLAM::Frame*, cv::Point2f>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			UVR_SLAM::Frame* pKFi = mit->first;
			if (pKFi->GetKeyFrameID() > maxKFid)
				continue;

			auto pt = mit->second;
			Eigen::Matrix<double, 2, 1> obs;
			obs << pt.x, pt.y;

			g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->GetKeyFrameID())));
			e->setMeasurement(obs);
			const float &invSigma2 = 1.0;
			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(thHuberMono);

			e->fx = pKFi->fx;
			e->fy = pKFi->fy;
			e->cx = pKFi->cx;
			e->cy = pKFi->cy;

			optimizer.addEdge(e);
			vpEdgesMono.push_back(e);
			vpEdgeKFMono.push_back(pKFi);
			vpMapPointEdgeMono.push_back(pMP);
		}

	}

	std::cout << "ba::setting::end" << std::endl;

	bStopBA = pMapOptimizer->isStopBA();
	if (bStopBA)
		return;
	/*if (pbStopFlag)
	if (*pbStopFlag)
	return;*/

	std::cout << "ba::setting::end2" << std::endl;

	optimizer.initializeOptimization();
	optimizer.optimize(5);

	std::cout << "ba::optimize::end" << std::endl;

	bStopBA = pMapOptimizer->isStopBA();

	std::cout << "ba::optimize::end2" << std::endl;

	bool bDoMore = true;
	if (bStopBA)
		bDoMore = false;
	/*if (pbStopFlag)
	if (*pbStopFlag)
	bDoMore = false;*/

	if (bDoMore)
	{

		// Check inlier observations
		for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
		{
			g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
			MapPoint* pMP = vpMapPointEdgeMono[i];

			if (pMP->isDeleted())
				continue;

			if (e->chi2()>5.991 || !e->isDepthPositive())
			{
				e->setLevel(1);
			}

			e->setRobustKernel(0);
		}

		// Optimize again without the outliers

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);

	}

	std::cout << "ba::optimize::end3" << std::endl;

	std::vector<std::pair<UVR_SLAM::Frame*, MapPoint*> > vToErase;
	vToErase.reserve(vpEdgesMono.size());

	// Check inlier observations       
	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	{
		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		MapPoint* pMP = vpMapPointEdgeMono[i];

		if (pMP->isDeleted())
			continue;

		if (e->chi2()>5.991 || !e->isDepthPositive())
		{
			UVR_SLAM::Frame* pKFi = vpEdgeKFMono[i];
			vToErase.push_back(std::make_pair(pKFi, pMP));
		}
	}

	std::cout << "ba::check::inlier::end" << std::endl;

	if (!vToErase.empty())
	{
		for (size_t i = 0; i<vToErase.size(); i++)
		{
			UVR_SLAM::Frame* pKFi = vToErase[i].first;
			MapPoint* pMPi = vToErase[i].second;
			if (pMPi->isDeleted()) {
				std::cout << "????????????????????????????????????????" << std::endl<<std::endl;
			}
			pMPi->RemoveDenseFrame(pKFi);
		}
	}

	std::cout << "ba::erase::end" << std::endl;

	// Recover optimized data

	//Keyframes
	for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	{
		UVR_SLAM::Frame* pKF = *lit;
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetKeyFrameID()));
		g2o::SE3Quat SE3quat = vSE3->estimate();

		cv::Mat R, t;
		cv::Mat Tcw = Converter::toCvMat(SE3quat);
		R = Tcw.rowRange(0, 3).colRange(0, 3);
		t = Tcw.rowRange(0, 3).col(3);
		pKF->SetPose(R, t);
	}

	std::cout << "ba::restore::kf::end" << std::endl;

	//Points
	for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	{
		MapPoint* pMP = *lit;
		if (!pMP || pMP->isDeleted())
			continue;

		////remove
		//int nConnectedThresh = 2;
		////////////////평면일 경우 쓰레시값 조절
		////if (pMP->GetMapPointType() == UVR_SLAM::MapPointType::PLANE_MP){
		////	//바로 만들어진 포인트는 삭제 금지.
		////	if (pMP->mnFirstKeyFrameID == nTargetID)
		////		continue;
		////	nConnectedThresh = 1;
		////}
		////////////////평면일 경우 쓰레시값 조절
		////else if (pMP->isNewMP())
		////	nConnectedThresh = 1;
		//int nConncted = 0;
		//if (pMP->GetMapPointType() == UVR_SLAM::PLANE_DENSE_MP) {
		//	nConncted = pMP->GetNumDensedFrames();
		//}
		//else {
		//	nConncted = pMP->GetNumConnectedFrames();
		//}
		//if (nConncted <= nConnectedThresh) {
		//	pMP->SetDelete(true);
		//	pMP->Delete();
		//	//pWindow->SetMapPoint(nullptr, idx);
		//	//pWindow->SetBoolInlier(false, idx);
		//	continue;
		//}

		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();
	}
	std::cout << "ba::restore::mp::end" << std::endl;
}
////Opticalflow 버전용
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////