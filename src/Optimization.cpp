
#include <Optimization.h>
#include <map>
#include <FrameWindow.h>
#include <opencv2/core/eigen.hpp>
#include <Edge.h>
#include <Optimizer.h>
//#include <PlaneBastedOptimization.h>
#include <PoseGraphOptimization.h>
#include <MatrixOperator.h>

int UVR_SLAM::Optimization::PoseOptimization(UVR_SLAM::Frame* pF, std::vector<UVR_SLAM::MapPoint*> mvpLocalMPs, std::vector<bool>& mvbLocalMPInliers, std::vector<cv::DMatch> mvMatchLocalMap, bool bStatus, int trial1, int trial2){
	cv::Mat mK;
	pF->mK.convertTo(mK, CV_64FC1);
	double fx = mK.at<double>(0, 0);
	double fy = mK.at<double>(1, 1);

	int nCurrFrameID = pF->GetFrameID();
	int nPoseJacobianSize = 6;
	int nResidualSize = 2;

	GraphOptimizer::Optimizer* mpOptimizer = new GraphOptimizer::LMOptimizer();

	cv::Mat Rinit, Tinit;
	Rinit = pF->GetRotation();
	Tinit = pF->GetTranslation();

	FrameVertex* mpVertex1 = new FrameVertex(Rinit, Tinit, nPoseJacobianSize);
	mpOptimizer->AddVertex(mpVertex1);

	//Add Edge
	const double deltaMono = sqrt(5.991);
	const double chiMono = 5.991;

	std::vector<PoseOptimizationEdge*> mvpEdges;
	std::vector<int> mvIndexes;

	for (int i = 0; i < mvMatchLocalMap.size(); i++) {

		int idx1 = mvMatchLocalMap[i].queryIdx;
		int idx2 = mvMatchLocalMap[i].trainIdx;

		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[idx1];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		if (!mvbLocalMPInliers[i])
			continue;
		mvIndexes.push_back(i);
		UVR_SLAM::PoseOptimizationEdge* pEdge1 = new UVR_SLAM::PoseOptimizationEdge(pMP->GetWorldPos(), nResidualSize);
		Eigen::Vector2d temp = Eigen::Vector2d();
		cv::Point2f pt1 = pF->mvKeyPoints[idx2].pt;
		temp(0) = pt1.x;
		temp(1) = pt1.y;
		pEdge1->SetMeasurement(temp);
		cv::cv2eigen(mK, pEdge1->K);
		pEdge1->AddVertex(mpVertex1);
		double info = (double)pF->mvInvLevelSigma2[pF->mvKeyPoints[idx2].octave];
		//std::cout << "information::" << info2 << std::endl;
		pEdge1->SetInformation(cv::Mat::eye(nResidualSize, nResidualSize, CV_64FC1)*info);
		pEdge1->fx = fx;
		pEdge1->fy = fy;
		GraphOptimizer::RobustKernel* huber = new GraphOptimizer::HuberKernel();
		huber->SetDelta(deltaMono);
		pEdge1->mpRobustKernel = huber;
		mpOptimizer->AddEdge(pEdge1);
		mvpEdges.push_back(pEdge1);
	}

	int nInlier = 0;
	for (int trial = 0; trial < trial1; trial++) {
		mpOptimizer->Optimize(trial2, 0, bStatus);
		nInlier = 0;
		for (int edgeIdx = 0; edgeIdx < mvpEdges.size(); edgeIdx++) {
			cv::DMatch match = mvMatchLocalMap[mvIndexes[edgeIdx]];
			int idx1 = match.queryIdx; //framewindow
			int idx2 = match.trainIdx; //frame
			if (!pF->mvbMPInliers[idx2]) {
				mvpEdges[edgeIdx]->CalcError();
			}
			if (mvpEdges[edgeIdx]->GetError() > chiMono || !mvpEdges[edgeIdx]->GetDepth()) {
				mvpEdges[edgeIdx]->SetLevel(1);
				pF->mvbMPInliers[idx2] = false;
				//pF->mvpMPs[idx2]->SetRecentTrackingFrameID(-1);
				mvbLocalMPInliers[idx1] = false;
			}
			else
			{
				mvpEdges[edgeIdx]->SetLevel(0);
				pF->mvbMPInliers[idx2] = true;
				//pF->mvpMPs[idx2]->SetRecentTrackingFrameID(nCurrFrameID);
				mvbLocalMPInliers[idx1] = true;
				nInlier++;
			}
		}
	}

	//mp inlier 
	//std::cout << "PoseOptimization::inlier=" << nInlier << std::endl;
	mpVertex1->RestoreData();
	pF->SetPose(mpVertex1->Rmat, mpVertex1->Tmat);
	if (bStatus)
		std::cout << "PoseOptimization::End" << std::endl;
	return nInlier;
}

int UVR_SLAM::Optimization::PoseOptimization(UVR_SLAM::FrameWindow* pWindow, UVR_SLAM::Frame* pF, std::vector<MapPoint*> mvpLocalMPs, std::vector<bool>& mvbLocalMapInliers, bool bStatus, int trial1, int trial2) {
	if(bStatus)
		std::cout << "PoseOptimization::Start" << std::endl;
	cv::Mat mK;
	pF->mK.convertTo(mK, CV_64FC1);
	double fx = mK.at<double>(0, 0);
	double fy = mK.at<double>(1, 1);

	int nPoseJacobianSize = 6;
	int nResidualSize = 2;

	GraphOptimizer::Optimizer* mpOptimizer = new GraphOptimizer::LMOptimizer();
	FrameVertex* mpVertex1 = new FrameVertex(pWindow->GetRotation(), pWindow->GetTranslation(), nPoseJacobianSize);
	mpOptimizer->AddVertex(mpVertex1);
	
	//Add Edge
	const double deltaMono = sqrt(5.991);
	const double chiMono = 5.991;

	std::vector<PoseOptimizationEdge*> mvpEdges;
	std::vector<int> mvIndexes;

	auto mvpMPs = pF->GetMapPoints();

	//for (int i = 0; i < pWindow->mvMatchInfos.size(); i++) {
	//	
	//	cv::DMatch match = pWindow->mvMatchInfos[i];
	//	if (!mvbLocalMapInliers[i])
	//		continue;
	//	
	//	int idx1 = match.queryIdx; //framewindow
	//	int idx2 = match.trainIdx; //frame

	//	UVR_SLAM::MapPoint* pMP = mvpMPs[idx2];
	//	if (!pMP)
	//		continue;
	//	if (pMP->isDeleted())
	//		continue;
	//	mvIndexes.push_back(i);

	//	UVR_SLAM::PoseOptimizationEdge* pEdge1 = new UVR_SLAM::PoseOptimizationEdge(pMP->GetWorldPos(),nResidualSize);
	//	Eigen::Vector2d temp = Eigen::Vector2d();
	//	cv::Point2f pt1 = pF->mvKeyPoints[idx2].pt;
	//	temp(0) = pt1.x;
	//	temp(1) = pt1.y;
	//	pEdge1->SetMeasurement(temp);
	//	cv::cv2eigen(mK, pEdge1->K);
	//	pEdge1->AddVertex(mpVertex1);
	//	double info = (double)pF->mvInvLevelSigma2[pF->mvKeyPoints[idx2].octave];
	//	//std::cout << "information::" << info2 << std::endl;
	//	pEdge1->SetInformation(cv::Mat::eye(nResidualSize, nResidualSize, CV_64FC1)*info);
	//	pEdge1->fx = fx;
	//	pEdge1->fy = fy;
	//	GraphOptimizer::RobustKernel* huber = new GraphOptimizer::HuberKernel();
	//	huber->SetDelta(deltaMono);
	//	pEdge1->mpRobustKernel = huber;
	//	mpOptimizer->AddEdge(pEdge1);
	//	mvpEdges.push_back(pEdge1);
	//}

	//int nInlier = 0;
	//for (int trial = 0; trial < trial1; trial++) {
	//	mpOptimizer->Optimize(trial2, 0, bStatus);
	//	nInlier = 0;
	//	for (int edgeIdx = 0; edgeIdx < mvpEdges.size(); edgeIdx++) {
	//		cv::DMatch match = pWindow->mvMatchInfos[mvIndexes[edgeIdx]];
	//		
	//		int idx1 = match.queryIdx; //framewindow
	//		int idx2 = match.trainIdx; //frame
	//		if (!pF->mvbMPInliers[idx2]) {
	//			mvpEdges[edgeIdx]->CalcError();
	//		}
	//		if (mvpEdges[edgeIdx]->GetError() > chiMono || !mvpEdges[edgeIdx]->GetDepth()) {
	//			mvpEdges[edgeIdx]->SetLevel(1);
	//			pF->mvbMPInliers[idx2] = false;
	//			mvbLocalMapInliers[idx1] = false;
	//		}
	//		else
	//		{
	//			mvpEdges[edgeIdx]->SetLevel(0);
	//			pF->mvbMPInliers[idx2] = true;
	//			mvbLocalMapInliers[idx1] = true;
	//			nInlier++;
	//		}
	//	}
	//}

	//mp inlier 
	//std::cout << "PoseOptimization::inlier=" << nInlier << std::endl;
	mpVertex1->RestoreData();
	pF->SetPose(mpVertex1->Rmat, mpVertex1->Tmat);
	pWindow->SetPose(mpVertex1->Rmat, mpVertex1->Tmat);
	if(bStatus)
		std::cout << "PoseOptimization::End" << std::endl;
	return 0;
}
int UVR_SLAM::Optimization::InitOptimization(UVR_SLAM::InitialData* data, std::vector<cv::DMatch> Matches, UVR_SLAM::Frame* pInitFrame1, UVR_SLAM::Frame* pInitFrame2, cv::Mat K, bool& bInit, int trial1, int trial2) {
	cv::Mat mK;
	K.convertTo(mK, CV_64FC1);
	double fx = mK.at<double>(0, 0);
	double fy = mK.at<double>(1, 1);

	int nPoseJacobianSize = 6;
	int nMapJacobianSize = 3;
	int nResidualSize = 2;
	GraphOptimizer::Optimizer* mpOptimizer = new GraphOptimizer::LMOptimizer();
	mpOptimizer->bSchur = true;
	FrameVertex* mpVertex1 = new FrameVertex(data->R0, data->t0, nPoseJacobianSize);
	mpVertex1->SetFixed(true);
	FrameVertex* mpVertex2 = new FrameVertex(data->R, data->t, nPoseJacobianSize);
	mpOptimizer->AddVertex(mpVertex1, true);
	mpOptimizer->AddVertex(mpVertex2, true);
	
	const double deltaMono = sqrt(5.991);
	const double chiMono = 5.991;

	//Add MapPoints
	std::vector<MapPointVertex*> mvpMPVertices(0);
	for (int i = 0; i < data->mvX3Ds.size(); i++) {
		if (!data->vbTriangulated[i])
			continue;
		MapPointVertex* mpVertex = new MapPointVertex(data->mvX3Ds[i], i, nMapJacobianSize);
		//std::cout << mpVertex->GetParam().rows() << std::endl;
		mpOptimizer->AddVertex(mpVertex, false);
		mvpMPVertices.push_back(mpVertex);
	}

	//Add Edge
	std::vector<EdgePoseNMap*> mvpEdges;
	for (int i = 0; i < mvpMPVertices.size(); i++) {
		int idx = mvpMPVertices[i]->GetIndex();
		
		UVR_SLAM::EdgePoseNMap* pEdge1 = new UVR_SLAM::EdgePoseNMap(nResidualSize);
		Eigen::Vector2d temp = Eigen::Vector2d();
		cv::Point2f pt1 = pInitFrame1->mvKeyPoints[Matches[idx].queryIdx].pt;
		temp(0) = pt1.x;
		temp(1) = pt1.y;
		pEdge1->SetMeasurement(temp);
		cv::cv2eigen(mK, pEdge1->K);
		
		pEdge1->AddVertex(mpVertex1);
		pEdge1->AddVertex(mvpMPVertices[i]);
		//pEdge1->SetSubHessian();
		double info1 = (double)pInitFrame1->mvInvLevelSigma2[pInitFrame1->mvKeyPoints[Matches[idx].queryIdx].octave];
		pEdge1->SetInformation(cv::Mat::eye(nResidualSize, nResidualSize, CV_64FC1)*info1);
		
		pEdge1->fx = fx;
		pEdge1->fy = fy;
		
		GraphOptimizer::RobustKernel* huber = new GraphOptimizer::HuberKernel();
		huber->SetDelta(deltaMono);
		pEdge1->mpRobustKernel = huber;
	
		mpOptimizer->AddEdge(pEdge1);
		mvpEdges.push_back(pEdge1);

		UVR_SLAM::EdgePoseNMap* pEdge2 = new UVR_SLAM::EdgePoseNMap(nResidualSize);
		Eigen::Vector2d temp2 = Eigen::Vector2d();
		cv::Point2f pt2 = pInitFrame2->mvKeyPoints[Matches[idx].trainIdx].pt;
		temp2(0) = pt2.x;
		temp2(1) = pt2.y;
		pEdge2->SetMeasurement(temp2);
		cv::cv2eigen(mK, pEdge2->K);

		pEdge2->AddVertex(mpVertex2);
		pEdge2->AddVertex(mvpMPVertices[i]);
		//pEdge2->SetSubHessian();
		double info2 = (double)pInitFrame2->mvInvLevelSigma2[pInitFrame2->mvKeyPoints[Matches[idx].trainIdx].octave];
		pEdge2->SetInformation(cv::Mat::eye(nResidualSize, nResidualSize, CV_64FC1)*info2);

		pEdge2->fx = fx;
		pEdge2->fy = fy;

		GraphOptimizer::RobustKernel* huber2 = new GraphOptimizer::HuberKernel();
		huber2->SetDelta(deltaMono);
		pEdge2->mpRobustKernel = huber2;

		mpOptimizer->AddEdge(pEdge2);
		mvpEdges.push_back(pEdge2);
	}

	for (int trial = 0; trial < trial1; trial++) {
		mpOptimizer->Optimize(trial2, 0, true);

		for (int i = 0; i < mvpMPVertices.size(); i++) {
			int eidx1 = i * 2;
			int eidx2 = eidx1 + 1;

			int idx = mvpMPVertices[i]->GetIndex();
			if (!data->vbTriangulated[idx]){
				mvpEdges[eidx1]->CalcError();
				mvpEdges[eidx2]->CalcError();
			}
			double err1 = mvpEdges[eidx1]->GetError();
			double err2 = mvpEdges[eidx2]->GetError();
			if (err1 > chiMono || err2 > chiMono || !mvpEdges[eidx1]->GetDepth() || !mvpEdges[eidx2]->GetDepth()) {
				data->vbTriangulated[idx] = false;
				mvpEdges[eidx1]->SetLevel(1);
				mvpEdges[eidx2]->SetLevel(1);
			}
			else {
				data->vbTriangulated[idx] = true;
				mvpEdges[eidx1]->SetLevel(0);
				mvpEdges[eidx2]->SetLevel(0);
			}
		}
	}
	
	//update
	mpVertex1->RestoreData();
	mpVertex2->RestoreData();
	int nRes = 0;
	data->SetRt(mpVertex2->Rmat, mpVertex2->Tmat);
	for (int i = 0; i < mvpMPVertices.size(); i++) {
		int idx = mvpMPVertices[i]->GetIndex();
		if (data->vbTriangulated[idx]) {
			nRes++;
			mvpMPVertices[i]->RestoreData();
			data->mvX3Ds[idx] = mvpMPVertices[i]->Xw.clone();
		}
	}
	if (nRes > 79) {
		bInit = true;
	}
	else {
		bInit = false;
	}

	std::cout << mpVertex1->Rmat << mpVertex1->Tmat << std::endl;
	std::cout << mpVertex2->Rmat << mpVertex2->Tmat << std::endl;

	return nRes;
}

void UVR_SLAM::Optimization::LocalBundleAdjustment(UVR_SLAM::FrameWindow* pWindow, int nTargetID,bool& bStopBA, int trial1, int trial2, bool bShowStatus) {
	//fixed frame
	//connected kf = 3
	//check newmp
	//KF
	if(bShowStatus)
		std::cout << "LocalBA::Start"<< std::endl;
	

	int nPoseJacobianSize = 6;
	int nMapJacobianSize = 3;
	int nResidualSize = 2;

	GraphOptimizer::Optimizer* mpOptimizer = new GraphOptimizer::LMOptimizer();
	mpOptimizer->bSchur = true;

	//Add vertices in window
	//check initial keyframe
	std::vector<FrameVertex*> mvpFrameVertices;
	std::vector<MapPointVertex*> mvpMapPointVertices;
	std::map<UVR_SLAM::Frame*, int> mmpFrames; int nFrame = 0; int nFixedFrames = 0;
	//std::vector<FrameVertex*> mvpFixedVertices;
	std::set<UVR_SLAM::Frame*> mspFixedFrames;

	std::vector<UVR_SLAM::Frame*> mvpFrames;
	std::vector<UVR_SLAM::MapPoint*> mvpMPs;
	std::vector<int> mvLocalMPIndex;

	auto mvpKFs = pWindow->GetLocalMapFrames();
	for (auto iter = mvpKFs.begin(); iter != mvpKFs.end(); iter++) {
		Frame* pF = *iter;
		if (pF->mnLocalBAID == nTargetID)
			continue;
		pF->mnLocalBAID = nTargetID;
		FrameVertex* mpVertex = new FrameVertex(pF->GetRotation(), pF->GetTranslation(), nPoseJacobianSize);
		mpOptimizer->AddVertex(mpVertex);
		mvpFrameVertices.push_back(mpVertex);
		mmpFrames.insert(std::make_pair(pF, nFrame++));
		mvpFrames.push_back(pF);
	}
	//K
	cv::Mat mK = mvpKFs[0]->mK.clone();
	mK.convertTo(mK, CV_64FC1);
	double fx = mK.at<double>(0, 0);
	double fy = mK.at<double>(1, 1);
	
	auto mvpLocalMPs = pWindow->GetLocalMap();
	//Add map point
	for (int i = 0; i < mvpLocalMPs.size(); i++) {

		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted()){
			continue;
		}
		/*MapPointVertex* mpVertex = new MapPointVertex(pMP->GetWorldPos(), i, nMapJacobianSize);
		mpOptimizer->AddVertex(mpVertex, false);
		mvpMapPointVertices.push_back(mpVertex);*/

		//fixed keyframe
		auto mmpConnectedFrames = pMP->GetConnedtedFrames();
		for (auto iter = mmpConnectedFrames.begin(); iter != mmpConnectedFrames.end(); iter++) {
			UVR_SLAM::Frame* pF = iter->first;

			if (pF->mnLocalBAID == nTargetID)
				continue;
			pF->mnLocalBAID = nTargetID;
			mmpFrames.insert(std::make_pair(pF, nFrame++));
			mspFixedFrames.insert(pF);
			nFixedFrames++;

			//auto findres = mmpFrames.find(pF);
			//if (findres == mmpFrames.end()) {
			//	mmpFrames.insert(std::make_pair(pF, nFrame++));
			//	mspFixedFrames.insert(pF);
			//	nFixedFrames++;
				
				
				/*auto findres2 = mspFixedVertices.find(pF);
				if (findres2 == mspFixedVertices.end()) {
				}
				FrameVertex* mpFVertex = new FrameVertex(pF->GetRotation(), pF->GetTranslation(), nPoseJacobianSize);
				mpFVertex->SetFixed(true);
				mpOptimizer->AddVertex(mpFVertex);*/
				
			//}
		}
	}
	
	for (auto iter = mspFixedFrames.begin(); iter != mspFixedFrames.end(); iter++) {
		UVR_SLAM::Frame* pF = *iter;
		FrameVertex* mpFVertex = new FrameVertex(pF->GetRotation(), pF->GetTranslation(), nPoseJacobianSize);
		mpFVertex->SetFixed(true);
		mvpFrameVertices.push_back(mpFVertex);
		mpOptimizer->AddVertex(mpFVertex);
	}

	for (int i = 0; i < mvpLocalMPs.size(); i++) {

		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		MapPointVertex* mpVertex = new MapPointVertex(pMP->GetWorldPos(), i, nMapJacobianSize);
		mpOptimizer->AddVertex(mpVertex, false);
		mvpMapPointVertices.push_back(mpVertex);
		mvpMPs.push_back(pMP);
		mvLocalMPIndex.push_back(i);
	}
	if (bShowStatus)
		std::cout << "LocalBA::FixedFrame::" << nFixedFrames << std::endl;
	
	//Add Edge
	const double deltaMono = sqrt(5.991);
	const double chiMono = 5.991;

	std::vector<EdgePoseNMap*> mvpEdges;
	typedef std::tuple<UVR_SLAM::MapPoint*, UVR_SLAM::Frame*, int> LocalBAEdgeData;
	std::vector<LocalBAEdgeData> mvTupleData;
	//std::vector<std::pair<UVR_SLAM::MapPoint*, int>> mvPairEdges;
	std::vector<bool> mvbEdges;
	
	//Add Edge
	for (int i = 0; i < mvpMapPointVertices.size(); i++) {
		int idx = mvpMapPointVertices[i]->GetIndex();
		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[idx];
		auto mmpConnectedFrames = pMP->GetConnedtedFrames();
		for (auto iter = mmpConnectedFrames.begin(); iter != mmpConnectedFrames.end(); iter++) {
			UVR_SLAM::Frame* pF = iter->first;
			int feature_idx = iter->second;
			int frame_idx = mmpFrames[pF];

			UVR_SLAM::EdgePoseNMap* pEdge = new UVR_SLAM::EdgePoseNMap(nResidualSize);
			Eigen::Vector2d temp = Eigen::Vector2d();
			cv::Point2f pt = pF->mvKeyPoints[feature_idx].pt;
			temp(0) = pt.x;
			temp(1) = pt.y;
			pEdge->SetMeasurement(temp);
			cv::cv2eigen(mK, pEdge->K);

			pEdge->AddVertex(mvpFrameVertices[frame_idx]);
			pEdge->AddVertex(mvpMapPointVertices[i]);

			double info1 = (double)pF->mvInvLevelSigma2[pF->mvKeyPoints[feature_idx].octave];
			pEdge->SetInformation(cv::Mat::eye(nResidualSize, nResidualSize, CV_64FC1)*info1);

			pEdge->fx = fx;
			pEdge->fy = fy;

			GraphOptimizer::RobustKernel* huber = new GraphOptimizer::HuberKernel();
			huber->SetDelta(deltaMono);
			pEdge->mpRobustKernel = huber;

			mpOptimizer->AddEdge(pEdge);
			mvpEdges.push_back(pEdge);
			mvbEdges.push_back(true);
			LocalBAEdgeData tempData = std::make_tuple(pMP, pF, feature_idx);
			mvTupleData.push_back(tempData);

		}

	}
	//Optimize
	for (int trial = 0; trial < trial1; trial++) {
		if (bStopBA)
			break;
		mpOptimizer->Optimize(trial2, 0, bShowStatus);
		for (int i = 0; i < mvpEdges.size(); i++) {
			EdgePoseNMap* pEdge = mvpEdges[i];
			if (!mvbEdges[i]) {
				mvpEdges[i]->CalcError();
			}
			double err = mvpEdges[i]->GetError();
			if (err > chiMono || !mvpEdges[i]->GetDepth()) {
				mvbEdges[i] = false;
				mvpEdges[i]->SetLevel(1);
			}
			else {
				mvbEdges[i] = true;
				mvpEdges[i]->SetLevel(0);
			}
		}
	}
	if (bShowStatus)
		std::cout << "Update Parameter::Start" << std::endl;
	//update parameter
	//std::vector<
	for (int i = 0; i < mvpEdges.size(); i++) {
		
		if (!mvbEdges[i]) {
			LocalBAEdgeData tempData = mvTupleData[i];
			UVR_SLAM::MapPoint* pMP = std::get<0>(tempData);
			UVR_SLAM::Frame* pF = std::get<1>(tempData);
			int feature_idx = std::get<2>(tempData);
			pMP->RemoveFrame(pF);
		}
	}
	for (int i = 0; i < mvpFrames.size(); i++) {
		mvpFrameVertices[i]->RestoreData();
		mvpFrames[i]->SetPose(mvpFrameVertices[i]->Rmat, mvpFrameVertices[i]->Tmat);
	}
	for (int i = 0; i <  mvpMPs.size(); i++) {
		int nConnectedThresh = 3;
		if (mvpMPs[i]->GetMapPointType() == UVR_SLAM::MapPointType::PLANE_MP)
			nConnectedThresh = 1;
		else if (mvpMPs[i]->isNewMP())
			nConnectedThresh = 2;
		if (mvpMPs[i]->GetNumConnectedFrames() < nConnectedThresh) {

			int idx = mvLocalMPIndex[i];
			UVR_SLAM::MapPoint* pMP = mvpLocalMPs[idx];
			pMP->SetDelete(true);
			pMP->Delete();
			//pWindow->SetMapPoint(nullptr, idx);
			//pWindow->SetBoolInlier(false, idx);
			continue;
		}
		
		mvpMapPointVertices[i]->RestoreData();
		mvpMPs[i]->SetWorldPos(mvpMapPointVertices[i]->Xw);
		//std::cout <<"Connected MPs = "<< mvpMPs[i]->GetNumConnectedFrames() << std::endl;
	}
	if (bShowStatus)
		std::cout << "Update Parameter::End" << std::endl;
}

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