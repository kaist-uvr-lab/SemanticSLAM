
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

///////////////////////////////////////////////////////////
////////백업
//g2o 버전
//void UVR_SLAM::Optimization::LocalBundleAdjustment(UVR_SLAM::MapOptimizer* pMapOptimizer, UVR_SLAM::Frame* pKF, UVR_SLAM::FrameWindow* pWindow) {
//	// Local KeyFrames: First Breath Search from Current Keyframe
//	std::list<UVR_SLAM::Frame*> lLocalKeyFrames;
//
//	std::cout << "ba::ssssss" << std::endl;
//
//	int nTargetID = pKF->GetFrameID();
//	lLocalKeyFrames.push_back(pKF);
//	pKF->mnLocalBAID = nTargetID;
//
//	int nn = 15;
//
//	const std::vector<UVR_SLAM::Frame*> vNeighKFs = pKF->GetConnectedKFs(nn);
//	for (int i = 0, iend = vNeighKFs.size(); i<iend; i++)
//	{
//		UVR_SLAM::Frame* pKFi = vNeighKFs[i];
//		if (pKFi->mnLocalBAID == nTargetID)
//			continue;
//		pKFi->mnLocalBAID = nTargetID;
//		lLocalKeyFrames.push_back(pKFi);
//	}
//	std::cout << "ba::aaaa" << std::endl;
//	// Local MapPoints seen in Local KeyFrames
//	std::list<MapPoint*> lLocalMapPoints;
//	for (std::list<UVR_SLAM::Frame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
//	{
//		std::vector<MapPoint*> vpMPs = (*lit)->GetMapPoints();
//		for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
//		{
//			MapPoint* pMP = *vit;
//			if (!pMP)
//				continue;
//			if (pMP->isDeleted()) {
//				continue;
//			}
//			if (pMP->GetNumConnectedFrames() <= 2)
//				continue;
//			if (pMP->mnLocalBAID == nTargetID)
//				continue;
//			lLocalMapPoints.push_back(pMP);
//			pMP->mnLocalBAID = nTargetID;
//		}
//	}
//	std::cout << "ba::bbbbb" << std::endl;
//	////dense map points 추가하기
//	int nDenseIdx = lLocalMapPoints.size();
//	auto mvpDenseMPs = pKF->GetDenseVectors();
//	for (int i = 0; i < mvpDenseMPs.size(); i++)
//	{
//		lLocalMapPoints.push_back(mvpDenseMPs[i]);
//	}
//	////dense map points 추가하기
//	std::cout << "ba::ccccc" << std::endl;
//	// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
//	std::list<UVR_SLAM::Frame*> lFixedCameras;
//	for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
//	{
//		UVR_SLAM::MapPoint* pMP = *lit;
//
//		if (pMP->GetMapPointType() == UVR_SLAM::PLANE_DENSE_MP) {
//			auto observations = pMP->GetConnedtedDenseFrames();
//			//map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
//			for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
//			{
//				UVR_SLAM::Frame* pKFi = mit->first;
//
//				if (pKFi->mnLocalBAID != nTargetID && pKFi->mnFixedBAID != nTargetID)
//				{
//					pKFi->mnFixedBAID = nTargetID;
//					lFixedCameras.push_back(pKFi);
//				}
//			}
//		}
//		else {
//			auto observations = pMP->GetConnedtedFrames();
//			//map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
//			for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
//			{
//				UVR_SLAM::Frame* pKFi = mit->first;
//
//				if (pKFi->mnLocalBAID != nTargetID && pKFi->mnFixedBAID != nTargetID)
//				{
//					pKFi->mnFixedBAID = nTargetID;
//					lFixedCameras.push_back(pKFi);
//				}
//			}
//		}
//
//		
//	}
//	std::cout << "ba::ddddd" << std::endl;
//	// Setup optimizer
//	g2o::SparseOptimizer optimizer;
//	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
//
//	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
//
//	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
//
//	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//	optimizer.setAlgorithm(solver);
//
//	std::cout << "ba::eeee" << std::endl;
//
//	bool bStopBA = pMapOptimizer->isStopBA();
//	if (bStopBA)
//		optimizer.setForceStopFlag(&bStopBA);
//
//	unsigned long maxKFid = 0;
//	std::cout << "ba::ffff" << std::endl;
//	// Set Local KeyFrame vertices
//	for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
//	{
//		UVR_SLAM::Frame* pKFi = *lit;
//		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
//
//		cv::Mat R, t;
//		pKFi->GetPose(R, t);
//		cv::Mat Tcw = cv::Mat::zeros(4,4, CV_32FC1);
//		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
//		t.copyTo(Tcw.col(3).rowRange(0, 3));
//
//		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
//		vSE3->setId(pKFi->GetKeyFrameID());
//		vSE3->setFixed(pKFi->GetKeyFrameID() == 0);
//		optimizer.addVertex(vSE3);
//		if (pKFi->GetKeyFrameID()>maxKFid)
//			maxKFid = pKFi->GetKeyFrameID();
//	}
//	std::cout << "ba::ggg::"<< maxKFid<< std::endl;
//	// Set Fixed KeyFrame vertices
//	for (auto lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
//	{
//		UVR_SLAM::Frame* pKFi = *lit;
//		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
//
//		cv::Mat R, t;
//		pKFi->GetPose(R, t);
//		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
//		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
//		t.copyTo(Tcw.col(3).rowRange(0, 3));
//
//		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
//		vSE3->setId(pKFi->GetKeyFrameID());
//		vSE3->setFixed(true);
//		optimizer.addVertex(vSE3);
//		if (pKFi->GetKeyFrameID()>maxKFid)
//			maxKFid = pKFi->GetKeyFrameID();
//	}
//	std::cout << "ba::hhh::" << maxKFid << std::endl;
//	// Set MapPoint vertices
//	const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size())*lLocalMapPoints.size();
//
//	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
//	vpEdgesMono.reserve(nExpectedSize);
//
//	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
//	vpEdgeKFMono.reserve(nExpectedSize);
//
//	std::vector<MapPoint*> vpMapPointEdgeMono;
//	vpMapPointEdgeMono.reserve(nExpectedSize);
//
//	const float thHuberMono = sqrt(5.991);
//	const float thHuberStereo = sqrt(7.815);
//	std::cout << "ba::iii" << std::endl;
//	for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
//	{
//		MapPoint* pMP = *lit;
//		if (!pMP)
//			continue;
//		if (pMP->isDeleted())
//			continue;
//		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
//		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
//		int id = pMP->mnMapPointID + maxKFid + 1;
//		vPoint->setId(id);
//		vPoint->setMarginalized(true);
//		optimizer.addVertex(vPoint);
//
//		if (pMP->GetMapPointType() != UVR_SLAM::PLANE_DENSE_MP) {
//			const auto observations = pMP->GetConnedtedFrames();
//			
//			//Set edges
//			for (std::map<UVR_SLAM::Frame*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
//			{
//				UVR_SLAM::Frame* pKFi = mit->first;
//				if (pKFi->GetKeyFrameID() > maxKFid)
//					continue;
//
//				const cv::KeyPoint &kpUn = pKFi->mvKeyPoints[mit->second];
//
//				Eigen::Matrix<double, 2, 1> obs;
//				obs << kpUn.pt.x, kpUn.pt.y;
//
//				g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
//				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
//				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->GetKeyFrameID())));
//				e->setMeasurement(obs);
//				const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
//				e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
//
//				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//				e->setRobustKernel(rk);
//				rk->setDelta(thHuberMono);
//
//				e->fx = pKFi->fx;
//				e->fy = pKFi->fy;
//				e->cx = pKFi->cx;
//				e->cy = pKFi->cy;
//
//				optimizer.addEdge(e);
//				vpEdgesMono.push_back(e);
//				vpEdgeKFMono.push_back(pKFi);
//				vpMapPointEdgeMono.push_back(pMP);
//			}
//		}else {
//			
//			const auto observations = pMP->GetConnedtedDenseFrames();
//			
//			//Set edges
//			for (std::map<UVR_SLAM::Frame*, cv::Point2f>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
//			{
//				UVR_SLAM::Frame* pKFi = mit->first;
//				if (pKFi->GetKeyFrameID() > maxKFid)
//					continue;
//
//				auto pt = mit->second;
//				Eigen::Matrix<double, 2, 1> obs;
//				obs << pt.x, pt.y;
//
//				g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
//				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
//				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->GetKeyFrameID())));
//				e->setMeasurement(obs);
//				const float &invSigma2 = 1.0;
//				e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
//
//				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//				e->setRobustKernel(rk);
//				rk->setDelta(thHuberMono);
//
//				e->fx = pKFi->fx;
//				e->fy = pKFi->fy;
//				e->cx = pKFi->cx;
//				e->cy = pKFi->cy;
//
//				optimizer.addEdge(e);
//				vpEdgesMono.push_back(e);
//				vpEdgeKFMono.push_back(pKFi);
//				vpMapPointEdgeMono.push_back(pMP);
//			}
//		}//if mappoint type
//		
//	}
//
//	std::cout << "ba::setting::end" << std::endl;
//
//	bStopBA = pMapOptimizer->isStopBA();
//	if (bStopBA)
//		return;
//	/*if (pbStopFlag)
//		if (*pbStopFlag)
//			return;*/
//
//	std::cout << "ba::setting::end2" << std::endl;
//
//	optimizer.initializeOptimization();
//	optimizer.optimize(5);
//
//	std::cout << "ba::optimize::end" << std::endl;
//
//	bStopBA = pMapOptimizer->isStopBA();
//
//	std::cout << "ba::optimize::end2" << std::endl;
//
//	bool bDoMore = true;
//	if (bStopBA)
//		bDoMore = false;
//	/*if (pbStopFlag)
//		if (*pbStopFlag)
//			bDoMore = false;*/
//
//	if (bDoMore)
//	{
//
//		// Check inlier observations
//		for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
//		{
//			g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
//			MapPoint* pMP = vpMapPointEdgeMono[i];
//
//			if (pMP->isDeleted())
//				continue;
//
//			if (e->chi2()>5.991 || !e->isDepthPositive())
//			{
//				e->setLevel(1);
//			}
//
//			e->setRobustKernel(0);
//		}
//
//		// Optimize again without the outliers
//
//		optimizer.initializeOptimization(0);
//		optimizer.optimize(10);
//
//	}
//
//	std::cout << "ba::optimize::end3" << std::endl;
//
//	std::vector<std::pair<UVR_SLAM::Frame*, MapPoint*> > vToErase;
//	vToErase.reserve(vpEdgesMono.size());
//
//	// Check inlier observations       
//	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
//	{
//		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
//		MapPoint* pMP = vpMapPointEdgeMono[i];
//
//		if (pMP->isDeleted())
//			continue;
//
//		if (e->chi2()>5.991 || !e->isDepthPositive())
//		{
//			UVR_SLAM::Frame* pKFi = vpEdgeKFMono[i];
//			vToErase.push_back(std::make_pair(pKFi, pMP));
//		}
//	}
//
//	std::cout << "ba::check::inlier::end" << std::endl;
//
//	if (!vToErase.empty())
//	{
//		for (size_t i = 0; i<vToErase.size(); i++)
//		{
//			UVR_SLAM::Frame* pKFi = vToErase[i].first;
//			MapPoint* pMPi = vToErase[i].second;
//			
//			if (pMPi->GetMapPointType() == UVR_SLAM::PLANE_DENSE_MP) {
//				pMPi->RemoveDenseFrame(pKFi);
//			}
//			else {
//				pMPi->RemoveFrame(pKFi);
//			}
//			
//			//pKFi->EraseMapPointMatch(pMPi);
//			//pMPi->EraseObservation(pKFi);
//
//		}
//	}
//
//	std::cout << "ba::erase::end" << std::endl;
//
//	// Recover optimized data
//
//	//Keyframes
//	for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
//	{
//		UVR_SLAM::Frame* pKF = *lit;
//		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetKeyFrameID()));
//		g2o::SE3Quat SE3quat = vSE3->estimate();
//
//		cv::Mat R, t;
//		cv::Mat Tcw = Converter::toCvMat(SE3quat);
//		R = Tcw.rowRange(0, 3).colRange(0, 3);
//		t = Tcw.rowRange(0, 3).col(3);
//		pKF->SetPose(R, t);
//	}
//
//	std::cout << "ba::restore::kf::end" << std::endl;
//
//	//Points
//	for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
//	{
//		MapPoint* pMP = *lit;
//		if (!pMP || pMP->isDeleted())
//			continue;
//
//		//remove
//		int nConnectedThresh = 2;
//		//////////////평면일 경우 쓰레시값 조절
//		//if (pMP->GetMapPointType() == UVR_SLAM::MapPointType::PLANE_MP){
//		//	//바로 만들어진 포인트는 삭제 금지.
//		//	if (pMP->mnFirstKeyFrameID == nTargetID)
//		//		continue;
//		//	nConnectedThresh = 1;
//		//}
//		//////////////평면일 경우 쓰레시값 조절
//		//else if (pMP->isNewMP())
//		//	nConnectedThresh = 1;
//		int nConncted = 0;
//		if (pMP->GetMapPointType() == UVR_SLAM::PLANE_DENSE_MP) {
//			nConncted = pMP->GetNumDensedFrames();
//		}
//		else {
//			nConncted = pMP->GetNumConnectedFrames();
//		}
//		if (nConncted <= nConnectedThresh) {
//			pMP->SetDelete(true);
//			pMP->Delete();
//			//pWindow->SetMapPoint(nullptr, idx);
//			//pWindow->SetBoolInlier(false, idx);
//			continue;
//		}
//
//		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + 1));
//		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
//		pMP->UpdateNormalAndDepth();
//	}
//	std::cout << "ba::restore::mp::end" << std::endl;
//}
//void UVR_SLAM::Optimization::LocalBundleAdjustmentWithPlane(UVR_SLAM::Map* pMap, UVR_SLAM::Frame *pKF, UVR_SLAM::FrameWindow* pWindow, bool* pbStopFlag)
//{
//	// Local KeyFrames: First Breath Search from Current Keyframe
//	std::list<UVR_SLAM::Frame*> lLocalKeyFrames;
//
//	int nTargetID = pKF->GetFrameID();
//	lLocalKeyFrames.push_back(pKF);
//	pKF->mnLocalBAID = nTargetID;
//
//	int nn = 15;
//
//	const std::vector<UVR_SLAM::Frame*> vNeighKFs = pKF->GetConnectedKFs(nn);
//	for (int i = 0, iend = vNeighKFs.size(); i<iend; i++)
//	{
//		UVR_SLAM::Frame* pKFi = vNeighKFs[i];
//		if (pKFi->mnLocalBAID == nTargetID)
//			continue;
//		pKFi->mnLocalBAID = nTargetID;
//		lLocalKeyFrames.push_back(pKFi);
//	}
//
//	// Local MapPoints seen in Local KeyFrames
//	std::list<MapPoint*> lLocalMapPoints;
//	for (std::list<UVR_SLAM::Frame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
//	{
//		std::vector<MapPoint*> vpMPs = (*lit)->GetMapPoints();
//		for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
//		{
//			MapPoint* pMP = *vit;
//			if (!pMP)
//				continue;
//			if (pMP->isDeleted()) {
//				continue;
//			}
//			if (pMP->GetNumConnectedFrames() == 1)
//				continue;
//			if (pMP->mnLocalBAID == nTargetID)
//				continue;
//			lLocalMapPoints.push_back(pMP);
//			pMP->mnLocalBAID = nTargetID;
//		}
//	}
//
//	// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
//	std::list<UVR_SLAM::Frame*> lFixedCameras;
//	for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
//	{
//		UVR_SLAM::MapPoint* pMP = *lit;
//
//		if (!pMP)
//			continue;
//		if (pMP->isDeleted())
//			continue;
//
//		auto observations = pMP->GetConnedtedFrames();
//		//map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
//		for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
//		{
//			UVR_SLAM::Frame* pKFi = mit->first;
//
//			if (pKFi->mnLocalBAID != nTargetID && pKFi->mnFixedBAID != nTargetID)
//			{
//				pKFi->mnFixedBAID = nTargetID;
//				lFixedCameras.push_back(pKFi);
//			}
//		}
//	}
//
//	// Setup optimizer
//	g2o::SparseOptimizer optimizer;
//	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
//
//	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
//
//	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
//
//	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//	optimizer.setAlgorithm(solver);
//
//	if (pbStopFlag)
//		optimizer.setForceStopFlag(pbStopFlag);
//
//	unsigned long maxKFid = 0;
//
//	// Set Local KeyFrame vertices
//	for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
//	{
//		UVR_SLAM::Frame* pKFi = *lit;
//		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
//
//		cv::Mat R, t;
//		pKFi->GetPose(R, t);
//		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
//		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
//		t.copyTo(Tcw.col(3).rowRange(0, 3));
//
//		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
//		vSE3->setId(pKFi->GetKeyFrameID());
//		vSE3->setFixed(pKFi->GetKeyFrameID() == 0);
//		optimizer.addVertex(vSE3);
//		if (pKFi->GetKeyFrameID()>maxKFid)
//			maxKFid = pKFi->GetKeyFrameID();
//	}
//
//	// Set Fixed KeyFrame vertices
//	for (auto lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
//	{
//		UVR_SLAM::Frame* pKFi = *lit;
//		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
//
//		cv::Mat R, t;
//		pKFi->GetPose(R, t);
//		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
//		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
//		t.copyTo(Tcw.col(3).rowRange(0, 3));
//
//		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
//		vSE3->setId(pKFi->GetKeyFrameID());
//		vSE3->setFixed(true);
//		optimizer.addVertex(vSE3);
//		if (pKFi->GetKeyFrameID()>maxKFid)
//			maxKFid = pKFi->GetKeyFrameID();
//	}
//
//	// Set Local KeyFrame vertices
//	/*for (std::list<SemanticSLAM::PlaneInformation*>::iterator lit = lLocalPlanes.begin(), lend = lLocalPlanes.end(); lit != lend; lit++) {
//		SemanticSLAM::PlaneInformation* pPlane = *lit;
//		g2o::PlaneVertex* vPlane = new g2o::PlaneVertex();
//		vPlane->setEstimate(Converter::toVector6d(pPlane->GetPlaneParam()));
//		vPlane->setId(maxKFid + pPlane->mnPlaneID);
//		vPlane->setFixed(false);
//		optimizer.addVertex(vPlane);
//		if (pPlane->mnPlaneID > maxPlaneid)
//			maxPlaneid = pPlane->mnPlaneID;
//	}*/
//
//	// Set MapPoint vertices
//	const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size())*lLocalMapPoints.size();
//
//	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
//	vpEdgesMono.reserve(nExpectedSize);
//
//	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
//	vpEdgeKFMono.reserve(nExpectedSize);
//
//	std::vector<MapPoint*> vpMapPointEdgeMono;
//	vpMapPointEdgeMono.reserve(nExpectedSize);
//
//	std::vector<MapPoint*> vpPlaneEdgeMP;
//	std::vector<g2o::PlaneBAEdgeOnlyMapPoint*> vpPlaneEdges;
//	vpPlaneEdges.reserve(nExpectedSize);
//
//	const float thHuberMono = sqrt(5.991);
//	const float thHuberStereo = sqrt(7.815);
//
//	int nPlaneID = pMap->mpFloorPlane->mnPlaneID;
//	cv::Mat normal;
//	float dist;
//	pMap->mpFloorPlane->GetParam(normal, dist);
//
//	for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
//	{
//		MapPoint* pMP = *lit;
//
//		if (!pMP)
//			continue;
//		if (pMP->isDeleted())
//			continue;
//		
//		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
//		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
//		int id = pMP->mnMapPointID + maxKFid + 1;
//		vPoint->setId(id);
//		vPoint->setMarginalized(true);
//		optimizer.addVertex(vPoint);
//		const auto observations = pMP->GetConnedtedFrames();
//		//Set edges
//		
//		for (std::map<UVR_SLAM::Frame*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
//		{
//			UVR_SLAM::Frame* pKFi = mit->first;
//			/*if (pKFi->GetFrameID() > nTargetID)
//				continue;*/
//			if (pKFi->GetKeyFrameID() > maxKFid)
//				continue;
//			const cv::KeyPoint &kpUn = pKFi->mvKeyPoints[mit->second];
//
//			Eigen::Matrix<double, 2, 1> obs;
//			obs << kpUn.pt.x, kpUn.pt.y;
//
//			g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
//
//			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
//			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->GetKeyFrameID())));
//			e->setMeasurement(obs);
//			const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
//			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
//
//			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//			e->setRobustKernel(rk);
//			rk->setDelta(thHuberMono);
//
//			e->fx = pKFi->fx;
//			e->fy = pKFi->fy;
//			e->cx = pKFi->cx;
//			e->cy = pKFi->cy;
//
//			optimizer.addEdge(e);
//			vpEdgesMono.push_back(e);
//			vpEdgeKFMono.push_back(pKFi);
//			vpMapPointEdgeMono.push_back(pMP);
//		}
//		//set plane edge
//		if (pMP->GetPlaneID() == nPlaneID) {
//			g2o::PlaneBAEdgeOnlyMapPoint* e = new g2o::PlaneBAEdgeOnlyMapPoint();
//			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
//
//			e->normal[0] = normal.at<float>(0);
//			e->normal[1] = normal.at<float>(1);
//			e->normal[2] = normal.at<float>(2);
//			e->dist = dist;
//
//			/*cv::Mat Xw = pMP->GetWorldPos();
//			e->Xw[0] = Xw.at<float>(0);
//			e->Xw[1] = Xw.at<float>(1);
//			e->Xw[2] = Xw.at<float>(2);*/
//
//			optimizer.addEdge(e);
//			vpPlaneEdgeMP.push_back(pMP);
//			vpPlaneEdges.push_back(e);
//
//		}
//	}
//	
//	if (pbStopFlag)
//		if (*pbStopFlag)
//			return;
//
//	optimizer.initializeOptimization();
//	optimizer.optimize(5);
//
//	bool bDoMore = true;
//
//	if (pbStopFlag)
//		if (*pbStopFlag)
//			bDoMore = false;
//
//	if (bDoMore)
//	{
//
//		// Check inlier observations
//		for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
//		{
//			g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
//			MapPoint* pMP = vpMapPointEdgeMono[i];
//			if (!pMP)
//				continue;
//			if (pMP->isDeleted())
//				continue;
//
//			if (e->chi2()>5.991 || !e->isDepthPositive())
//			{
//				e->setLevel(1);
//			}
//
//			e->setRobustKernel(0);
//		}
//
//		// Optimize again without the outliers
//
//		optimizer.initializeOptimization(0);
//		optimizer.optimize(10);
//
//	}
//
//	std::vector<std::pair<UVR_SLAM::Frame*, MapPoint*> > vToErase;
//	vToErase.reserve(vpEdgesMono.size());
//
//	// Check inlier observations       
//	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
//	{
//		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
//		MapPoint* pMP = vpMapPointEdgeMono[i];
//		if (!pMP)
//			continue;
//		if (pMP->isDeleted())
//			continue;
//
//		if (e->chi2()>5.991 || !e->isDepthPositive())
//		{
//			UVR_SLAM::Frame* pKFi = vpEdgeKFMono[i];
//			vToErase.push_back(std::make_pair(pKFi, pMP));
//		}
//	}
//
//	for (size_t i = 0, iend = vpPlaneEdges.size(); i < iend; i++) {
//		g2o::PlaneBAEdgeOnlyMapPoint* e = vpPlaneEdges[i];
//		MapPoint* pMP = vpPlaneEdgeMP[i];
//		if (!pMP)
//			continue;
//		if (pMP->isDeleted())
//			continue;
//		//if(pMP->GetConnedtedFrames().size() > 1)
//		//std::cout << "err:" << pMP->GetConnedtedFrames().size() << std::endl;
//		if (e->chi2() > 0.001)
//			e->setLevel(1);
//		e->setRobustKernel(0);
//	}
//
//	if (!vToErase.empty())
//	{
//		for (size_t i = 0; i<vToErase.size(); i++)
//		{
//			UVR_SLAM::Frame* pKFi = vToErase[i].first;
//			MapPoint* pMPi = vToErase[i].second;
//			if (!pMPi)
//				continue;
//			if (pMPi->isDeleted())
//				continue;
//			pMPi->RemoveFrame(pKFi);
//
//			//pKFi->EraseMapPointMatch(pMPi);
//			//pMPi->EraseObservation(pKFi);
//
//		}
//	}
//
//	// Recover optimized data
//
//	//Keyframes
//	for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
//	{
//		UVR_SLAM::Frame* pKF = *lit;
//		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetKeyFrameID()));
//		g2o::SE3Quat SE3quat = vSE3->estimate();
//
//		cv::Mat R, t;
//		cv::Mat Tcw = Converter::toCvMat(SE3quat);
//		R = Tcw.rowRange(0, 3).colRange(0, 3);
//		t = Tcw.rowRange(0, 3).col(3);
//		pKF->SetPose(R, t);
//	}
//
//	//Points
//	for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
//	{
//		MapPoint* pMP = *lit;
//		if (!pMP)
//			continue;
//		if (pMP->isDeleted())
//			continue;
//		//remove
//		int nConnectedThresh = 2;
//		/////////평면인경우 쓰레시 홀딩 조절
//		//if (pMP->GetMapPointType() == UVR_SLAM::MapPointType::PLANE_MP)
//		//	nConnectedThresh = 1;
//		/////////평면인경우 쓰레시 홀딩 조절
//		//else if (pMP->isNewMP())
//		//	nConnectedThresh = 1;
//		if (pMP->GetNumConnectedFrames() < nConnectedThresh) {
//			pMP->SetDelete(true);
//			pMP->Delete();
//			//pWindow->SetMapPoint(nullptr, idx);
//			//pWindow->SetBoolInlier(false, idx);
//			continue;
//		}
//		
//		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + 1));
//		
//		//cv::Mat tempori = pMP->GetWorldPos();
//		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
//		pMP->UpdateNormalAndDepth();
//		
//		/*if(pMP->mnFirstKeyFrameID == pKF->GetKeyFrameID() && pMP->GetPlaneID() == nPlaneID){
//			cv::Mat temp = pMP->GetWorldPos();
//			tempori = normal.t()*tempori;
//			temp = normal.t()*temp;
//			float val1 = tempori.at<float>(0) + dist;
//			float val2 = temp.at<float>(0)+dist;
//			std::cout << val1 << ", " << val2 << std::endl;
//		}*/
//	}
//	////////////////////////////////////////////////////////////////////
//}
//void UVR_SLAM::Optimization::InitBundleAdjustment(const std::vector<UVR_SLAM::Frame*> &vpKFs, const std::vector<UVR_SLAM::MapPoint *> &vpMP, int nIterations)
//{
//	std::vector<bool> vbNotIncludedMP;
//	vbNotIncludedMP.resize(vpMP.size());
//
//	g2o::SparseOptimizer optimizer;
//	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
//
//	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
//
//	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
//
//	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//	optimizer.setAlgorithm(solver);
//
//	long unsigned int maxKFid = 0;
//
//	// Set KeyFrame vertices
//	for (size_t i = 0; i<vpKFs.size(); i++)
//	{
//		UVR_SLAM::Frame* pKF = vpKFs[i];
//		
//		cv::Mat R, t;
//		pKF->GetPose(R, t);
//		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
//		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
//		t.copyTo(Tcw.col(3).rowRange(0, 3));
//		
//		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
//		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
//		vSE3->setId(pKF->GetKeyFrameID());
//		vSE3->setFixed(pKF->GetKeyFrameID() == 0);
//		optimizer.addVertex(vSE3);
//		if (pKF->GetKeyFrameID()>maxKFid)
//			maxKFid = pKF->GetKeyFrameID();
//	}
//
//	const float thHuber2D = sqrt(5.99);
//	const float thHuber3D = sqrt(7.815);
//
//	// Set MapPoint vertices
//	for (size_t i = 0; i<vpMP.size(); i++)
//	{
//		MapPoint* pMP = vpMP[i];
//		if (pMP->isDeleted())
//			continue;
//		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
//		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
//		const int id = pMP->mnMapPointID + maxKFid + 1;
//		vPoint->setId(id);
//		vPoint->setMarginalized(true);
//		optimizer.addVertex(vPoint);
//
//		//const std::map<UVR_SLAM::Frame*, int> observations = pMP->GetConnedtedFrames();
//		const auto observations = pMP->GetConnedtedDenseFrames();
//
//		int nEdges = 0;
//		//SET EDGES
//		for (auto mit = observations.begin(); mit != observations.end(); mit++)
//		{
//
//			UVR_SLAM::Frame* pKF = mit->first;
//			if (pKF->GetKeyFrameID()>maxKFid)
//				continue;
//
//			nEdges++;
//
//			/*const cv::KeyPoint &kpUn = pKF->mvKeyPoints[mit->second];
//
//			Eigen::Matrix<double, 2, 1> obs;
//			obs << kpUn.pt.x, kpUn.pt.y;
//
//			g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
//
//			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
//			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->GetKeyFrameID())));
//			e->setMeasurement(obs);
//			const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
//			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);*/
//
//			const auto pt = mit->second;
//			Eigen::Matrix<double, 2, 1> obs;
//			obs << pt.x, pt.y;
//
//			g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
//			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
//			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->GetKeyFrameID())));
//			e->setMeasurement(obs);
//			const float &invSigma2 = 1.0;
//			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
//
//			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//			e->setRobustKernel(rk);
//			rk->setDelta(thHuber2D);
//
//			e->fx = pKF->fx;
//			e->fy = pKF->fy;
//			e->cx = pKF->cx;
//			e->cy = pKF->cy;
//			optimizer.addEdge(e);
//		}
//
//		if (nEdges == 0)
//		{
//			optimizer.removeVertex(vPoint);
//			vbNotIncludedMP[i] = true;
//		}
//		else
//		{
//			vbNotIncludedMP[i] = false;
//		}
//	}
//
//	// Optimize!
//	optimizer.initializeOptimization();
//	optimizer.optimize(nIterations);
//
//	// Recover optimized data
//
//	//Keyframes
//	for (size_t i = 0; i<vpKFs.size(); i++)
//	{
//		UVR_SLAM::Frame* pKF = vpKFs[i];
//		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetKeyFrameID()));
//		g2o::SE3Quat SE3quat = vSE3->estimate();
//		
//		cv::Mat R, t;
//		cv::Mat Tcw = Converter::toCvMat(SE3quat);
//		R = Tcw.rowRange(0, 3).colRange(0, 3);
//		t = Tcw.rowRange(0, 3).col(3);
//		std::cout<<"ID::Before::"<< pKF->GetKeyFrameID() << pKF->GetRotation() << ", " << pKF->GetTranslation().t() << std::endl;
//		pKF->SetPose(R, t);
//		//if (i == 0)
//		{
//			std::cout <<"ID::After::"<<pKF->GetKeyFrameID()<< R << ", " << t.t() << std::endl;
//		}
//	}
//
//	//Points
//	for (size_t i = 0; i<vpMP.size(); i++)
//	{
//		if (vbNotIncludedMP[i])
//			continue;
//
//		MapPoint* pMP = vpMP[i];
//
//		if (pMP->isDeleted())
//			continue;
//		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + 1));
//		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
//	}
//
//}
////////백업
///////////////////////////////////////////////////////////





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////Opticalflow 버전용
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

	const float deltaMono = sqrt(5.991);
	const float deltaStereo = sqrt(7.815);

	{
		for (int i = 0; i<N; i++)
		{
			MapPoint* pMP = vpMPs[i];
			if (!pMP || pMP->isDeleted() || pMP->GetRecentTrackingFrameID() != nTargetID)
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
void UVR_SLAM::Optimization::OpticalLocalBundleAdjustmentWithPlane(UVR_SLAM::MapOptimizer* pMapOptimizer, PlaneProcessInformation* pPlaneInfo, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::Frame*> vpKFs, std::vector<UVR_SLAM::Frame*> vpFixedKFs) {

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	bool bStopBA = pMapOptimizer->isStopBA();
	if (bStopBA)
		optimizer.setForceStopFlag(&bStopBA);

	unsigned long maxKFid = 0;

	for (int i = 0; i < vpKFs.size(); i++) {
		auto pKFi = vpKFs[i];
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
	for (int i = 0; i < vpFixedKFs.size(); i++)
	{
		UVR_SLAM::Frame* pKFi = vpFixedKFs[i];
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

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////평면 조인트 최적화
	////평면 조인트 최적화 관련
	std::vector<MapPoint*> vpPlaneEdgeMP;
	std::vector<UVR_SLAM::PlaneInformation*> vpPlaneEdgePlane;
	std::vector<g2o::PlaneBAEdge*> vpPlaneEdge;
	////평면 조인트 최적화 관련
	////Plane Vertex 추가
	int maxPlaneId = 0;
	bool bPlane = false;
	std::vector<g2o::PlaneVertex*> vPlaneVertices;
	
	/*auto pFloor = pPlaneInfo->GetPlane(1);
	auto pCeil = pPlaneInfo->GetPlane(2);*/

	auto planes = pPlaneInfo->GetPlanes();
	bPlane = true;
	for (auto iter = planes.begin(); iter != planes.end(); iter++) {
		auto plane = iter->second;
		auto pid = iter->first;
		g2o::PlaneVertex* vertex = new g2o::PlaneVertex();
		vertex->setEstimate(Converter::toVector6d(plane->GetParam()));
		vertex->setId(maxKFid + pid);
		vertex->setFixed(pid == 1);
		optimizer.addVertex(vertex);

		//////correlation
		if (pid >= 2) {
			int cor = 2;
			if (pid > 2)
				cor = 1;
			g2o::PlaneCorrelationBAEdge* e = new g2o::PlaneCorrelationBAEdge();
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(1 + maxKFid)));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pid + maxKFid)));
			e->type = cor;
			e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
			optimizer.addEdge(e);
		}
		//////correlation
		
		vPlaneVertices.push_back(vertex);
		if (pid > maxPlaneId)
			maxPlaneId = pid;
	}

	/*g2o::PlaneVertex *vpFloor, * vpCeil;
	if (pFloor) {
		bPlane = true;
		vpFloor = new g2o::PlaneVertex();
		vpFloor->setEstimate(Converter::toVector6d(pFloor->GetParam()));
		vpFloor->setId(maxKFid + pFloor->mnPlaneID);
		vpFloor->setFixed(false);
		optimizer.addVertex(vpFloor);
		vPlaneVertices.push_back(vpFloor);
		if (pFloor->mnPlaneID > maxPlaneId)
			maxPlaneId = pFloor->mnPlaneID;
	}
	if (pCeil) {
		vpCeil = new g2o::PlaneVertex();
		vpCeil->setEstimate(Converter::toVector6d(pCeil->GetParam()));
		vpCeil->setId(maxKFid + pCeil->mnPlaneID);
		vpCeil->setFixed(false);
		optimizer.addVertex(vpCeil);
		vPlaneVertices.push_back(vpCeil);
		if (pCeil->mnPlaneID > maxPlaneId)
			maxPlaneId = pCeil->mnPlaneID;
	}*/
	
	////Plane Vertex 추가
	////////////평면 조인트 최적화
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Set MapPoint vertices
	const int nExpectedSize = (vpKFs.size() + vpFixedKFs.size())*vpMPs.size();

	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	std::vector<MapPoint*> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);
	const float thHuberPlane = sqrt(0.01);
	
	for (int i = 0; i < vpMPs.size(); i++)
	{
		MapPoint* pMP = vpMPs[i];
		if (!pMP || pMP->isDeleted())
			continue;
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		int id = pMP->mnMapPointID +maxPlaneId + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const auto observations = pMP->GetConnedtedFrames();

		//Set edges
		for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto pMatch = mit->first;
			auto pKFi = pMatch->mpRefFrame;
			if (pKFi->GetKeyFrameID() > maxKFid)
				continue;
			int idx = mit->second;
			auto pt = pMatch->mvMatchingPts[idx];
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
		/////Set Planar Edge
		//int ptype = pMP->GetRecentLayoutFrameID();
		int ptype = pMP->GetPlaneID();
		auto findres = planes.find(ptype);
		if (findres != planes.end()) {
			int pid = findres->first;

			g2o::PlaneBAEdge* e = new g2o::PlaneBAEdge();
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pid + maxKFid))); ///이부분은 추후 평면별로 변경이 필요함.
			e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(thHuberPlane);

			optimizer.addEdge(e);
			vpPlaneEdge.push_back(e);
			vpPlaneEdgeMP.push_back(pMP);
		}
		//if (bPlane && ptype > 0) {
		//	
		//	int pid;
		//	if (ptype == 1 && pFloor) {
		//		pid = pFloor->mnPlaneID;
		//	}
		//	else if (ptype == 2 && pCeil) {
		//		pid = pCeil->mnPlaneID;
		//	}
		//	else {
		//		continue;
		//	}
		//	
		//	g2o::PlaneBAEdge* e = new g2o::PlaneBAEdge();
		//	e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
		//	e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pid + maxKFid))); ///이부분은 추후 평면별로 변경이 필요함.
		//	e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

		//	g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		//	e->setRobustKernel(rk);
		//	rk->setDelta(thHuberPlane);

		//	optimizer.addEdge(e);
		//	vpPlaneEdge.push_back(e);
		//	vpPlaneEdgeMP.push_back(pMP);
		//	//vpPlaneEdgePlane.push_back(pEstimator->GetPlane(pMP->GetPlaneID()));
		//}
		/////Set Planar Edge
	}
	
	bStopBA = pMapOptimizer->isStopBA();
	if (bStopBA)
		return;

	optimizer.initializeOptimization();
	optimizer.optimize(5);

	bStopBA = pMapOptimizer->isStopBA();
	bool bDoMore = true;
	if (bStopBA)
		bDoMore = false;

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

		for (size_t i = 0, iend = vpPlaneEdge.size(); i < iend; i++) {
			g2o::PlaneBAEdge* e = vpPlaneEdge[i];
			MapPoint* pMP = vpPlaneEdgeMP[i];
			if (pMP->isDeleted())
				continue;
			if (e->chi2() > 0.01) {
				pMP->SetPlaneID(0);
				//std::cout << e->chi2() <<"::"<<pMP->GetConnedtedFrames().size()<< std::endl;
				//e->setLevel(1);
			}
			else {
				//pMP->SetPlaneID(1);
			}
			e->setRobustKernel(0);
		}

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);

	}

	std::vector<std::pair<UVR_SLAM::MatchInfo*, MapPoint*> > vToErase;
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
			vToErase.push_back(std::make_pair(pKFi->mpMatchInfo, pMP));
		}
	}
	///////////////////////////////
	//////테스트 코드
	cv::Mat mat = cv::Mat::zeros(0, 4, CV_32FC1);
	///////////////////////////////
	int nBadPlane = 0;
	std::vector<UVR_SLAM::MapPoint*> vErasePlanarMPs;
	for (size_t i = 0, iend = vpPlaneEdge.size(); i < iend; i++) {
		g2o::PlaneBAEdge* e = vpPlaneEdge[i];
		MapPoint* pMP = vpPlaneEdgeMP[i];

		////////테스트
		//cv::Mat temp = pMP->GetWorldPos();
		//temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		//mat.push_back(temp.t());
		////////테스트
		if (pMP->isDeleted())
			continue;
		if (e->chi2() > 0.05) {
			nBadPlane++;
			pMP->SetPlaneID(0);
			//std::cout << e->chi2() <<"::"<<pMP->GetConnedtedFrames().size()<< std::endl;
			//e->setLevel(1);
		}
		else {
			//pMP->SetPlaneID(1);
		}
		if (e->chi2() > 0.1) {
			vErasePlanarMPs.push_back(pMP);
		}
		
		//e->setRobustKernel(0);
	}
	std::cout << "ba::plane::" <<planes.size()<<"::"<< nBadPlane << ", " << vpPlaneEdge.size() << std::endl;
	////////테스트
	//if (pPlaneInfo && mat.rows > 0) {
	//	cv::Mat pMat = pPlaneInfo->GetParam();
	//	std::cout << "BA::PLANE::BEFORE::PARAM::" << pMat.t()<<", "<< vpPlaneEdge.size()<< std::endl;
	//	auto val = cv::sum(abs(mat*pMat));
	//	std::cout << "BA::PLANE::BEFORE::" << val.val[0]/ mat.rows << std::endl;
	//}
	////////테스트
	
	if (!vToErase.empty())
	{
		for (size_t i = 0; i<vToErase.size(); i++)
		{
			auto pMatch = vToErase[i].first;
			MapPoint* pMPi = vToErase[i].second;
			pMPi->RemoveFrame(pMatch);
		}
	}
	//////평면 거리가 큰 포인트 삭제
	/*if (!vErasePlanarMPs.empty()) {
		for (int i = 0; i < vErasePlanarMPs.size(); i++) {
			auto pMP = vErasePlanarMPs[i];
			pMP->Delete();
		}
	}*/
	//////평면 거리가 큰 포인트 삭제

	// Recover optimized data
	//Keyframes
	for (int i = 0; i < vpKFs.size(); i++)
	{
		UVR_SLAM::Frame* pKF = vpKFs[i];
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetKeyFrameID()));
		g2o::SE3Quat SE3quat = vSE3->estimate();

		cv::Mat R, t;
		cv::Mat Tcw = Converter::toCvMat(SE3quat);
		R = Tcw.rowRange(0, 3).colRange(0, 3);
		t = Tcw.rowRange(0, 3).col(3);
		pKF->SetPose(R, t);
	}
	
	for (int i = 0; i < vpMPs.size(); i++)
	{
		MapPoint* pMP = vpMPs[i];
		if (!pMP || pMP->isDeleted())
			continue;
		if (pMP->GetConnedtedFrames().size() < 3)
			std::cout << "BA::커넥티드 에러" << std::endl;
		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + maxPlaneId + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();

	}
	
	///////////////////////////////////////////////
	////평면 값 복원
	for (auto iter = planes.begin(); iter != planes.end(); iter++) {
		int pid = iter->first;
		auto plane = iter->second;
		g2o::PlaneVertex* vPlane = static_cast<g2o::PlaneVertex*>(optimizer.vertex(pid + maxKFid));
		plane->SetParam(Converter::toCvMat(vPlane->estimate()).rowRange(0, 4));
	}
	/*if (pFloor) {
		g2o::PlaneVertex* vPlane = static_cast<g2o::PlaneVertex*>(optimizer.vertex(pFloor->mnPlaneID + maxKFid));
		pFloor->SetParam(Converter::toCvMat(vPlane->estimate()).rowRange(0, 4));
	}
	if (pCeil) {
		g2o::PlaneVertex* vPlane = static_cast<g2o::PlaneVertex*>(optimizer.vertex(pCeil->mnPlaneID + maxKFid));
		pCeil->SetParam(Converter::toCvMat(vPlane->estimate()).rowRange(0, 4));
	}*/
	//if (pPlaneInfo) {

	//	////////테스트
	//	//cv::Mat mat = cv::Mat::zeros(0, 4, CV_32FC1);
	//	//for (size_t i = 0, iend = vpPlaneEdge.size(); i < iend; i++) {
	//	//	g2o::PlaneBAEdge* e = vpPlaneEdge[i];
	//	//	MapPoint* pMP = vpPlaneEdgeMP[i];
	//	//	if (!pMP || pMP->isDeleted())
	//	//		continue;

	//	//	cv::Mat temp = pMP->GetWorldPos();
	//	//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
	//	//	mat.push_back(temp.t());
	//	//}
	//	//std::cout << mat.size() << std::endl;
	//	////////테스트

	//	/*g2o::PlaneVertex* vPlane = static_cast<g2o::PlaneVertex*>(optimizer.vertex(pPlaneInfo->mnPlaneID + maxKFid));
	//	pPlaneInfo->SetParam(Converter::toCvMat(vPlane->estimate()).rowRange(0, 4));*/

	//	////////테스트
	//	//if (mat.rows > 0){
	//	//	cv::Mat pMat = pPlaneInfo->GetParam();
	//	//	auto val = cv::sum(abs(mat*pMat));
	//	//	std::cout << "BA::PLANE::AFTER::PARAM::" << pMat.t() << std::endl;
	//	//	std::cout << "BA::PLANE::AFTER::" << val.val[0] / mat.rows << std::endl;
	//	//}
	//	////////테스트
	//}
	////평면 값 복원
	///////////////////////////////////////////////
}
void UVR_SLAM::Optimization::OpticalLocalBundleAdjustment(UVR_SLAM::MapOptimizer* pMapOptimizer, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::Frame*> vpKFs, std::vector<UVR_SLAM::Frame*> vpFixedKFs) {

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	bool bStopBA = pMapOptimizer->isStopBA();
	if (bStopBA)
		optimizer.setForceStopFlag(&bStopBA);

	unsigned long maxKFid = 0;
	
	for (int i = 0; i < vpKFs.size(); i++) {
		auto pKFi = vpKFs[i];
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
	for (int i = 0; i < vpFixedKFs.size(); i++)
	{
		UVR_SLAM::Frame* pKFi = vpFixedKFs[i];
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
	
	// Set MapPoint vertices
	const int nExpectedSize = (vpKFs.size()+vpFixedKFs.size())*vpMPs.size();

	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	std::vector<MapPoint*> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);
	
	for (int i =0; i < vpMPs.size(); i++)
	{
		MapPoint* pMP = vpMPs[i];
		if (!pMP || pMP->isDeleted())
			continue;
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		int id = pMP->mnMapPointID + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const auto observations = pMP->GetConnedtedFrames();

		//Set edges
		for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto pMatch = mit->first;
			auto pKFi = pMatch->mpRefFrame;
			if (pKFi->GetKeyFrameID() > maxKFid)
				continue;
			int idx = mit->second;
			auto pt = pMatch->mvMatchingPts[idx];
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

	bStopBA = pMapOptimizer->isStopBA();
	if (bStopBA)
		return;

	optimizer.initializeOptimization();
	optimizer.optimize(5);

	bStopBA = pMapOptimizer->isStopBA();
	bool bDoMore = true;
	if (bStopBA)
		bDoMore = false;

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

	std::vector<std::pair<UVR_SLAM::MatchInfo*, MapPoint*> > vToErase;
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
			vToErase.push_back(std::make_pair(pKFi->mpMatchInfo, pMP));
		}
	}

	if (!vToErase.empty())
	{
		for (size_t i = 0; i<vToErase.size(); i++)
		{
			auto pMatch = vToErase[i].first;
			MapPoint* pMPi = vToErase[i].second;
			pMPi->RemoveFrame(pMatch);
		}
	}

	// Recover optimized data

	//Keyframes
	for (int i = 0; i < vpKFs.size(); i++)
	{
		UVR_SLAM::Frame* pKF = vpKFs[i];
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetKeyFrameID()));
		g2o::SE3Quat SE3quat = vSE3->estimate();

		cv::Mat R, t;
		cv::Mat Tcw = Converter::toCvMat(SE3quat);
		R = Tcw.rowRange(0, 3).colRange(0, 3);
		t = Tcw.rowRange(0, 3).col(3);
		pKF->SetPose(R, t);
	}

	for (int i = 0; i < vpMPs.size(); i++)
	{
		MapPoint* pMP = vpMPs[i];
		if (!pMP || pMP->isDeleted())
			continue;

		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();
	}
}
void UVR_SLAM::Optimization::OpticalLocalBundleAdjustment(UVR_SLAM::MapOptimizer* pMapOptimizer, UVR_SLAM::Frame* pKF, UVR_SLAM::FrameWindow* pWindow) {
	//// Local KeyFrames: First Breath Search from Current Keyframe
	//std::list<UVR_SLAM::Frame*> lLocalKeyFrames;

	//std::cout << "ba::ssssss" << std::endl;

	//int nTargetID = pKF->GetFrameID();
	//lLocalKeyFrames.push_back(pKF);
	//pKF->mnLocalBAID = nTargetID;

	//int nn = 15;

	//const std::vector<UVR_SLAM::Frame*> vNeighKFs = pKF->GetConnectedKFs(nn);
	//std::cout << "ba::connected kf::" << vNeighKFs.size() << std::endl;
	//for (int i = 0, iend = vNeighKFs.size(); i<iend; i++)
	//{
	//	UVR_SLAM::Frame* pKFi = vNeighKFs[i];
	//	if (pKFi->mnLocalBAID == nTargetID)
	//		continue;
	//	pKFi->mnLocalBAID = nTargetID;
	//	lLocalKeyFrames.push_back(pKFi);
	//}
	//std::cout << "ba::aaaa" << std::endl;
	//// Local MapPoints seen in Local KeyFrames

	////dense mp의 커넥티드 수는 고려하지 않음.
	//std::list<MapPoint*> lLocalMapPoints;
	//for (std::list<UVR_SLAM::Frame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	//{
	//	std::vector<MapPoint*> vpMPs = (*lit)->GetDenseVectors();
	//	for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
	//	{
	//		MapPoint* pMP = *vit;
	//		if (!pMP)
	//			continue;
	//		if (pMP->isDeleted()) {
	//			continue;
	//		}
	//		/*if (pMP->GetNumConnectedFrames() <= 2)
	//			continue;*/
	//		if (pMP->mnLocalBAID == nTargetID)
	//			continue;
	//		lLocalMapPoints.push_back(pMP);
	//		pMP->mnLocalBAID = nTargetID;
	//	}
	//}
	//std::cout << "ba::bbbbb" << std::endl;
	//
	//// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
	//std::list<UVR_SLAM::Frame*> lFixedCameras;
	//for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	//{
	//	UVR_SLAM::MapPoint* pMP = *lit;
	//	auto observations = pMP->GetConnedtedDenseFrames();
	//	//map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
	//	for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
	//	{
	//		UVR_SLAM::Frame* pKFi = mit->first;

	//		if (pKFi->mnLocalBAID != nTargetID && pKFi->mnFixedBAID != nTargetID)
	//		{
	//			pKFi->mnFixedBAID = nTargetID;
	//			lFixedCameras.push_back(pKFi);
	//		}
	//	}
	//}
	//std::cout << "ba::ddddd" << std::endl;
	//// Setup optimizer
	//g2o::SparseOptimizer optimizer;
	//g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	//linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	//g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	//g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	//optimizer.setAlgorithm(solver);

	//std::cout << "ba::eeee" << std::endl;

	//bool bStopBA = pMapOptimizer->isStopBA();
	//if (bStopBA)
	//	optimizer.setForceStopFlag(&bStopBA);

	//unsigned long maxKFid = 0;
	//std::cout << "ba::ffff" << std::endl;
	//// Set Local KeyFrame vertices
	//for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	//{
	//	UVR_SLAM::Frame* pKFi = *lit;
	//	g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

	//	cv::Mat R, t;
	//	pKFi->GetPose(R, t);
	//	cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
	//	R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
	//	t.copyTo(Tcw.col(3).rowRange(0, 3));

	//	vSE3->setEstimate(Converter::toSE3Quat(Tcw));
	//	vSE3->setId(pKFi->GetKeyFrameID());
	//	vSE3->setFixed(pKFi->GetKeyFrameID() == 0);
	//	optimizer.addVertex(vSE3);
	//	if (pKFi->GetKeyFrameID()>maxKFid)
	//		maxKFid = pKFi->GetKeyFrameID();
	//}
	//std::cout << "ba::ggg::" << maxKFid << std::endl;
	//// Set Fixed KeyFrame vertices
	//for (auto lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
	//{
	//	UVR_SLAM::Frame* pKFi = *lit;
	//	g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

	//	cv::Mat R, t;
	//	pKFi->GetPose(R, t);
	//	cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
	//	R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
	//	t.copyTo(Tcw.col(3).rowRange(0, 3));

	//	vSE3->setEstimate(Converter::toSE3Quat(Tcw));
	//	vSE3->setId(pKFi->GetKeyFrameID());
	//	vSE3->setFixed(true);
	//	optimizer.addVertex(vSE3);
	//	if (pKFi->GetKeyFrameID()>maxKFid)
	//		maxKFid = pKFi->GetKeyFrameID();
	//}
	//std::cout << "ba::hhh::" << maxKFid << std::endl;
	//// Set MapPoint vertices
	//const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size())*lLocalMapPoints.size();

	//std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	//vpEdgesMono.reserve(nExpectedSize);

	//std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	//vpEdgeKFMono.reserve(nExpectedSize);

	//std::vector<MapPoint*> vpMapPointEdgeMono;
	//vpMapPointEdgeMono.reserve(nExpectedSize);

	//const float thHuberMono = sqrt(5.991);
	//const float thHuberStereo = sqrt(7.815);
	//std::cout << "ba::iii" << std::endl;
	//for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	//{
	//	MapPoint* pMP = *lit;
	//	if (!pMP)
	//		continue;
	//	if (pMP->isDeleted())
	//		continue;
	//	g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
	//	vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
	//	int id = pMP->mnMapPointID + maxKFid + 1;
	//	vPoint->setId(id);
	//	vPoint->setMarginalized(true);
	//	optimizer.addVertex(vPoint);

	//	const auto observations = pMP->GetConnedtedDenseFrames();

	//	//Set edges
	//	for (std::map<UVR_SLAM::Frame*, cv::Point2f>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
	//	{
	//		UVR_SLAM::Frame* pKFi = mit->first;
	//		if (pKFi->GetKeyFrameID() > maxKFid)
	//			continue;

	//		auto pt = mit->second;
	//		Eigen::Matrix<double, 2, 1> obs;
	//		obs << pt.x, pt.y;

	//		g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
	//		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
	//		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->GetKeyFrameID())));
	//		e->setMeasurement(obs);
	//		const float &invSigma2 = 1.0;
	//		e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

	//		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
	//		e->setRobustKernel(rk);
	//		rk->setDelta(thHuberMono);

	//		e->fx = pKFi->fx;
	//		e->fy = pKFi->fy;
	//		e->cx = pKFi->cx;
	//		e->cy = pKFi->cy;

	//		optimizer.addEdge(e);
	//		vpEdgesMono.push_back(e);
	//		vpEdgeKFMono.push_back(pKFi);
	//		vpMapPointEdgeMono.push_back(pMP);
	//	}

	//}

	//std::cout << "ba::setting::end" << std::endl;

	//bStopBA = pMapOptimizer->isStopBA();
	//if (bStopBA)
	//	return;
	///*if (pbStopFlag)
	//if (*pbStopFlag)
	//return;*/

	//std::cout << "ba::setting::end2" << std::endl;

	//optimizer.initializeOptimization();
	//optimizer.optimize(5);

	//std::cout << "ba::optimize::end" << std::endl;

	//bStopBA = pMapOptimizer->isStopBA();

	//std::cout << "ba::optimize::end2" << std::endl;

	//bool bDoMore = true;
	//if (bStopBA)
	//	bDoMore = false;
	///*if (pbStopFlag)
	//if (*pbStopFlag)
	//bDoMore = false;*/

	//if (bDoMore)
	//{

	//	// Check inlier observations
	//	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	//	{
	//		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
	//		MapPoint* pMP = vpMapPointEdgeMono[i];

	//		if (pMP->isDeleted())
	//			continue;

	//		if (e->chi2()>5.991 || !e->isDepthPositive())
	//		{
	//			e->setLevel(1);
	//		}

	//		e->setRobustKernel(0);
	//	}

	//	// Optimize again without the outliers

	//	optimizer.initializeOptimization(0);
	//	optimizer.optimize(10);

	//}

	//std::cout << "ba::optimize::end3" << std::endl;

	//std::vector<std::pair<UVR_SLAM::Frame*, MapPoint*> > vToErase;
	//vToErase.reserve(vpEdgesMono.size());

	//// Check inlier observations       
	//for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	//{
	//	g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
	//	MapPoint* pMP = vpMapPointEdgeMono[i];

	//	if (pMP->isDeleted())
	//		continue;

	//	if (e->chi2()>5.991 || !e->isDepthPositive())
	//	{
	//		UVR_SLAM::Frame* pKFi = vpEdgeKFMono[i];
	//		vToErase.push_back(std::make_pair(pKFi, pMP));
	//	}
	//}

	//std::cout << "ba::check::inlier::end" << std::endl;

	//if (!vToErase.empty())
	//{
	//	for (size_t i = 0; i<vToErase.size(); i++)
	//	{
	//		UVR_SLAM::Frame* pKFi = vToErase[i].first;
	//		MapPoint* pMPi = vToErase[i].second;
	//		pMPi->RemoveDenseFrame(pKFi);
	//	}
	//}

	//std::cout << "ba::erase::end" << std::endl;

	//// Recover optimized data

	////Keyframes
	//for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
	//{
	//	UVR_SLAM::Frame* pKF = *lit;
	//	g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->GetKeyFrameID()));
	//	g2o::SE3Quat SE3quat = vSE3->estimate();

	//	cv::Mat R, t;
	//	cv::Mat Tcw = Converter::toCvMat(SE3quat);
	//	R = Tcw.rowRange(0, 3).colRange(0, 3);
	//	t = Tcw.rowRange(0, 3).col(3);
	//	pKF->SetPose(R, t);
	//}

	//std::cout << "ba::restore::kf::end" << std::endl;

	////Points
	//for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
	//{
	//	MapPoint* pMP = *lit;
	//	if (!pMP || pMP->isDeleted())
	//		continue;

	//	////remove
	//	//int nConnectedThresh = 2;
	//	////////////////평면일 경우 쓰레시값 조절
	//	////if (pMP->GetMapPointType() == UVR_SLAM::MapPointType::PLANE_MP){
	//	////	//바로 만들어진 포인트는 삭제 금지.
	//	////	if (pMP->mnFirstKeyFrameID == nTargetID)
	//	////		continue;
	//	////	nConnectedThresh = 1;
	//	////}
	//	////////////////평면일 경우 쓰레시값 조절
	//	////else if (pMP->isNewMP())
	//	////	nConnectedThresh = 1;
	//	//int nConncted = 0;
	//	//if (pMP->GetMapPointType() == UVR_SLAM::PLANE_DENSE_MP) {
	//	//	nConncted = pMP->GetNumDensedFrames();
	//	//}
	//	//else {
	//	//	nConncted = pMP->GetNumConnectedFrames();
	//	//}
	//	//if (nConncted <= nConnectedThresh) {
	//	//	pMP->SetDelete(true);
	//	//	pMP->Delete();
	//	//	//pWindow->SetMapPoint(nullptr, idx);
	//	//	//pWindow->SetBoolInlier(false, idx);
	//	//	continue;
	//	//}

	//	g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + 1));
	//	pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
	//	pMP->UpdateNormalAndDepth();
	//}
	//std::cout << "ba::restore::mp::end" << std::endl;
}
////Opticalflow 버전용
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////