#include <PlanarOptimization.h>
#include <Converter.h>
#include <map>
#include <opencv2/core/eigen.hpp>
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


void UVR_SLAM::PlanarOptimization::OpticalLocalBundleAdjustmentWithPlane(UVR_SLAM::MapOptimizer* pMapOptimizer, PlaneProcessInformation* pPlaneInfo, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::Frame*> vpKFs, std::vector<UVR_SLAM::Frame*> vpFixedKFs) {

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
	int nTargetID = vpKFs[0]->mnLocalBAID;
	for (int i = 0; i < vpKFs.size(); i++) {
		auto pKFi = vpKFs[i];
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();

		cv::Mat R, t;
		pKFi->GetPose(R, t);
		cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
		R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
		t.copyTo(Tcw.col(3).rowRange(0, 3));

		vSE3->setEstimate(Converter::toSE3Quat(Tcw));
		vSE3->setId(pKFi->mnKeyFrameID);
		vSE3->setFixed(pKFi->mnKeyFrameID == 0);
		optimizer.addVertex(vSE3);
		if (pKFi->mnKeyFrameID>maxKFid)
			maxKFid = pKFi->mnKeyFrameID;
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
		vSE3->setId(pKFi->mnKeyFrameID);
		vSE3->setFixed(true);
		optimizer.addVertex(vSE3);
		if (pKFi->mnKeyFrameID>maxKFid)
			maxKFid = pKFi->mnKeyFrameID;
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

	std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpPlanarPoseEdges;

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
		int id = pMP->mnMapPointID + maxPlaneId + maxKFid + 1;
		int octave = pMP->mnOctave;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const auto observations = pMP->GetConnedtedFrames();

		int ptype = pMP->GetPlaneID();
		auto findres = planes.find(ptype);
		bool bPlanarMP = false;
		if (findres != planes.end()) {
			int pid = findres->first;

			g2o::PlaneBAEdge* e = new g2o::PlaneBAEdge();
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pid + maxKFid))); ///이부분은 추후 평면별로 변경이 필요함.
			e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(thHuberPlane);
			
			e->computeError();
			optimizer.addEdge(e);
			vpPlaneEdge.push_back(e);
			vpPlaneEdgeMP.push_back(pMP);
			bPlanarMP = true;
		}

		//Set edges
		////평면에 포함되는 경우에는 엣지를 다르게 설정
		for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto pMatch = mit->first;
			auto pKFi = pMatch->mpRefFrame;
			if (pKFi->mnLocalBAID != nTargetID)
				continue;
			int idx = mit->second;
			auto pt = pMatch->mvMatchingPts[idx];
			Eigen::Matrix<double, 2, 1> obs;
			obs << pt.x, pt.y;
			if (bPlanarMP) {
				g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnKeyFrameID)));
				e->setMeasurement(obs);
				int octave = pMP->mnOctave;
				const float invSigma2 = pKFi->mvInvLevelSigma2[octave];
				e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

				/*g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(thHuberMono);*/

				e->fx = pKFi->fx;
				e->fy = pKFi->fy;
				e->cx = pKFi->cx;
				e->cy = pKFi->cy;
				cv::Mat Xw = pMP->GetWorldPos();
				e->Xw[0] = Xw.at<float>(0);
				e->Xw[1] = Xw.at<float>(1);
				e->Xw[2] = Xw.at<float>(2);
				optimizer.addEdge(e);
				e->computeError();
				vpPlanarPoseEdges.push_back(e);
			}
			else {
				g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnKeyFrameID)));
				e->setMeasurement(obs);
				const float &invSigma2 = pKFi->mvInvLevelSigma2[octave];
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
		/////Set Planar Edge
		//int ptype = pMP->GetRecentLayoutFrameID();

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

	double sumError = 0.0;
	////////테스트 코드
	////평균 포즈 에러 테스트
	double poseErr = 0.0;
	for (int i = 0; i < vpPlanarPoseEdges.size(); i++)
	{
		auto e = vpPlanarPoseEdges[i];
		poseErr += e->chi2();
	}
	std::cout << "Pose Error::0::" << poseErr / vpPlanarPoseEdges.size() << std::endl;
	for (size_t i = 0, iend = vpPlaneEdge.size(); i < iend; i++) {
		g2o::PlaneBAEdge* e = vpPlaneEdge[i];
		MapPoint* pMP = vpPlaneEdgeMP[i];
		sumError += e->error()[0];
		if (pMP->isDeleted())
			continue;
	}
	std::cout << "error::0::" << sumError / vpPlaneEdge.size() << std::endl;
	sumError = 0.0;
	////////테스트 코드

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
		//////////////////////////인라이어 체크
		////평면 엣지

		////기본 엣지
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
			sumError += e->error()[0];
			if (e->chi2() > 0.01) {
				//pMP->SetPlaneID(0);
				//e->setLevel(1);
				//std::cout << e->chi2() <<"::"<<pMP->GetConnedtedFrames().size()<< std::endl;
			}
			else {
				//pMP->SetPlaneID(1);
			}
			e->setRobustKernel(0);
		}
		std::cout << "error::1::" << sumError/vpPlaneEdge.size() << std::endl;
		sumError = 0.0;
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

	////평균 포즈 에러 테스트
	poseErr = 0.0;
	for (int i = 0; i < vpPlanarPoseEdges.size(); i++)
	{
		auto e = vpPlanarPoseEdges[i];
		poseErr += e->chi2();
	}
	std::cout << "Pose Error::1::" << poseErr / vpPlanarPoseEdges.size() << std::endl;
	///////////////////////////////
	//////테스트 코드
	cv::Mat mat = cv::Mat::zeros(0, 4, CV_32FC1);
	///////////////////////////////
	int nBadPlane = 0;
	std::vector<UVR_SLAM::MapPoint*> vErasePlanarMPs;
	for (size_t i = 0, iend = vpPlaneEdge.size(); i < iend; i++) {
		g2o::PlaneBAEdge* e = vpPlaneEdge[i];
		MapPoint* pMP = vpPlaneEdgeMP[i];
		sumError += e->error()[0];
		if (pMP->isDeleted())
			continue;
	}
	std::cout << "error::2::" << sumError / vpPlaneEdge.size() << std::endl;
	std::cout << "ba::plane::" << planes.size() << "::" << nBadPlane << ", " << vpPlaneEdge.size() << std::endl;
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
			pMatch->RemoveMP();
			pMPi->DisconnectFrame(pMatch);

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
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnKeyFrameID));
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