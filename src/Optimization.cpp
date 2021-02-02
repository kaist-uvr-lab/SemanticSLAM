
#include <Optimization.h>
#include <Converter.h>
#include <map>
#include <Frame.h>
#include <opencv2/core/eigen.hpp>
//#include <Edge.h>
//#include <Optimizer.h>
//#include <PlaneBastedOptimization.h>
#include <PoseGraphOptimization.h>
#include <MatrixOperator.h>
#include <CandidatePoint.h>
#include <PlaneBA.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <System.h>
#include <Map.h>
#include <MapPoint.h>
#include <MapOptimizer.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/types/types_seven_dof_expmap.h"

//////Refinement
bool UVR_SLAM::Optimization::PointRefinement(UVR_SLAM::Map* pMap, UVR_SLAM::Frame* pCurrKF, UVR_SLAM::CandidatePoint* pCP, cv::Mat X3D, 
	std::map<UVR_SLAM::MatchInfo*, int> observations, std::set<UVR_SLAM::Frame*> spKFs, int thMinKF, float thHuberMono) {
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_3_2::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_3_2::PoseMatrixType>();

	g2o::BlockSolver_3_2 * solver_ptr = new g2o::BlockSolver_3_2(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	std::vector<g2o::BAEdgeOnlyMapPoint*> vpEdgesMono;
	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	std::vector<int> vnIDXs;
	g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
	//cv::Mat X3D = pMPi->GetWorldPos();
	vPoint->setEstimate(Converter::toVector3d(X3D));
	const int id = 0;// i + maxKFid + 1;
	vPoint->setId(id);
	//vPoint->setMarginalized(true);
	optimizer.addVertex(vPoint);

	int numEdges = 0;
	//Set edges
	for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
	{
		auto pMatch = mit->first;
		auto pKFi = pMatch->mpRefFrame;
		/*if (!spKFs.count(pKFi))
			continue;*/

		int idx = mit->second;
		auto pt = pMatch->mvMatchingPts[idx];
		Eigen::Matrix<double, 2, 1> obs;
		obs << pt.x, pt.y;

		g2o::BAEdgeOnlyMapPoint* e = new g2o::BAEdgeOnlyMapPoint();
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
		//e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnKeyFrameID)));
		e->setMeasurement(obs);

		const float &invSigma2 = pKFi->mvInvLevelSigma2[pCP->octave];

		e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		e->setRobustKernel(rk);
		rk->setDelta(thHuberMono);

		e->fx = pKFi->fx;
		e->fy = pKFi->fy;
		e->cx = pKFi->cx;
		e->cy = pKFi->cy;

		cv::Mat R, t;
		pKFi->GetPose(R, t);
		e->SetPose(R, t);

		optimizer.addEdge(e);
		vpEdgesMono.push_back(e);
		vpEdgeKFMono.push_back(pKFi);
		vnIDXs.push_back(idx);
	}

	optimizer.initializeOptimization();
	optimizer.optimize(10);

	int nres = 0;
	std::vector<bool> vbTemp(vpEdgeKFMono.size(), false);
	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	{
		g2o::BAEdgeOnlyMapPoint* e = vpEdgesMono[i];
		
		if (e->chi2() < 5.991 && e->isDepthPositive())
		{
			vbTemp[i] = true;
			nres++;
		}
	}
	float ratio = ((float)nres) / vbTemp.size();
	if (nres >= thMinKF && ratio > 0.5f) {
		////MP 생성
		int label = pCP->GetLabel();
		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(0));
		cv::Mat X = (Converter::toCvMat(vPoint->estimate()));
		auto pMP = new UVR_SLAM::MapPoint(pMap, pCurrKF, pCP, X, cv::Mat(), label, pCP->octave);
		//std::cout << "ratio::" << ratio << std::endl;
		for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
		{
			if (!vbTemp[i])
				continue;
			auto pKFi = vpEdgeKFMono[i];
			pMP->ConnectFrame(pKFi->mpMatchInfo, vnIDXs[i]);
			if(pKFi->mnFrameID != pCurrKF->mnFrameID)
				pCurrKF->mmKeyFrameCount[pKFi]++;
		}

		return true;
		/*pMP->SetOptimization(true);
		pSystem->mlpNewMPs.push_back(pMP);*/
	}
	return false;
}
bool UVR_SLAM::Optimization::ObjectPointRefinement(UVR_SLAM::Map* pMap, UVR_SLAM::MapPoint* pMP, std::vector<Frame*> vpKFs,
	std::set<UVR_SLAM::Frame*> spKFs, int thMinKF, float thHuberMono) {
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_3_2::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_3_2::PoseMatrixType>();

	g2o::BlockSolver_3_2 * solver_ptr = new g2o::BlockSolver_3_2(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	std::vector<g2o::BAEdgeOnlyMapPoint*> vpEdgesMono;
	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	std::vector<int> vnIDXs;
	g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
	vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
	const int id = 0;// i + maxKFid + 1;
	vPoint->setId(id);
	optimizer.addVertex(vPoint);

	int numEdges = 0;
	//Set edges
	auto observations = pMP->GetConnedtedFrames();
	for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
	{
		auto pMatch = mit->first;
		auto pKFi = pMatch->mpRefFrame;
		if (!spKFs.count(pKFi))
			continue;

		int idx = mit->second;
		auto pt = pMatch->mvMatchingPts[idx];
		Eigen::Matrix<double, 2, 1> obs;
		obs << pt.x, pt.y;

		g2o::BAEdgeOnlyMapPoint* e = new g2o::BAEdgeOnlyMapPoint();
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
		//e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnKeyFrameID)));
		e->setMeasurement(obs);

		const float &invSigma2 = pKFi->mvInvLevelSigma2[pMP->mnOctave];

		e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		e->setRobustKernel(rk);
		rk->setDelta(thHuberMono);

		e->fx = pKFi->fx;
		e->fy = pKFi->fy;
		e->cx = pKFi->cx;
		e->cy = pKFi->cy;

		cv::Mat R, t;
		pKFi->GetPose(R, t);
		e->SetPose(R, t);

		optimizer.addEdge(e);
		vpEdgesMono.push_back(e);
		vpEdgeKFMono.push_back(pKFi);
		vnIDXs.push_back(idx);
	}
	if (vpEdgesMono.size() < 2)
		return true;
	optimizer.initializeOptimization();
	optimizer.optimize(10);

	int nres = 0;
	std::vector<bool> vbTemp(vpEdgeKFMono.size(), false);
	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	{
		g2o::BAEdgeOnlyMapPoint* e = vpEdgesMono[i];

		if (e->chi2() < 5.991 && e->isDepthPositive())
		{
			vbTemp[i] = true;
			nres++;
		}
		else {
			auto pKFi = vpEdgeKFMono[i]->mpMatchInfo;
			pMP->DisconnectFrame(pKFi);
		}
	}

	g2o::VertexSBAPointXYZ* vPoint2 = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(0));
	cv::Mat X = (Converter::toCvMat(vPoint2->estimate()));
	pMP->SetWorldPos(X);

	float ratio = ((float)nres) / vbTemp.size();
	if (nres >= thMinKF && ratio > 0.5f) {
		////MP 생성
		/*g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(0));
		cv::Mat X = (Converter::toCvMat(vPoint->estimate()));
		pMP->SetWorldPos(X);*/
		//auto pMP = new UVR_SLAM::MapPoint(pMap, pCurrKF, pCP, X, cv::Mat(), label, pCP->octave);
		////std::cout << "ratio::" << ratio << std::endl;
		//for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
		//{
		//	if (!vbTemp[i])
		//		continue;
		//	auto pKFi = vpEdgeKFMono[i];
		//	pMP->ConnectFrame(pKFi->mpMatchInfo, vnIDXs[i]);
		//	if (pKFi->mnFrameID != pCurrKF->mnFrameID)
		//		pCurrKF->mmKeyFrameCount[pKFi]++;
		//}

		return true;
		/*pMP->SetOptimization(true);
		pSystem->mlpNewMPs.push_back(pMP);*/
	}
	return false;
}

int UVR_SLAM::Optimization::ObjectPointRefinement(UVR_SLAM::Map* pMap, std::vector<MapPoint*> vpObjectMPs, std::vector<Frame*> vpKFs) {
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	auto spKFs = pMap->GetWindowFramesSet(3);
	
	long unsigned int maxKFid = 0;
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
		//vSE3->setFixed(pKFi->mnKeyFrameID != nCurrKeyFrameID);
		optimizer.addVertex(vSE3);
		if (pKFi->mnKeyFrameID>maxKFid)
			maxKFid = pKFi->mnKeyFrameID;
	}

	// Set MapPoint vertices
	const int nExpectedSize = (vpKFs.size())*vpObjectMPs.size();
	
	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	std::vector<MapPoint*> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);

	/*std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	std::vector<UVR_SLAM::MatchInfo*> vpEdgeKFMono;
	std::vector<int> vnVertexIDXs;
	std::vector<int> vnConnected;
	std::vector<int> vnEdgeConnectedIDXs;
	const float thHuberMono = sqrt(5.991);*/

	//LOCK
	std::unique_lock<std::mutex> lock2(pMap->mMutexMapUpdate);

	////새로 추가된 맵포인트 설정
	for (size_t i = 0; i<vpObjectMPs.size(); i++)
	{
		auto pMPi = vpObjectMPs[i];
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		cv::Mat X3D = pMPi->GetWorldPos();
		vPoint->setEstimate(Converter::toVector3d(X3D));
		const int id = i + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const auto observations = pMPi->GetConnedtedFrames();
		int octave = pMPi->mnOctave;
		int numEdges = 0;
		//Set edges
		for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto pMatch = mit->first;
			auto pKFi = pMatch->mpRefFrame;
			if (!spKFs.count(pKFi))
				continue;

			int idx = mit->second;
			auto pt = pMatch->mvMatchingPts[idx];
			Eigen::Matrix<double, 2, 1> obs;
			obs << pt.x, pt.y;

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
		}
	}

	//for (int i = 0; i < vpObjectMPs.size(); i++)
	//{
	//	MapPoint* pMP = vpObjectMPs[i];
	//	if (!pMP || pMP->isDeleted())
	//		continue;
	//	g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
	//	vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
	//	int id = pMP->mnMapPointID;
	//	vPoint->setId(id);
	//	vPoint->setMarginalized(false);
	//	optimizer.addVertex(vPoint);
	//	int octave = pMP->mnOctave;

	//	//평면 프로젝션
	//	cv::Mat X3D = pMP->GetWorldPos();
	//	/*float dist = pNormal.dot(X3D) + pDist;
	//	X3D -= (dist*pNormal);
	//	pMP->SetWorldPos(X3D);*/

	//	const auto observations = pMP->GetConnedtedFrames();

	//	//Set edges
	//	for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
	//	{
	//		auto pMatch = mit->first;
	//		auto pKFi = pMatch->mpRefFrame;
	//		if (!spKFs.count(pKFi))
	//			continue;
	//		////이거 테스트 필요함.
	//		/*if (pKFi->GetKeyFrameID() > maxKFid)
	//		continue;*/
	//		int idx = mit->second;
	//		auto pt = pMatch->mvMatchingPts[idx];
	//		Eigen::Matrix<double, 2, 1> obs;
	//		obs << pt.x, pt.y;

	//		g2o::BAEdgeOnlyMapPoint* e = new g2o::BAEdgeOnlyMapPoint();
	//		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
	//		e->setMeasurement(obs);

	//		const float &invSigma2 = pKFi->mvInvLevelSigma2[octave];

	//		e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
	//		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
	//		e->setRobustKernel(rk);
	//		rk->setDelta(thHuberMono);

	//		e->fx = pKFi->fx;
	//		e->fy = pKFi->fy;
	//		e->cx = pKFi->cx;
	//		e->cy = pKFi->cy;
	//		
	//		cv::Mat R, t;
	//		pKFi->GetPose(R, t);
	//		e->SetPose(R, t);

	//		optimizer.addEdge(e);
	//		vpEdgesMono.push_back(e);
	//		vpEdgeKFMono.push_back(pKFi);
	//		vpMapPointEdgeMono.push_back(pMP);

	//	}
	//}
	optimizer.initializeOptimization();
	optimizer.optimize(20);
	std::vector<std::pair<UVR_SLAM::MatchInfo*, MapPoint*> > vToErase;
	
	for (int i = 0; i < vpObjectMPs.size(); i++)
	{
		MapPoint* pMP = vpObjectMPs[i];
		if (!pMP || pMP->isDeleted())
			continue;

		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i+maxKFid+1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		pMP->UpdateNormalAndDepth();
	}

}
////맵포인트를 평면에 노말을 이용해서 붙이고, 포즈를 수정하고, 그 후 나머지 맵포인트를 수정
int UVR_SLAM::Optimization::PlanarPoseRefinement(UVR_SLAM::Map* pMap, std::vector<MapPoint*> vpPlanarMPs, std::vector<Frame*> vpKFs) {

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	auto spKFs = pMap->GetWindowFramesSet(3);

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
		vSE3->setId(pKFi->mnKeyFrameID);
		vSE3->setFixed(pKFi->mnKeyFrameID == 0);
		optimizer.addVertex(vSE3);
		if (pKFi->mnKeyFrameID>maxKFid)
			maxKFid = pKFi->mnKeyFrameID;
	}

	// Set MapPoint vertices
	const int nExpectedSize = (vpKFs.size())*vpPlanarMPs.size();

	std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	std::vector<MapPoint*> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);

	//LOCK
	std::unique_lock<std::mutex> lock2(pMap->mMutexMapUpdate);

	for (int i = 0; i < vpPlanarMPs.size(); i++)
	{
		MapPoint* pMP = vpPlanarMPs[i];
		if (!pMP || pMP->isDeleted())
			continue;
		/*g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		int id = pMP->mnMapPointID + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);*/
		int octave = pMP->mnOctave;

		//평면 프로젝션
		cv::Mat X3D = pMP->GetWorldPos();
		//cv::Mat X3D2 = pMP->GetWorldPos();
		/*float dist = pNormal.dot(X3D) + pDist;
		X3D -= (dist*pNormal);
		pMP->SetWorldPos(X3D);*/

		const auto observations = pMP->GetConnedtedFrames();

		//Set edges
		for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto pMatch = mit->first;
			auto pKFi = pMatch->mpRefFrame;
			if (!spKFs.count(pKFi))
				continue;
			////이거 테스트 필요함.
			/*if (pKFi->GetKeyFrameID() > maxKFid)
			continue;*/
			int idx = mit->second;
			auto pt = pMatch->mvMatchingPts[idx];
			Eigen::Matrix<double, 2, 1> obs;
			obs << pt.x, pt.y;

			g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnKeyFrameID)));
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
			e->Xw[0] = X3D.at<float>(0);
			e->Xw[1] = X3D.at<float>(1);
			e->Xw[2] = X3D.at<float>(2);

			optimizer.addEdge(e);
			vpEdgesMono.push_back(e);
			vpEdgeKFMono.push_back(pKFi);
			vpMapPointEdgeMono.push_back(pMP);

			/*cv::Mat R, t;
			pKFi->GetPose(R, t);
			cv::Mat temp1 = R*X3D + t;
			cv::Mat temp2 = R*X3D2 + t;
			temp1 /= temp1.at<float>(2);
			temp1 = pKFi->mK*temp1;
			cv::Point2f pt1(temp1.at<float>(0), temp1.at<float>(1));
			temp2 /= temp2.at<float>(2);
			temp2 = pKFi->mK*temp2;
			cv::Point2f pt2(temp2.at<float>(0), temp2.at<float>(1));
			std::cout <<pMP->mnMapPointID<<"::"<< pt1 << " " << pt2 <<"::"<<pt<< std::endl;*/
		}
	}
	optimizer.initializeOptimization();
	optimizer.optimize(20);
	std::vector<std::pair<UVR_SLAM::MatchInfo*, MapPoint*> > vToErase;
	vToErase.reserve(vpEdgesMono.size());

	// Check inlier observations       
	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	{
		g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];
		MapPoint* pMP = vpMapPointEdgeMono[i];

		if (pMP->isDeleted())
			continue;

		if (e->chi2()>5.991 || !e->isDepthPositive())
		{
			UVR_SLAM::Frame* pKFi = vpEdgeKFMono[i];
			vToErase.push_back(std::make_pair(pKFi->mpMatchInfo, pMP));
			//std::cout << "Refinement::" <<pMP->mnMapPointID<<"="<<pMP->GetConnedtedFrames().size()<<"::"<< e->chi2()<< std::endl;
		}
	}
	std::cout << "PlanarRefinement::" <<vpPlanarMPs.size()<<" "<<vpEdgesMono.size()<<" "<<vToErase .size()<< std::endl;
	
	if (!vToErase.empty())
	{
		for (size_t i = 0; i<vToErase.size(); i++)
		{
			auto pMatch = vToErase[i].first;
			MapPoint* pMPi = vToErase[i].second;
			//pMatch->RemoveMP();
			pMPi->DisconnectFrame(pMatch);
		}
	}
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
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////Opticalflow 버전용
int UVR_SLAM::Optimization::PoseOptimization(Map* pMap, Frame *pFrame, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool>& vbInliers, std::vector<float> vInvLevelSigma2)
{
	//std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	int nInitialCorrespondences = 0;
	int nTargetID = pFrame->mnFrameID;

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
	const int N = vpMPs.size();//vpMPs.size();

	std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
	std::vector<size_t> vnIndexEdgeMono;
	vpEdgesMono.reserve(N);
	vnIndexEdgeMono.reserve(N);
	const float deltaMono = sqrt(5.991);
	{
		for (int i = 0; i<N; i++)
		{
			auto pMP = vpMPs[i];
			if (!pMP || pMP->isDeleted()) {
				vbInliers[i] = false;
				continue;
			}
			nInitialCorrespondences++;

			Eigen::Matrix<double, 2, 1> obs;
			const cv::Point2f pt = vpPts[i];

			obs << pt.x, pt.y;
			g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
			e->setMeasurement(obs);
			int octave = pMP->mnOctave;
			const float invSigma2 = vInvLevelSigma2[octave];
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
			//std::cout << Xw.t()<<", "<<pt << std::endl;
			optimizer.addEdge(e);
			vpEdgesMono.push_back(e);
			vnIndexEdgeMono.push_back(i);
			//vbInliers[i] = true;
		}
	}
	if (nInitialCorrespondences < 10) {
		std::cout << "PoseOptimization::Error::Init=" << nInitialCorrespondences << "|| CP=" << N << std::endl;
		return 0;
	}
	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	const float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };
	const int its[4] = { 10,10,10,10 };
	int nBad = 0;
	float maxDepth = pFrame->mfMedianDepth + pFrame->mfRange;
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
			if (!vbInliers[idx])
			{
				e->computeError();
			}

			const float chi2 = e->chi2();
			float depth = e->GetDepth();
			if (chi2>chi2Mono[it] || depth <= 0.0)// || depth > maxDepth)
			{
				//vpCPs[idx]->GetMP()->mnTrackingID = -1;
				vbInliers[idx] = false;
				e->setLevel(1);
				nBad++;
			}
			else
			{
				//vpCPs[idx]->GetMP()->mnTrackingID = nTargetID;
				vbInliers[idx] = true;
				e->setLevel(0);
			}
			if (it == 2)
				e->setRobustKernel(0);
		}

	}
	if (nInitialCorrespondences - nBad < 10) {
		std::cout << "PoseOptimization::Error::Init=" << nInitialCorrespondences << ", bad=" << nBad << "=CP=" << N << std::endl;
		return 0;
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


int UVR_SLAM::Optimization::PoseOptimization(Map* pMap, Frame *pFrame, std::vector<UVR_SLAM::CandidatePoint*> vpCPs, std::vector<cv::Point2f> vpPts, std::vector<bool>& vbInliers, std::vector<float> vInvLevelSigma2)
{
	//std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	int nInitialCorrespondences = 0;
	int nTargetID = pFrame->mnFrameID;

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
	const int N = vpCPs.size();//vpMPs.size();
	
	std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
	std::vector<size_t> vnIndexEdgeMono;
	vpEdgesMono.reserve(N);
	vnIndexEdgeMono.reserve(N);
	const float deltaMono = sqrt(5.991);
	{
		for (int i = 0; i<N; i++)
		{
			auto pCP = vpCPs[i];
			MapPoint* pMP = pCP->GetMP();

			if (!pMP || pMP->isDeleted())
				continue;
			nInitialCorrespondences++;

			Eigen::Matrix<double, 2, 1> obs;
			const cv::Point2f pt = vpPts[i];
			
			obs << pt.x, pt.y;
			g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
			e->setMeasurement(obs);
			int octave = pMP->mnOctave;
			const float invSigma2 = vInvLevelSigma2[octave];
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
			//std::cout << Xw.t()<<", "<<pt << std::endl;
			optimizer.addEdge(e);
			vpEdgesMono.push_back(e);
			vnIndexEdgeMono.push_back(i);
			vbInliers[i] = true;
		}
	}
	if (nInitialCorrespondences < 10) {
		std::cout << "PoseOptimization::Error::Init=" << nInitialCorrespondences << "|| CP=" << N << std::endl;
		return 0;
	}
	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	const float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };
	const int its[4] = { 10,10,10,10 };
	int nBad = 0;
	float maxDepth = pFrame->mfMedianDepth + pFrame->mfRange;
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
			if (!vbInliers[idx])
			{
				e->computeError();
			}

			const float chi2 = e->chi2();
			float depth = e->GetDepth();
			if (chi2>chi2Mono[it] || depth <= 0.0 || depth > maxDepth)
			{
				vpCPs[idx]->GetMP()->mnTrackingID = -1;
				vbInliers[idx] = false;
				e->setLevel(1);
				nBad++;
			}
			else
			{
				vpCPs[idx]->GetMP()->mnTrackingID = nTargetID;
				vbInliers[idx] = true;
				e->setLevel(0);
			}
			if (it == 2)
				e->setRobustKernel(0);
		}
		
	}
	if (nInitialCorrespondences - nBad < 10) {
		std::cout << "PoseOptimization::Error::Init="<< nInitialCorrespondences << ", bad=" << nBad << "=CP=" << N << std::endl;
		return 0;
	}
	//std::cout << "PoseOptimization::nBad::" << nBad <<"::"<< nInitialCorrespondences << std::endl;
	/*for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++) {

		const size_t idx = vnIndexEdgeMono[i];
		auto pCPi = vpCPs[idx];
		auto pMPi = pCPi->GetMP();
		if (!pMPi || pMPi->isDeleted())
			continue;
		pMPi->IncreaseVisible();
		if (!vbInliers[i]){
			continue;
		}
		pMPi->IncreaseFound();
	}*/

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
	//g2o::SparseOptimizer optimizer;
	//g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	//linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

	//g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	//g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	//optimizer.setAlgorithm(solver);

	//int nInitialCorrespondences = 0;
	//int nTargetID = pFrame->mnFrameID;

	//// Set Frame vertex
	//cv::Mat R, t;
	//pFrame->GetPose(R, t);
	//cv::Mat Tcw = cv::Mat::zeros(4, 4, CV_32FC1);
	//R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
	//t.copyTo(Tcw.rowRange(0, 3).col(3));
	//g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
	//vSE3->setEstimate(Converter::toSE3Quat(Tcw));
	//vSE3->setId(0);
	//vSE3->setFixed(false);
	//optimizer.addVertex(vSE3);

	//// Set MapPoint vertices
	//const int N = vpMPs.size();

	//std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
	//std::vector<size_t> vnIndexEdgeMono;
	//vpEdgesMono.reserve(N);
	//vnIndexEdgeMono.reserve(N);

	//const float deltaMono = sqrt(5.991);
	//const float deltaStereo = sqrt(7.815);
	//auto mvInvLevelSigma2 = pFrame->mpMatchInfo->mpTargetFrame->mvInvLevelSigma2;
	//{
	//	for (int i = 0; i<N; i++)
	//	{
	//		MapPoint* pMP = vpMPs[i];
	//		if (!pMP || pMP->isDeleted() || pMP->GetRecentTrackingFrameID() != nTargetID)
	//			continue;
	//		nInitialCorrespondences++;
	//		//pFrame->mvbMPInliers[i] = true;

	//		Eigen::Matrix<double, 2, 1> obs;
	//		const cv::Point2f pt = vpPts[i];
	//		obs << pt.x, pt.y;
	//		//std::cout << pt << std::endl;
	//		g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

	//		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
	//		e->setMeasurement(obs);
	//		int octave = pMP->mnOctave;
	//		const float invSigma2 = mvInvLevelSigma2[octave];
	//		e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

	//		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
	//		e->setRobustKernel(rk);
	//		rk->setDelta(deltaMono);

	//		e->fx = pFrame->fx;
	//		e->fy = pFrame->fy;
	//		e->cx = pFrame->cx;
	//		e->cy = pFrame->cy;
	//		cv::Mat Xw = pMP->GetWorldPos();
	//		e->Xw[0] = Xw.at<float>(0);
	//		e->Xw[1] = Xw.at<float>(1);
	//		e->Xw[2] = Xw.at<float>(2);

	//		optimizer.addEdge(e);

	//		vpEdgesMono.push_back(e);
	//		vnIndexEdgeMono.push_back(i);
	//	}
	//}

	//if (nInitialCorrespondences<3)
	//	return 0;

	//// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	//// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	//const float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };
	//const float chi2Stereo[4] = { 7.815,7.815,7.815, 7.815 };
	//const int its[4] = { 10,10,10,10 };

	//int nBad = 0;
	//for (size_t it = 0; it<4; it++)
	//{

	//	vSE3->setEstimate(Converter::toSE3Quat(Tcw)); //이건가??
	//	optimizer.initializeOptimization(0);
	//	optimizer.optimize(its[it]);

	//	nBad = 0;
	//	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	//	{
	//		g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

	//		const size_t idx = vnIndexEdgeMono[i];

	//		if (!vbInliers[idx])
	//		{
	//			e->computeError();
	//		}

	//		const float chi2 = e->chi2();

	//		if (chi2>chi2Mono[it] || !e->isDepthPositive())
	//		{
	//			vbInliers[idx] = false;
	//			e->setLevel(1);
	//			nBad++;
	//		}
	//		else
	//		{
	//			vbInliers[idx] = true;
	//			e->setLevel(0);
	//		}
	//		if (it == 2)
	//			e->setRobustKernel(0);
	//	}

	//	if (optimizer.edges().size()<10)
	//		break;
	//}

	//// Recover optimized pose and return number of inliers
	//g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
	//g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
	//cv::Mat pose = Converter::toCvMat(SE3quat_recov);
	//R = pose.rowRange(0, 3).colRange(0, 3);
	//t = pose.rowRange(0, 3).col(3);
	//pFrame->SetPose(R, t);

	//return nInitialCorrespondences - nBad;
}

void UVR_SLAM::Optimization::OpticalLocalBundleAdjustment(UVR_SLAM::Map* pMap, UVR_SLAM::MapOptimizer* pMapOptimizer, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::Frame*> vpKFs, std::vector<UVR_SLAM::Frame*> vpFixedKFs) {

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
	
	// Set MapPoint vertices
	const int nExpectedSize = (vpKFs.size()+vpFixedKFs.size())*vpMPs.size();

	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	vpEdgesMono.reserve(nExpectedSize);

	std::vector<UVR_SLAM::Frame*> vpEdgeKFMono;
	vpEdgeKFMono.reserve(nExpectedSize);

	std::vector<MapPoint*> vpMapPointEdgeMono;
	vpMapPointEdgeMono.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	
	for (int i =0; i < vpMPs.size(); i++)
	{
		MapPoint* pMP = vpMPs[i];
		if (!pMP || pMP->isDeleted())
			continue;
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
		int id = pMP->mnMapPointID + maxKFid + 1;
		int octave = pMP->mnOctave;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);

		const auto observations = pMP->GetObservations();

		//Set edges
		for (std::map<UVR_SLAM::Frame*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto pKFi = mit->first;
			if (pKFi->isDeleted())
				std::cout << "BA::deleted kf" << std::endl;
			if (pKFi->mnLocalBAID != nTargetID)
				continue;
			////이거 테스트 필요함.
			/*if (pKFi->GetKeyFrameID() > maxKFid)
				continue;*/
			int idx = mit->second;
			auto pt = pKFi->mvPts[idx];
			Eigen::Matrix<double, 2, 1> obs;
			obs << pt.x, pt.y;

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
	
	std::unique_lock<std::mutex> lock2(pMap->mMutexMapUpdate);
	
	if (!vToErase.empty())
	{
	
		for (size_t i = 0; i<vToErase.size(); i++)
		{
			auto pKF = vToErase[i].first;
			MapPoint* pMPi = vToErase[i].second;
			pMPi->EraseObservation(pKF);
		}
	}
	
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

		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnMapPointID + maxKFid + 1));
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
		//pMP->UpdateNormalAndDepth();
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

void UVR_SLAM::Optimization::LocalOptimization(System* pSystem, Map* pMap, Frame* pCurrKF) {

	//std::unique_lock<std::mutex> lock(pMap->mMutexMapOptimization);
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	auto vpKFs = pMap->GetWindowFramesVector(3);
	auto spKFs = pMap->GetWindowFramesSet(3);
	if (!spKFs.count(pCurrKF)) {
		vpKFs.push_back(pCurrKF);
		spKFs.insert(pCurrKF);
	}

	std::cout << "opt::-1" << std::endl;

	long unsigned int maxKFid = 0;
	int nCurrKeyFrameID = pCurrKF->mnKeyFrameID;
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
		vSE3->setFixed(pKFi->mnKeyFrameID != nCurrKeyFrameID);
		optimizer.addVertex(vSE3);
		if (pKFi->mnKeyFrameID>maxKFid)
			maxKFid = pKFi->mnKeyFrameID;
	}

	std::vector<CandidatePoint*> vpVertexCPs;
	std::vector<g2o::VertexSBAPointXYZ*> vpVertices;
	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	std::vector<UVR_SLAM::MatchInfo*> vpEdgeKFMono;
	std::vector<int> vnVertexIDXs;
	std::vector<int> vnConnected;
	std::vector<int> vnEdgeConnectedIDXs;
	
	const float thHuberMono = sqrt(5.991);

	auto vpCPs = pCurrKF->mpMatchInfo->mvpMatchingCPs;
	auto vPTs = pCurrKF->mpMatchInfo->mvMatchingPts;

	std::vector<bool> vbInliers(vpCPs.size(), true);
	std::cout << "opt::0" << std::endl;
	////새로 추가된 맵포인트 설정
	for (size_t i = 0, iend = vpCPs.size(); i < iend; i++)
	{
		auto pCPi = (vpCPs[i]);
		auto pMPi = pCPi->GetMP();
		if (!pMPi || pMPi->isDeleted())
			continue;

		int vIDX = vpVertexCPs.size();

		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		cv::Mat X3D = pMPi->GetWorldPos();
		vPoint->setEstimate(Converter::toVector3d(X3D));
		const int id = pCPi->mnCandidatePointID + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		//vPoint->setFixed(pMPi && pMPi->GetQuality() && !pMPi->isDeleted());
		optimizer.addVertex(vPoint);

		const auto observations = pCPi->GetFrames();
		int octave = pCPi->octave;
		int numEdges = 0;
		//Set edges
		for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto pMatch = mit->first;
			auto pKFi = pMatch->mpRefFrame;
			if (spKFs.find(pKFi) == spKFs.end())
				continue;

			int idx = mit->second;
			auto pt = pMatch->mvMatchingPts[idx];
			Eigen::Matrix<double, 2, 1> obs;
			obs << pt.x, pt.y;

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
			vnVertexIDXs.push_back(vIDX);
			vpEdgeKFMono.push_back(pMatch);
			vnEdgeConnectedIDXs.push_back(idx);
			numEdges++;
		}
		vpVertices.push_back(vPoint);
		vpVertexCPs.push_back(pCPi);
		vnConnected.push_back(numEdges);
	}
	////새로 추가된 맵포인트 설정

	optimizer.initializeOptimization();
	optimizer.optimize(20);
	std::cout << "opt::1" << std::endl;
	//std::unique_lock<std::mutex> lock2(pMap->mMutexMapUdpate);
	std::cout << "opt::2" << std::endl;
	////체크
	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	{
		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		int vIdx = vnVertexIDXs[i];

		auto pCPi = vpVertexCPs[vIdx];
		auto pMPi = pCPi->GetMP();
		bool bMP = pMPi && !pMPi->isDeleted();
		auto pMatch = vpEdgeKFMono[i];
		bool bConnect = true;
		if (e->chi2()>5.991 || !e->isDepthPositive())
		{
			bConnect = false;
		}
		if (bMP) {
			if (bConnect) {
				auto cIdx = vnEdgeConnectedIDXs[i];
				pMPi->ConnectFrame(pMatch, cIdx);
			}else
				pMPi->DisconnectFrame(pMatch);
		}
	}
	std::cout << "opt::3" << std::endl;
	//Curr KF 포즈 수정
	cv::Mat R, t;
	{
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pCurrKF->mnKeyFrameID));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		cv::Mat Tcw = Converter::toCvMat(SE3quat);
		R = Tcw.rowRange(0, 3).colRange(0, 3);
		t = Tcw.rowRange(0, 3).col(3);
		pCurrKF->SetPose(R, t);
	}
	
	////포인트 복원
	
	
	std::cout << "opt::4" << std::endl;
	for (size_t i = 0, iend = vpVertexCPs.size(); i < iend; i++)
	{
		int idx = vnVertexIDXs[i];
		auto pCPi = vpVertexCPs[i];
		
		//g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(idx + maxKFid + 1));
		cv::Mat Xw = Converter::toCvMat(vpVertices[i]->estimate()).clone();
		auto pMPi = pCPi->GetMP();
		if(pMPi && !pMPi->isDeleted())
			pMPi->SetWorldPos(Xw);
	}
	std::cout << "opt::5" << std::endl;
}

void UVR_SLAM::Optimization::LocalOptimization(System* pSystem, Map* pMap, Frame* pCurrKF, std::vector<cv::Mat>& vX3Ds, std::vector<CandidatePoint*> vpCPs, std::vector<bool>& vbInliers) {

	//std::unique_lock<std::mutex> lock(pMap->mMutexMapOptimization);
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	auto vpKFs = pMap->GetWindowFramesVector(3);
	auto spKFs = pMap->GetWindowFramesSet(3);
	if (!spKFs.count(pCurrKF)) {
		vpKFs.push_back(pCurrKF);
		spKFs.insert(pCurrKF);
	}
	
	
	long unsigned int maxKFid = 0;
	int nCurrKeyFrameID = pCurrKF->mnKeyFrameID;
	{
		//std::unique_lock<std::mutex> lock2(pMap->mMutexMapUdpate);
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
			vSE3->setFixed(pKFi->mnKeyFrameID != nCurrKeyFrameID);
			optimizer.addVertex(vSE3);
			if (pKFi->mnKeyFrameID>maxKFid)
				maxKFid = pKFi->mnKeyFrameID;
		}
	}

	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	std::vector<UVR_SLAM::MatchInfo*> vpEdgeKFMono;
	std::vector<int> vnVertexIDXs;
	std::vector<int> vnConnected;
	std::vector<int> vnEdgeConnectedIDXs;
	const float thHuberMono = sqrt(5.991);

	////새로 추가된 맵포인트 설정
	for (size_t i = 0; i<vX3Ds.size(); i++)
	{
		auto pCPi = std::move(vpCPs[i]);
		auto pMPi = pCPi->GetMP();
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		cv::Mat X3D = std::move(vX3Ds[i]);
		vPoint->setEstimate(Converter::toVector3d(X3D));
		const int id = i + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		//vPoint->setFixed(pMPi && pMPi->GetQuality() && !pMPi->isDeleted());
		optimizer.addVertex(vPoint);

		const auto observations = pCPi->GetFrames();
		int octave = pCPi->octave;
		int numEdges = 0;
		//Set edges
		for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto pMatch = mit->first;
			auto pKFi = pMatch->mpRefFrame;
			if (!spKFs.count(pKFi))
				continue;

			int idx = mit->second;
			auto pt = pMatch->mvMatchingPts[idx];
			Eigen::Matrix<double, 2, 1> obs;
			obs << pt.x, pt.y;

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
			vnVertexIDXs.push_back(i);
			vpEdgeKFMono.push_back(pMatch);
			vnEdgeConnectedIDXs.push_back(idx);
			numEdges++;
		}
		vnConnected.push_back(numEdges);
	}
	////새로 추가된 맵포인트 설정

	optimizer.initializeOptimization();
	optimizer.optimize(20);

	////체크
	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	{
		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		int vIdx = vnVertexIDXs[i];

		if (e->chi2()>5.991 || !e->isDepthPositive())
		{
			vnConnected[vIdx]--;
			if (vnConnected[vIdx] < pSystem->mnThreshMinKF)
				vbInliers[vIdx] = false;
		}
	}


	//Curr KF 포즈 수정
	cv::Mat R, t;
	{
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pCurrKF->mnKeyFrameID));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		cv::Mat Tcw = Converter::toCvMat(SE3quat);
		R = Tcw.rowRange(0, 3).colRange(0, 3);
		t = Tcw.rowRange(0, 3).col(3);
		//std::unique_lock<std::mutex> lock2(pMap->mMutexMapUdpate);
		std::unique_lock<std::mutex> lock2(pMap->mMutexMapUpdate);
		pCurrKF->SetPose(R, t);
	}
	cv::Mat Rcw2 = R.row(2);
	Rcw2 = Rcw2.t();
	float zcw = t.at<float>(2);
	
	float fMaxTh = pCurrKF->mfMedianDepth + pCurrKF->mfRange; //fMean + 1.15*fstddev; //1.654		//float thresh = 1.15;//1.284;// *dStdDev + dMean; //1.654
	float fMinTh = pCurrKF->mfMedianDepth - pCurrKF->mfRange;// fMean - 1.15*fstddev;


	int N = 0;
	for (size_t i = 0, iend = vX3Ds.size(); i < iend; i++)
	{
		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i + maxKFid + 1));
		vX3Ds[i] = Converter::toCvMat(vPoint->estimate()).clone();
		cv::Mat X3D = std::move(vX3Ds[i]);
		float z = (float)Rcw2.dot(X3D) + zcw;
		if (vbInliers[i] && (z > 0.0 && z < fMaxTh)) {//&& (depth > fMinTh && depth < fMaxTh)
			pMap->AddReinit(X3D);
			auto pCPi = vpCPs[i];
			auto pMPi = pCPi->GetMP();
			if (!pMPi || pMPi->isDeleted()) {
				//new mp
				int label = pCPi->GetLabel();
				auto pMP = new UVR_SLAM::MapPoint(pMap, pCurrKF, pCPi, X3D, cv::Mat(), label, pCPi->octave);
				pMP->SetOptimization(true);
				pSystem->mlpNewMPs.push_back(pMP);
				N++;
			}
			/*else if (pMPi && pMPi->GetQuality() && !pMPi->isDeleted()) {
			pMPi->SetWorldPos(std::move(X3D));
			}*/
		}

	}
	std::cout << "LM::New::" << N << std::endl;

	//커넥션 체크
	for (int i = 0; i < vpEdgesMono.size(); i++) {
		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		int vIdx = vnVertexIDXs[i];
		if (vbInliers[vIdx]) {
			auto pCPi = vpCPs[vIdx];
			auto pMPi = pCPi->GetMP();
			if (pMPi && pMPi->GetQuality() && !pMPi->isDeleted()) {
				auto pMatch = vpEdgeKFMono[i];
				if (e->chi2() > 5.991 || !e->isDepthPositive()) {
					pMPi->DisconnectFrame(pMatch);
					continue;
				}
				//connect
				auto cIdx = vnEdgeConnectedIDXs[i];
				pMPi->ConnectFrame(pMatch, cIdx);
			}
		}
	}
}

void UVR_SLAM::Optimization::LocalOptimization(System* pSystem, Map* pMap, Frame* pCurrKF, std::vector<cv::Mat>& vX3Ds, std::vector<CandidatePoint*> vpCPs, std::vector<bool>& vbInliers, std::vector<bool>& vbInliers2, float scale, float fMedianDepth, float fMeanDepth, float fStdDev) {
	
	//std::unique_lock<std::mutex> lock(pMap->mMutexMapOptimization);
	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	auto vpKFs = pMap->GetWindowFramesVector(3);
	auto spKFs = pMap->GetWindowFramesSet(3);
	if (!spKFs.count(pCurrKF)) {
		vpKFs.push_back(pCurrKF);
		spKFs.insert(pCurrKF);
	}

	long unsigned int maxKFid = 0;
	int nCurrKeyFrameID = pCurrKF->mnKeyFrameID;// vpKFs[vpKFs.size() - 1]->mnKeyFrameID;//pCurrKF->mnKeyFrameID;
	
	{
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
			//vSE3->setFixed(pKFi->mnKeyFrameID==0);
			vSE3->setFixed(pKFi->mnKeyFrameID != nCurrKeyFrameID);
			optimizer.addVertex(vSE3);
			if (pKFi->mnKeyFrameID>maxKFid)
				maxKFid = pKFi->mnKeyFrameID;
		}
	}
	std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
	std::vector<UVR_SLAM::MatchInfo*> vpEdgeKFMono;
	std::vector<int> vnVertexIDXs;
	std::vector<int> vnConnected;
	std::vector<int> vnEdgeConnectedIDXs;
	const float thHuberMono = sqrt(5.991);

	////새로 추가된 맵포인트 설정
	for (size_t i = 0; i<vX3Ds.size(); i++)
	{
		auto pCPi = std::move(vpCPs[i]);
		auto pMPi = pCPi->GetMP();
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		cv::Mat X3D = std::move(vX3Ds[i]); 
		vPoint->setEstimate(Converter::toVector3d(X3D));
		const int id = i + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		vPoint->setFixed(pMPi && pMPi->GetQuality() && !pMPi->isDeleted());
		optimizer.addVertex(vPoint);

		const auto observations = pCPi->GetFrames();
		int octave = pCPi->octave;
		int numEdges = 0;
		//Set edges
		for (std::map<UVR_SLAM::MatchInfo*, int>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto pMatch = mit->first;
			auto pKFi = pMatch->mpRefFrame;
			if (!spKFs.count(pKFi))
				continue;
			
			int idx = mit->second;
			auto pt = pMatch->mvMatchingPts[idx];
			Eigen::Matrix<double, 2, 1> obs;
			obs << pt.x, pt.y;

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
			vnVertexIDXs.push_back(i);
			vpEdgeKFMono.push_back(pMatch);
			vnEdgeConnectedIDXs.push_back(idx);
			numEdges++;
		}
		vnConnected.push_back(numEdges);
	}
	////새로 추가된 맵포인트 설정
	optimizer.initializeOptimization();
	optimizer.optimize(20);

	////체크
	for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
	{
		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		int vIdx = vnVertexIDXs[i];

		if (e->chi2()>5.991 || !e->isDepthPositive())
		{
			vnConnected[vIdx]--;
			if (vnConnected[vIdx] < pSystem->mnThreshMinKF)
				vbInliers[vIdx] = false;
		}
	}
	
	
	//Curr KF 포즈 수정
	cv::Mat R, t;
	{
		/*g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pCurrKF->mnKeyFrameID));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		cv::Mat Tcw = Converter::toCvMat(SE3quat);
		R = Tcw.rowRange(0, 3).colRange(0, 3);
		t = Tcw.rowRange(0, 3).col(3);
		std::unique_lock<std::mutex> lock2(pMap->mMutexMapUdpate);
		pCurrKF->SetPose(R, t);*/
	}
	
	
	////포인트 복원
	//평균 뎁스와 편차 계산
	/*std::vector<float> vfDepths;
	std::vector<int> vnTempIDXs;
	*/
	pCurrKF->GetPose(R, t);
	cv::Mat Rcw2 = R.row(2);
	Rcw2 = Rcw2.t();
	float zcw = t.at<float>(2);
	

	float fMaxTh = pCurrKF->mfMedianDepth + pCurrKF->mfRange; //fMean + 1.15*fstddev; //1.654		//float thresh = 1.15;//1.284;// *dStdDev + dMean; //1.654
	float fMinTh = pCurrKF->mfMedianDepth - pCurrKF->mfRange;// fMean - 1.15*fstddev;
	int N = 0;
	for (size_t i = 0, iend = vX3Ds.size(); i < iend; i++)
	{
		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i + maxKFid + 1));
		vX3Ds[i] = Converter::toCvMat(vPoint->estimate()).clone();
		cv::Mat X3D = std::move(vX3Ds[i]);
		float z = (float)Rcw2.dot(X3D) + zcw;
		if (vbInliers[i] && (z > 0.0f && z < fMaxTh)) {//&& (depth > fMinTh && depth < fMaxTh)
			pMap->AddReinit(X3D);
			auto pCPi = vpCPs[i];
			auto pMPi = pCPi->GetMP();
			if (!pMPi || pMPi->isDeleted()) {
				//new mp
				int label = pCPi->GetLabel();
				auto pMP = new UVR_SLAM::MapPoint(pMap, pCurrKF, pCPi, X3D, cv::Mat(), label, pCPi->octave);
				pMP->SetOptimization(true);
				pSystem->mlpNewMPs.push_back(pMP);
				N++;
			}
			/*else if (pMPi && pMPi->GetQuality() && !pMPi->isDeleted()) {
				pMPi->SetWorldPos(std::move(X3D));
			}*/
		}

	}
	std::cout << "LM::New<<" << N << std::endl;
	//커넥션 체크
	for (int i = 0; i < vpEdgesMono.size(); i++) {
		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		int vIdx = vnVertexIDXs[i];
		if (vbInliers[vIdx]) {
			auto pCPi = vpCPs[vIdx];
			auto pMPi = pCPi->GetMP();
			if (pMPi && pMPi->GetQuality() && !pMPi->isDeleted()) {
				auto pMatch = vpEdgeKFMono[i];
				if (e->chi2() > 5.991 || !e->isDepthPositive()){
					pMPi->DisconnectFrame(pMatch);
					continue;
				}
				//connect
				auto cIdx = vnEdgeConnectedIDXs[i];
				pMPi->ConnectFrame(pMatch, cIdx);
			}
		}
	}

}

void UVR_SLAM::Optimization::PoseRecoveryOptimization(Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, std::vector<cv::Point2f> vCurrPTs, std::vector<cv::Point2f> vPrevPTs, std::vector<cv::Point2f> vPPrevPTs, std::vector<cv::Mat>& vP3Ds) {
	std::vector<bool> vbNotIncludedMP;
	vbNotIncludedMP.resize(vP3Ds.size());

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	long unsigned int maxKFid = pCurrKF->mnFrameID;

	///////KF Curr추가
	cv::Mat Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);
	cv::Mat TCurrcw = cv::Mat::zeros(4, 4, CV_32FC1);
	Rcurr.copyTo(TCurrcw.rowRange(0, 3).colRange(0, 3));
	Tcurr.copyTo(TCurrcw.col(3).rowRange(0, 3));
	g2o::VertexSE3Expmap * vSE3Curr = new g2o::VertexSE3Expmap();
	vSE3Curr->setEstimate(Converter::toSE3Quat(TCurrcw));
	vSE3Curr->setId(pCurrKF->mnFrameID);
	vSE3Curr->setFixed(false);
	optimizer.addVertex(vSE3Curr);
	///////KF Curr추가

	///////KF Prev추가
	cv::Mat Rprev, Tprev;
	pPrevKF->GetPose(Rprev, Tprev);
	cv::Mat TPrevcw = cv::Mat::zeros(4, 4, CV_32FC1);
	Rprev.copyTo(TPrevcw.rowRange(0, 3).colRange(0, 3));
	Tprev.copyTo(TPrevcw.col(3).rowRange(0, 3));
	g2o::VertexSE3Expmap * vSE3Prev = new g2o::VertexSE3Expmap();
	vSE3Prev->setEstimate(Converter::toSE3Quat(TPrevcw));
	vSE3Prev->setId(pPrevKF->mnFrameID);
	vSE3Prev->setFixed(true);
	optimizer.addVertex(vSE3Prev);
	///////KF Prev추가

	///////KF PPrev추가
	cv::Mat Rpprev, Tpprev;
	pPPrevKF->GetPose(Rpprev, Tpprev);
	cv::Mat TPprevcw = cv::Mat::zeros(4, 4, CV_32FC1);
	Rpprev.copyTo(TPprevcw.rowRange(0, 3).colRange(0, 3));
	Tpprev.copyTo(TPprevcw.col(3).rowRange(0, 3));
	g2o::VertexSE3Expmap * vSE3Pprev = new g2o::VertexSE3Expmap();
	vSE3Pprev->setEstimate(Converter::toSE3Quat(TPprevcw));
	vSE3Pprev->setId(pPPrevKF->mnFrameID);
	vSE3Pprev->setFixed(true);
	optimizer.addVertex(vSE3Pprev);
	///////KF PPrev추가

	const float thHuber2D = sqrt(5.99);
	const float thHuber3D = sqrt(7.815);
	////새로 추가된 맵포인트 설정
	for (size_t i = 0; i<vP3Ds.size(); i++)
	{
		
		g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
		vPoint->setEstimate(Converter::toVector3d(vP3Ds[i]));
		const int id = i + maxKFid + 1;
		vPoint->setId(id);
		vPoint->setMarginalized(true);
		optimizer.addVertex(vPoint);
		
		////KFcurr edge
		{
			Eigen::Matrix<double, 2, 1> obs;
			obs << vCurrPTs[i].x, vCurrPTs[i].y;

			g2o::EdgeSE3ProjectXYZ* eCurr= new g2o::EdgeSE3ProjectXYZ();
			eCurr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			eCurr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pCurrKF->mnFrameID)));
			eCurr->setMeasurement(obs);
			eCurr->setInformation(Eigen::Matrix2d::Identity());

			g2o::RobustKernelHuber* rkCurr = new g2o::RobustKernelHuber;
			eCurr->setRobustKernel(rkCurr);
			rkCurr->setDelta(thHuber2D);

			eCurr->fx = pCurrKF->fx;
			eCurr->fy = pCurrKF->fy;
			eCurr->cx = pCurrKF->cx;
			eCurr->cy = pCurrKF->cy;
			optimizer.addEdge(eCurr);
		}
		////KFcurr edge

		////KFprev edge
		{
			Eigen::Matrix<double, 2, 1> obsPrev;
			obsPrev << vPrevPTs[i].x, vPrevPTs[i].y;

			g2o::EdgeSE3ProjectXYZ* ePrev = new g2o::EdgeSE3ProjectXYZ();
			ePrev->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			ePrev->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pPrevKF->mnFrameID)));
			ePrev->setMeasurement(obsPrev);
			ePrev->setInformation(Eigen::Matrix2d::Identity());

			g2o::RobustKernelHuber* rkPrev = new g2o::RobustKernelHuber;
			ePrev->setRobustKernel(rkPrev);
			rkPrev->setDelta(thHuber2D);

			ePrev->fx = pCurrKF->fx;
			ePrev->fy = pCurrKF->fy;
			ePrev->cx = pCurrKF->cx;
			ePrev->cy = pCurrKF->cy;
			optimizer.addEdge(ePrev);
		}
		////KFprev edge

		////KFpprev edge
		{
			Eigen::Matrix<double, 2, 1> obsPprev;
			obsPprev << vPPrevPTs[i].x, vPPrevPTs[i].y;

			g2o::EdgeSE3ProjectXYZ* ePprev = new g2o::EdgeSE3ProjectXYZ();
			ePprev->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			ePprev->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pPPrevKF->mnFrameID)));
			ePprev->setMeasurement(obsPprev);
			ePprev->setInformation(Eigen::Matrix2d::Identity());

			g2o::RobustKernelHuber* rkPprev = new g2o::RobustKernelHuber;
			ePprev->setRobustKernel(rkPprev);
			rkPprev->setDelta(thHuber2D);

			ePprev->fx = pCurrKF->fx;
			ePprev->fy = pCurrKF->fy;
			ePprev->cx = pCurrKF->cx;
			ePprev->cy = pCurrKF->cy;
			optimizer.addEdge(ePprev);
		}
		////KFprev edge

		vbNotIncludedMP[i] = false;
	}
	////새로 추가된 맵포인트 설정

	optimizer.initializeOptimization();
	optimizer.optimize(10);

	//Curr KF 포즈 수정
	{
		g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pCurrKF->mnFrameID));
		g2o::SE3Quat SE3quat = vSE3->estimate();
		cv::Mat R, t;
		cv::Mat Tcw = Converter::toCvMat(SE3quat);
		R = Tcw.rowRange(0, 3).colRange(0, 3);
		t = Tcw.rowRange(0, 3).col(3);
		pCurrKF->SetPose(R, t);
	}
	////포인트 복원
	for (size_t i = 0; i<vP3Ds.size(); i++)
	{
		if (vbNotIncludedMP[i])
			continue;
		g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i + maxKFid + 1));
		vP3Ds[i] = Converter::toCvMat(vPoint->estimate()).clone();
	}
	////포인트 복원
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////