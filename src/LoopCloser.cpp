#include "LoopCloser.h"
#include "Frame.h"
#include <MapPoint.h>
#include "System.h"
#include <Converter.h>
#include <Matcher.h>
#include <KeyframeDatabase.h>
#include <Sim3Solver.h>
#include <Optimization.h>
#include "Map.h"

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
	LoopCloser::LoopCloser(System* pSys):mpSystem(pSys), mbFixScale(false), mbProcessing(false), mnCovisibilityConsistencyTh(3){
	}
	LoopCloser::~LoopCloser() {}
	void LoopCloser::Init() {
		mpMap = mpSystem->mpMap;
		mpKeyFrameDatabase = mpSystem->mpKeyframeDatabase;
		mpVoc = mpSystem->mpDBoWVoc;
		mpMatcher = mpSystem->mpMatcher;
		mK = mpSystem->mK.clone();
		mnWidth = mpSystem->mnWidth;
		mnHeight = mpSystem->mnHeight;
		mInvK = mpSystem->mInvK.clone();
	}
	void UVR_SLAM::LoopCloser::RunWithMappingServer() {
		std::cout << "MappingServer::LoopCloser::Start" << std::endl;
		while (true) {

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
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		mpTargetFrame = mKFQueue.front();
		mKFQueue.pop();
		mpTargetFrame->ComputeBoW();
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