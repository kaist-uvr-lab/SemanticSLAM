#include <Initializer.h>
#include <FrameWindow.h>
#include <LocalMapper.h>
#include <MatrixOperator.h>

//���� �Ķ����ȭ. ������.
int N_matching_init_therah = 120; //80
int N_thresh_init_triangulate = 120; //80

UVR_SLAM::Initializer::Initializer() :mbInit(false), mpInitFrame1(nullptr), mpInitFrame2(nullptr) {
}
UVR_SLAM::Initializer::Initializer(cv::Mat _K):mK(_K),mbInit(false), mpInitFrame1(nullptr), mpInitFrame2(nullptr){
}
UVR_SLAM::Initializer::~Initializer(){

}

void UVR_SLAM::Initializer::Init() {
	mpInitFrame1 = nullptr;
	mpInitFrame2 = nullptr;
	mbInit = false;
}

void UVR_SLAM::Initializer::SetLocalMapper(LocalMapper* pMapper) {
	mpLocalMapper = pMapper;
}

void UVR_SLAM::Initializer::SetMatcher(UVR_SLAM::Matcher* pMatcher) {
	mpMatcher = pMatcher;
}

void UVR_SLAM::Initializer::SetFrameWindow(UVR_SLAM::FrameWindow* pWindow) {
	mpFrameWindow = pWindow;
}

bool UVR_SLAM::Initializer::Initialize(Frame* pFrame, bool& bReset, int w, int h) {
	//std::cout << "Initializer::Initialize::Start" << std::endl;
	
	if (!mpInitFrame1) {
		mpInitFrame1 = pFrame;
		return mbInit;
	}
	else {
		//��Ī�� ������ mpInitFrame1�� mpInitFrame2�� ��ü
		cv::Mat F;
		std::vector<cv::DMatch> tempMatches, resMatches;
		mpInitFrame2 = pFrame;
		if (mpInitFrame2->GetFrameID() - mpInitFrame1->GetFrameID() < 3)
			return mbInit;
		int count = mpMatcher->MatchingProcessForInitialization(mpInitFrame1, mpInitFrame2, F, tempMatches);
		//int count = mpMatcher->SearchForInitialization(mpInitFrame1, mpInitFrame2, tempMatches, 100);
		if (count < N_matching_init_therah) {
			delete mpInitFrame1;
			mpInitFrame1 = mpInitFrame2;
			return mbInit;
		}

		//Fã��
		std::vector<bool> mvInliers;
		float score;

		if ((int)tempMatches.size() >= 8) {
			mpMatcher->FindFundamental(mpInitFrame1, mpInitFrame2, tempMatches, mvInliers, score, F);
			F.convertTo(F, CV_32FC1);
		}
		if ((int)tempMatches.size() < 8 || F.empty()) {
			F.release();
			F = cv::Mat::zeros(0, 0, CV_32FC1);
			delete mpInitFrame1;
			mpInitFrame1 = mpInitFrame2;
			return mbInit;
		}

		for (unsigned long i = 0; i < tempMatches.size(); i++) {
			//if(inlier_mask.at<uchar>((int)i)) {
			if (mvInliers[i]) {

				cv::Point2f pt1 = mpInitFrame1->mvKeyPoints[tempMatches[i].queryIdx].pt;
				cv::Point2f pt2 = mpInitFrame2->mvKeyPoints[tempMatches[i].trainIdx].pt;
				//init->mvnCPMatchingIdx.push_back(vMatches[i].queryIdx);
				//curr->mvnCPMatchingIdx.push_back(vMatches[i].trainIdx);
				resMatches.push_back(tempMatches[i]);
			}
		}

		//std::cout << "matching res = " << count<<", "<< resMatches.size() << std::endl;

		std::vector<UVR_SLAM::InitialData*> vCandidates;
		UVR_SLAM::InitialData *mC1 = new UVR_SLAM::InitialData(count);
		UVR_SLAM::InitialData *mC2 = new UVR_SLAM::InitialData(count);
		UVR_SLAM::InitialData *mC3 = new UVR_SLAM::InitialData(count);
		UVR_SLAM::InitialData *mC4 = new UVR_SLAM::InitialData(count);
		vCandidates.push_back(mC1);
		vCandidates.push_back(mC2);
		vCandidates.push_back(mC3);
		vCandidates.push_back(mC4);
		
		SetCandidatePose(F, resMatches, vCandidates);
		int resIDX = SelectCandidatePose(vCandidates);

		cv::Mat vis1 = mpInitFrame1->GetOriginalImage();
		cv::Mat vis2 = mpInitFrame2->GetOriginalImage();
		cv::Point2f ptBottom = cv::Point2f(0, vis1.rows);
		cv::Rect mergeRect1 = cv::Rect(0, 0, vis1.cols, vis1.rows);
		cv::Rect mergeRect2 = cv::Rect(0, vis1.rows, vis1.cols, vis1.rows);
		cv::Mat debugging = cv::Mat::zeros(vis1.rows * 2, vis1.cols, vis1.type());
		vis1.copyTo(debugging(mergeRect1));
		vis2.copyTo(debugging(mergeRect2));
		//cvtColor(vis1, vis1, CV_8UC3);

		cv::RNG rng = cv::RNG(12345);

		

		if (resIDX > 0 && vCandidates[resIDX]->nGood > N_thresh_init_triangulate) {
			std::cout << vCandidates[resIDX]->nGood << std::endl;
			mpInitFrame1->SetPose(vCandidates[resIDX]->R0, vCandidates[resIDX]->t0);
			mpInitFrame2->SetPose(vCandidates[resIDX]->R, vCandidates[resIDX]->t); //�ι�° �������� median depth�� �����ؾ� ��.

			//Ű���������� ����
			mpInitFrame1->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
			mpInitFrame2->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);

			//�����쿡 �� ���� Ű������ �ֱ�
			//20.01.02 deque���� list�� ������.
			mpFrameWindow->AddFrame(mpInitFrame1);
			mpFrameWindow->AddFrame(mpInitFrame2);

			mpInitFrame1->mTrackedDescriptor = cv::Mat::zeros(0, mpInitFrame1->matDescriptor.cols, mpInitFrame1->matDescriptor.type());
			mpInitFrame2->mTrackedDescriptor = cv::Mat::zeros(0, mpInitFrame2->matDescriptor.cols, mpInitFrame2->matDescriptor.type());
			//mpFrameWindow->push_back(mpInitFrame1);
			//mpFrameWindow->push_back(mpInitFrame2);

			//������Ʈ ���� �� Ű�����Ӱ� ����
			int nMatch = 0;
			std::vector<MapPoint*> vpMPs;
			std::vector<UVR_SLAM::Frame*> vpKFs;
			vpKFs.push_back(mpInitFrame1);
			vpKFs.push_back(mpInitFrame2);
			for (int i = 0; i < vCandidates[resIDX]->mvX3Ds.size(); i++) {
				if (vCandidates[resIDX]->vbTriangulated[i]) {
					int idx1 = resMatches[i].queryIdx;
					int idx2 = resMatches[i].trainIdx;
					UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpInitFrame1, vCandidates[resIDX]->mvX3Ds[i], mpInitFrame2->matDescriptor.row(idx2));
					pNewMP->AddFrame(mpInitFrame1, idx1);
					pNewMP->AddFrame(mpInitFrame2, idx2);
					pNewMP->mnFirstKeyFrameID = mpInitFrame2->GetKeyFrameID();

					//update
					mpInitFrame1->mTrackedDescriptor.push_back(mpInitFrame1->matDescriptor.row(idx1));
					mpInitFrame1->mvTrackedIdxs.push_back(idx1);

					mpInitFrame2->mTrackedDescriptor.push_back(mpInitFrame2->matDescriptor.row(idx2));
					mpInitFrame2->mvTrackedIdxs.push_back(idx2);

					//local map�� ��
					mpLocalMapper->mlpNewMPs.push_back(pNewMP);

					nMatch++;

					vpMPs.push_back(pNewMP);
				}
			}


			//����ȭ ���� �� Map ����
			//UVR_SLAM::Optimization::InitOptimization(vCandidates[resIDX], resMatches, mpInitFrame1, mpInitFrame2, mK, bInitOpt);
			UVR_SLAM::Optimization::InitBundleAdjustment(vpKFs, vpMPs, 20);

			//calculate median depth
			float medianDepth;
			mpInitFrame1->ComputeSceneMedianDepth(medianDepth);
			float invMedianDepth = 1.0f / medianDepth;

			if (medianDepth < 0.0 || mpInitFrame2->TrackedMapPoints(1) < 100){
				mbInit = false;
				bReset = true;
				return mbInit;
			}
			//���� ������Ʈ
			cv::Mat R, t;
			mpInitFrame2->GetPose(R, t);
			mpInitFrame2->SetPose(R, t*invMedianDepth);

			//������Ʈ ������Ʈ
			for (int i = 0; i < vpMPs.size(); i++) {
				UVR_SLAM::MapPoint* pMP = vpMPs[i];
				if (!pMP)
					continue;
				if (pMP->isDeleted())
					continue;
				pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
				pMP->UpdateNormalAndDepth();
			}

			//������ ���ø�, ���� ����
			mpFrameWindow->SetPose(R, t*invMedianDepth);
			mpFrameWindow->SetLocalMap(mpInitFrame2->GetFrameID());
			mpFrameWindow->SetLastFrameID(mpInitFrame2->GetFrameID());
			mpFrameWindow->mnLastMatches = nMatch;
			mbInit = true;

			//if (mbInit) {
			//	//test
			//	for (int i = 0; i < vCandidates[resIDX]->vbTriangulated.size(); i++) {
			//		if (!vCandidates[resIDX]->vbTriangulated[i])
			//			continue;
			//		int idx1 = resMatches[i].queryIdx;
			//		int idx2 = resMatches[i].trainIdx;

			//		UVR_SLAM::MapPoint* pMP = mpInitFrame1->mvpMPs[idx1];
			//		cv::Mat pCam1;
			//		cv::Point2f p2D1;
			//		bool bProjection1 = pMP->Projection(p2D1, pCam1, mpInitFrame1->GetRotation(), mpInitFrame1->GetTranslation(), mpInitFrame1->mK, w, h);
			//		cv::Mat pCam2;
			//		cv::Point2f p2D2;
			//		bool bProjection2 = pMP->Projection(p2D2, pCam2, mpInitFrame2->GetRotation(), mpInitFrame2->GetTranslation(), mpInitFrame2->mK, w, h);

			//		cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

			//		circle(debugging, p2D1, 2, color, -1);
			//		line(debugging, p2D1, mpInitFrame1->mvKeyPoints[idx1].pt, cv::Scalar(0, 255, 255));

			//		circle(debugging, p2D2+ ptBottom, 2, color, -1);
			//		line(debugging, p2D2+ ptBottom, mpInitFrame2->mvKeyPoints[idx2].pt+ ptBottom, cv::Scalar(0, 255, 255));

			//		//line(debugging, p2D1, p2D2 + ptBottom, cv::Scalar(255, 255, 0), 1);
			//		/*imshow("Initialization::Results::Frame::1", vis1);
			//		imshow("Initialization::Results::Frame::2", vis2);
			//		cv::waitKey(0);*/
			//	}
			//}
			std::cout << "Initializer::" << nMatch << std::endl;
		}

		//imshow("Initialization::Frame::1", debugging);
		//imshow("Initialization::Frame::2", vis2);
		waitKey(1);

	}
	//std::cout << "Initializer::Initialize::End" << std::endl;
	return mbInit;
}

void UVR_SLAM::Initializer::SetCandidatePose(cv::Mat F, std::vector<cv::DMatch> Matches, std::vector<UVR_SLAM::InitialData*>& vCandidates) {
	//E
	Mat E = mK.t()*F*mK;;
	//Decompose E

	float th = 4.0f;
	Mat R1, R2, t1, t2;
	DecomposeE(E, R1, R2, t1, t2);
	vCandidates[0]->SetRt(R1, t1);
	vCandidates[1]->SetRt(R2, t1);
	vCandidates[2]->SetRt(R1, t2);
	vCandidates[3]->SetRt(R2, t2);
	CheckRT(Matches, vCandidates[0], th);
	CheckRT(Matches, vCandidates[1], th);
	CheckRT(Matches, vCandidates[2], th);
	CheckRT(Matches, vCandidates[3], th);
}

void UVR_SLAM::Initializer::DecomposeE(cv::Mat E, cv::Mat &R1, cv::Mat& R2, cv::Mat& t1, cv::Mat& t2){
	cv::Mat u, w, vt;
	cv::SVD::compute(E, w, u, vt);

	u.col(2).copyTo(t1); // or UZU.t()xt=�� 0�̿���
	float tempNorm = (float)cv::norm(t1);
	t1 = t1 / tempNorm;
	t2 = -1.0f*t1.clone();

	cv::Mat W = cv::Mat::zeros(3, 3, CV_32FC1);
	W.at<float>(0, 1) = -1.0f;
	W.at<float>(1, 0) = 1.0f;
	W.at<float>(2, 2) = 1.0f;

	R1 = u*W*vt;
	if (cv::determinant(R1)<0.0)
		R1 = -R1;

	R2 = u*W.t()*vt;
	if (cv::determinant(R2)<0.0)
		R2 = -R2;
}
void UVR_SLAM::Initializer::CheckRT(std::vector<cv::DMatch> Matches, UVR_SLAM::InitialData* candidate, float th2) {

	//vector map�� ����� �����ΰ��� �ʿ���.

	candidate->vbTriangulated = std::vector<bool>(Matches.size(), false);
	candidate->mvX3Ds = std::vector<cv::Mat>(Matches.size(), cv::Mat::zeros(3, 1, CV_32FC1));
	//candidate->vMap3D = std::vector<UVR::MapPoint*>(pInitFrame->mvnCPMatchingIdx.size(), nullptr);
	//candidate->vP3D.resize(pKF->mvnMatchingIdx.size());

	std::vector<float> vCosParallax;
	//vCosParallax.reserve(pKF->mvnMatchingIdx.size());

	cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
	cv::Mat t = cv::Mat::zeros(3, 1, CV_32FC1);

	// Camera 1 Projection Matrix K[I|0]
	cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
	mK.copyTo(P1.rowRange(0, 3).colRange(0, 3));
	cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

	// Camera 2 Projection Matrix K[R|t]
	cv::Mat P2(3, 4, CV_32F);
	candidate->R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
	candidate->t.copyTo(P2.rowRange(0, 3).col(3));
	P2 = mK*P2;

	cv::Mat O2 = -candidate->R.t()*candidate->t;

	for (unsigned long i = 0; i < Matches.size(); i++)
	{

		const cv::KeyPoint &kp1 = mpInitFrame1->mvKeyPoints[Matches[i].queryIdx];
		const cv::KeyPoint &kp2 = mpInitFrame2->mvKeyPoints[Matches[i].trainIdx];
		cv::Mat X3D;

		if (!Triangulate(kp1.pt, kp2.pt, P1, P2, X3D))
			continue;

		float cosParallax;

		bool res = CheckCreatedPoints(X3D, kp1.pt, kp2.pt, O1, O2, R, t, candidate->R, candidate->t, cosParallax, th2,th2);
		if (res) {
			vCosParallax.push_back(cosParallax);
			//candidate->vMap3D[i] = new MapPoint(X3D);
			/*
			{
			//�׻� �ʱ�ȭ �ÿ� �����ؾ� �� ��.
			cv::Mat Ow = -R.t()*t;
			cv::Mat PC = X3D-Ow;
			float dist = cv::norm(PC);
			int level = kp2.octave;
			int nLevels = pF->mnScaleLevels;
			float levelScaleFactor = pF->mvScaleFactors[level];
			candidate->vMap3D[i]->mfMaxDistance = dist*levelScaleFactor;
			candidate->vMap3D[i]->mfMinDistance = candidate->vMap3D[i]->mfMaxDistance / pF->mvScaleFactors[nLevels-1];
			candidate->vMap3D[i]->mNormalVector = X3D-Ow;
			candidate->vMap3D[i]->mNormalVector = candidate->vMap3D[i]->mNormalVector / cv::norm(candidate->vMap3D[i]->mNormalVector);
			}
			*/
			if (cosParallax<0.99998f) {
				candidate->vbTriangulated[i] = true;
				candidate->mvX3Ds[i] = X3D.clone();
				candidate->nGood++;
			}
		}
	}
	
	if (candidate->nGood>0)
	{
		std::sort(vCosParallax.begin(), vCosParallax.end());
		int idx = 50;
		int nParallaxSize = (int)vCosParallax.size();
		if (idx > nParallaxSize - 1) {
			idx = nParallaxSize - 1;
		}
		candidate->parallax = (float)(acos(vCosParallax[idx])*UVR_SLAM::MatrixOperator::rad2deg);
	}
	else {
		candidate->parallax = 0.0f;
	}
}

bool UVR_SLAM::Initializer::Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2, cv::Mat& x3D){
	cv::Mat A(4, 4, CV_32F);

	A.row(0) = pt1.x*P1.row(2) - P1.row(0);
	A.row(1) = pt1.y*P1.row(2) - P1.row(1);
	A.row(2) = pt2.x*P2.row(2) - P2.row(0);
	A.row(3) = pt2.y*P2.row(2) - P2.row(1);

	cv::Mat u, w, vt;
	cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	x3D = vt.row(3).t();

	float a = w.at<float>(3);
	if (a < 0.001)
		return false;
	if (abs(x3D.at<float>(3)) <= 0.001)
		return false;

	//if (abs(x3D.at<float>(3)) < 0.01)
	//	std::cout << "abc:" << x3D.at<float>(3) << std::endl;

	x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
	
	return true;
}
bool UVR_SLAM::Initializer::CheckCreatedPoints(cv::Mat X3D, cv::Point2f kp1, cv::Point2f kp2, cv::Mat O1, cv::Mat O2, cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, float& cosParallax, float th1, float th2) {
	if (!std::isfinite(X3D.at<float>(0)) || !std::isfinite(X3D.at<float>(1)) || !std::isfinite(X3D.at<float>(2)))
	{
		return false;
	}

	cv::Mat p3dC1 = R1*X3D + t1;
	cv::Mat p3dC2 = R2*X3D + t2;
	// Check parallax
	cv::Mat normal1 = p3dC1 - O1;
	float dist1 = (float)cv::norm(normal1);
	cv::Mat normal2 = p3dC1 - O2;
	float dist2 = (float)cv::norm(normal2);

	cosParallax = ((float)normal1.dot(normal2)) / (dist1*dist2);

	if (cosParallax >= 0.99998f)
		return false;
	//std::cout << p3dC1 << ", " << p3dC2 << ", " << th1 <<", "<< th2 << std::endl;
	// Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
	if (p3dC1.at<float>(2) <= 0.0f || p3dC2.at<float>(2) <= 0.0f )
		return false;

	// Check reprojection error in first image
	cv::Mat reproj1 = mK*p3dC1;
	reproj1 /= p3dC1.at<float>(2);
	float squareError1 = (reproj1.at<float>(0) - kp1.x)*(reproj1.at<float>(0) - kp1.x) + (reproj1.at<float>(1) - kp1.y)*(reproj1.at<float>(1) - kp1.y);
	
	//std::cout << squareError1 << std::endl;
	if (squareError1>th1)
		return false;

	// Check reprojection error in second image
	cv::Mat reproj2 = mK*p3dC2;
	reproj2 /= p3dC2.at<float>(2);
	float squareError2 = (reproj2.at<float>(0) - kp2.x)*(reproj2.at<float>(0) - kp2.x) + (reproj2.at<float>(1) - kp2.y)*(reproj2.at<float>(1) - kp2.y);
	
	if (squareError2>th2)
		return false;
	return true;
}

int UVR_SLAM::Initializer::SelectCandidatePose(std::vector<UVR_SLAM::InitialData*>& vCandidates) {
	//int SelectCandidatePose(UVR::InitialData* c1, UVR::InitialData* c2, UVR::InitialData* c3, UVR::InitialData* c4){
	float minParallax = 1.0f;
	int   minTriangulated = 80;

	unsigned long maxIdx = (unsigned long)-1;
	int nMaxGood = -1;
	for (unsigned long i = 0; i < vCandidates.size(); i++) {
		if (vCandidates[i]->nGood > nMaxGood) {
			maxIdx = i;
			nMaxGood = vCandidates[i]->nGood;
		}
	}
	std::cout << "max::" << nMaxGood << std::endl;
	int nsimilar = 0;
	int th_good = (int)(0.7f*(float)nMaxGood);
	int nMinGood = (int)(0.9f*(float)vCandidates[0]->nMinGood);
	if (nMinGood < minTriangulated) {
		nMinGood = minTriangulated;
	}
	for (unsigned long i = 0; i < vCandidates.size(); i++) {
		if (vCandidates[i]->nGood>th_good) {
			nsimilar++;
		}
	}

	int res = -1;
	if (vCandidates[maxIdx]->parallax > minParallax && nMaxGood<nMinGood && nsimilar == 1) {
		res = (int)maxIdx;
	}

	return res;
}