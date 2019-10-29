#include <Initializer.h>
#include <FrameWindow.h>
#include <MatrixOperator.h>

//추후 파라메터화. 귀찮아.
int N_matching_init_therah = 80;
int N_thresh_init_triangulate = 80;

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

void UVR_SLAM::Initializer::SetMatcher(UVR_SLAM::Matcher* pMatcher) {
	mpMatcher = pMatcher;
}

void UVR_SLAM::Initializer::SetFrameWindow(UVR_SLAM::FrameWindow* pWindow) {
	mpFrameWindow = pWindow;
}

bool UVR_SLAM::Initializer::Initialize(Frame* pFrame, int w, int h) {
	//std::cout << "Initializer::Initialize::Start" << std::endl;
	
	if (!mpInitFrame1) {
		mpInitFrame1 = pFrame;
		return mbInit;
	}
	else {
		//매칭이 적으면 mpInitFrame1을 mpInitFrame2로 교체
		cv::Mat F;
		std::vector<cv::DMatch> resMatches;
		mpInitFrame2 = pFrame;
		int count = mpMatcher->MatchingProcessForInitialization(mpInitFrame1, mpInitFrame2, F, resMatches);
		if (count < N_matching_init_therah) {
			delete mpInitFrame1;
			mpInitFrame1 = mpInitFrame2;
			return mbInit;
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
		//cvtColor(vis1, vis1, CV_8UC3);

		cv::RNG rng = cv::RNG(12345);

		if (resIDX > 0 && vCandidates[resIDX]->nGood > N_thresh_init_triangulate) {

			std::cout << "Initialization::EpipolarGeometry" << std::endl;
			std::cout << "R::" << vCandidates[resIDX]->R << ", " << vCandidates[resIDX]->t << std::endl;

			//최적화 수행 후 Map 생성
			bool bInitOpt = false;
			UVR_SLAM::Optimization::InitOptimization(vCandidates[resIDX], resMatches, mpInitFrame1, mpInitFrame2, mK, bInitOpt);
			if (!bInitOpt)
				return mbInit;
			std::cout << "Initialization::Optimization" << std::endl;
			std::cout << "R::" << vCandidates[resIDX]->R << ", " << vCandidates[resIDX]->t << std::endl;
			//scale 잡아주기
			//median depth check
			std::vector<float> vDepths;
			for (int i = 0; i < vCandidates[resIDX]->mvX3Ds.size(); i++) {
				if (vCandidates[resIDX]->vbTriangulated[i]) {
					cv::Mat x = vCandidates[resIDX]->R*vCandidates[resIDX]->mvX3Ds[i]+ vCandidates[resIDX]->t;
					float depth = x.at<float>(2);
					vDepths.push_back(depth);
				}
			}
			std::nth_element(vDepths.begin(), vDepths.begin() + vDepths.size() / 2, vDepths.end());
			float medianDepth = vDepths[(vDepths.size()) / 2];
			float invMedianDepth = 1.0f / medianDepth;

			mpInitFrame1->SetPose(vCandidates[resIDX]->R0, vCandidates[resIDX]->t0);
			mpInitFrame2->SetPose(vCandidates[resIDX]->R, vCandidates[resIDX]->t*invMedianDepth);

			for (int i = 0; i < vCandidates[resIDX]->mvX3Ds.size(); i++) {
				if (vCandidates[resIDX]->vbTriangulated[i]) {
					int idx1 = resMatches[i].queryIdx;
					int idx2 = resMatches[i].trainIdx;
					UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(vCandidates[resIDX]->mvX3Ds[i]* invMedianDepth, mpInitFrame2->matDescriptor.row(idx2));
					pNewMP->AddFrame(mpInitFrame1, idx1);
					pNewMP->AddFrame(mpInitFrame2, idx2);
				}
			}
			//queue에 두개 넣기
			mpInitFrame1->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
			mpInitFrame2->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
			mpFrameWindow->push_back(mpInitFrame1);
			mpFrameWindow->push_back(mpInitFrame2);
			
			mpFrameWindow->SetPose(vCandidates[resIDX]->R, vCandidates[resIDX]->t);
			mpFrameWindow->SetLocalMap();
			mpFrameWindow->SetVectorInlier(mpFrameWindow->LocalMapSize, false);

			mbInit = true;

			if (mbInit) {
				for (int i = 0; i < vCandidates[resIDX]->vbTriangulated.size(); i++) {
					if (!vCandidates[resIDX]->vbTriangulated[i])
						continue;
					int idx1 = resMatches[i].queryIdx;
					int idx2 = resMatches[i].trainIdx;

					UVR_SLAM::MapPoint* pMP = mpInitFrame1->GetMapPoint(idx1);
					cv::Mat pCam1;
					cv::Point2f p2D1;
					bool bProjection1 = pMP->Projection(p2D1, pCam1, mpInitFrame1->GetRotation(), mpInitFrame1->GetTranslation(), mpInitFrame1->mK, w, h);
					cv::Mat pCam2;
					cv::Point2f p2D2;
					bool bProjection2 = pMP->Projection(p2D2, pCam2, mpInitFrame2->GetRotation(), mpInitFrame2->GetTranslation(), mpInitFrame2->mK, w, h);

					cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

					circle(vis1, p2D1, 3, color, -1);
					circle(vis2, p2D2, 3, color, -1);
					/*imshow("Initialization::Results::Frame::1", vis1);
					imshow("Initialization::Results::Frame::2", vis2);
					cv::waitKey(0);*/
				}
			}
		}

		imshow("Initialization::Frame::1", vis1);
		imshow("Initialization::Frame::2", vis2);
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

	u.col(2).copyTo(t1); // or UZU.t()xt=가 0이여서
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

	//vector map을 대신할 무엇인가가 필요함.

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
		Triangulate(kp1.pt, kp2.pt, P1, P2, X3D);

		float cosParallax;
		bool res = CheckCreatedPoints(X3D, kp1.pt, kp2.pt, O1, O2, R, t, candidate->R, candidate->t, cosParallax, th2);
		if (res) {
			vCosParallax.push_back(cosParallax);
			//candidate->vMap3D[i] = new MapPoint(X3D);
			/*
			{
			//항상 초기화 시에 수행해야 할 듯.
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

void UVR_SLAM::Initializer::Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2, cv::Mat& x3D){
	cv::Mat A(4, 4, CV_32F);

	A.row(0) = pt1.x*P1.row(2) - P1.row(0);
	A.row(1) = pt1.y*P1.row(2) - P1.row(1);
	A.row(2) = pt2.x*P2.row(2) - P2.row(0);
	A.row(3) = pt2.y*P2.row(2) - P2.row(1);

	cv::Mat u, w, vt;
	cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	x3D = vt.row(3).t();
	x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}
bool UVR_SLAM::Initializer::CheckCreatedPoints(cv::Mat X3D, cv::Point2f kp1, cv::Point2f kp2, cv::Mat O1, cv::Mat O2, cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, float& cosParallax, float th2) {
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

	// Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
	if (p3dC1.at<float>(2) <= 0.0f && cosParallax<0.99998f)
		return false;

	// Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
	if (p3dC2.at<float>(2) <= 0.0f && cosParallax<0.99998f)
		return false;

	// Check reprojection error in first image
	cv::Mat reproj1 = mK*p3dC1;
	reproj1 /= p3dC1.at<float>(2);
	float squareError1 = (reproj1.at<float>(0) - kp1.x)*(reproj1.at<float>(0) - kp1.x) + (reproj1.at<float>(1) - kp1.y)*(reproj1.at<float>(1) - kp1.y);
	if (squareError1>th2)
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