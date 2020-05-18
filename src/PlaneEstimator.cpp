#include <PlaneEstimator.h>
#include <random>
#include <System.h>
#include <Map.h>
#include <Plane.h>
#include <Frame.h>
#include <SegmentationData.h>
#include <MapPoint.h>
#include <Matcher.h>
#include <Visualizer.h>
#include <Initializer.h>
#include <MatrixOperator.h>

static int nPlaneID = 0;

UVR_SLAM::PlaneEstimator::PlaneEstimator() :mbDoingProcess(false), mnProcessType(0), mpLayoutFrame(nullptr){
}
UVR_SLAM::PlaneEstimator::PlaneEstimator(Map* pMap,std::string strPath,cv::Mat K, cv::Mat K2, int w, int h) : mK(K), mK2(K2),mbDoingProcess(false), mnWidth(w), mnHeight(h), mnProcessType(0), mpLayoutFrame(nullptr),
mpPrevFrame(nullptr), mpPPrevFrame(nullptr), mpTargetFrame(nullptr)
{
	cv::FileStorage fSettings(strPath, cv::FileStorage::READ);
	mnRansacTrial = fSettings["Layout.trial"];
	mfThreshPlaneDistance = fSettings["Layout.dist"];
	mfThreshPlaneRatio = fSettings["Layout.ratio"];
	mfThreshNormal = fSettings["Layout.normal"];

	//mnNeedFloorMPs = fSettings["Layout.nfloor"];
	//mnNeedWallMPs = fSettings["Layout.nwall"];
	//mnNeedCeilMPs = fSettings["Layout.nceil"];
	//mnConnect = fSettings["Layout.nconnect"];
	fSettings.release();

	mpMap = pMap;
}
void UVR_SLAM::PlaneInformation::SetParam(cv::Mat m) {
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
	matPlaneParam = cv::Mat::zeros(4, 1, CV_32FC1);
	matPlaneParam = m.clone();

	normal = matPlaneParam.rowRange(0, 3);
	distance = matPlaneParam.at<float>(3);
	norm = normal.dot(normal);
	normal /= norm;
	distance /= norm;
	norm = 1.0;

	normal.copyTo(matPlaneParam.rowRange(0, 3));
	matPlaneParam.at<float>(3) = distance;
}
void UVR_SLAM::PlaneInformation::SetParam(cv::Mat n, float d){
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
	matPlaneParam = cv::Mat::zeros(4, 1, CV_32FC1);
	normal = n.clone();
	distance = d;
	norm = normal.dot(normal);
	normal.copyTo(matPlaneParam.rowRange(0, 3));
	matPlaneParam.at<float>(3) = d;
}
void UVR_SLAM::PlaneInformation::GetParam(cv::Mat& n, float& d){
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
	n = normal.clone();
	d = distance;
}
cv::Mat UVR_SLAM::PlaneInformation::GetParam() {
	std::unique_lock<std::mutex> lockTemp(mMutexParam);
	return matPlaneParam.clone();
}
UVR_SLAM::PlaneEstimator::~PlaneEstimator() {}

///////////////////////////////////////////////////////////////////////////////
//�⺻ �Լ���
void UVR_SLAM::PlaneEstimator::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}

void UVR_SLAM::PlaneEstimator::SetVisualizer(UVR_SLAM::Visualizer* pVis) {
	mpVisualizer = pVis;
}

void UVR_SLAM::PlaneEstimator::SetInitializer(UVR_SLAM::Initializer* pInitializer) {
	mpInitializer = pInitializer;
}
void UVR_SLAM::PlaneEstimator::SetMatcher(UVR_SLAM::Matcher* pMatcher) {
	mpMatcher = pMatcher;
}

void UVR_SLAM::PlaneEstimator::SetBoolDoingProcess(bool b, int ptype) {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	mbDoingProcess = b;
	mnProcessType = ptype;
}
bool UVR_SLAM::PlaneEstimator::isDoingProcess() {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	return mbDoingProcess;
}

void UVR_SLAM::PlaneEstimator::InsertKeyFrame(UVR_SLAM::Frame *pKF)
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mKFQueue.push(pKF);
}

bool UVR_SLAM::PlaneEstimator::CheckNewKeyFrames()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	return(!mKFQueue.empty());
}

void UVR_SLAM::PlaneEstimator::ProcessNewKeyFrame()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mpPPrevFrame = mpPrevFrame;
	mpPrevFrame = mpTargetFrame;
	mpTargetFrame = mKFQueue.front();
	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_LAYOUT_FRAME);
	//mpSystem->SetPlaneFrameID(mpTargetFrame->GetKeyFrameID());

	//�ӽ÷� �ּ� ó��
	/*if (mpMap->isFloorPlaneInitialized()) {
		mpTargetFrame->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpTargetFrame, mpMap->mpFloorPlane);
		mpTargetFrame->mpPlaneInformation->Calculate();
	}*/

	mKFQueue.pop();
}
void UVR_SLAM::PlaneEstimator::Reset() {
	mpTargetFrame = mpPrevFrame;
	mpPPrevFrame = nullptr;
	mpPrevFrame = nullptr;
}
///////////////////////////////////////////////////////////////////////////////

void UVR_SLAM::PlaneEstimator::Run() {

	std::string mStrPath;

	std::vector<UVR_SLAM::PlaneInformation*> mvpPlanes;

	UVR_SLAM::Frame* pTestF = nullptr;
	UVR_SLAM::PlaneInformation* mpGlobalFloor = nullptr;
	UVR_SLAM::PlaneInformation* mpGlobalCeil = nullptr;

	std::vector<UVR_SLAM::MapPoint*> mvpFloorMPs, mvpWallMPs, mvpCeilMPs;
	std::vector<int> vIndices;

	int mnFontFace = (2);
	double mfFontScale = (0.6);
	int mnDebugFontSize = 20;
	int nFrameSize;
	int nTrial = 500;
	float pidst = 0.05;//0.01
	float pratio = 0.25;

	int nLayoutFloor = 1;
	int nLayoutCeil = 2;
	int nLayoutWall = 3;

	float sumFloor = 0.0;
	int nFloor = 0;

	float sumCeil = 0.0;
	int nCeil = 0;

	while (true) {

		if (CheckNewKeyFrames()) {
			
			//���� ���丮 �� ȹ��
			SetBoolDoingProcess(true,0);
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			ProcessNewKeyFrame();
			std::cout << "pe::start::"<<mpTargetFrame->GetKeyFrameID()<< std::endl;

			///////////////////////////////��� ���� �ð�ȭ ���� ����
			cv::Mat debug = cv::Mat::zeros(1000, 640, CV_8UC1);
			int nDebugRows = 20;
			///////////////////////////////��� ���� �ð�ȭ ���� ����

			/////////////////////////////���� �����ӿ��� �̿��� ������ ����
			int nTargetID = mpTargetFrame->GetKeyFrameID();
			auto matchInfo = mpTargetFrame->mpMatchInfo;
			//���� ������
			auto prevFrame = mpTargetFrame->mpMatchInfo->mpTargetFrame;
			auto prevMatchInfo = prevFrame->mpMatchInfo;
			//�������� ������
			auto prevPrevFrame = prevFrame->mpMatchInfo->mpTargetFrame;
			auto prevPrevMatchInfo = prevPrevFrame->mpMatchInfo;
			/////////////////////////////���� �����ӿ��� �̿��� ������ ����

			//////////////////////////���� ��� ������ ȹ��
			////������ ��� ���� ȹ��
			auto vpPlaneInfos = mpMap->GetPlaneInfos();
			int nSize = vpPlaneInfos.size() - 1;
			PlaneInformation* pLastInfo;
			if (nSize >= 0) {
				pLastInfo = vpPlaneInfos[nSize]->GetPlane(1);
			}
			//////////////////////////���� ��� ������ ȹ��

			/////////////////////////�ĺ� ����Ʈ �߰�
			//GetRecentLayoutFrameID�� ������Ʈ ���̺�� ������.
			std::vector<UVR_SLAM::Frame*> vpKFs;// = prevFrame->GetConnectedKFs(10);
			vpKFs.push_back(prevFrame);
			vpKFs.push_back(prevPrevFrame);
			vpKFs.push_back(mpTargetFrame);
			for (int i = 0; i < vpKFs.size(); i++)
			{
				auto pKFi = vpKFs[i];
				auto matchInfo = pKFi->mpMatchInfo;
				int tempFrameID = pKFi->GetRecentTrackedFrameID();
				
				for (int k = 0; k < matchInfo->mvpMatchingMPs.size(); k++) {
					auto pMP = matchInfo->mvpMatchingMPs[k];
					int label = matchInfo->mvObjectLabels[k];

					//if (!pMP || pMP->isDeleted() || pMP->GetRecentTrackingFrameID() < tempFrameID)
					if (!pMP || pMP->isDeleted())
						continue;
					
					if (label == 150) {
						if (pMP->GetRecentLayoutFrameID() != nLayoutFloor && pMP->GetNumConnectedFrames() >= 7) {
							pMP->SetRecentLayoutFrameID(nLayoutFloor);
							mvpFloorMPs.push_back(pMP);
						}
						pMP->SetPlaneID(1);
					}
					else if (label == 255 && pMP->GetRecentLayoutFrameID() != nLayoutWall) {
						if (pMP->GetNumConnectedFrames()>=7) {
							pMP->SetRecentLayoutFrameID(nLayoutWall);
							mvpWallMPs.push_back(pMP);
						}
					}
					else if (label == 100) {
						if (pMP->GetRecentLayoutFrameID() != nLayoutCeil && pMP->GetNumConnectedFrames() >= 7) {
							pMP->SetRecentLayoutFrameID(nLayoutCeil);
							mvpCeilMPs.push_back(pMP);
						}
						pMP->SetPlaneID(2);
					}
				}
			}
			//�����ε��� ����
			int nMax = max(mvpFloorMPs.size(), max(mvpWallMPs.size(), mvpCeilMPs.size()));
			if (nMax > vIndices.size()) {
				for (int i = vIndices.size(); i < nMax; i++) {
					vIndices.push_back(i);
				}
			}
			////////�߰� ���� ���
			std::stringstream ababab;
			ababab << "Curr::" <<mvpFloorMPs.size() << "||" << mvpWallMPs.size() << "||" << mvpCeilMPs.size();
			cv::putText(debug, ababab.str(), cv::Point2f(0, nDebugRows), mnFontFace, mfFontScale, cv::Scalar::all(255)); nDebugRows += mnDebugFontSize;
			/////////////////////////�ĺ� ����Ʈ �߰�

			/////////////////////////�����ϰ� ����
			std::random_device rd;
			std::mt19937 g(rd());
			std::vector<UVR_SLAM::MapPoint*> vpFloorMPs, vpCeilMPs, vpWallMPs;
			auto vTempIndices1 = std::vector<int>(vIndices.begin(), vIndices.end());
			std::shuffle(vTempIndices1.begin(), vTempIndices1.begin()+mvpFloorMPs.size(), g);
			int nFloorSize = min(2500, mvpFloorMPs.size());
			for (int i = 0; i < nFloorSize; i++) {
				vpFloorMPs.push_back(mvpFloorMPs[vTempIndices1[i]]);
			}
			auto vTempIndices2 = std::vector<int>(vIndices.begin(), vIndices.end());
			std::shuffle(vTempIndices2.begin(), vTempIndices2.begin() + mvpCeilMPs.size(), g);
			int nCeilSize = min(2500, mvpCeilMPs.size());
			for (int i = 0; i < nCeilSize; i++) {
				vpCeilMPs.push_back(mvpCeilMPs[vTempIndices2[i]]);
			}
			/////////////////////////�����ϰ� ����

			////////////////////���� �����ӿ����� �ĺ� ����Ʈ �߰�
			std::vector<UVR_SLAM::MapPoint*> vpCurrFloorMPs, vpCurrCeilMPs, vpCurrWallMPs;
			for (int k = 0; k < prevPrevMatchInfo->mvpMatchingMPs.size(); k++) {
				auto pMP = prevPrevMatchInfo->mvpMatchingMPs[k];
				int label = prevPrevMatchInfo->mvObjectLabels[k];

				if (!pMP || pMP->isDeleted())
					continue;
				if (label == 255) {
					vpCurrWallMPs.push_back(pMP);
				}
				if (label == 150) {
					vpCurrFloorMPs.push_back(pMP);
					//pMP->SetPlaneID(pF);
				}
				if (label == 100) {
					vpCurrCeilMPs.push_back(pMP);
					//pMP->SetPlaneID(2);
				}
			}
			////////////////////���� �����ӿ����� �ĺ� ����Ʈ �߰�

			/////////////////////////////////////////////////////////////////////
			////����� Ű������ ȹ��
			std::vector<UVR_SLAM::MapPoint*> mvpOutlierFloorMPs, mvpOutlierWallMPs, mvpOutlierCeilMPs;
			std::vector<UVR_SLAM::MapPoint*> vpCurrOutlierFloorMPs, vpCurrOutlierCeilMPs, vpCurrOutlierWallMPs1, vpCurrOutlierWallMPs2;
			//std::cout << "pe::candidate::" << mvpFloorMPs.size() << std::endl;
			if (mvpFloorMPs.size() < 50) {
				SetBoolDoingProcess(false, 0);
				continue;
			}
			////�ĺ� ����Ʈ ȹ��
			/////////////////////////////////////////////////////////////////////
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////������ ������ ������ ����
			//bool bCurrFloor = false;
			//bool bCurrCeil = false;
			//bool bCurrWall1 = false;
			//bool bCurrWall2 = false;
			//
			//UVR_SLAM::PlaneInformation* pCurrFloor = new UVR_SLAM::PlaneInformation();
			//UVR_SLAM::PlaneInformation* pCurrCeil = new UVR_SLAM::PlaneInformation();
			//UVR_SLAM::PlaneInformation* pCurrWall1 = new UVR_SLAM::PlaneInformation();
			//UVR_SLAM::PlaneInformation* pCurrWall2 = new UVR_SLAM::PlaneInformation();

			//if (vpCurrFloorMPs.size() > 100) {
			//	bCurrFloor = UVR_SLAM::PlaneInformation::PlaneInitialization2(pCurrFloor, vpCurrFloorMPs, vpCurrOutlierFloorMPs, prevFrame->GetFrameID(), 1500, 0.05, 0.1);
			//	if(bCurrFloor){
			//		std::stringstream ss;
			//		auto p = pCurrFloor->GetParam();
			//		ss << "Floor::Curr::" << pCurrFloor->mvpMPs.size() << ", " << vpCurrOutlierFloorMPs.size() << ", " << vpCurrFloorMPs.size() << std::fixed << std::setprecision(3) << "::[" << p.at<float>(0) << " " << p.at<float>(1) << " " << p.at<float>(2) << " " << p.at<float>(3) << "] ";
			//		cv::putText(debug, ss.str(), cv::Point2f(0, nDebugRows), mnFontFace, mfFontScale, cv::Scalar::all(255)); nDebugRows += mnDebugFontSize;
			//	}
			//}
			//////////������ ������ ������ ����
			///////////////////////////////////////////////////////////////////////////////////////////////////////

			/////////////////////////////�ٴ� ��� ����
			UVR_SLAM::PlaneProcessInformation* pPlaneInfo = new UVR_SLAM::PlaneProcessInformation();
			UVR_SLAM::PlaneInformation* pFloor = new UVR_SLAM::PlaneInformation();
			//vpCurrFloorMPs
			//vpFloorMPs : ��ü ����Ʈ�� ���ؼ� �����ϰ� �Ϻθ� �̾Ƽ� ��� ã��
			bool bFloorRes = UVR_SLAM::PlaneInformation::PlaneInitialization(pFloor, vpCurrFloorMPs, mvpOutlierFloorMPs, prevFrame->GetFrameID(), 1500, pidst, 0.1);
			//float orid;
			if (bFloorRes) {
				////������� �Ÿ��� ��ȯ�ϴ� ��
				//cv::Mat n;
				//float d;
				//pFloor->GetParam(n, d);
				////orid = d;
				//sumFloor += d; nFloor++;
				//d = sumFloor / nFloor;
				//pFloor->SetParam(n, d);
				////������� �Ÿ��� ��ȯ�ϴ� ��
				mpGlobalFloor = pFloor;
				////////��� �Ÿ��� ����غ���
				//auto pp = pFloor->GetParam();
				//pp.at<float>(3) = 0.0;
				//cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
				//for (int i = 0; i < mvpFloorMPs.size(); i++) {
				//	UVR_SLAM::MapPoint* pMP = mvpFloorMPs[i];
				//	if (!pMP || pMP->isDeleted())
				//		continue;
				//	cv::Mat temp = pMP->GetWorldPos();
				//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
				//	mMat.push_back(temp.t());
				//}
				//abs(mMat*pp);
			}
			else if(mpGlobalFloor){
				bFloorRes = true;
				pFloor = mpGlobalFloor;
			}
			////��� ���� ���� ���
			if (bFloorRes) {

				pPlaneInfo->SetReferenceFrame(mpTargetFrame);
				pPlaneInfo->AddPlane(pFloor, 1);
				mpTargetFrame->mpPlaneInformation = pPlaneInfo;
				
				ababab.str("");
				auto p = pFloor->GetParam();
				ababab << "Floor::" << pFloor->mvpMPs.size() << std::fixed << std::setprecision(3) << "::[" << p.at<float>(0) << " " << p.at<float>(1) << " " << p.at<float>(2) << " " << p.at<float>(3) << "] " <<"::" << nFloor;
				cv::putText(debug, ababab.str(), cv::Point2f(0, nDebugRows), mnFontFace, mfFontScale, cv::Scalar::all(255)); nDebugRows += mnDebugFontSize;
			}
			/////////////////////////////�ٴ� ��� ����

			/////////////////////////////õ�� ��� ����
			//vpCurrCeilMPs : ���������ӿ����� ã�� ���
			//vpCeilMPs : ��ü õ�� �� �����ϰ� �Ϻ� �����ؼ� ��� ����
			bool bCeilRes = false;
			UVR_SLAM::PlaneInformation* pCeil = new UVR_SLAM::PlaneInformation();
			if (bFloorRes && mvpCeilMPs.size() > 50) {
				std::cout << vpCeilMPs.size() << std::endl;
				bCeilRes = UVR_SLAM::PlaneInformation::PlaneInitialization(pCeil, vpCurrCeilMPs, mvpOutlierCeilMPs, prevFrame->GetFrameID(), 1500, pidst, 0.1);
				std::cout << vpCeilMPs.size() << std::endl;
				if(bCeilRes){
					cv::Mat n;
					float d;
					pCeil->GetParam(n, d);
					sumCeil += d; nCeil++;
					d = sumCeil / nCeil;
					pCeil->SetParam(n, d);
					mpGlobalCeil = pCeil;
				}
				else if (mpGlobalCeil) {
					bCeilRes = true;
					pCeil = mpGlobalCeil;
				}
			}
			////õ�� ���� ���� ���
			if (bCeilRes) {
				pPlaneInfo->AddPlane(pCeil, 2);

				std::stringstream ss;
				auto p = pCeil->GetParam();
				float cos_val = pFloor->CalcCosineSimilarity(pCeil);
				float dist_val = pFloor->CalcPlaneDistance(pCeil);
				ss << "Ceil::" << pCeil->mvpMPs.size() << std::fixed << std::setprecision(3) << "::[" << p.at<float>(0) << " " << p.at<float>(1) << " " << p.at<float>(2) << " " << p.at<float>(3) << "] " << cos_val << ", " << dist_val;
				cv::putText(debug, ss.str(), cv::Point2f(0, nDebugRows), mnFontFace, mfFontScale, cv::Scalar::all(255)); nDebugRows += mnDebugFontSize;
			}
			/////////////////////////////õ�� ��� ����
			
			///////////////////////�ٴڰ� õ�� �Ÿ��� �̿��� ������ ����
			//std::stringstream ssScale;
			//ssScale << "Scale::";
			//if (mpGlobalFloor && mpGlobalCeil) {
			//	float fd, cd;
			//	cv::Mat fn, cn;
			//	mpGlobalFloor->GetParam(fn, fd);
			//	mpGlobalCeil->GetParam(cn, cd);
			//	cv::Mat currCenter = mpTargetFrame->GetCameraCenter();
			//	float fdist = currCenter.dot(fn) + fd;
			//	float cdist = currCenter.dot(cn) + cd;
			//	ssScale << std::fixed << std::setprecision(3) <<currCenter.at<float>(0)<<", "<<currCenter.at<float>(1)<<", "<<currCenter.at<float>(2)<<"::"<< fdist << ", " << cdist;
			//}
			///////////////////////�ٴڰ� õ�� �Ÿ��� �̿��� ������ ����

			////////////////////////////���� �� ����
			int nCeil = 0;
			bool bFrontWall = false;
			UVR_SLAM::PlaneInformation* pFrontWall = new UVR_SLAM::PlaneInformation();
			/*
			for (int k = 0; k < prevMatchInfo->mvpMatchingMPs.size(); k++) {
				auto pMP = prevMatchInfo->mvpMatchingMPs[k];
				int label = prevMatchInfo->mvObjectLabels[k];
				
				if (!pMP || pMP->isDeleted())
					continue;
				if (label == 100) {
					nCeil++;
				}
				else if (label == 255)
					pFrontWall->tmpMPs.push_back(pMP);
			}
			if (bRes && nCeil < 20) {
				bFrontWall = UVR_SLAM::PlaneInformation::PlaneInitialization(pFrontWall, pFrontWall->tmpMPs, pFrontWall->tmpOutlierMPs, prevFrame->GetFrameID(), nTrial, pidst, 0.1);
			}*/
			////////////////////////////���� �� ����

			/////////////////////////////�� ���1 ����
			bool bWallRes1 = false;
			UVR_SLAM::PlaneInformation* pWall1 = new UVR_SLAM::PlaneInformation();
			if (bFloorRes && vpCurrWallMPs.size() > 50) {
				bWallRes1 = UVR_SLAM::PlaneInformation::PlaneInitialization(pWall1, vpCurrWallMPs, vpCurrOutlierWallMPs1, prevFrame->GetFrameID(), nTrial, 0.01, 0.1);
			}
			if (bWallRes1) {
				pPlaneInfo->AddPlane(pWall1, pWall1->mnPlaneID);

				for (int i = 0; i < pWall1->mvpMPs.size(); i++) {
					pWall1->mvpMPs[i]->SetPlaneID(pWall1->mnPlaneID);
				}
				/*float d;
				cv::Mat n;
				pWall1->GetParam(n, d);
				for(int j = 0; j < 1500; j++)
					for (int i = 0; i < mvpFloorMPs.size(); i++) {
						auto pMP = mvpFloorMPs[i];
						if (!pMP || pMP->isDeleted())
							continue;
						auto X3D = pMP->GetWorldPos();
						auto pdist = X3D.dot(n) + d;
					}*/

				std::stringstream ss;
				auto p = pWall1->GetParam();
				float cos_val = pFloor->CalcCosineSimilarity(pWall1);
				float dist_val = pFloor->CalcPlaneDistance(pWall1);
				ss << "Wall::1::" << pWall1->mvpMPs.size() << std::fixed << std::setprecision(3) << "::[" << p.at<float>(0) << " " << p.at<float>(1) << " " << p.at<float>(2) << " " << p.at<float>(3) << "] " << cos_val << ", " << dist_val;
				cv::putText(debug, ss.str(), cv::Point2f(0, nDebugRows), mnFontFace, mfFontScale, cv::Scalar::all(255)); nDebugRows += mnDebugFontSize;
			}
			/////////////////////////////�� ���1 ����

			/////////////////////////////�� ���2 ����
			bool bWallRes2 = false;
			UVR_SLAM::PlaneInformation* pWall2 = new UVR_SLAM::PlaneInformation();
			if (bWallRes1 && vpCurrOutlierWallMPs1.size() > 50) {
				bWallRes2 = UVR_SLAM::PlaneInformation::PlaneInitialization(pWall2, vpCurrOutlierWallMPs1, vpCurrOutlierWallMPs2, prevFrame->GetFrameID(), nTrial, 0.01, 0.1);
			
				if (bWallRes2) {
					float cos_val = pWall1->CalcCosineSimilarity(pWall2);
					float dist_val = pWall1->CalcPlaneDistance(pWall2);
					if (cos_val < 0.999 || dist_val < 0.1) {
						bWallRes2 = false;
					}
					else {
						pPlaneInfo->AddPlane(pWall2, pWall2->mnPlaneID);

						for (int i = 0; i < pWall2->mvpMPs.size(); i++) {
							pWall2->mvpMPs[i]->SetPlaneID(pWall2->mnPlaneID);
						}

						auto p = pWall2->GetParam();
						std::stringstream ss;
						ss << "Wall::2::" << pWall2->mvpMPs.size() << std::fixed << std::setprecision(3) << "::[" << p.at<float>(0) << " " << p.at<float>(1) << " " << p.at<float>(2) << " " << p.at<float>(3) << "] " << cos_val << ", " << dist_val;
						cv::putText(debug, ss.str(), cv::Point2f(0, nDebugRows), mnFontFace, mfFontScale, cv::Scalar::all(255)); nDebugRows += mnDebugFontSize;
					}
				}
			}
			if(pPlaneInfo->GetPlanes().size() > 0)
				mpMap->AddPlaneInfo(pPlaneInfo);
			/////////////////////////////�� ���2 ����

			///////////////////////////////��� ���� ��
						
			if (bFrontWall) {
				std::stringstream ss;
				auto p = pFrontWall->GetParam();
				float cos_val = pFloor->CalcCosineSimilarity(pFrontWall);
				float dist_val = pFloor->CalcPlaneDistance(pFrontWall);
				ss << "Front::" << pFrontWall->mvpMPs.size() << std::fixed << std::setprecision(3) << "::[" << p.at<float>(0) << " " << p.at<float>(1) << " " << p.at<float>(2) << " " << p.at<float>(3) << "] " << cos_val << ", " << dist_val;
				cv::putText(debug, ss.str(), cv::Point2f(0, nDebugRows), mnFontFace, mfFontScale, cv::Scalar::all(255)); nDebugRows += mnDebugFontSize;
			}
			
			if (bFloorRes) {
				//auto planeInfo1 = new UVR_SLAM::PlaneProcessInformation(mpTargetFrame, pFloor);
				//if (bCeilRes)
				//	planeInfo1->AddPlane(pCeil, 2);
				//mpTargetFrame->mpPlaneInformation = planeInfo1;//vpPlaneInfos[nSize]; //planeInfo1
				//mpMap->AddPlaneInfo(planeInfo1);
				/*std::stringstream aaass;
				aaass << "Curr::" << pFloor->mvpMPs.size() << ", " << mvpFloorMPs.size() << "||" << mvpCeilMPs.size();
				cv::putText(debug, aaass.str(), cv::Point2f(0, nDebugRows), mnFontFace, mfFontScale, cv::Scalar::all(255)); nDebugRows += mnDebugFontSize;*/
				int n = 40;
				for (int i = nSize; i >= 0 && n > 0; i--, n--) {
					std::stringstream ss;
					auto pPlaneInfo = vpPlaneInfos[i];
					auto pinfo = pPlaneInfo->GetPlane(1);
					auto pF = pPlaneInfo->GetReferenceFrame();
					auto p = pinfo->GetParam();
					float cos_val = pFloor->CalcCosineSimilarity(pinfo);
					float dist_val = pFloor->CalcPlaneDistance(pinfo);
					ss << "Prev::" <<std::setw(4)<<std::setfill('0')<< pF->GetKeyFrameID() << std::fixed << std::setprecision(3) << "::[" << p.at<float>(0)<<" "<< p.at<float>(1)<<" "<< p.at<float>(2)<<" "<< p.at<float>(3) <<"] "<< cos_val << ", " << dist_val;
					cv::putText(debug, ss.str(), cv::Point2f(0, nDebugRows), mnFontFace, mfFontScale, cv::Scalar::all(255));
					nDebugRows += mnDebugFontSize;
				}
			}
			imshow("Output::PE::PARAM", debug);
			///////////////////////////////��� ���� ��
			////////////////////////////////////��� ã�� ����
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
			/////////////////////////////////////////////////////////////////////
			////�ƿ����̾� �����
			for (auto iter = mvpFloorMPs.begin(); iter != mvpFloorMPs.end();) {
				auto pMPi = *iter;
				if (!pMPi || pMPi->isDeleted()) {
					iter = mvpFloorMPs.erase(iter);
				}
				else
					iter++;
			}
			for (auto iter = mvpCeilMPs.begin(); iter != mvpCeilMPs.end();) {
				auto pMPi = *iter;
				if (!pMPi || pMPi->isDeleted()) {
					iter = mvpCeilMPs.erase(iter);
				}
				else
					iter++;
			}
			////�ƿ����̾� �����
			/////////////////////////////////////////////////////////////////////
			

			////�ð� üũ
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto leduration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float letime = leduration / 1000.0;
			std::stringstream ss;
			ss << std::fixed << std::setprecision(3) << "Layout:" << mpTargetFrame->GetKeyFrameID() << " t=" << letime;
			mpSystem->SetPlaneString(ss.str());
			SetBoolDoingProcess(false, 0);
			continue;

			/////////////Conneted Component Labeling & object labeling
			cv::Mat segmented = prevFrame->matLabeled.clone();
			std::chrono::high_resolution_clock::time_point saa = std::chrono::high_resolution_clock::now();
			cv::Mat imgStructure = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgWall = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgFloor = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgCeil = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat imgObject = cv::Mat::zeros(segmented.size(), CV_8UC1);
			cv::Mat seg_color = cv::Mat::zeros(segmented.size(), CV_8UC3);
			int minY = imgWall.rows;
			int maxY = 0;
			int minFloorX = imgWall.cols;
			int maxFloorX = 0;
			int minCeilX = imgWall.cols;
			int maxCeilX = 0;

			bool bMinX = false;
			bool bMaxX = false;
			bool bMinCeilY = false;
			bool bMaxFloorY = false;

			//////////////////////////////////////////////////////////////////////////////////
			////��ü ���̺�

			for (int i = 0; i < segmented.rows; i++) {
				for (int j = 0; j < segmented.cols; j++) {
					seg_color.at<Vec3b>(i, j) = UVR_SLAM::ObjectColors::mvObjectLabelColors[segmented.at<uchar>(i, j)];
					int val = segmented.at<uchar>(i, j);
					switch (val) {
					case 255:
						imgWall.at<uchar>(i, j) = 255;
						imgStructure.at<uchar>(i, j) = 255;
						break;
					case 150:
						imgFloor.at<uchar>(i, j) = 255;
						imgStructure.at<uchar>(i, j) = 150;
						break;
					case 100:
						imgStructure.at<uchar>(i, j) = 100;
						imgCeil.at<uchar>(i, j) = 255;
						break;
					case 20:
						break;
					case 50:
						imgObject.at<uchar>(i, j) = 255;
						imgStructure.at<uchar>(i, j) = 50;
						break;

					}
				}
			}
			
			////////////////////�ٴڰ� ���� ������ �Լ�.
			////https://webnautes.tistory.com/823
			cv::Mat floorCCL, ceilCCL, sumCCL;
			cv::Mat floorStat, ceilStat;
			bool b1= ConnectedComponentLabeling(imgFloor, floorCCL, floorStat);
			bool b2 = ConnectedComponentLabeling(imgCeil, ceilCCL, ceilStat);
			std::cout << "ccl : " << b1 << ", " << b2 << std::endl;
			sumCCL = floorCCL + ceilCCL;
			maxY = floorStat.at<int>(CC_STAT_TOP);
			minY = ceilStat.at<int>(CC_STAT_TOP) + ceilStat.at<int>(CC_STAT_HEIGHT);
			if (minY != segmented.rows)
				bMinCeilY = true;
			if (maxY != 0)
				bMaxFloorY = true;
			////////////////////�ٴڰ� ���� ������ �Լ�.
			///////////////Conneted Component Labeling & object labeling

			//////////////////////////////////////////////////////////////////////////
			///////////Line Segments Detection
			////////�ٴ�, ���� ������ ������ �������� ������ ����
			//////������ ������ ������ ����Ʈ�� ����� �׽�Ʈ �ϴµ� �̿�
			//std::vector<cv::Vec4i> tFloorLines, tCeilLines, tlines;
			//std::vector<Line*> lines;
			//{

			//	cv::Ptr<cv::LineSegmentDetector> pLSD = createLineSegmentDetector();
			//	pLSD->detect(floorCCL, tFloorLines);
			//	pLSD->detect(ceilCCL, tCeilLines);
			//	std::cout << tFloorLines.size() << ", " << tCeilLines.size() << std::endl;
			//	tlines.insert(tlines.begin(), tFloorLines.begin(), tFloorLines.end());
			//	tlines.insert(tlines.end(), tCeilLines.begin(), tCeilLines.end());
			//	//pLSD->detect(tempFloor, tlines);
			//								   //graph.convertTo(graph, CV_8UC3);
			//								   //cv::cvtColor(graph, graph, CV_GRAY2BGR);
			//	for (int i = 0; i < tlines.size(); i++) {
			//		Vec4i v = tlines[i];
			//		Point2f from(v[0] * 2, v[1] * 2);
			//		Point2f to(v[2] * 2, v[3] * 2);
			//		Point2f diff = from - to;
			//		float dist = sqrt(diff.dot(diff));
			//		
			//		if (dist < 25 * 2)
			//			continue;
			//		/*if (to.y < maxY + 3 || from.y < maxY + 3)
			//			continue;*/
			//		float slope = abs(LineProcessor::CalcSlope(from, to));
			//		if (slope > 3.0)
			//			continue;
			//		Line* line = new Line(mpTargetFrame, from, to);
			//		//line->SetLinePts();
			//		lines.push_back(line);
			//	}
			//	//�ش� ������ Ÿ�� �����ӿ� ������.
			//	//mpTargetFrame->SetLines(lines);

			//}
			///////////Line Segments Detection

			///////////////���� ���� ��Ī�� ����
			//cv::Mat lineImg = cv::Mat::zeros(prevFrame->GetOriginalImage().size(), CV_8UC1);
			//for (int i = 0; i < lines.size(); i++) {
			//	auto line = lines[i];
			//	cv::line(lineImg, line->from, line->to, cv::Scalar(255, 255, 255), 5);
			//}
			//std::vector<cv::Point2f> vPtsOnLines;
			//for (int i = 0; i < prevMatchInfo->mvMatchingPts.size(); i++) {
			//	auto pt = prevMatchInfo->mvMatchingPts[i];
			//	if (lineImg.at<uchar>(pt)) {
			//		vPtsOnLines.push_back(pt);
			//	}
			//}
			///////////////���� ���� ��Ī�� ����
			////////////////////////////////////////////////////////////////

			////Ȯ��
			cv::Mat vis = prevFrame->GetOriginalImage();
			maxY *= 2;
			minY *= 2;
			if (bMinCeilY) {
				cv::line(vis, cv::Point2f(0, minY), cv::Point2f(vis.cols, minY), cv::Scalar(255, 0, 0), 1);
			}
			if (bMaxFloorY) {
				cv::line(vis, cv::Point2f(0, maxY), cv::Point2f(vis.cols, maxY), cv::Scalar(255, 0, 0), 1);
			}

			//////////////////////////////////////////////////////////////////
			//////Hough lines and vanishing points
			cv::Mat filtered, edge;
			std::vector<cv::Vec4i> vlines;
			std::vector<cv::Vec2f> vflines;
			auto gray = prevFrame->GetFrame();
			GaussianBlur(gray, filtered, cv::Size(5, 5), 0.0);
			cv::Canny(filtered, edge, 50, 200);
			cv::HoughLinesP(edge, vlines, 1, CV_PI / 180, 50, 50, 10);
			for (int i = 0; i < vlines.size(); i++) {
				auto line = vlines[i];
				auto from = cv::Point2f(line[0], line[1]);
				auto to = cv::Point2f(line[2], line[3]);
				Line* aline = new Line(mpTargetFrame, mnHeight, from, to);
				//		//line->SetLinePts();
				//		lines.push_back(line);
				cv::line(vis, aline->fromExt, aline->toExt, cv::Scalar(0, 255, 0), 1);
				cv::line(vis, aline->from, aline->to, cv::Scalar(0, 255, 255), 2);
			}
			//cv::HoughLines(edge, vflines, 1, CV_PI / 180, 150);
			//for (int i = 0; i < vflines.size(); i++) {
			//	auto line = vflines[i];
			//	/*auto pt1 = cv::Point2f(line[0], line[1]);
			//	auto pt2 = cv::Point2f(line[2], line[3]);*/
			//	
			//	float rho = line[0];
			//	float theta = line[1];
			//	double a = cos(theta), b = sin(theta);
			//	double x0 = a*rho, y0 = b*rho;
			//	
			//	Point pt1, pt2;
			//	pt1.x = cvRound(x0 + 1000 * (-b));
			//	pt1.y = cvRound(y0 + 1000 * (a));
			//	pt2.x = cvRound(x0 - 1000 * (-b));
			//	pt2.y = cvRound(y0 - 1000 * (a));
			//	cv::line(vis, pt1, pt2, cv::Scalar(255, 0, 0), 1);
			//}
			imshow("edge", edge);
			//////Hough lines and vanishing points
			//////////////////////////////////////////////////////////////////

			/*for (int i = 0; i < lines.size(); i++) {
				auto line = lines[i];
				cv::line(vis, line->from, line->to, cv::Scalar(0, 255, 255), 2);
			}
			for (int i = 0; i < vPtsOnLines.size(); i++) {
				auto pt = vPtsOnLines[i];
				cv::circle(vis, pt, 2, cv::Scalar(0, 0, 255), -1);
			}	*/
			imshow("layout test", vis);
			waitKey(1);
			////Ȯ��
			
			

			////////////////////////////////////////////////////////////////////////

			std::cout << "pe::end::" << std::endl;
			SetBoolDoingProcess(false, 0);
			continue;
			if (!mpPrevFrame)
				std::cout << "null::prev" << std::endl;
			if (!mpPPrevFrame)
				std::cout << "null::pprev" << std::endl;

			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////200412
			//////���� ����!!!
			//if(mpPPrevFrame && mpPrevFrame){
			//	std::cout << "pe::test::matching::prev::start" << std::endl;
			//	cv::Mat invP, invT, invK;
			//	mpPrevFrame->mpPlaneInformation->Calculate();
			//	mpPrevFrame->mpPlaneInformation->GetInformation(invP, invT, invK);

			//	cv::Point2f ptBottom = cv::Point2f(0, mpPrevFrame->GetOriginalImage().rows);
			//	cv::Mat debbuging;
			//	std::vector<std::pair<cv::Point2f, cv::Point2f>> vPairMatches;
			//	mpMatcher->OpticalMatchingForMapping(mpPrevFrame, mpTargetFrame, vPairMatches, debbuging);
			//	/////�� ������ ������ ��Ī 
			//	for (int j = 0; j < vPairMatches.size(); j++) {
			//		if (!mpPrevFrame->isInImage(vPairMatches[j].first.x, vPairMatches[j].first.y) || !mpPrevFrame->isInImage(vPairMatches[j].second.x, vPairMatches[j].second.y)) {
			//			std::cout << "�̹��� ��!!" << std::endl;
			//			continue;
			//		}
			//		/*if (!mpPrevFrame->isInImage(vPairMatches[j].first.x, vPairMatches[j].first.y)) {
			//			std::cout << "MP::GetDenseMP::image boundary error" << std::endl;
			//			continue;
			//		}
			//		if (!mpPrevFrame->isInImage(vPairMatches[j].second.x, vPairMatches[j].second.y)) {
			//			std::cout << "MP::GetDenseMP::image boundary error" << std::endl;
			//			continue;
			//		}*/
			//		auto pMP1 =  mpPrevFrame->GetDenseMP(vPairMatches[j].first);
			//		auto pMP2 =  mpTargetFrame->GetDenseMP(vPairMatches[j].second);
			//		auto pt1 = vPairMatches[j].first;
			//		auto pt2 = vPairMatches[j].second;

			//		bool bMP1 = pMP1 && !pMP1->isDeleted();
			//		bool bMP2 = pMP2 && !pMP2->isDeleted();

			//		if (bMP1 && bMP2) {
			//			if (pMP1->mnMapPointID == pMP2->mnMapPointID) {
			//				auto apt1 = mpPrevFrame->Projection(pMP1->GetWorldPos());
			//				auto apt2 = mpTargetFrame->Projection(pMP1->GetWorldPos());
			//				cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(255, 0, 255), 1);
			//				//cv::line(debbuging, pt1, apt1, cv::Scalar(255, 255, 0), 1);
			//				//cv::line(debbuging, pt2 + ptBottom, apt2 + ptBottom, cv::Scalar(255, 255, 0), 1);
			//			}
			//			else {
			//				cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(255, 255, 0), 1);
			//			}
			//		}
			//		else if (bMP1) {
			//			auto apt1 = mpPrevFrame->Projection(pMP1->GetWorldPos());
			//			auto apt2 = mpTargetFrame->Projection(pMP1->GetWorldPos());
			//			cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(0, 255, 0), 1);
			//			cv::line(debbuging, pt1, apt1, cv::Scalar(255, 0, 0), 1);
			//			cv::line(debbuging, pt2 + ptBottom, apt2 + ptBottom, cv::Scalar(255, 0, 0), 1);
			//		}
			//		else if (!bMP1 && !bMP2) {

			//			if (mpPrevFrame->matLabeled.at<uchar>(pt1.y / 2, pt1.x / 2) == 150) {
			//				cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(0, 255, 255), 1);
			//				cv::Mat X3D;
			//				bool bRes = PlaneInformation::CreatePlanarMapPoint(pt1, invP, invT, invK, X3D);
			//				if (bRes)
			//				{
			//					UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpPrevFrame, X3D, cv::Mat());
			//					pNewMP->SetPlaneID(mpMap->mpFloorPlane->mnPlaneID);
			//					//pNewMP->SetWorldPos(X3D);
			//					pNewMP->SetMapPointType(UVR_SLAM::PLANE_DENSE_MP);
			//					pNewMP->AddDenseFrame(mpPrevFrame, pt1);
			//					pNewMP->AddDenseFrame(mpTargetFrame, pt2);
			//					mpTargetFrame->mvMatchingPts.push_back(pt2);
			//					mpTargetFrame->mvpMatchingMPs.push_back(pNewMP);
			//					mpSystem->mlpNewMPs.push_back(pNewMP);
			//				}
			//			}
			//			else {
			//				//cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(0, 0, 255), 1);
			//			}
			//		}
			//	}
			//	mpTargetFrame->SetBoolMapping(true);
			//	std::cout << "pe::test::matching::prev::end" << std::endl;
			//	//std::cout << "pe::matching::end2" << std::endl;
			//	/////�� ������ ������ ��Ī 
			//	std::stringstream sss;
			//	sss << mpSystem->GetDirPath(0) << "/kfmatching/" << mpTargetFrame->GetKeyFrameID() << "_" << mpPrevFrame->GetKeyFrameID() << ".jpg";
			//	imwrite(sss.str(), debbuging);
			//	//vPairMatches.clear();
			//	//std::cout << "pe::test::matching::pprev::start" << std::endl;
			//	//mpMatcher->OpticalMatchingForMapping(mpPPrevFrame, mpTargetFrame, vPairMatches, debbuging);
			//	//std::cout << "pe::test::matching::pprev::aa" << std::endl;
			//	////std::cout << "pe::matching::end3" << std::endl;
			//	///////�� ������ ������ ��Ī
			//	//cv::Mat invP, invT, invK;
			//	//mpPPrevFrame->mpPlaneInformation->Calculate();
			//	//mpPPrevFrame->mpPlaneInformation->GetInformation(invP, invT, invK);
			//	//for (int j = 0; j < vPairMatches.size(); j++) {
			//	//	if (!mpPrevFrame->isInImage(vPairMatches[j].first.x, vPairMatches[j].first.y) || !mpPrevFrame->isInImage(vPairMatches[j].second.x, vPairMatches[j].second.y)) {
			//	//		std::cout << "�̹��� ��!!" << std::endl;
			//	//		continue;
			//	//	}
			//	//	auto pMP1 = mpPPrevFrame->GetDenseMP(vPairMatches[j].first);
			//	//	auto pMP2 = mpTargetFrame->GetDenseMP(vPairMatches[j].second);
			//	//	auto pt1 = vPairMatches[j].first;
			//	//	auto pt2 = vPairMatches[j].second;

			//	//	bool bMP1 = pMP1 && !pMP1->isDeleted();
			//	//	bool bMP2 = pMP2 && !pMP2->isDeleted();

			//	//	if (bMP1 && bMP2) {
			//	//		if (pMP1->mnMapPointID == pMP2->mnMapPointID) {
			//	//			auto apt1 = mpPPrevFrame->Projection(pMP1->GetWorldPos());
			//	//			auto apt2 = mpTargetFrame->Projection(pMP1->GetWorldPos());
			//	//			cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(255, 0, 255), 1);
			//	//			//cv::line(debbuging, pt1, apt1, cv::Scalar(255, 255, 0), 1);
			//	//			//cv::line(debbuging, pt2 + ptBottom, apt2 + ptBottom, cv::Scalar(255, 255, 0), 1);
			//	//		}
			//	//		else {
			//	//			cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(255, 255, 0), 1);
			//	//		}
			//	//	}else if (bMP1) {
			//	//		auto apt1 = mpPPrevFrame->Projection(pMP1->GetWorldPos());
			//	//		auto apt2 = mpTargetFrame->Projection(pMP1->GetWorldPos());
			//	//		cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(0, 255, 0), 1);
			//	//		cv::line(debbuging, pt1, apt1, cv::Scalar(255, 0, 0), 1);
			//	//		cv::line(debbuging, pt2 + ptBottom, apt2 + ptBottom, cv::Scalar(255, 0, 0), 1);
			//	//	}
			//	//	else if (!bMP1 && !bMP2) {
			//	//		
			//	//		if (mpPPrevFrame->matLabeled.at<uchar>(pt1.y / 2, pt1.x / 2) == 150) {
			//	//			cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(0, 255, 255), 1);
			//	//			cv::Mat X3D;
			//	//			bool bRes = PlaneInformation::CreatePlanarMapPoint(pt1, invP, invT, invK, X3D);
			//	//			if (bRes)
			//	//			{
			//	//				UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpPPrevFrame, X3D, cv::Mat());
			//	//				pNewMP->SetPlaneID(mpMap->mpFloorPlane->mnPlaneID);
			//	//				//pNewMP->SetWorldPos(X3D);
			//	//				pNewMP->SetMapPointType(UVR_SLAM::PLANE_DENSE_MP);
			//	//				pNewMP->AddDenseFrame(mpPPrevFrame, pt1);
			//	//				pNewMP->AddDenseFrame(mpTargetFrame, pt2);
			//	//				mpTargetFrame->mvMatchingPts.push_back(pt2);
			//	//				mpTargetFrame->mvpMatchingMPs.push_back(pNewMP);
			//	//				mpSystem->mlpNewMPs.push_back(pNewMP);
			//	//			}
			//	//		}
			//	//		else {
			//	//			//cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(0, 0, 255), 1);
			//	//		}
			//	//	}
			//	//	mpTargetFrame->SetBoolMapping(true);
			//	//	/*if (pMP1 && !pMP1->isDeleted()) {
			//	//		auto apt1 = mpPPrevFrame->Projection(pMP1->GetWorldPos());
			//	//		auto apt2 = mpTargetFrame->Projection(pMP1->GetWorldPos());
			//	//		cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(255, 0, 255), 1);
			//	//		cv::line(debbuging, pt1, apt1, cv::Scalar(255, 255, 0), 1);
			//	//		cv::line(debbuging, pt2 + ptBottom, apt2 + ptBottom, cv::Scalar(255, 255, 0), 1);
			//	//	}
			//	//	if (pMP2  && !pMP2->isDeleted()) {
			//	//		auto apt1 = mpPPrevFrame->Projection(pMP2->GetWorldPos());
			//	//		auto apt2 = mpTargetFrame->Projection(pMP2->GetWorldPos());
			//	//		cv::line(debbuging, pt1, pt2 + ptBottom, cv::Scalar(255, 0, 255), 1);
			//	//		cv::line(debbuging, pt1, apt1, cv::Scalar(0, 255, 255), 1);
			//	//		cv::line(debbuging, pt2 + ptBottom, apt2 + ptBottom, cv::Scalar(0, 255, 255), 1);
			//	//	}*/
			//	//}
			//	//std::cout << "pe::test::matching::pprev::end" << std::endl;
			//	////std::cout << "pe::matching::end4" << std::endl;
			//	///////�� ������ ������ ��Ī
			//	//sss.str("");
			//	//sss << mpSystem->GetDirPath(0) << "/kfmatching/" << mpTargetFrame->GetKeyFrameID() << "_" << mpPPrevFrame->GetKeyFrameID() << ".jpg";
			//	//imwrite(sss.str(), debbuging);
			//	////std::cout << "pe::test::end" << std::endl;
			//	////continue;
			//}
			//std::cout << "pe::end" << std::endl;
			//SetBoolDoingProcess(false, 0);
			//continue;
			//////200412
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
			////////��� ����Ʈ ����
			//std::vector<cv::Mat> vPlanarMaps;
			//vPlanarMaps = std::vector<cv::Mat>(mpTargetFrame->mvKeyPoints.size(), cv::Mat::zeros(0, 0, CV_8UC1));
			//
			////�ʱ�ȭ �Ŀ� ����Ǳ� ������ ������ �ʱ�ȭ�� �ȵǾ������� �׳� �����ص� ��.
			//if (!mpMap->isFloorPlaneInitialized()) {
			//	////unlock & notify
			//	std::unique_lock<std::mutex> lockPlanar(mpSystem->mMutexUsePlaneEstimation);
			//	mpSystem->mbPlaneEstimationEnd = true;
			//	lockPlanar.unlock();
			//	mpSystem->cvUsePlaneEstimation.notify_all();
			//	std::cout << "pe::end::s::" << mpTargetFrame->GetKeyFrameID() << std::endl;
			//	continue;
			//}

			//mStrPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());
			////���� ���̾ƿ� �����ϴ� Ű������ ����
			//int nTargetID = mpTargetFrame->GetFrameID();
			//mpFrameWindow->SetLastLayoutFrameID(nTargetID);
			//cv::Mat vImg = mpTargetFrame->GetOriginalImage();
			////matching test

			////////////////////////////////////////////////////////////////////////////////
			////////////���� ����
			////�ٴ� �� �� ������ ���� ����. 
			//
			////////////���� ����
			////////////////////////////////////////////////////////////////////////////////
			//std::chrono::high_resolution_clock::time_point line_start = std::chrono::high_resolution_clock::now();
			///*cv::Mat img = mpTargetFrame->GetOriginalImage();
			//cv::Mat gray, filtered, edge;
			//std::vector<cv::Vec4i> vecLines;
			//cv::cvtColor(img, gray, CV_BGR2GRAY);
			//GaussianBlur(gray, filtered, cv::Size(5, 5), 0.0);
			//Canny(filtered, edge, 100, 200);
			//cv::HoughLinesP(edge, vecLines, 1, CV_PI / 180.0, 30, 30, 3);*/
			//
			////imshow("line", img);
			//std::chrono::high_resolution_clock::time_point line_end = std::chrono::high_resolution_clock::now();
			//auto lineduration = std::chrono::duration_cast<std::chrono::milliseconds>(line_end - line_start).count();
			//float line_time = lineduration / 1000.0;

			/////////////////////////////////////////////
			//////���� Ű�����ӿ��� ������ ������Ʈ ������Ʈ
			////int nUpdateT
			//int nPrevTest = 0;
			//bool bInitFloorPlane = mpMap->isFloorPlaneInitialized();//true;
			//auto pFloor = mpMap->mpFloorPlane;
			//int nPrevTest2 = 0;
			//cv::Mat pmat;
			//if (bInitFloorPlane) {

			//	////update dense map
			//	auto mvpDenseMPs = mpPrevFrame->GetDenseVectors();
			//	cv::Mat debugging;
			//	std::vector<std::pair<int, cv::Point2f>> vPairs;
			//	std::vector<bool> vbInliers;
			//	int mnDenseMatching = mpMatcher->DenseMatchingWithEpiPolarGeometry(mpTargetFrame, mpPrevFrame, mvpDenseMPs, vPairs, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugging);
			//	int nInc = 1;
			//	if (vPairs.size() > 1000) {
			//		nInc = 20;
			//	}
			//	
			//	for (int i = 0; i < vPairs.size(); i+= nInc) {
			//		auto idx = vPairs[i].first;
			//		auto pt = vPairs[i].second;
			//	
			//		mvpDenseMPs[idx]->AddDenseFrame(mpTargetFrame, pt);
			//		//mpTargetFrame->AddDenseMP(mvpDenseMPs[idx], pt);
			//	}
			//	////update dense map

			//	////���� �ö� ����Ʈ�� ������ ����Ʈ�� ���ؼ� Ʈ��ŷ�� ������ ����Ʈ�� ���ؼ� ���� ��� ���Ϳ� ���Խ�Ŵ.
			//	for (int i = 0; i < pFloor->tmpMPs.size(); i++) {
			//		UVR_SLAM::MapPoint* pMP = pFloor->tmpMPs[i];
			//		if (!pMP)
			//			continue;
			//		if (pMP->isDeleted())
			//			continue;
			//		if (pMP->GetNumConnectedFrames() > 1){
			//			pFloor->mvpMPs.push_back(pMP);
			//			nPrevTest2++;
			//		}
			//	}
			//	pFloor->tmpMPs.clear();
			//	UpdatePlane(pFloor, nTargetID, mnRansacTrial, mfThreshPlaneDistance, mfThreshPlaneRatio);
			//	pmat = pFloor->GetParam().t();
			//}else
			//	pmat = cv::Mat::zeros(1, 4, CV_32FC1);
			//////���� Ű�����ӿ��� ������ ������Ʈ ������Ʈ
			/////////////////////////////////////////////
			//
			////////////////////////////////////////////////////////////////
			//////lock
			////local �ʿ��� Ű����Ʈ ���� ���� ���� ����
			//std::unique_lock<std::mutex> lock(mpSystem->mMutexUsePlaneEstimation);
			//mpSystem->mbPlaneEstimationEnd = false;

			//////labeling ���� ������ ���
			//{
			//	std::unique_lock<std::mutex> lock(mpSystem->mMutexUseSegmentation);
			//	while (!mpSystem->mbSegmentationEnd) {
			//		mpSystem->cvUseSegmentation.wait(lock);
			//	}
			//}
			//////labeling ���� ������ ���

			//std::cout << "pe::2" << std::endl;
			//////////////////////////////////////////////////////////////////////////
			////
			//

			//std::cout << "pe::4" << std::endl;
			//
			////////LINE �߰� ����.
			////WallLineTest
			//if (bInitFloorPlane) {
			//	//////�ٴ� ��� �� ����
			//	UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(mpTargetFrame, pFloor, vPlanarMaps);
			//	//�ٴ� ���� ��� ����
			//	//UVR_SLAM::PlaneInformation::CreateDensePlanarMapPoint(mpTargetFrame->mDenseMap, imgStructure, mpTargetFrame, mpSystem->mnPatchSize);
			//	UVR_SLAM::PlaneInformation::CreateDensePlanarMapPoint(mpTargetFrame->mvX3Ds, imgStructure, mpTargetFrame, mpSystem->mnPatchSize);
			//	//////�ٴ� ��� �� ����

			//	bool bInitWallPlane = mpMap->isWallPlaneInitialized();
			//	auto mvFrames = mpTargetFrame->GetConnectedKFs(10);
			//	auto mvWallPlanes = mpMap->GetWallPlanes();
			//	std::vector<WallPlane*> mvpNewWallPlanes;

			//	cv::Mat normal1;
			//	float dist1;
			//	pFloor->GetParam(normal1, dist1);
			//	cv::Mat planeParam = pFloor->GetParam().clone();
			//	
			//	///����� �����ӿ� ���ؼ� ���� ������Ʈ
			//	for (int i = 0; i < mvFrames.size(); i++) {
			//		/*if (!mvFrames[i]->mpPlaneInformation)
			//			mvFrames[i]->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mvFrames[i], plane);*/
			//		if (!mvFrames[i]->mpPlaneInformation){
			//			std::cout << "error!!!!"<<mpTargetFrame->GetKeyFrameID()<<", "<< mvFrames[i]->GetKeyFrameID() << std::endl;
			//		}
			//		mvFrames[i]->mpPlaneInformation->Calculate();
			//	}

			//	//���� �������� ���ο� ���ؼ� ���� �� ���� ��
			//	if (bInitWallPlane) {
			//		////update wall plane parameters
			//		for (int i = 0; i < mvWallPlanes.size(); i++) {

			//			//������Ʈ�� ������ ������ ���Ե� ��츸 ������.
			//			for (int k = 0; k < mvFrames.size(); k++) {
			//				if (mvWallPlanes[i]->isContain(mvFrames[k])) {
			//					auto line_iter = mvWallPlanes[i]->GetLines(mvFrames[k]);
			//					for (auto iter = line_iter.first; iter != line_iter.second; iter++) {
			//						auto line = iter->second;
			//						line->SetLinePts();
			//					}
			//				}
			//			}
			//			cv::Mat parama = UVR_SLAM::PlaneInformation::PlaneLineEstimator(mvWallPlanes[i], pFloor);
			//			mvWallPlanes[i]->SetParam(parama);
			//		}
			//		////update wall plane parameters
			//	}

			//	////���� ���� �˰���
			//	////���� �������� �ӽ� ���� ������.
			//	////������ ���� ��.
			//	//{
			//	//	auto mvpCurrLines = mpTargetFrame->Getlines();
			//	//	//std::cout << "wall::line::" << mvpCurrLines.size() << std::endl;
			//	//	
			//	//	cv::Mat invCurrP, invCurrPlane, invK;
			//	//	mpTargetFrame->mpPlaneInformation->Calculate();
			//	//	mpTargetFrame->mpPlaneInformation->GetInformation(invCurrPlane, invCurrP, invK);

			//	//	cv::Mat R, t;
			//	//	mpTargetFrame->GetPose(R, t);

			//	//	//���� ���������κ��� ���� ����
			//	//	for (int li = 0; li < mvpCurrLines.size(); li++) {
			//	//		auto line = mvpCurrLines[li];
			//	//		if (mvWallPlanes.size() == 0) {
			//	//			UVR_SLAM::WallPlane* tempWallPlane = new UVR_SLAM::WallPlane();
			//	//			tempWallPlane->AddLine(mvpCurrLines[li], mpTargetFrame);
			//	//			tempWallPlane->SetRecentKeyFrameID(mpTargetFrame->GetKeyFrameID());
			//	//			mvWallPlanes.push_back(tempWallPlane);
			//	//			mvpNewWallPlanes.push_back(tempWallPlane);
			//	//		}
			//	//		else {
			//	//			bool bInsert = true;
			//	//			bool bConnect = false;
			//	//			for (int j = 0; j < mvWallPlanes.size(); j++) {

			//	//				//���߿� �ϳ����� ����ǵ� Ŀ��Ʈ.
			//	//				for (int jk = 0; jk < mvFrames.size(); jk++) {
			//	//					UVR_SLAM::Frame* pKF = mvFrames[jk];

			//	//					if (!mvWallPlanes[j]->isContain(pKF))
			//	//						continue;

			//	//					auto line_iter = mvWallPlanes[j]->GetLines(pKF);

			//	//					for (auto iter = line_iter.first; iter != line_iter.second; iter++) {
			//	//						cv::Mat invP, invT, invK;
			//	//						pKF->mpPlaneInformation->GetInformation(invP, invT, invK);
			//	//						auto tempLine = iter->second;
			//	//						cv::Mat X3Dfrom = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(tempLine->from, invP, invT, invK);
			//	//						cv::Mat X3Dto = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(tempLine->to, invP, invT, invK);
			//	//						//���� �̹����� ��������

			//	//						cv::Mat temp1 = R*X3Dfrom + t;
			//	//						temp1 = mpTargetFrame->mK*temp1;
			//	//						cv::Point2f pt1(temp1.at<float>(0) / temp1.at<float>(2), temp1.at<float>(1) / temp1.at<float>(2));
			//	//						cv::Mat temp2 = R*X3Dto + t;
			//	//						temp2 = mpTargetFrame->mK*temp2;
			//	//						cv::Point2f pt2(temp2.at<float>(0) / temp2.at<float>(2), temp2.at<float>(1) / temp2.at<float>(2));
			//	//						//cv::line(vImg, pt1, pt2, cv::Scalar(255, 0, 0), 1);

			//	//						float d1 = line->CalcLineDistance(pt1);
			//	//						float d2 = line->CalcLineDistance(pt2);

			//	//						//connect
			//	//						if (d1 + d2 < 6.0) {

			//	//							mvWallPlanes[j]->AddLine(line, mpTargetFrame);
			//	//							mvWallPlanes[j]->SetRecentKeyFrameID(mpTargetFrame->GetKeyFrameID());

			//	//							//UVR_SLAM::PlaneInformation::CreateWallMapPoints(mpTargetFrame, mvWallPlanes[j], line, vPlanarMaps, mpSystem);
			//	//							//UVR_SLAM::PlaneInformation::CreateDenseWallPlanarMapPoint(mpTargetFrame->mvX3Ds, imgStructure, mpTargetFrame, mvWallPlanes[j], line, mpSystem->mnPatchSize);

			//	//							bConnect = true;
			//	//							bInsert = false;
			//	//							break;
			//	//						}
			//	//					}
			//	//					if (bConnect)
			//	//						break;
			//	//				}
			//	//				if (bConnect)
			//	//					break;
			//	//			}
			//	//			if (bInsert) {
			//	//				//std::cout << "insert!!!!" << std::endl;
			//	//				UVR_SLAM::WallPlane* tempWallPlane = new UVR_SLAM::WallPlane();
			//	//				//tempWallPlane->SetParam(param);
			//	//				tempWallPlane->AddLine(mvpCurrLines[li], mpTargetFrame);
			//	//				//tempWallPlane->AddFrame(mpTargetFrame);
			//	//				tempWallPlane->SetRecentKeyFrameID(mpTargetFrame->GetKeyFrameID());
			//	//				mvWallPlanes.push_back(tempWallPlane);
			//	//				mvpNewWallPlanes.push_back(tempWallPlane);
			//	//			}
			//	//		}
			//	//	}
			//	//}

			//	//std::cout << "pe::6" << std::endl;
			//	/////////���ο� ��� ����
			//	//////���� �������� ��� �߰� �� ����
			//	////���� �����ӵ��� ���Ḹ �����ϱ� 
			//	//std::vector<cv::Mat> wallParams;
			//	//std::vector<cv::Mat> twallParams;
			//	////mvFrames.push_back(mpTargetFrame);

			//	//cv::Mat R, t;
			//	//mpTargetFrame->GetPose(R, t);

			//	////���Ӱ� �߰��� �� ���ο� ���ؼ��� ������.
			//	//for (int k = 0; k < mvFrames.size(); k++) {
			//	//	
			//	//	cv::Mat invP, invK, invT;
			//	//	mvFrames[k]->mpPlaneInformation->GetInformation(invP, invT, invK);

			//	//	auto mvpLines = mvFrames[k]->Getlines();

			//	//	bool bConnect = false;
			//	//	for (int i = 0; i < mvpLines.size(); i++) {

			//	//		if (mvpLines[i]->mnPlaneID > 0)
			//	//			continue;

			//	//		auto tempLine = mvpLines[i];
			//	//		cv::Mat X3Dfrom = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(tempLine->from, invP, invT, invK);
			//	//		cv::Mat X3Dto = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(tempLine->to, invP, invT, invK);

			//	//		cv::Mat temp1 = R*X3Dfrom + t;
			//	//		temp1 = mpTargetFrame->mK*temp1;
			//	//		cv::Point2f pt1(temp1.at<float>(0) / temp1.at<float>(2), temp1.at<float>(1) / temp1.at<float>(2));
			//	//		cv::Mat temp2 = R*X3Dto + t;
			//	//		temp2 = mpTargetFrame->mK*temp2;
			//	//		cv::Point2f pt2(temp2.at<float>(0) / temp2.at<float>(2), temp2.at<float>(1) / temp2.at<float>(2));

			//	//		for (int j = 0; j < mvpNewWallPlanes.size(); j++) {
			//	//			auto iterTargetKF = mvpNewWallPlanes[j]->GetLines(mpTargetFrame);

			//	//			for (auto iter = iterTargetKF.first; iter != iterTargetKF.second; iter++) {
			//	//				auto line = iter->second;
			//	//				float d1 = line->CalcLineDistance(pt1);
			//	//				float d2 = line->CalcLineDistance(pt2);

			//	//				if (d1 + d2 < 6.0) {

			//	//					mvpNewWallPlanes[j]->AddLine(line, mpTargetFrame);
			//	//					mvpNewWallPlanes[j]->SetRecentKeyFrameID(mpTargetFrame->GetKeyFrameID());

			//	//					//UVR_SLAM::PlaneInformation::CreateWallMapPoints(mpTargetFrame, mvWallPlanes[j], line, vPlanarMaps, mpSystem);
			//	//					//UVR_SLAM::PlaneInformation::CreateDenseWallPlanarMapPoint(mpTargetFrame->mvX3Ds, imgStructure, mpTargetFrame, mvWallPlanes[j], line, mpSystem->mnPatchSize);

			//	//					bConnect = true;
			//	//					break;
			//	//				}
			//	//			}
			//	//			if (bConnect)
			//	//				break;
			//	//		}
			//	//		
			//	//	}//for
			//	//}
			//	/////////���ο� ��� ����
			//	//std::cout << "pe::6_2" << std::endl;
			//	//for (int i = 0; i < mvpNewWallPlanes.size(); i++) {
			//	//	//if (mvWallPlanes[i]->GetSize()> 3 && mvWallPlanes[i]->mnPlaneID < 0) {
			//	//	if (mvpNewWallPlanes[i]->GetSize()> 3) {
			//	//		mvpNewWallPlanes[i]->CreateWall();
			//	//		mpMap->AddWallPlane(mvpNewWallPlanes[i]);
			//	//		cv::Mat parama = UVR_SLAM::PlaneInformation::PlaneLineEstimator(mvpNewWallPlanes[i], pFloor);
			//	//		mvpNewWallPlanes[i]->SetParam(parama);
			//	//		if (!mpMap->isWallPlaneInitialized()) {
			//	//			mpMap->SetWallPlaneInitialization(true);
			//	//		}
			//	//	}
			//	//}
			//	//std::cout << "pe::6_3" << std::endl;
			//	//for (int i = 0; i < mvWallPlanes.size(); i++) {
			//	//	if (mvWallPlanes[i]->mnPlaneID < 0)
			//	//		continue;
			//	//	auto iterTargetKF = mvWallPlanes[i]->GetLines(mpTargetFrame);

			//	//	for (auto iter = iterTargetKF.first; iter != iterTargetKF.second; iter++) {
			//	//		auto line = iter->second;
			//	//		UVR_SLAM::PlaneInformation::CreateWallMapPoints(mpTargetFrame, mvWallPlanes[i], line, vPlanarMaps, mpSystem);
			//	//		UVR_SLAM::PlaneInformation::CreateDenseWallPlanarMapPoint(mpTargetFrame->mvX3Ds, imgStructure, mpTargetFrame, mvWallPlanes[i], line, mpSystem->mnPatchSize);
			//	//	}
			//	//}
			//	////���� ���� �˰���

			//}
			////WallLineTest
			////////////////////////////////////////////////////////////////
			//
			//std::cout << "pe::7" << std::endl;
			//////////��Ī �׽�Ʈ
			//if (bInitFloorPlane) {

			//	//
			//	cv::Mat aadebug;
			//	std::vector<cv::DMatch> aavMatches;
			//	std::vector<bool> aavbInliers;
			//	std::vector<std::pair<int, cv::Point2f>> aavPairs;
			//	mpMatcher->MatchingWithOptiNEpi(mpPrevFrame, mpTargetFrame, vPlanarMaps, aavbInliers, aavMatches, aavPairs, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, aadebug);
			//	//


			//	////////////////////////////////
			//	//���� ������Ʈ ���� �ð�
			//	///////////////////////////////
			//	//���� �������� ���� ������
			//	//���� �������� ���� ���������� ���������ӿ��� ����Ʈ�� ������.

			//	cv::Mat debugImg;
			//	
			//	std::vector<UVR_SLAM::Frame*> mvpKFs = mpTargetFrame->GetConnectedKFs(5);
			//	//mvpKFs.push_back(mpPrevFrame);
			//	//mvpKFs.push_back(mpPPrevFrame);
			//	for (int ki = 0; ki < mvpKFs.size(); ki++) {
			//		///////dense
			//		//cv::Mat debugging;
			//		////mpMatcher->DenseMatchingWithEpiPolarGeometry(mpTargetFrame, mvpKFs[ki], mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugging);
			//		//mpMatcher->DenseMatchingWithEpiPolarGeometry(mvpKFs[ki], mpTargetFrame,mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugging);
			//		//std::string base = mpSystem->GetDirPath(0);
			//		//std::stringstream ssss;
			//		//ssss << base << "/dense/dense_"<< mpTargetFrame->GetKeyFrameID() << "_" << mvpKFs[ki]->GetKeyFrameID() << ".jpg";
			//		//imwrite(ssss.str(), debugging);
			//		///////dense

			//		std::vector<cv::DMatch> vMatches;
			//		std::vector<bool> vbInliers;
			//		UVR_SLAM::Frame* pKFi = mvpKFs[ki];
			//		std::cout << "pe::matching::0" << std::endl;
			//		std::vector<std::pair<int, cv::Point2f>> vPairs;
			//		mpMatcher->MatchingWithEpiPolarGeometry(pKFi, mpTargetFrame, vPlanarMaps, vbInliers,vMatches, vPairs, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugImg);
			//		//mpMatcher->DenseMatchingWithEpiPolarGeometry(pKFi, mpTargetFrame, vPlanarMaps, vPairs, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugImg);
			//		std::cout << "pe::matching::1" << std::endl;
			//		/*std::stringstream ss;
			//		std::cout << "pe::" << mStrPath << std::endl;
			//		std::cout << "pe::kf::" << pKFi->GetKeyFrameID() << std::endl;
			//		ss << mpSystem->GetDirPath(0) << "/kfmatching/" << mpTargetFrame->GetKeyFrameID() << "_" << pKFi->GetKeyFrameID() << ".jpg";
			//		imwrite(ss.str(), debugImg);*/
			//		std::cout << "pe::matching::2" << std::endl;

			//		for (int i = 0; i < vPairs.size(); i++) {
			//			//���� ������� Ȯ���� �����.
			//			auto idx = vPairs[i].first;
			//			auto pt = vPairs[i].second;

			//			UVR_SLAM::MapPoint* pMP1 = pKFi->GetDenseMP(pt);
			//			UVR_SLAM::MapPoint* pMP2 = mpTargetFrame->GetDenseMP(mpTargetFrame->mvKeyPoints[idx].pt);

			//			bool b1 = pMP1 && !pMP1->isDeleted();
			//			bool b2 = pMP2 && !pMP2->isDeleted();

			//			if (!b1 && !b2) {
			//				UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, vPlanarMaps[idx], mpTargetFrame->matDescriptor.row(idx), UVR_SLAM::PLANE_DENSE_MP);
			//				pNewMP->SetPlaneID(pFloor->mnPlaneID);
			//				pNewMP->SetObjectType(pFloor->mnPlaneType);
			//				pNewMP->AddDenseFrame(pKFi, pt);
			//	 			pNewMP->AddDenseFrame(mpTargetFrame, mpTargetFrame->mvKeyPoints[idx].pt);
			//				pNewMP->UpdateNormalAndDepth();
			//				pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
			//				mpSystem->mlpNewMPs.push_back(pNewMP);
			//				pFloor->tmpMPs.push_back(pNewMP); //�̰͵� �ٴڰ� ���� ���� �ʿ䰡 ����.
			//			}
			//			else if (b1 && b2) {
			//				if (pMP1->mnMapPointID != pMP2->mnMapPointID) {
			//					/*pMP1->FuseDenseMP(pMP2);
			//					pMP2->SetWorldPos(vPlanarMaps[idx]);
			//					std::cout << "Dense::Fuse::" << pMP1->GetNumDensedFrames() << ", " << pMP2->GetNumDensedFrames() << std::endl;*/
			//				}
			//			}
			//			else if (b1) {
			//				pMP1->AddDenseFrame(mpTargetFrame, mpTargetFrame->mvKeyPoints[idx].pt);
			//				pMP1->SetWorldPos(vPlanarMaps[idx]);
			//			}
			//			else if (b2) {
			//				pMP2->AddDenseFrame(pKFi, pt);
			//				pMP2->SetWorldPos(vPlanarMaps[idx]);
			//			}
			//			
			//		}

			//		for (int i = 0; i < vbInliers.size(); i++) {
			//			if (vbInliers[i]) {
			//				int idx1 = vMatches[i].trainIdx;
			//				int idx2 = vMatches[i].queryIdx;

			//				UVR_SLAM::MapPoint* pMP1 = pKFi->mvpMPs[idx1];
			//				UVR_SLAM::MapPoint* pMP2 = mpTargetFrame->mvpMPs[idx2];

			//				bool b1 = pMP1 && !pMP1->isDeleted();
			//				bool b2 = pMP2 && !pMP2->isDeleted();

			//				if (!b1 && !b2)
			//				{
			//					UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, vPlanarMaps[idx2], mpTargetFrame->matDescriptor.row(idx2), UVR_SLAM::PLANE_MP);
			//					pNewMP->SetPlaneID(pFloor->mnPlaneID);
			//					pNewMP->SetObjectType(pFloor->mnPlaneType);
			//					pNewMP->AddFrame(pKFi, idx1);
			//					pNewMP->AddFrame(mpTargetFrame, idx2);
			//					pNewMP->UpdateNormalAndDepth();
			//					pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
			//					mpSystem->mlpNewMPs.push_back(pNewMP);
			//					pFloor->tmpMPs.push_back(pNewMP);
			//				}
			//				else if (b1 && b2) {
			//					if(pMP1->mnMapPointID != pMP2->mnMapPointID){
			//						pMP1->Fuse(pMP2);
			//						pMP2->SetWorldPos(vPlanarMaps[idx2]);
			//					}
			//				}
			//				else if (b1) {
			//					pMP1->AddFrame(mpTargetFrame, idx2);
			//					pMP1->SetWorldPos(vPlanarMaps[idx2]);
			//				}
			//				else if (b2) {
			//					pMP2->AddFrame(pKFi, idx1);
			//					pMP2->SetWorldPos(vPlanarMaps[idx2]);
			//				}
			//			}
			//			else {
			//				if (vPlanarMaps[i].rows == 0)
			//					continue;
			//				/*UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, vPlanarMaps[i], mpTargetFrame->matDescriptor.row(i), UVR_SLAM::PLANE_MP);
			//				pNewMP->SetPlaneID(pFloor->mnPlaneID);
			//				pNewMP->SetObjectType(pFloor->mnPlaneType);
			//				pNewMP->AddFrame(mpTargetFrame, i);
			//				pNewMP->UpdateNormalAndDepth();
			//				pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
			//				mpSystem->mlpNewMPs.push_back(pNewMP);
			//				pFloor->tmpMPs.push_back(pNewMP);*/
			//			}
			//			
			//		}

			//		//���� �������� ����
			//		auto currLines = mpTargetFrame->Getlines();
			//		/*for (int cli = 0; cli < currLines.size(); cli++) {
			//			cv::Point2f pt1 = UVR_SLAM::PlaneInformation::CalcLinePoint(0.0, currLines[cli]->mLineEqu);
			//			cv::Point2f pt2 = UVR_SLAM::PlaneInformation::CalcLinePoint(mnHeight, currLines[cli]->mLineEqu);
			//			cv::line(vImg, pt1, pt2, cv::Scalar(0, 0, 255), 1);
			//		}*/

			//		///////////////////////////////////
			//		////////�ð�ȭ �׽�Ʈ
			//		//cv::Mat invP, invT, invK;
			//		//auto lines = pKFi->Getlines();
			//		//for (int li = 0; li < lines.size(); li++) {
			//		//	lines[li]->mpFrame->mpPlaneInformation->Calculate();
			//		//	lines[li]->mpFrame->mpPlaneInformation->GetInformation(invP, invT, invK);
			//		//	cv::Mat X3Dfrom = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(lines[li]->from, invP, invT, invK);
			//		//	cv::Mat X3Dto = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(lines[li]->to, invP, invT, invK);
			//		//	//���� �̹����� ��������
			//		//	cv::Mat R, t;
			//		//	mpTargetFrame->GetPose(R, t);
			//		//	cv::Mat temp1 = R*X3Dfrom + t;
			//		//	temp1 = mpTargetFrame->mK*temp1;
			//		//	cv::Point2f pt1(temp1.at<float>(0) / temp1.at<float>(2), temp1.at<float>(1) / temp1.at<float>(2));
			//		//	cv::Mat temp2 = R*X3Dto + t;
			//		//	temp2 = mpTargetFrame->mK*temp2;
			//		//	cv::Point2f pt2(temp2.at<float>(0) / temp2.at<float>(2), temp2.at<float>(1) / temp2.at<float>(2));
			//		//	cv::line(vImg, pt1, pt2, cv::Scalar(255, 0, 0), 1);

			//		//	//line ���� ���.
			//		//	for (int cli = 0; cli < currLines.size(); cli++) {
			//		//		float d1 = currLines[cli]->CalcLineDistance(pt1);
			//		//		float d2 = currLines[cli]->CalcLineDistance(pt2);
			//		//		float err = d1 + d2;
			//		//		if (err < 6.0) {
			//		//			//connect

			//		//			for (int lj = 0; lj < lines[li]->mvpLines.size(); lj++) {
			//		//				currLines[cli]->mvpLines.push_back(lines[li]->mvpLines[lj]);

			//		//			}
			//		//			currLines[cli]->mvpLines.push_back(lines[li]);
			//		//			//visualize all lines
			//		//			cv::line(vImg, pt1, pt2, cv::Scalar(255, 0, 255), 1);
			//		//		}
			//		//		std::cout << "line error == "<<li<<"="<<cli<<"=="<< d1 << ", " << d2 <<"::"<<pt1<<", "<<pt2<<"=="<<currLines[cli]->mLineEqu<< std::endl;
			//		//	}
			//		//}
			//		////����� ���� �ð�ȭ
			//		//for (int cli = 0; cli < currLines.size(); cli++) {
			//		//	for (int li = 0; li < currLines[cli]->mvpLines.size(); li++) {
			//		//		auto line = currLines[cli]->mvpLines[li];
			//		//		line->mpFrame->mpPlaneInformation->Calculate();
			//		//		line->mpFrame->mpPlaneInformation->GetInformation(invP, invT, invK);
			//		//		cv::Mat X3Dfrom = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(line->from, invP, invT, invK);
			//		//		cv::Mat X3Dto = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(line->to, invP, invT, invK);
			//		//		//���� �̹����� ��������
			//		//		cv::Mat R, t;
			//		//		mpTargetFrame->GetPose(R, t);
			//		//		cv::Mat temp1 = R*X3Dfrom + t;
			//		//		temp1 = mpTargetFrame->mK*temp1;
			//		//		float depth1 = temp1.at<float>(2);
			//		//		cv::Point2f pt1(temp1.at<float>(0) / temp1.at<float>(2), temp1.at<float>(1) / temp1.at<float>(2));
			//		//		cv::Mat temp2 = R*X3Dto + t;
			//		//		temp2 = mpTargetFrame->mK*temp2;
			//		//		cv::Point2f pt2(temp2.at<float>(0) / temp2.at<float>(2), temp2.at<float>(1) / temp2.at<float>(2));
			//		//		float depth2 = temp2.at<float>(2);
			//		//		if (depth1 < 0.0 || depth2 < 0.0)
			//		//			continue;
			//		//		cv::line(vImg, pt1, pt2, cv::Scalar(255, 255, 0), 1);
			//		//	}
			//		//}
			//		////////�ð�ȭ �׽�Ʈ
			//		///////////////////////////////////
			//		std::cout << "pe::matching::3" << std::endl;
			//	}
			//	mpMap->SetCurrFrame(mpTargetFrame);
			//	//////////////////////////////////////////////////////////////////////////////////


			//	//std::vector<cv::DMatch> vMatches;
			//	//////��Ī ����� ȹ��. ���⿡���� �� ������ ������ ��Ī ����� ����. ������Ʈ ���� ���� �Ǵ�x
			//	//mpMatcher->MatchingWithEpiPolarGeometry(mpPrevFrame, mpTargetFrame, pFloor, vPlanarMaps, vMatches, debugImg);
			//	//////fuse�� ������ ������ ������.
			//	//for (int i = 0; i < vMatches.size(); i++) {
			//	//	int idx1 = vMatches[i].trainIdx;
			//	//	int idx2 = vMatches[i].queryIdx;

			//	//	UVR_SLAM::MapPoint* pMP1 = mpPrevFrame->mvpMPs[idx1];
			//	//	UVR_SLAM::MapPoint* pMP2 = mpTargetFrame->mvpMPs[idx2];

			//	//	bool b1 = pMP1 && !pMP1->isDeleted();
			//	//	bool b2 = pMP2 && !pMP2->isDeleted();

			//	//	if (!b1 && !b2)
			//	//	{
			//	//		UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, vPlanarMaps[idx2], mpTargetFrame->matDescriptor.row(idx2), UVR_SLAM::PLANE_MP);
			//	//		pNewMP->SetPlaneID(pFloor->mnPlaneID);
			//	//		pNewMP->SetObjectType(pFloor->mnPlaneType);
			//	//		pNewMP->AddFrame(mpPrevFrame, idx1);
			//	//		pNewMP->AddFrame(mpTargetFrame, idx2);
			//	//		pNewMP->UpdateNormalAndDepth();
			//	//		pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
			//	//		mpSystem->mlpNewMPs.push_back(pNewMP);
			//	//		pFloor->tmpMPs.push_back(pNewMP);
			//	//	}
			//	//	else if (b1 && b2) {
			//	//		if (pMP1->GetConnedtedFrames() > pMP2->GetConnedtedFrames()) {
			//	//			pMP2->Fuse(pMP1);
			//	//		}
			//	//		else
			//	//			pMP1->Fuse(pMP2);

			//	//		/*if (pMPinKF->GetConnedtedFrames()>pMP->GetConnedtedFrames())
			//	//		pMP->Fuse(pMPinKF);
			//	//		else
			//	//		pMPinKF->Fuse(pMP);*/
			//	//	}
			//	//	else if (b1) {
			//	//		pMP1->AddFrame(mpTargetFrame, idx2);
			//	//	}
			//	//	else if (b2) {
			//	//		pMP2->AddFrame(mpPrevFrame, idx1);
			//	//	}
			//	//}
			//}
			//////////��Ī �׽�Ʈ
			//
			//std::cout << "pe::8" << std::endl;
			////////�ٴ�, ���� ������ ������ �������� ������ ����
			//////������ ������ ������ ����Ʈ�� ����� �׽�Ʈ �ϴµ� �̿�
			//////unlock & notify
			//mpSystem->mbPlaneEstimationEnd = true;
			//lock.unlock();
			//mpSystem->cvUsePlaneEstimation.notify_all();
			//////////////////////////////////////////////////////////////////////////
			//			
			//
			//
			////////test
			////////save txt
			///*std::ofstream f;
			//std::stringstream sss;
			//sss << mStrPath.c_str() << "/plane.txt";
			//f.open(sss.str().c_str());
			//mvpMPs = mpTargetFrame->GetMapPoints();
			//mvpOPs = mpTargetFrame->GetObjectVector();
			//if(bLocalFloor)
			//	for (int j = 0; j < mvpMPs.size(); j++) {
			//		UVR_SLAM::MapPoint* pMP = mvpMPs[j];
			//		if (!pMP) {
			//			continue;
			//		}
			//		if (pMP->isDeleted()) {
			//			continue;
			//		}
			//		cv::Mat Xw = pMP->GetWorldPos();
			//	
			//		if (pMP->GetPlaneID() > 0) {
			//			if (pMP->GetPlaneID() == tempFloorID)
			//			{
			//				f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) << " 1" << std::endl;
			//			}
			//			else if (pMP->GetPlaneID() == tempWallID) {
			//				f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) << " 2" << std::endl;
			//			}
			//		}
			//		else
			//			f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) << " 0" << std::endl;
			//	}
			//f.close();*/
			////////save txt
			//
			//////////////////image
			//
			//if(bMaxFloorY)
			//	cv::line(vImg, cv::Point2f(0, maxY * 2), cv::Point2f(vImg.cols, maxY * 2), cv::Scalar(255, 0, 0), 2);
			//cv::Mat R, t;
			//mpTargetFrame->GetPose(R, t);
			///*for (int j = 0; j < mvpDummys.size(); j++) {
			//	auto pMP = mvpDummys[j];
			//	cv::Mat temp = R*pMP->GetWorldPos() + t;
			//	temp = mK*temp;
			//	cv::Point2f pt(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
			//	cv::circle(vImg, pt, 3, cv::Scalar(0, 255, 255), -1);
			//}*/
			//
			////for (int j = 0; j < mvpMPs.size(); j++) {
			////	auto type = mvpOPs[j];
			////	int wtype;
			////	int x;
			////	switch (type) {
			////	case ObjectType::OBJECT_WALL:
			////		x = mpTargetFrame->mvKeyPoints[j].pt.x;
			////		wtype = 2;
			////		if (bMinX && x < minFloorX) {
			////			wtype = 1;
			////		}
			////		if (bMaxX && x > maxFloorX) {
			////			wtype = 3;
			////		}
			////		if (wtype == 1) {
			////			cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 0, 255), -1);
			////		}
			////		else if (wtype == 3) {
			////			cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 0), -1);
			////		}
			////		else {
			////			cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(255,0, 0), -1);
			////		}
			////		break;
			////	case ObjectType::OBJECT_CEILING:
			////		//cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 0), -1);
			////		break;
			////	case ObjectType::OBJECT_NONE:
			////		//cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 0, 0), -1);
			////		break;
			////	case ObjectType::OBJECT_FLOOR:
			////		cv::circle(vImg, mpTargetFrame->mvKeyPoints[j].pt, 1, cv::Scalar(0, 255, 255), -1);
			////		break;
			////	}
			////}
			//
			///*sss.str("");
			//sss << mStrPath.c_str() << "/plane.jpg";
			//cv::imwrite(sss.str(), vImg);*/
			//imshow("Output::PlaneEstimation", vImg); cv::waitKey(1);
			//std::cout << "pe::end::" << mpTargetFrame->GetKeyFrameID() << std::endl;
			//SetBoolDoingProcess(false, 1);
		}
	}
}

//////////////////////////////////////////////////////////////////////
cv::Mat UVR_SLAM::PlaneInformation::PlaneLineEstimator(WallPlane* pWall, PlaneInformation* pFloor) {
	cv::Mat mat = cv::Mat::zeros(0,3,CV_32FC1);

	auto lines = pWall->GetLines();
	for (int i = 0; i < lines.size(); i++) {
		mat.push_back(lines[i]->GetLinePts());
	}
	
	std::random_device rn;
	std::mt19937_64 rnd(rn());
	std::uniform_int_distribution<int> range(0, mat.rows - 1);

	//ransac
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;
	cv::Mat param, paramStatus;
	//ransac

	for (int n = 0; n < 1000; n++) {

		cv::Mat arandomPts = cv::Mat::zeros(0, 4, CV_32FC1);
		//select pts
		for (int k = 0; k < 3; k++) {
			int randomNum = range(rnd);
			cv::Mat temp = mat.row(randomNum).clone();
			arandomPts.push_back(temp);
		}//select

		 //SVD
		cv::Mat X;
		cv::Mat w, u, vt;
		cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
		X = vt.row(2).clone();
		cv::transpose(X, X);

		cv::Mat checkResidual = abs(mat*X) < 0.01;
		checkResidual = checkResidual / 255;
		int temp_inlier = cv::countNonZero(checkResidual);

		if (max_num_inlier < temp_inlier) {
			max_num_inlier = temp_inlier;
			param = X.clone();
			paramStatus = checkResidual.clone();
		}
	}
	
	/////////
	cv::Mat newMat = cv::Mat::zeros(0,3, CV_32FC1);
	int nInc = 1;
	if (mat.rows > 1000)
		nInc *= 10;
	for (int i = 0; i < mat.rows; i+= nInc) {
		int checkIdx = paramStatus.at<uchar>(i);
		if (checkIdx == 0)
			continue;
		newMat.push_back(mat.row(i));
	}
	
	cv::Mat w, u, vt;
	cv::SVD::compute(newMat, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	
	param = vt.row(2).clone();
	cv::transpose(param, param);
	
	/////////
	float yval = 0.0;
	cv::Mat pParam = pFloor->matPlaneParam.clone();
	//std::cout << param.t() <<pParam.t()<< std::endl;
	cv::Mat normal;
	float dist;
	pFloor->GetParam(normal, dist);
	cv::Mat s = cv::Mat::zeros(4, 1, CV_32FC1);
	s.at<float>(3) = 1.0;
	s.at<float>(2) = -param.at<float>(2) / param.at<float>(1);
	yval = pParam.dot(s);
	s.at<float>(1) = -yval / pParam.at<float>(1);

	cv::Mat e = cv::Mat::zeros(4, 1, CV_32FC1);
	e.at<float>(0) = 1.0;
	e.at<float>(3) = 1.0;
	e.at<float>(2) = -(param.at<float>(0)+param.at<float>(2)) / param.at<float>(1);
	yval = pParam.dot(e);
	e.at<float>(1) = -yval / pParam.at<float>(1);
	//std::cout << "????" << std::endl;
	//std::cout << s.dot(pParam) << ", " << e.dot(pParam) << std::endl;
	param = UVR_SLAM::PlaneInformation::PlaneWallEstimator(s.rowRange(0,3), e.rowRange(0,3), normal, cv::Mat(), cv::Mat(), cv::Mat());
	//std::cout << "test : " << param.t() << "::" << max_num_inlier << ", " << mat.rows << std::endl;
	//std::cout << "line param : " << param.t() << std::endl;
	return param;
} 

//��� ���� ���� �Լ���
bool UVR_SLAM::PlaneInformation::PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::MapPoint*>& vpOutlierMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
	//RANSAC
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;

	cv::Mat param, paramStatus;

	//�ʱ� ��Ʈ���� ����
	cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
	std::vector<int> vIdxs;
	for(int i = 0; i < vpMPs.size(); i++){
		UVR_SLAM::MapPoint* pMP = vpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		cv::Mat temp = pMP->GetWorldPos();
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		mMat.push_back(temp.t());
		vIdxs.push_back(i);
	}
	if (mMat.rows < 10)
		return false;
	std::random_device rn;
	std::mt19937_64 rnd(rn());
	std::uniform_int_distribution<int> range(0, mMat.rows-1);

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

		calcUnitNormalVector(X);
		//reversePlaneSign(X);

		/*cv::Mat X2 = vt.col(3).clone();
		calcUnitNormalVector(X2);
		reversePlaneSign(X2);
		std::cout << sum(abs(mMatFromMap*X)) << " " << sum(abs(mMatFromMap*X2)) << std::endl;*/

		//cv::Mat checkResidual = abs(mMatCurrMap*X);
		//threshold(checkResidual, checkResidual, thresh_plane_distance, 1.0, cv::THRESH_BINARY_INV);
		cv::Mat checkResidual = abs(mMat*X) < thresh_distance;
		checkResidual = checkResidual / 255;
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
		cv::Mat pParam = param.clone();
		pPlane->matPlaneParam = pParam.clone();
		pPlane->mnPlaneID = ++nPlaneID;

		cv::Mat normal = pParam.rowRange(0, 3);
		float dist = pParam.at<float>(3);
		pPlane->SetParam(normal, dist);
		//pPlane->norm = sqrt(pPlane->normal.dot(pPlane->normal));

		for (int i = 0; i < mMat.rows; i++) {
			int checkIdx = paramStatus.at<uchar>(i);
			//std::cout << checkIdx << std::endl;
			UVR_SLAM::MapPoint* pMP = vpMPs[vIdxs[i]];
			if (checkIdx == 0){
				vpOutlierMPs.push_back(pMP);
				continue;
			}
			if (pMP && !pMP->isDeleted()) {
				//��鿡 ���� ���̺��� �ʿ���.
				//pMP->SetRecentLayoutFrameID(nTargetID);
				//pMP->SetPlaneID(pPlane->mnPlaneID);
				pPlane->mvpMPs.push_back(pMP);
				tempMat.push_back(mMat.row(i));
			}
		}
		//��� ���� ����.

		cv::Mat X;
		cv::Mat w, u, vt;
		cv::SVD::compute(tempMat, w, u, vt, cv::SVD::FULL_UV);
		X = vt.row(3).clone();
		cv::transpose(X, X);
		UVR_SLAM::PlaneInformation::calcUnitNormalVector(X);
		int idx = pPlane->GetNormalType(X);
		if (X.at<float>(idx) > 0.0)
			X *= -1.0;
		pPlane->mnCount = pPlane->mvpMPs.size();
		//std::cout <<"PLANE::"<< planeRatio << std::endl;
		//std::cout << "Update::" << pPlane->matPlaneParam.t() << ", " << X.t() <<", "<<pPlane->mvpMPs.size()<<", "<<nReject<< std::endl;
		pPlane->SetParam(X.rowRange(0, 3), X.at<float>(3));

		return true;
	}
	else
	{
		//std::cout << "failed" << std::endl;
		return false;
	}
}
bool UVR_SLAM::PlaneInformation::PlaneInitialization2(UVR_SLAM::PlaneInformation* pPlane, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::MapPoint*>& vpOutlierMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
	//RANSAC
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;

	cv::Mat param, paramStatus;

	//�ʱ� ��Ʈ���� ����
	cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
	std::vector<int> vIdxs;
	for (int i = 0; i < vpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = vpMPs[i];
		if (!pMP || pMP->isDeleted())
			continue;
		cv::Mat temp = pMP->GetWorldPos();
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		mMat.push_back(temp.t());
		vIdxs.push_back(i);
	}
	if (mMat.rows < 10)
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

		calcUnitNormalVector(X);
	
		cv::Mat checkResidual = abs(mMat*X) < thresh_distance;
		checkResidual = checkResidual / 255;
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
		cv::Mat pParam = param.clone();
		pPlane->matPlaneParam = pParam.clone();
		pPlane->mnPlaneID = ++nPlaneID;

		cv::Mat normal = pParam.rowRange(0, 3);
		float dist = pParam.at<float>(3);
		pPlane->SetParam(normal, dist);
		//pPlane->norm = sqrt(pPlane->normal.dot(pPlane->normal));

		for (int i = 0; i < mMat.rows; i++) {
			int checkIdx = paramStatus.at<uchar>(i);
			//std::cout << checkIdx << std::endl;
			UVR_SLAM::MapPoint* pMP = vpMPs[vIdxs[i]];
			if (checkIdx == 0) {
				vpOutlierMPs.push_back(pMP);
				continue;
			}
			if (pMP && !pMP->isDeleted()) {
				//��鿡 ���� ���̺��� �ʿ���.
				//pMP->SetRecentLayoutFrameID(nTargetID);
				//pMP->SetPlaneID(pPlane->mnPlaneID);
				pPlane->mvpMPs.push_back(pMP);
				tempMat.push_back(mMat.row(i));
			}
		}
		//��� ���� ����.

		cv::Mat X;
		cv::Mat w, u, vt;
		cv::SVD::compute(tempMat, w, u, vt, cv::SVD::FULL_UV);
		X = vt.row(3).clone();
		cv::transpose(X, X);
		UVR_SLAM::PlaneInformation::calcUnitNormalVector(X);
		int idx = pPlane->GetNormalType(X);
		if (X.at<float>(idx) > 0.0)
			X *= -1.0;
		pPlane->mnCount = pPlane->mvpMPs.size();
		//std::cout <<"PLANE::"<< planeRatio << std::endl;
		//std::cout << "Update::" << pPlane->matPlaneParam.t() << ", " << X.t() <<", "<<pPlane->mvpMPs.size()<<", "<<nReject<< std::endl;
		pPlane->SetParam(X.rowRange(0, 3), X.at<float>(3));

		return true;
	}
	else
	{
		//std::cout << "failed" << std::endl;
		return false;
	}
}
void UVR_SLAM::PlaneEstimator::UpdatePlane(PlaneInformation* pPlane, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
	auto mvpMPs = std::vector<UVR_SLAM::MapPoint*>(pPlane->mvpMPs.begin(), pPlane->mvpMPs.end());
	std::vector<int> vIdxs(0);
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;
	//�ʱ� ��Ʈ���� ����
	cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
	int nStartIdx = 0;
	if (mvpMPs.size() > 5000) {
		nStartIdx = 1000;
	}
	for (int i = nStartIdx; i < mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		cv::Mat temp = pMP->GetWorldPos();
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		mMat.push_back(temp.t());
		vIdxs.push_back(i);
	}
	if (mMat.rows == 0)
		return;

	cv::Mat param, paramStatus;
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
		UVR_SLAM::PlaneInformation::calcUnitNormalVector(X);
		if (X.at<float>(1) > 0.0)
			X *= -1.0;

		cv::Mat checkResidual = abs(mMat*X) < 0.001;
		checkResidual = checkResidual / 255;
		int temp_inlier = cv::countNonZero(checkResidual);

		if (max_num_inlier < temp_inlier) {
			max_num_inlier = temp_inlier;
			param = X.clone();
			paramStatus = checkResidual.clone();
		}
	}//trial

	if (max_num_inlier == 0)
		return;

	float planeRatio = ((float)max_num_inlier / mMat.rows);
	
	if (planeRatio > thresh_ratio) {
		int nReject = 0;
		pPlane->mvpMPs.clear();
		cv::Mat tempMat = cv::Mat::zeros(0, 4, CV_32FC1);
		
		for (int i = 0; i < mMat.rows; i++) {
			int checkIdx = paramStatus.at<uchar>(i);
			UVR_SLAM::MapPoint* pMP = mvpMPs[vIdxs[i]];
			if (checkIdx == 0){
				pMP->SetPlaneID(-1);
				nReject++;
				continue;
			}
			if (pMP) {
				//��鿡 ���� ���̺��� �ʿ���.
				if (pMP->isDeleted())
					continue;
				pMP->SetRecentLayoutFrameID(nTargetID);
				pMP->SetPlaneID(pPlane->mnPlaneID);
				pPlane->mvpMPs.push_back(pMP);
				tempMat.push_back(mMat.row(i));
			}
		}
		//��� ���� ����.
		
		cv::Mat X;
		cv::Mat w, u, vt;
		cv::SVD::compute(tempMat, w, u, vt, cv::SVD::FULL_UV);
		X = vt.row(3).clone();
		cv::transpose(X, X);
		UVR_SLAM::PlaneInformation::calcUnitNormalVector(X);
		if (X.at<float>(1) > 0.0)
			X *= -1.0;

		//std::cout << "Update::" << pPlane->matPlaneParam.t() << ", " << X.t() <<", "<<pPlane->mvpMPs.size()<<", "<<nReject<< std::endl;
		pPlane->SetParam(X.rowRange(0, 3), X.at<float>(3));

		return;
	}
	else
	{
		return;
	}
}

//GroundPlane�� ���� ���, type == 1�̸� ��, �ƴϸ� õ��
bool UVR_SLAM::PlaneEstimator::PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, UVR_SLAM::PlaneInformation* GroundPlane, int type, std::vector<UVR_SLAM::MapPoint*> vpMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio) {
	//RANSAC
	std::vector<int> vIdxs(0);
	int max_num_inlier = 0;
	cv::Mat best_plane_param;
	cv::Mat inlier;

	cv::Mat param, paramStatus;

	//�ʱ� ��Ʈ���� ����
	cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
	for (int i = 0; i < vpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = vpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		cv::Mat temp = pMP->GetWorldPos();
		temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		mMat.push_back(temp.t());
		vIdxs.push_back(i);
	}
	if (mMat.rows == 0)
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
		UVR_SLAM::PlaneInformation::calcUnitNormalVector(X);

		float val = GroundPlane->CalcCosineSimilarity(X);
		//std::cout << "cos::" << val << std::endl;
		if (type == 1) {
			//�ٴڰ� ��	
			if (abs(val) > mfThreshNormal)
				continue;
		}
		else {
			//�ٴڰ� õ��
			if (1.0 - abs(val) > mfThreshNormal)
				continue;
		}

		cv::Mat checkResidual = abs(mMat*X) < thresh_distance;
		checkResidual = checkResidual / 255;
		int temp_inlier = cv::countNonZero(checkResidual);

		if (max_num_inlier < temp_inlier) {
			max_num_inlier = temp_inlier;
			param = X.clone();
			paramStatus = checkResidual.clone();
		}
	}//trial

	if (max_num_inlier == 0)
		return false;

	float planeRatio = ((float)max_num_inlier / mMat.rows);
	//std::cout << "PLANE INIT : " << max_num_inlier << ", " << paramStatus.rows << "::" << cv::countNonZero(paramStatus) << " " << spMPs.size() << "::" << planeRatio << std::endl;

	//cv::Mat checkResidual2 = mMat*param > 2 * thresh_distance; checkResidual2 /= 255; checkResidual2 *= 2;
	//paramStatus += checkResidual2;

	if (planeRatio > thresh_ratio) {
		cv::Mat pParam = param.clone();
		pPlane->mnPlaneID = ++nPlaneID;
		pPlane->SetParam(pParam.rowRange(0, 3), pParam.at<float>(3));

		for (int i = 0; i < mMat.rows; i++) {
			int checkIdx = paramStatus.at<uchar>(i);
			//std::cout << checkIdx << std::endl;
			UVR_SLAM::MapPoint* pMP = vpMPs[vIdxs[i]];
			if (checkIdx == 0)
				continue;
			if (pMP) {
				//��鿡 ���� ���̺��� �ʿ���.
				if (pMP->isDeleted())
					continue;

				pMP->SetRecentLayoutFrameID(nTargetID);
				pMP->SetPlaneID(pPlane->mnPlaneID);
				pPlane->mvpMPs.push_back(pMP);
			}
		}
		//��� ���� ����.
		return true;
	}
	else
	{
		//std::cout << "failed" << std::endl;
		return false;
	}
}

bool UVR_SLAM::PlaneInformation::calcUnitNormalVector(cv::Mat& X) {
	float sum = sqrt(X.at<float>(0)*X.at<float>(0) + X.at<float>(1)*X.at<float>(1) + X.at<float>(2)*X.at<float>(2));
	//cout<<"befor X : "<<X<<endl;
	if (sum != 0) {
		X.at<float>(0, 0) = X.at<float>(0, 0) / sum;
		X.at<float>(1, 0) = X.at<float>(1, 0) / sum;
		X.at<float>(2, 0) = X.at<float>(2, 0) / sum;
		X.at<float>(3, 0) = X.at<float>(3, 0) / sum;
		//cout<<"after X : "<<X<<endl;
		return true;
	}
	return false;
}
int UVR_SLAM::PlaneInformation::GetNormalType(cv::Mat X) {
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

void UVR_SLAM::PlaneEstimator::reversePlaneSign(cv::Mat& param) {
	if (param.at<float>(3, 0) < 0.0) {
		param *= -1.0;
	}
}
//��� ���� ���� �Լ���
//////////////////////////////////////////////////////////////////////

//�÷�Ŀ ���� �������� ���� �Լ�
cv::Mat UVR_SLAM::PlaneInformation::FlukerLineProjection(cv::Mat P1, cv::Mat P2, cv::Mat R, cv::Mat t, cv::Mat K, float& m) {
	cv::Mat PLw1, Lw1, NLw1;
	PLw1 = P1*P2.t() - P2*P1.t();
	Lw1 = cv::Mat::zeros(6, 1, CV_32FC1);
	Lw1.at<float>(3) = PLw1.at<float>(2, 1);
	Lw1.at<float>(4) = PLw1.at<float>(0, 2);
	Lw1.at<float>(5) = PLw1.at<float>(1, 0);
	NLw1 = PLw1.col(3).rowRange(0, 3);
	NLw1.copyTo(Lw1.rowRange(0, 3));

	//Line projection test : Ni
	cv::Mat T2 = cv::Mat::zeros(6, 6, CV_32FC1);
	R.copyTo(T2.rowRange(0, 3).colRange(0, 3));
	R.copyTo(T2.rowRange(3, 6).colRange(3, 6));
	cv::Mat tempSkew = cv::Mat::zeros(3, 3, CV_32FC1);
	tempSkew.at<float>(0, 1) = -t.at<float>(2);
	tempSkew.at<float>(1, 0) = t.at<float>(2);
	tempSkew.at<float>(0, 2) = t.at<float>(1);
	tempSkew.at<float>(2, 0) = -t.at<float>(1);
	tempSkew.at<float>(1, 2) = -t.at<float>(0);
	tempSkew.at<float>(2, 1) = t.at<float>(0);
	tempSkew *= R;
	tempSkew.copyTo(T2.rowRange(0, 3).colRange(3, 6));
	cv::Mat Lc = T2*Lw1;
	cv::Mat Nc = Lc.rowRange(0, 3);
	cv::Mat res = K*Nc;
	if (res.at<float>(0) < 0)
		res *= -1;
	if (res.at<float>(0) != 0)
		m = res.at<float>(1) / res.at<float>(0);
	else
		m = 9999.0;
	return res.clone();
}
cv::Mat UVR_SLAM::PlaneInformation::FlukerLineProjection(PlaneInformation* P, cv::Mat R, cv::Mat t, cv::Mat K, float& m) {
	cv::Mat PLw1, Lw1, NLw1;
	PLw1 = this->matPlaneParam*P->matPlaneParam.t() - P->matPlaneParam*this->matPlaneParam.t();
	Lw1 = cv::Mat::zeros(6, 1, CV_32FC1);
	Lw1.at<float>(3) = PLw1.at<float>(2, 1);
	Lw1.at<float>(4) = PLw1.at<float>(0, 2);
	Lw1.at<float>(5) = PLw1.at<float>(1, 0);
	NLw1 = PLw1.col(3).rowRange(0, 3);
	NLw1.copyTo(Lw1.rowRange(0, 3));

	//Line projection test : Ni
	cv::Mat T2 = cv::Mat::zeros(6, 6, CV_32FC1);
	R.copyTo(T2.rowRange(0, 3).colRange(0, 3));
	R.copyTo(T2.rowRange(3, 6).colRange(3, 6));
	cv::Mat tempSkew = cv::Mat::zeros(3, 3, CV_32FC1);
	tempSkew.at<float>(0, 1) = -t.at<float>(2);
	tempSkew.at<float>(1, 0) = t.at<float>(2);
	tempSkew.at<float>(0, 2) = t.at<float>(1);
	tempSkew.at<float>(2, 0) = -t.at<float>(1);
	tempSkew.at<float>(1, 2) = -t.at<float>(0);
	tempSkew.at<float>(2, 1) = t.at<float>(0);
	tempSkew *= R;
	tempSkew.copyTo(T2.rowRange(0, 3).colRange(3, 6));
	cv::Mat Lc = T2*Lw1;
	cv::Mat Nc = Lc.rowRange(0, 3);
	cv::Mat res = K*Nc;
	if (res.at<float>(0) < 0)
		res *= -1;
	if (res.at<float>(0) != 0)
		m = res.at<float>(1) / res.at<float>(0);
	else
		m = 9999.0;
	return res.clone();
}

cv::Point2f UVR_SLAM::PlaneInformation::CalcLinePoint(float y, cv::Mat mLine) {
	float x = 0.0;
	if (mLine.at<float>(0) != 0)
		x = (-mLine.at<float>(2) - mLine.at<float>(1)*y) / mLine.at<float>(0);
	return cv::Point2f(x, y);
}

void UVR_SLAM::PlaneInformation::CalcFlukerLinePoints(cv::Point2f& sPt, cv::Point2f& ePt, float f1, float f2, cv::Mat mLine) {
	sPt = CalcLinePoint(f1, mLine);
	ePt = CalcLinePoint(f2, mLine);
}

///////////////
////Pluker Lines
//cv::Mat P1 = (cv::Mat_<float>(4, 1) << 0, 1, 0, 0);
//cv::Mat P2 = (cv::Mat_<float>(4, 1) << 1, 0, 0, -0.36);

//cv::Mat PLw = P1*P2.t() - P2*P1.t();
//cv::Mat Lw = cv::Mat::zeros(6, 1, CV_32FC1);
//Lw.at<float>(3) = PLw.at<float>(2, 1);
//Lw.at<float>(4) = PLw.at<float>(0, 2);
//Lw.at<float>(5) = PLw.at<float>(1, 0);
//cv::Mat NLw = PLw.col(3).rowRange(0, 3);
//NLw.copyTo(Lw.rowRange(0, 3));
//std::cout << PLw << Lw << std::endl;
////Line projection test : Ni
//cv::Mat T2 = cv::Mat::zeros(6, 6, CV_32FC1);
//R.copyTo(T2.rowRange(0, 3).colRange(0, 3));
//R.copyTo(T2.rowRange(3, 6).colRange(3, 6));
//cv::Mat tempSkew = cv::Mat::zeros(3, 3, CV_32FC1);
//tempSkew.at<float>(0, 1) = -t.at<float>(2);
//tempSkew.at<float>(1, 0) = t.at<float>(2);
//tempSkew.at<float>(0, 2) = t.at<float>(1);
//tempSkew.at<float>(2, 0) = -t.at<float>(1);
//tempSkew.at<float>(1, 2) = -t.at<float>(0);
//tempSkew.at<float>(2, 1) = t.at<float>(0);
//tempSkew *= R;
//tempSkew.copyTo(T2.rowRange(0, 3).colRange(3, 6));
//cv::Mat Lc = T2*Lw;
//cv::Mat Nc = Lc.rowRange(0, 3);
//cv::Mat Ni = mK2*Nc;
//std::cout << Ni << std::endl;

//float x1 = 0;
//float y1 = 0;
//if (Ni.at<float>(0) != 0)
//	x1 = -Ni.at<float>(2) / Ni.at<float>(0);

//float x2 = 0;
//float y2 = 480;
//if (Ni.at<float>(0) != 0)
//	x2 = (-Ni.at<float>(2) - Ni.at<float>(1)*y2) / Ni.at<float>(0);
//cv::line(vis, cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Scalar(255, 0, 0), 2);
////Pluker Lines
///////////////

//keyframe(p)���� ���� local map(this)�� ����
void UVR_SLAM::PlaneInformation::Merge(PlaneInformation* p, int nID, float thresh) {
	//p�� ���ϴ� MP �� ���� ��鿡 ���ϴ� �͵� �߰�
	//map point vector ����
	//update param

	int n1 = p->mvpMPs.size();
	int n2 = mvpMPs.size();

	for (int i = 0; i < mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		/*if (pMP->GetRecentLocalMapID() < nID) {
			continue;
		}*/
		if (pMP->GetPlaneID() == p->mnPlaneID) {
			continue;
		}
		//distance ���
		cv::Mat X3D = pMP->GetWorldPos();
		cv::Mat normal = p->matPlaneParam.rowRange(0, 3);
		float dist = p->matPlaneParam.at<float>(3);
		float res = abs(normal.dot(X3D) + dist);

		if (res < thresh)
			p->mvpMPs.push_back(pMP);
	}
	mvpMPs = std::vector<MapPoint*>(p->mvpMPs.begin(), p->mvpMPs.end());

	std::cout << "Merge::" << n1 << ", " << n2 << "::" << mvpMPs.size() << std::endl;
}

//this : keyframe
//p : localmap
float UVR_SLAM::PlaneInformation::CalcOverlapMPs(PlaneInformation* p, int nID) {
	std::map<UVR_SLAM::MapPoint*, int> mmpMPs;
	int nCount = 0;
	int nTotal = 0;

	for (int i = 0; i < p->mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = p->mvpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		if (pMP->GetRecentLocalMapID()>=nID) {
			nTotal++;
		}
		if (pMP->GetPlaneID() == mnPlaneID) {
			nCount++;
		}
	}
	std::cout << "Association::Overlap::" << nCount << ", " << nTotal <<"::"<<p->mvpMPs.size()<<", "<<mvpMPs.size()<< std::endl;
	return ((float)nCount) / nTotal;
}

bool CheckZero(float val) {
	if (abs(val) < 1e-6) {
		return true;
	}
	return false;
}

float UVR_SLAM::PlaneInformation::CalcCosineSimilarity(PlaneInformation* p) {
	
	float d1 = this->norm;
	float d2 = p->norm;
	if (CheckZero(d1) || CheckZero(d2))
		return 0.0;
	return abs(normal.dot(p->normal) / (d1*d2));
}

float UVR_SLAM::PlaneInformation::CalcCosineSimilarity(cv::Mat P) {

	float d1 = this->norm;
	cv::Mat tempNormal = P.rowRange(0, 3);

	float d2 = sqrt(tempNormal.dot(tempNormal));
	
	if (CheckZero(d1) || CheckZero(d2))
		return 0.0;
	return abs(normal.dot(tempNormal) / (d1*d2));
}


float UVR_SLAM::PlaneInformation::CalcPlaneDistance(PlaneInformation* p) {
	return abs((distance) - (p->distance));
	//return abs(abs(distance) - abs(p->distance));
}

float UVR_SLAM::PlaneInformation::CalcPlaneDistance(cv::Mat X) {
	return X.dot(this->normal) + distance;
}

float UVR_SLAM::PlaneInformation::CalcCosineSimilarity(cv::Mat P1, cv::Mat P2){
	cv::Mat normal1 = P1.rowRange(0, 3);
	cv::Mat normal2 = P2.rowRange(0, 3);
	float d1 = sqrt(normal1.dot(normal1));
	float d2 = sqrt(normal2.dot(normal2));
	if (CheckZero(d1) || CheckZero(d2))
		return 0.0;
	return abs(normal1.dot(normal2)) / (d1*d2);
}

//�� ����� �Ÿ��� ��
float UVR_SLAM::PlaneInformation::CalcPlaneDistance(cv::Mat X1, cv::Mat X2){
	float d1 = (X1.at<float>(3));
	float d2 = (X2.at<float>(3));

	return abs(d1 - d2);

}
bool UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(cv::Point2f pt, cv::Mat invP, cv::Mat invT, cv::Mat invK, cv::Mat& X3D) {
	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	temp = invK*temp;
	cv::Mat matDepth = -invP.at<float>(3) / (invP.rowRange(0, 3).t()*temp);
	float depth = matDepth.at<float>(0);
	if (depth < 0) {
		//depth *= -1.0;
		return false;
	}
	temp *= depth;
	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	cv::Mat estimated = invT*temp;
	X3D = estimated.rowRange(0, 3);
	return true;
}

bool UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(cv::Point2f pt, cv::Mat invPw, cv::Mat invT, cv::Mat invK, cv::Mat fNormal, float fDist, cv::Mat& X3D) {
	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	temp = invK*temp;
	cv::Mat matDepth = -invPw.at<float>(3) / (invPw.rowRange(0, 3).t()*temp);
	float depth = matDepth.at<float>(0);
	if (depth < 0) {
		//depth *= -1.0;
		return false;
	}
	temp *= depth;
	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	cv::Mat estimated = invT*temp;
	X3D = estimated.rowRange(0, 3);
	float val = X3D.dot(fNormal) + fDist;
	if (val < 0.0)
		return false;
	return true;
}

bool UVR_SLAM::PlaneEstimator::ConnectedComponentLabeling(cv::Mat img, cv::Mat& dst, cv::Mat& stat) {
	dst = img.clone();
	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(img, img_labels, stats, centroids, 8, CV_32S);

	if (numOfLables == 0)
		return false;

	int maxArea = 0;
	int maxIdx = 0;
	//�󺧸� �� �̹����� ���� ���簢������ �ѷ��α� 
	for (int j = 1; j < numOfLables; j++) {
		int area = stats.at<int>(j, CC_STAT_AREA);
		if (area > maxArea) {
			maxArea = area;
			maxIdx = j;
		}
	}
	/*int left = stats.at<int>(maxIdx, CC_STAT_LEFT);
	int top = stats.at<int>(maxIdx, CC_STAT_TOP);
	int width = stats.at<int>(maxIdx, CC_STAT_WIDTH);
	int height = stats.at<int>(maxIdx, CC_STAT_HEIGHT);*/

	for (int j = 1; j < numOfLables; j++) {
		if (j == maxIdx)
			continue;
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);

		rectangle(dst, Point(left, top), Point(left + width, top + height), Scalar(0, 0, 0), -1);
	}
	stat = stats.row(maxIdx).clone();
	return true;
}

cv::Mat UVR_SLAM::PlaneInformation::CalcPlaneRotationMatrix(cv::Mat P) {
	//euler zxy
	cv::Mat Nidealfloor = cv::Mat::zeros(3, 1, CV_32FC1);
	cv::Mat normal = P.rowRange(0, 3);

	Nidealfloor.at<float>(1) = -1.0;
	float nx = P.at<float>(0);
	float ny = P.at<float>(1);
	float nz = P.at<float>(2);

	float d1 = atan2(nx, -ny);
	float d2 = atan2(-nz, sqrt(nx*nx + ny*ny));
	cv::Mat R = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngles(d1, d2, 0.0, "ZXY");


	///////////�ѹ� �� �����°� �׽�Ʈ
	/*cv::Mat R1 = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngle(d1, 'z');
	cv::Mat R2 = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngle(d2, 'x');
	cv::Mat Nnew = R2.t()*R1.t()*normal;
	float d3 = atan2(Nnew.at<float>(0), Nnew.at<float>(2));
	cv::Mat Rfinal = UVR_SLAM::MatrixOperator::RotationMatrixFromEulerAngles(d1, d2, d3, "ZXY");*/
	///////////�ѹ� �� �����°� �׽�Ʈ

	/*cv::Mat test1 = R*Nidealfloor;
	cv::Mat test3 = R.t()*normal;
	std::cout << "ATEST::" << P.t() << test1.t() << test3.t()<< std::endl;*/


	/*
	cv::Mat test2 = Rfinal*Nidealfloor;
	cv::Mat test4 = Rfinal.t()*normal;
	std::cout << d1 << ", " << d2 << ", " << d3 << std::endl;
	std::cout << "ATEST::" << P.t() << test1.t() << test2.t() << test3.t() << test4.t() << std::endl;*/

	return R;
}

////////////////////////////////////////////////////////////////////////
//////////�ڵ� ���
void UVR_SLAM::PlaneEstimator::CreatePlanarMapPoints(std::vector<UVR_SLAM::MapPoint*> mvpMPs, std::vector<UVR_SLAM::ObjectType> mvpOPs, UVR_SLAM::PlaneInformation* pPlane, cv::Mat invT) {
	
	//int nTargetID = mpTargetFrame->GetFrameID();
	//
	//cv::Mat invP1 = invT.t()*pPlane->GetParam();

	//float minDepth = FLT_MAX;
	//float maxDepth = 0.0f;
	////create new mp in current frame
	//for (int j = 0; j < mvpMPs.size(); j++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//	auto oType = mvpOPs[j];

	//	if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR)
	//		continue;
	//	cv::Point2f pt = mpTargetFrame->mvKeyPoints[j].pt;
	//	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	//	temp = mK.inv()*temp;
	//	cv::Mat matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
	//	float depth = matDepth.at<float>(0);
	//	if (depth < 0){
	//		//depth *= -1.0;
	//		continue;
	//	}
	//	temp *= depth;
	//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	//	/*if (maxDepth < depth)
	//		maxDepth = depth;
	//	if (minDepth > depth)
	//		minDepth = depth;*/

	//	cv::Mat estimated = invT*temp;
	//	if (pMP) {
	//		pMP->SetWorldPos(estimated.rowRange(0, 3));
	//	}
	//	else {
	//		UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), mpTargetFrame->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
	//		pNewMP->SetPlaneID(pPlane->mnPlaneID);
	//		pNewMP->SetObjectType(pPlane->mnPlaneType);
	//		pNewMP->AddFrame(mpTargetFrame, j);
	//		pNewMP->UpdateNormalAndDepth();
	//		pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
	//		mpSystem->mlpNewMPs.push_back(pNewMP);
	//		//pPlane->mvpMPs.push_back(pNewMP);
	//		pPlane->tmpMPs.push_back(pNewMP);
	//		//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
	//	}
	//}

	///*cv::Mat R, t;
	//mpTargetFrame->GetPose(R, t);

	//for (int j = 0; j < mvpMPs.size(); j++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//	auto oType = mvpOPs[j];
	//	if (!pMP)
	//		continue;
	//	if (pMP->isDeleted())
	//		continue;
	//	if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR) {
	//		cv::Mat X3D = pMP->GetWorldPos();
	//		cv::Mat Xcam = R*X3D + t;
	//		float depth = Xcam.at<float>(2);
	//		if (depth < 0.0 || depth > maxDepth)
	//			pMP->SetDelete(true);
	//	}
	//}

	//mpTargetFrame->SetDepthRange(minDepth, maxDepth);*/
}

void UVR_SLAM::PlaneEstimator::CreateWallMapPoints(std::vector<UVR_SLAM::MapPoint*> mvpMPs, std::vector<UVR_SLAM::ObjectType> mvpOPs, UVR_SLAM::PlaneInformation* pPlane, cv::Mat invT, int wtype,int MinX, int MaxX, bool b1, bool b2) {

	//int nTargetID = mpTargetFrame->GetFrameID();

	//cv::Mat invP1 = invT.t()*pPlane->GetParam();
	//	
	////create new mp in current frame
	//for (int j = 0; j < mvpMPs.size(); j++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//	auto oType = mvpOPs[j];

	//	if (oType != UVR_SLAM::ObjectType::OBJECT_WALL)
	//		continue;
	//	cv::Point2f pt = mpTargetFrame->mvKeyPoints[j].pt;
	//	int x = pt.x;
	//	if (wtype == 1) {
	//		if (x > MinX)
	//			continue;
	//	}
	//	else if (wtype == 2) {
	//		if (x < MinX || x > MaxX)
	//			continue;
	//	}
	//	else {
	//		if (x < MaxX)
	//			continue;
	//	}
	//	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	//	temp = mK.inv()*temp;
	//	cv::Mat matDepth = -invP1.at<float>(3) / (invP1.rowRange(0, 3).t()*temp);
	//	float depth = matDepth.at<float>(0);
	//	if (depth < 0.0){
	//		//depth *= -1.0;
	//		continue;
	//	}
	//	temp *= depth;
	//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	//	cv::Mat estimated = invT*temp;
	//	if (pMP) {
	//		pMP->SetWorldPos(estimated.rowRange(0, 3));
	//	}
	//	else {
	//		UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(mpTargetFrame, estimated.rowRange(0, 3), mpTargetFrame->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
	//		pNewMP->SetPlaneID(pPlane->mnPlaneID);
	//		pNewMP->SetObjectType(pPlane->mnPlaneType);
	//		pNewMP->AddFrame(mpTargetFrame, j);
	//		pNewMP->UpdateNormalAndDepth();
	//		pNewMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
	//		mpSystem->mlpNewMPs.push_back(pNewMP);
	//		pPlane->mvpMPs.push_back(pNewMP);
	//		//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
	//	}
	//}

}

void CheckWallNormal(cv::Mat& normal) {
	float a = (normal.at<float>(0));
	float c = (normal.at<float>(2));
	int idx;
	if (abs(a) > abs(c)) {
		if (a < 0)
			normal *= -1.0;
	}
	else {
		if (c < 0)
			normal *= -1.0;
	}
}

cv::Mat UVR_SLAM::PlaneInformation::PlaneWallEstimator(cv::Mat s, cv::Mat e, cv::Mat normal1, cv::Mat invP, cv::Mat invT, cv::Mat invK) {
	cv::Mat normal2 = s - e;
	normal2 = normal2.rowRange(0, 3);
	float norm2 = sqrt(normal2.dot(normal2));
	normal2 /= norm2;
	auto normal3 = normal1.cross(normal2);
	float norm3 = sqrt(normal3.dot(normal3));
	normal3 /= norm3;

	cv::Mat matDist = normal3.t()*s.rowRange(0, 3);

	normal3.push_back(-matDist);
	CheckWallNormal(normal3);
	return normal3;
}

cv::Mat UVR_SLAM::PlaneInformation::PlaneWallEstimator(UVR_SLAM::Line* line, cv::Mat normal1, cv::Mat invP, cv::Mat invT, cv::Mat invK) {

	Point2f from = line->from;
	Point2f to = line->to;

	cv::Mat s = CreatePlanarMapPoint(from, invP, invT, invK);
	cv::Mat e = CreatePlanarMapPoint(to, invP, invT, invK);

	cv::Mat normal2 = s - e;
	normal2 = normal2.rowRange(0, 3);
	float norm2 = sqrt(normal2.dot(normal2));
	normal2 /= norm2;
	auto normal3 = normal1.cross(normal2);
	float norm3 = sqrt(normal3.dot(normal3));
	normal3 /= norm3;

	cv::Mat matDist = normal3.t()*s.rowRange(0, 3);

	normal3.push_back(-matDist);
	CheckWallNormal(normal3);
	return normal3;
}

cv::Mat UVR_SLAM::PlaneInformation::PlaneWallEstimator(cv::Vec4i line, cv::Mat normal1, cv::Mat invP, cv::Mat invT, cv::Mat invK) {
	
	Point2f from(line[0], line[1]);
	Point2f to(line[2], line[3]);

	cv::Mat s = CreatePlanarMapPoint(from, invP, invT, invK);
	cv::Mat e = CreatePlanarMapPoint(to, invP, invT, invK);

	cv::Mat normal2 = s - e;
	normal2 = normal2.rowRange(0, 3);
	float norm2 = sqrt(normal2.dot(normal2));
	normal2 /= norm2;
	auto normal3 = normal1.cross(normal2);
	float norm3 = sqrt(normal3.dot(normal3));
	normal3 /= norm3;

	cv::Mat matDist = normal3.t()*s.rowRange(0, 3);

	normal3.push_back(-matDist);
	CheckWallNormal(normal3);
	return normal3;  
}
//3������? 4��������?����
cv::Mat UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(cv::Point2f pt, cv::Mat invP, cv::Mat invT, cv::Mat invK){
	cv::Mat temp1 = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	temp1 = invK*temp1;
	cv::Mat matDepth1 = -invP.at<float>(3) / (invP.rowRange(0, 3).t()*temp1);
	float depth = matDepth1.at<float>(0);
	if (depth < 0)
		depth *= -1.0;
	temp1 *= depth;
	temp1.push_back(cv::Mat::ones(1, 1, CV_32FC1));
	cv::Mat estimated = invT*temp1;
	return estimated.rowRange(0,3);
}

void UVR_SLAM::PlaneInformation::CreatePlanarMapPoints(Frame* pF, System* pSystem) {
	
	//int nTargetID = pF->GetFrameID();
	//cv::Mat invT, invK, invP;
	//pF->mpPlaneInformation->Calculate();
	//pF->mpPlaneInformation->GetInformation(invP, invT, invK);
	//auto pPlane = pF->mpPlaneInformation->GetFloorPlane();
	//
	//auto mvpMPs = pF->GetMapPoints();
	//auto mvpOPs = pF->GetObjectVector();
	//
	//float minDepth = FLT_MAX;
	//float maxDepth = 0.0f;
	////create new mp in current frame
	//for (int j = 0; j < mvpMPs.size(); j++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//	auto oType = mvpOPs[j];

	//	if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR)
	//		continue;
	//	cv::Point2f pt = pF->mvKeyPoints[j].pt;
	//	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	//	temp = invK*temp;
	//	cv::Mat matDepth = -invP.at<float>(3) / (invP.rowRange(0, 3).t()*temp);
	//	float depth = matDepth.at<float>(0);
	//	if (depth < 0) {
	//		//depth *= -1.0;
	//		continue;
	//	}
	//	temp *= depth;
	//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	//	/*if (maxDepth < depth)
	//	maxDepth = depth;
	//	if (minDepth > depth)
	//	minDepth = depth;*/

	//	cv::Mat estimated = invT*temp;
	//	if (pMP) {
	//		pMP->SetWorldPos(estimated.rowRange(0, 3));
	//	}
	//	else {
	//		UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(pF, estimated.rowRange(0, 3), pF->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
	//		pNewMP->SetPlaneID(pPlane->mnPlaneID);
	//		pNewMP->SetObjectType(pPlane->mnPlaneType);
	//		pNewMP->AddFrame(pF, j);
	//		pNewMP->UpdateNormalAndDepth();
	//		pNewMP->mnFirstKeyFrameID = pF->GetKeyFrameID();
	//		pSystem->mlpNewMPs.push_back(pNewMP);
	//		//pPlane->mvpMPs.push_back(pNewMP);
	//		pPlane->tmpMPs.push_back(pNewMP);
	//		//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
	//	}
	//}
	//
	///*cv::Mat R, t;
	//mpTargetFrame->GetPose(R, t);

	//for (int j = 0; j < mvpMPs.size(); j++) {
	//UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//auto oType = mvpOPs[j];
	//if (!pMP)
	//continue;
	//if (pMP->isDeleted())
	//continue;
	//if (oType != UVR_SLAM::ObjectType::OBJECT_FLOOR) {
	//cv::Mat X3D = pMP->GetWorldPos();
	//cv::Mat Xcam = R*X3D + t;
	//float depth = Xcam.at<float>(2);
	//if (depth < 0.0 || depth > maxDepth)
	//pMP->SetDelete(true);
	//}
	//}

	//mpTargetFrame->SetDepthRange(minDepth, maxDepth);*/
}
void UVR_SLAM::PlaneInformation::CreateDensePlanarMapPoint(std::vector<cv::Mat>& vX3Ds, cv::Mat label_map, Frame* pF, int nPatchSize) {
	//dense_map = cv::Mat::zeros(pF->mnMaxX, pF->mnMaxY, CV_32FC3);
	//cv::resize(label_map, label_map, pF->GetOriginalImage().size());
	int nTargetID = pF->GetFrameID();
	cv::Mat invT, invPfloor, invK;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);

	auto pFloor = pF->mpPlaneInformation->GetPlane(1);
	cv::Mat matFloorParam = pFloor->GetParam();
	cv::Mat normal;
	float dist;
	pFloor->GetParam(normal, dist);

	int inc = nPatchSize / 2;
	int mX = pF->mnMaxX - inc;
	int mY = pF->mnMaxY - inc;
	for (int x = inc; x < mX; x += nPatchSize) {
		for (int y = inc; y < mY; y += nPatchSize) {
			cv::Point2f pt(x, y);
			int label = label_map.at<uchar>(y, x);
			if (label == 150) {
				cv::Mat X3D;
				bool bRes = PlaneInformation::CreatePlanarMapPoint(pt, invPfloor, invT, invK, X3D);
				if (bRes)
				{
					vX3Ds.push_back(X3D);
					//dense_map.at<Vec3f>(y, x) = cv::Vec3f(X3D.at<float>(0), X3D.at<float>(1), X3D.at<float>(2));
				}
			}
		}
	}
}
void UVR_SLAM::PlaneInformation::CreateDenseWallPlanarMapPoint(std::vector<cv::Mat>& vX3Ds, cv::Mat label_map, Frame* pF, WallPlane* pWall, Line* pLine, int nPatchSize) {

	//dense_map = cv::Mat::zeros(pF->mnMaxX, pF->mnMaxY, CV_32FC3);
	//cv::resize(label_map, label_map, pF->GetOriginalImage().size());
	int nTargetID = pF->GetFrameID();
	cv::Mat invT, invPfloor, invK;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);
	cv::Mat invPwawll = invT.t()*pWall->GetParam();

	auto pFloor = pF->mpPlaneInformation->GetPlane(1);
	cv::Mat matFloorParam = pFloor->GetParam();
	cv::Mat normal;
	float dist;
	pFloor->GetParam(normal, dist);

	int inc = nPatchSize / 2;
	int mX = pF->mnMaxX - inc;
	int mY = pF->mnMaxY - inc;

	int sX = (pLine->to.x / nPatchSize)*nPatchSize+nPatchSize;
	int eX = (pLine->from.x / nPatchSize)*nPatchSize;

	for (int x = sX; x <= eX; x += nPatchSize) {
		for (int y = inc; y < mY; y += nPatchSize) {
			cv::Point2f pt(x, y);
			int label = label_map.at<uchar>(y, x);
			if (label == 255) {
				/*if (x < pLine->to.x || x > pLine->from.x)
					continue;*/
				cv::Mat temp = (cv::Mat_<float>(3, 1) << x, y, 1);
				temp = invK*temp;
				cv::Mat matDepth = -invPwawll.at<float>(3) / (invPwawll.rowRange(0, 3).t()*temp);
				float depth = matDepth.at<float>(0);
				if (depth < 0.0) {
					//depth *= -1.0;
					continue;
				}
				temp *= depth;
				temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

				cv::Mat estimated = invT*temp;
				estimated = estimated.rowRange(0, 3);
				//check on floor
				float val = estimated.dot(normal) + dist;

				if (val < 0.0)
					continue;
				vX3Ds.push_back(estimated);
				//dense_map.at<Vec3f>(y, x) = cv::Vec3f(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
			}
		}
	}
}
void UVR_SLAM::PlaneInformation::CreateDensePlanarMapPoint(cv::Mat& dense_map, cv::Mat label_map, Frame* pF, WallPlane* pWall, Line* pLine, int nPatchSize) {
	
	//dense_map = cv::Mat::zeros(pF->mnMaxX, pF->mnMaxY, CV_32FC3);
	//cv::resize(label_map, label_map, pF->GetOriginalImage().size());
	int nTargetID = pF->GetFrameID();
	cv::Mat invT, invPfloor, invK;
	pF->mpPlaneInformation->Calculate();
	pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);
	cv::Mat invPwawll = invT.t()*pWall->GetParam();

	auto pFloor = pF->mpPlaneInformation->GetPlane(1);
	cv::Mat matFloorParam = pFloor->GetParam();
	cv::Mat normal;
	float dist;
	pFloor->GetParam(normal, dist);

	int inc = nPatchSize / 2;
	int mX = pF->mnMaxX - inc;
	int mY = pF->mnMaxY - inc;
	for (int x = inc; x < mX; x += nPatchSize) {
		for (int y = inc; y < mY; y+= nPatchSize) {
			cv::Point2f pt(x, y);
			int label = label_map.at<uchar>(y,x);
			if (label == 255) {
				if (x < pLine->to.x || x > pLine->from.x)
					continue;
				cv::Mat temp = (cv::Mat_<float>(3, 1) << x, y, 1);
				temp = invK*temp;
				cv::Mat matDepth = -invPwawll.at<float>(3) / (invPwawll.rowRange(0, 3).t()*temp);
				float depth = matDepth.at<float>(0);
				if (depth < 0.0) {
					//depth *= -1.0;
					continue;
				}
				temp *= depth;
				temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

				cv::Mat estimated = invT*temp;
				estimated = estimated.rowRange(0, 3);
				//check on floor
				float val = estimated.dot(normal) + dist;

				if (val < 0.0)
					continue;

				dense_map.at<Vec3f>(y, x) = cv::Vec3f(estimated.at<float>(0), estimated.at<float>(1), estimated.at<float>(2));
			}
		}
	}
}

void UVR_SLAM::PlaneInformation::CreateWallMapPoints(Frame* pF, WallPlane* pWall, Line* pLine, std::vector<cv::Mat>& vPlanarMaps, System* pSystem) {

	//int nTargetID = pF->GetFrameID();
	//cv::Mat invT, invPfloor, invK;
	//pF->mpPlaneInformation->Calculate();
	//pF->mpPlaneInformation->GetInformation(invPfloor, invT, invK);
	//cv::Mat invPwawll = invT.t()*pWall->GetParam();

	//auto pFloor = pF->mpPlaneInformation->GetFloorPlane();
	//cv::Mat matFloorParam = pFloor->GetParam();
	//cv::Mat normal;
	//float dist;
	//pFloor->GetParam(normal, dist);

	//auto mvpMPs = pF->GetMapPoints();
	//auto mvpOPs = pF->GetObjectVector();
	//int count = 0;
	////create new mp in current frame
	//for (int j = 0; j < mvpMPs.size(); j++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[j];
	//	if (pMP)
	//		continue;
	//	auto oType = mvpOPs[j];
	//	if (oType != UVR_SLAM::ObjectType::OBJECT_WALL)
	//		continue;
	//	cv::Point2f pt = pF->mvKeyPoints[j].pt;
	//	int x = pt.x;
	//	if (x < pLine->to.x || x > pLine->from.x)
	//		continue;
	//	
	//	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	//	temp = invK*temp;
	//	cv::Mat matDepth = -invPwawll.at<float>(3) / (invPwawll.rowRange(0, 3).t()*temp);
	//	float depth = matDepth.at<float>(0);
	//	if (depth < 0.0) {
	//		//depth *= -1.0;
	//		continue;
	//	}
	//	temp *= depth;
	//	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	//	cv::Mat estimated = invT*temp;
	//	estimated = estimated.rowRange(0, 3);
	//	//check on floor
	//	float val = estimated.dot(normal) + dist;
	//	
	//	if (val < 0.0)
	//		continue;
	//	count++;
	//	std::cout <<"planar::"<< val << estimated.t()<< std::endl;
	//	vPlanarMaps[j] = estimated.clone();

	//	//if (pMP) {
	//	//	pMP->SetWorldPos(estimated.rowRange(0, 3));
	//	//}
	//	//else {
	//	//	UVR_SLAM::MapPoint* pNewMP = new UVR_SLAM::MapPoint(pF, estimated.rowRange(0, 3), pF->matDescriptor.row(j), UVR_SLAM::PLANE_MP);
	//	//	pNewMP->SetPlaneID(0);
	//	//	pNewMP->SetObjectType(UVR_SLAM::ObjectType::OBJECT_WALL);
	//	//	pNewMP->AddFrame(pF, j);
	//	//	pNewMP->UpdateNormalAndDepth();
	//	//	pNewMP->mnFirstKeyFrameID = pF->GetKeyFrameID();
	//	//	pSystem->mlpNewMPs.push_back(pNewMP);
	//	//	//pPlane->mvpMPs.push_back(pNewMP);
	//	//	//mpFrameWindow->AddMapPoint(pNewMP, nTargetID);
	//	//}
	//}
	//std::cout << "wall::cre::" << count << std::endl;

}

//vector size = keypoint size
void UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(Frame* pTargetF, PlaneInformation* pFloor, std::vector<cv::Mat>& vPlanarMaps) {
	
	//cv::Mat invP, invT, invK;
	//pTargetF->mpPlaneInformation->Calculate();
	//pTargetF->mpPlaneInformation->GetInformation(invP, invT, invK);

	//auto mvpMPs = pTargetF->GetMapPoints();
	//auto mvpOPs = pTargetF->GetObjectVector();
	//
	////��Ī Ȯ���ϱ�
	//for (int i = 0; i < pTargetF->mvKeyPoints.size(); i++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[i];
	//	if (mvpOPs[i] != ObjectType::OBJECT_FLOOR)
	//		continue;
	//	/*if (pMP && pMP->isDeleted()) {
	//		vPlanarMaps[i] = pMP->GetWorldPos();
	//		continue;
	//	}*/
	//	cv::Mat X3D;
	//	bool bRes = PlaneInformation::CreatePlanarMapPoint(pTargetF->mvKeyPoints[i].pt, invP, invT, invK, X3D);
	//	if (bRes)
	//		vPlanarMaps[i] = X3D.clone();
	//}
}
//////////�ڵ� ���
////////////////////////////////////////////////////////////////////////
