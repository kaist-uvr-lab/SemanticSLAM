#include <FrameWindow.h>
#include <System.h>
#include <MapPoint.h>

UVR_SLAM::FrameWindow::FrameWindow():mnWindowSize(10), mnLastSemanticFrame(-1), mnLastLayoutFrame(-1),mnQueueSize(0),mbUseLocalMap(false) {}
UVR_SLAM::FrameWindow::FrameWindow(int _size) : mnWindowSize(_size), mnLastSemanticFrame(-1), mnLastLayoutFrame(-1), mnQueueSize(0), mbUseLocalMap(false) {}
UVR_SLAM::FrameWindow::~FrameWindow() {}


void UVR_SLAM::FrameWindow::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}

//디큐 사이즈가 크거나 0일 때에 대한 처리가 필요함
size_t UVR_SLAM::FrameWindow::size() {
	std::unique_lock<std::mutex>(mMutexDeque);
	return mpDeque.size();
}
std::deque<UVR_SLAM::Frame*>::iterator UVR_SLAM::FrameWindow::GetBeginIterator() {
	std::unique_lock<std::mutex>(mMutexDeque);
	return mpDeque.begin();
}
std::deque<UVR_SLAM::Frame*>::iterator UVR_SLAM::FrameWindow::GetEndIterator() {
	std::unique_lock<std::mutex>(mMutexDeque);
	return mpDeque.end();
}
////////////////////////////////////////////////////////////////////////
int UVR_SLAM::FrameWindow::GetQueueSize(){
	std::unique_lock<std::mutex>(mMutexDeque);
	return mnQueueSize;
}
UVR_SLAM::Frame* UVR_SLAM::FrameWindow::GetQueueLastFrame(){
	std::unique_lock<std::mutex>(mMutexDeque);
	return mpQueue.back();
}
////////////////////////////////////////////////////////////////////////
std::vector<UVR_SLAM::Frame*> UVR_SLAM::FrameWindow::GetAllFrames() {
	std::unique_lock<std::mutex>(mMutexDeque);
	return std::vector<UVR_SLAM::Frame*>(mpDeque.begin(), mpDeque.end());
}
std::set<UVR_SLAM::Frame*> UVR_SLAM::FrameWindow::GetAllFrameSets() {
	std::unique_lock<std::mutex>(mMutexDeque);
	return std::set<UVR_SLAM::Frame*>(mpDeque.begin(), mpDeque.end());
}
void UVR_SLAM::FrameWindow::clear() {
	std::unique_lock<std::mutex>(mMutexDeque);
	mpDeque.clear();
}

bool UVR_SLAM::FrameWindow::isEmpty() {
	std::unique_lock<std::mutex>(mMutexDeque);
	return mpDeque.empty();
}
void UVR_SLAM::FrameWindow::push_front(UVR_SLAM::Frame* pFrame) {
	std::unique_lock<std::mutex>(mMutexDeque);
	mpDeque.push_front(pFrame);
}
void UVR_SLAM::FrameWindow::push_back(UVR_SLAM::Frame* pFrame) {

	std::unique_lock<std::mutex>(mMutexDeque);
	if (mpDeque.size() == mnWindowSize){
		UVR_SLAM::Frame* pLast = mpDeque.back();
		mpDeque.pop_front();
		mnLastSemanticFrame--;
		mnLastLayoutFrame--;
		//윈도우에서 벗어난 큐를 다른 곳에 넣는 과정
		mpQueue.push(pLast);
		mnQueueSize++;
	}
	mpDeque.push_back(pFrame);
	
}
void UVR_SLAM::FrameWindow::pop_front() {
	std::unique_lock<std::mutex>(mMutexDeque);
	mpDeque.pop_front();
}
void UVR_SLAM::FrameWindow::pop_back() {
	std::unique_lock<std::mutex>(mMutexDeque);
	mpDeque.pop_back();
}
UVR_SLAM::Frame* UVR_SLAM::FrameWindow::front() {
	std::unique_lock<std::mutex>(mMutexDeque);
	return mpDeque.front();
}
UVR_SLAM::Frame* UVR_SLAM::FrameWindow::back() {
	std::unique_lock<std::mutex>(mMutexDeque);
	return mpDeque.back();
}
UVR_SLAM::Frame* UVR_SLAM::FrameWindow::GetFrame(int idx) {
	std::unique_lock<std::mutex>(mMutexDeque);
	return mpDeque[idx];
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void UVR_SLAM::FrameWindow::SetPose(cv::Mat _R, cv::Mat _t) {
	std::unique_lock<std::mutex>(mMutexPose);
	R = _R.clone();
	t = _t.clone();
}
void UVR_SLAM::FrameWindow::GetPose(cv::Mat &_R, cv::Mat& _t) {
	std::unique_lock<std::mutex>(mMutexPose);
	_R = R.clone();
	_t = t.clone();
}
cv::Mat UVR_SLAM::FrameWindow::GetRotation() {
	std::unique_lock<std::mutex>(mMutexPose);
	return R;
}
cv::Mat UVR_SLAM::FrameWindow::GetTranslation() {
	std::unique_lock<std::mutex>(mMutexPose);
	return t;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
void UVR_SLAM::FrameWindow::SetMapPoint(UVR_SLAM::MapPoint* pMP, int idx) {
	std::unique_lock<std::mutex>(mMutexLocalMPs);
	mvpLocalMPs[idx] = pMP;
}

std::vector<UVR_SLAM::MapPoint*> UVR_SLAM::FrameWindow::GetLocalMap() {
	std::unique_lock<std::mutex> lockMP(mMutexLocalMPs);
	return std::vector<UVR_SLAM::MapPoint*>(mvpLocalMPs.begin(), mvpLocalMPs.end());
}
cv::Mat UVR_SLAM::FrameWindow::GetLocalMapDescriptor() {
	std::unique_lock<std::mutex> lockMP(mMutexLocalMPs);
	return descLocalMap.clone();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//프레임 카운트는 키프레임과 키프레임 사이의 누적된 프레임 수를 의미함.
//이게 꼭 여기 있어야 하나?
//트래커에 옮기는 것은 어떨까?
void UVR_SLAM::FrameWindow::SetLastFrameID(int id){
	std::unique_lock<std::mutex> lockMP(mMutexLastFrameID);
	mnLastFrameID = id;
}
int  UVR_SLAM::FrameWindow::GetLastFrameID(){
	std::unique_lock<std::mutex> lockMP(mMutexLastFrameID);
	return mnLastFrameID;
}
void UVR_SLAM::FrameWindow::SetLastLayoutFrameID(int id) {
	std::unique_lock<std::mutex> lockMP(mMutexLastLayoutFrameID);
	mnLastLayoutFrameID = id;
}
int  UVR_SLAM::FrameWindow::GetLastLayoutFrameID() {
	std::unique_lock<std::mutex> lockMP(mMutexLastLayoutFrameID);
	return mnLastLayoutFrameID;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//마지막 프레임 같은 경우는 프레임이 추가 될 때 수행되므로 뮤텍스 디큐와 연관되는게 맞음.

void UVR_SLAM::FrameWindow::SetLastSemanticFrameIndex() {
	std::unique_lock<std::mutex> lockMP(mMutexDeque);
	mnLastSemanticFrame = mpDeque.size()-1;
}
int UVR_SLAM::FrameWindow::GetLastSemanticFrameIndex() {
	std::unique_lock<std::mutex> lockMP(mMutexDeque);
	return mnLastSemanticFrame;
}
//void UVR_SLAM::FrameWindow::SetLastLayoutFrameIndex() {
//	std::unique_lock<std::mutex> lockMP(mMutexDeque);
//	mnLastLayoutFrame = mpDeque.size() - 1;
//}
//int UVR_SLAM::FrameWindow::GetLastLayoutFrameIndex() {
//	std::unique_lock<std::mutex> lockMP(mMutexDeque);
//	return mnLastLayoutFrame;
//}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void UVR_SLAM::FrameWindow::SetLocalMap(int nTargetID) {

	std::unique_lock<std::mutex> lock(mpSystem->mMutexUseLocalMap);
	while (!mpSystem->mbTrackingEnd){
		mpSystem->cvUseLocalMap.wait(lock);
	}
	mpSystem->mbLocalMapUpdateEnd = false;

	std::unique_lock<std::mutex>(mMutexLocalMPs);
	
	UVR_SLAM::Frame* pLastF = mlpFrames.back();
	descLocalMap = cv::Mat::zeros(0, pLastF->matDescriptor.cols, pLastF->matDescriptor.type());
	mvpLocalMPs.clear();
	mspWallMPs.clear();
	mspFloorMPs.clear();
	mspCeilMPs.clear();
	//SetLastFrameID(nTargetID);

	//20.01.02 deque에서 list로 변경함.
	for (auto iter = mlpFrames.begin(); iter != mlpFrames.end(); iter++) {
		UVR_SLAM::Frame* pF = *iter;
		for (int i = 0; i < pF->mvKeyPoints.size(); i++) {
			UVR_SLAM::MapPoint *pMP = pF->mvpMPs[i];
			if (!pMP)
				continue;
			if (pMP->isDeleted())
				continue;
			
			////퓨즈에서 다른 프레임과 매칭을 하지 못하면 삭제
			/*if (pMP->mnFirstKeyFrameID + 3 > nTargetID && pMP->GetMapPointType() == MapPointType::PLANE_MP &&  pMP->GetNumConnectedFrames() == 1){
				pMP->SetDelete(true);
				pMP->Delete();
				continue;
			}*/

			if (pMP->GetRecentLocalMapID() == nTargetID)
				continue;

			switch (pMP->GetObjectType()) {
			case  ObjectType::OBJECT_WALL:
				mspWallMPs.insert(pMP);
				break;
			case ObjectType::OBJECT_FLOOR:
				mspFloorMPs.insert(pMP);
				break;
			case ObjectType::OBJECT_CEILING:
				mspCeilMPs.insert(pMP);
				break;
			default:
				break;
			}

			mvpLocalMPs.push_back(pMP);
			descLocalMap.push_back(pMP->GetDescriptor());
			pMP->SetRecentLocalMapID(nTargetID);
			
		}
	}
	mpSystem->mbLocalMapUpdateEnd = true;
	mpSystem->cvUseLocalMap.notify_one();
	
}
////Planar MapPoint도 즉각적으로 추가하기 위함.
void UVR_SLAM::FrameWindow::AddMapPoint(UVR_SLAM::MapPoint* pMP, int nTargetID) {
	std::unique_lock<std::mutex>(mMutexLocalMPs);
	mvpLocalMPs.push_back(pMP);
	descLocalMap.push_back(pMP->GetDescriptor());
	pMP->SetRecentLocalMapID(nTargetID);
}

void UVR_SLAM::FrameWindow::SetDummyPoints(std::vector<UVR_SLAM::MapPoint*> vpMPs){
	std::unique_lock<std::mutex>(mMutexDummyMPs);
	mvpDummyMPs = std::vector<UVR_SLAM::MapPoint*>(vpMPs.begin(), vpMPs.end());
}
std::vector<UVR_SLAM::MapPoint*> UVR_SLAM::FrameWindow::GetDummyPoints(){
	std::unique_lock<std::mutex>(mMutexDummyMPs);
	return std::vector<UVR_SLAM::MapPoint*>(mvpDummyMPs.begin(), mvpDummyMPs.end());
}
void UVR_SLAM::FrameWindow::ClearDummyMPs(){
	std::unique_lock<std::mutex>(mMutexDummyMPs);
	mvpDummyMPs.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool UVR_SLAM::FrameWindow::CalcFrameDistanceWithBOW(UVR_SLAM::Frame* pF) {
	/////200310
	/*std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

	bool res = true;
	auto mvpKFs = mpSystem->GetLoopFrames();
	auto mvNeighKFs = pF->GetConnectedKFs();

	std::set<int> msKeyFrameIDs;
	for (int i = 0; i < mvNeighKFs.size(); i++) {
		msKeyFrameIDs.insert(mvNeighKFs[i]->GetKeyFrameID());
	}

	for (int i = 0; i < mvpKFs.size(); i++) {
		UVR_SLAM::Frame* pKFi = mvpKFs[i];
		auto find_res = msKeyFrameIDs.find(pKFi->GetKeyFrameID());
		if (find_res != msKeyFrameIDs.end())
			continue;
		double score = pKFi->Score(pF);
		if (score > 0.01) {
			cv::Mat resImg = pKFi->GetOriginalImage();
			cv::resize(resImg, resImg, resImg.size() / 2);
			imshow("Output::LoopFrame", resImg);
			cv::waitKey(1);
		}
	}

	std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
	auto du_test = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
	double t_test = du_test / 1000.0;
	std::cout << "loop test :: " << t_test << std::endl;*/


	////UVR_SLAM::Frame* pF1 = mpDeque.front();
	////UVR_SLAM::Frame* pF2 = mpDeque.back();
	//////std::cout << "Frame Type = " << (int)pF1->GetType()<<", "<<(int)pF2->GetType() <<", "<<pF2->CheckFrameType(UVR_SLAM::FLAG_SEGMENTED_FRAME)<<", "<<pF1->CheckFrameType(UVR_SLAM::FLAG_SEGMENTED_FRAME)<< std::endl;
	////std::cout << "DBOW::SCORE1::" << pF1->Score(pF2)<<std::endl;

	////윈도우 내의 프레임 중 하나라도 값이 0.01보다 크면 false를 return함.
	//std::unique_lock<std::mutex>(mMutexDeque);
	
	//for (auto iter = mpDeque.begin(); iter != mpDeque.end(); iter++) {
	//	UVR_SLAM::Frame* pDequeFrame = *iter;
	//	double score = pDequeFrame->Score(pF);
	//	
	//	if (score > 0.01)
	//		res = false;
	//	//std::cout << "DBOW::SCORE::" <<score<< std::endl;
	//}
	////std::cout << "BOW check::" << res << std::endl;
	

	//return res;
	return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void UVR_SLAM::FrameWindow::AddFrame(Frame* pF){
	mlpFrames.push_back(pF);
}
void UVR_SLAM::FrameWindow::ClearLocalMapFrames(){
	mlpFrames.clear();
}
std::vector<UVR_SLAM::Frame*> UVR_SLAM::FrameWindow::GetLocalMapFrames() {
	return std::vector<UVR_SLAM::Frame*>(mlpFrames.begin(), mlpFrames.end());
}

void UVR_SLAM::FrameWindow::SetLocalMapInliers(std::vector<bool> vInliers){
	std::unique_lock<std::mutex>(mMutexLocalMapInliers);
	mvbLocalMaPInliers = std::vector<bool>(vInliers.begin(), vInliers.end());
}
std::vector<bool> UVR_SLAM::FrameWindow::GetLocalMapInliers(){
	std::unique_lock<std::mutex>(mMutexLocalMapInliers);
	return std::vector<bool>(mvbLocalMaPInliers.begin(), mvbLocalMaPInliers.end());
}

bool UVR_SLAM::FrameWindow::isUseLocalMap() {
	std::unique_lock<std::mutex>(mMutexUseLocalMap);
	return mbUseLocalMap;
}
void UVR_SLAM::FrameWindow::SetUseLocalMap(bool bUse) {
	std::unique_lock<std::mutex>(mMutexUseLocalMap);
	mbUseLocalMap = bUse;
}