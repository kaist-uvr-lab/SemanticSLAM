#include <FrameWindow.h>
#include <System.h>
#include <MapPoint.h>

UVR_SLAM::FrameWindow::FrameWindow():mnWindowSize(10), LocalMapSize(0), mnFrameCount(0), mnLastSemanticFrame(-1),mnQueueSize(0){}
UVR_SLAM::FrameWindow::FrameWindow(int _size) : mnWindowSize(_size), LocalMapSize(0), mnFrameCount(0), mnLastSemanticFrame(-1), mnQueueSize(0) {}
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

void UVR_SLAM::FrameWindow::SetMapPoint(UVR_SLAM::MapPoint* pMP, int idx) {
	std::unique_lock<std::mutex>(mMutexMP);
	mvpLocalMPs[idx] = pMP;
}

UVR_SLAM::MapPoint* UVR_SLAM::FrameWindow::GetMapPoint(int idx) {
	std::unique_lock<std::mutex>(mMutexMP);
	return mvpLocalMPs[idx];
}

void UVR_SLAM::FrameWindow::SetPose(cv::Mat _R, cv::Mat _t){
	std::unique_lock<std::mutex>(mMutexPose);
	R = _R.clone();
	t = _t.clone();
}
void UVR_SLAM::FrameWindow::GetPose(cv::Mat &_R, cv::Mat& _t) {
	std::unique_lock<std::mutex>(mMutexPose);
	_R = R.clone();
	_t = t.clone();
}
cv::Mat UVR_SLAM::FrameWindow::GetRotation(){
	std::unique_lock<std::mutex>(mMutexPose);
	return R;
}
cv::Mat UVR_SLAM::FrameWindow::GetTranslation() {
	std::unique_lock<std::mutex>(mMutexPose);
	return t;
}

void UVR_SLAM::FrameWindow::SetBoolInlier(bool b, int idx) {
	std::unique_lock<std::mutex> lockMP(mMutexBoolInliers);
	mvbLocalMPInliers[idx] = b;
}
bool UVR_SLAM::FrameWindow::GetBoolInlier(int idx){
	std::unique_lock<std::mutex> lockMP(mMutexBoolInliers);
	return mvbLocalMPInliers[idx];
}
void UVR_SLAM::FrameWindow::SetVectorInlier(int size, bool b){
	std::unique_lock<std::mutex> lockMP(mMutexBoolInliers);
	mvbLocalMPInliers = std::vector<bool>(size, b);
}


void UVR_SLAM::FrameWindow::IncrementFrameCount() {
	std::unique_lock<std::mutex> lockMP(mMutexFrameCount);
	mnFrameCount++;
}
void UVR_SLAM::FrameWindow::SetFrameCount(int nCount) {
	std::unique_lock<std::mutex> lockMP(mMutexFrameCount);
	mnFrameCount = nCount;
}
int UVR_SLAM::FrameWindow::GetFrameCount() {
	std::unique_lock<std::mutex> lockMP(mMutexFrameCount);
	return mnFrameCount;
}

void UVR_SLAM::FrameWindow::SetLastSemanticFrameIndex() {
	std::unique_lock<std::mutex> lockMP(mMutexDeque);
	mnLastSemanticFrame = mpDeque.size()-1;
}
int UVR_SLAM::FrameWindow::GetLastSemanticFrameIndex() {
	std::unique_lock<std::mutex> lockMP(mMutexDeque);
	return mnLastSemanticFrame;
}

void UVR_SLAM::FrameWindow::SetLocalMap() {
	descLocalMap = cv::Mat::zeros(0, mpDeque.front()->matDescriptor.cols, mpDeque.front()->matDescriptor.type());
	mvpLocalMPs.clear();
	mspLocalMPs.clear();
	for (auto iter = mpDeque.begin(); iter != mpDeque.end(); iter++) {
		UVR_SLAM::Frame* pF = *iter;
		for (int i = 0; i < pF->mvKeyPoints.size(); i++) {
			UVR_SLAM::MapPoint *pMP = pF->GetMapPoint(i);
			if (!pMP)
				continue;
			auto findres = mspLocalMPs.find(pMP);
			if (findres == mspLocalMPs.end()) {
				pMP->SetFrameWindowIndex(mvpLocalMPs.size());
				mspLocalMPs.insert(pMP);
				mvpLocalMPs.push_back(pMP);
				descLocalMap.push_back(pMP->GetDescriptor());
			}
		}
	}
	LocalMapSize = mvpLocalMPs.size();
}

bool UVR_SLAM::FrameWindow::CalcFrameDistanceWithBOW(UVR_SLAM::Frame* pF) {
	
	//UVR_SLAM::Frame* pF1 = mpDeque.front();
	//UVR_SLAM::Frame* pF2 = mpDeque.back();
	////std::cout << "Frame Type = " << (int)pF1->GetType()<<", "<<(int)pF2->GetType() <<", "<<pF2->CheckType(UVR_SLAM::FLAG_SEGMENTED_FRAME)<<", "<<pF1->CheckType(UVR_SLAM::FLAG_SEGMENTED_FRAME)<< std::endl;
	//std::cout << "DBOW::SCORE1::" << pF1->Score(pF2)<<std::endl;

	//윈도우 내의 프레임 중 하나라도 값이 0.01보다 크면 false를 return함.
	bool res = true;
	for (auto iter = mpDeque.begin(); iter != mpDeque.end(); iter++) {
		UVR_SLAM::Frame* pDequeFrame = *iter;
		double score = pDequeFrame->Score(pF);
		
		if (score > 0.01)
			res = false;
		//std::cout << "DBOW::SCORE::" <<score<< std::endl;
	}
	//std::cout << "BOW check::" << res << std::endl;
	return res;
}