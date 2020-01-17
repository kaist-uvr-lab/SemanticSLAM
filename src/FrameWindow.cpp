#include <FrameWindow.h>
#include <System.h>
#include <MapPoint.h>

UVR_SLAM::FrameWindow::FrameWindow():mnWindowSize(10), LocalMapSize(0), mnLastSemanticFrame(-1), mnLastLayoutFrame(-1),mnQueueSize(0),mdFuseTime(0.0), mdPETime(0.0), mbUseLocalMap(false) {}
UVR_SLAM::FrameWindow::FrameWindow(int _size) : mnWindowSize(_size), LocalMapSize(0), mnLastSemanticFrame(-1), mnLastLayoutFrame(-1), mnQueueSize(0), mdFuseTime(0.0), mdPETime(0.0), mbUseLocalMap(false) {}
UVR_SLAM::FrameWindow::~FrameWindow() {}


void UVR_SLAM::FrameWindow::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}

//��ť ����� ũ�ų� 0�� ���� ���� ó���� �ʿ���
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
		//�����쿡�� ��� ť�� �ٸ� ���� �ִ� ����
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
//������ ī��Ʈ�� Ű�����Ӱ� Ű������ ������ ������ ������ ���� �ǹ���.
//�̰� �� ���� �־�� �ϳ�?
//Ʈ��Ŀ�� �ű�� ���� ���?
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
void UVR_SLAM::FrameWindow::SetFuseTime(double d){
	std::unique_lock<std::mutex> lockMP(mMutexFuseTime);
	mdFuseTime = d;
}
double UVR_SLAM::FrameWindow::GetFuseTime(){
	std::unique_lock<std::mutex> lockMP(mMutexFuseTime);
	return mdFuseTime;
}
void UVR_SLAM::FrameWindow::SetPETime(double d){
	std::unique_lock<std::mutex> lockMP(mMutexPETime);
	mdPETime = d;
}
double UVR_SLAM::FrameWindow::GetPETime(){
	std::unique_lock<std::mutex> lockMP(mMutexPETime);
	return mdPETime;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////
//������ ������ ���� ���� �������� �߰� �� �� ����ǹǷ� ���ؽ� ��ť�� �����Ǵ°� ����.

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
	//SetLastFrameID(nTargetID);

	//20.01.02 deque���� list�� ������.
	for (auto iter = mlpFrames.begin(); iter != mlpFrames.end(); iter++) {
		UVR_SLAM::Frame* pF = *iter;
		for (int i = 0; i < pF->mvKeyPoints.size(); i++) {
			UVR_SLAM::MapPoint *pMP = pF->mvpMPs[i];
			if (!pMP)
				continue;
			if (pMP->isDeleted())
				continue;
			if (pMP->GetRecentLocalMapID() == nTargetID)
				continue;
			mvpLocalMPs.push_back(pMP);
			descLocalMap.push_back(pMP->GetDescriptor());
			pMP->SetRecentLocalMapID(nTargetID);
			//AddMapPoint(pMP);
			/*auto findres = mspLocalMPs.find(pMP);
			if (findres == mspLocalMPs.end()) {
				pMP->SetFrameWindowIndex(mvpLocalMPs.size());
				mspLocalMPs.insert(pMP);
				mvpLocalMPs.push_back(pMP);
				descLocalMap.push_back(pMP->GetDescriptor());
			}*/ 
		}
	}
	LocalMapSize = mvpLocalMPs.size();
	mpSystem->mbLocalMapUpdateEnd = true;
	mpSystem->cvUseLocalMap.notify_one();
	
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool UVR_SLAM::FrameWindow::CalcFrameDistanceWithBOW(UVR_SLAM::Frame* pF) {
	
	//UVR_SLAM::Frame* pF1 = mpDeque.front();
	//UVR_SLAM::Frame* pF2 = mpDeque.back();
	////std::cout << "Frame Type = " << (int)pF1->GetType()<<", "<<(int)pF2->GetType() <<", "<<pF2->CheckFrameType(UVR_SLAM::FLAG_SEGMENTED_FRAME)<<", "<<pF1->CheckFrameType(UVR_SLAM::FLAG_SEGMENTED_FRAME)<< std::endl;
	//std::cout << "DBOW::SCORE1::" << pF1->Score(pF2)<<std::endl;

	//������ ���� ������ �� �ϳ��� ���� 0.01���� ũ�� false�� return��.
	std::unique_lock<std::mutex>(mMutexDeque);
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