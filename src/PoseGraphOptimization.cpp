#include <PoseGraphOptimization.h>

#include <opencv2/core/eigen.hpp>
//#include <Pose.h>

UVR_SLAM::FrameVertex::FrameVertex(cv::Mat _R, cv::Mat _t, int _size) :Vertex(0, _size, false) {
	//mpPose = new UVR_SLAM::Pose(_R, _t);
	cv::cv2eigen(_R, R);
	cv::cv2eigen(_t, t);
	SetParam();
	bLeft = true;
	//ResetHessian();
}

//UVR_SLAM::FrameVertex::FrameVertex(UVR_SLAM::Pose* pPose, int _size) :Vertex(0, _size, false) {
//	mpPose = pPose;
//	SetParam();
//	bLeft = true;
//	//ResetHessian();
//}

UVR_SLAM::FrameVertex::~FrameVertex() {}

void UVR_SLAM::FrameVertex::SetParam() {
	//4x4 param
	//param = cv::Mat::eye(4, 4, CV_64FC1);
	
	//R.copyTo(param(cv::Rect(0, 0, 3, 3)));
	//t.copyTo(param(cv::Rect(3, 0, 1, 3)));
}

void UVR_SLAM::FrameVertex::RestoreData() {
	//if (bFixed)
	//	return;
	//항상 포즈 결과는 업데이트가 됨
	//mpPose->Set(GetRotationMatrix(), GetTranslationMatrix());
	cv::eigen2cv(R, Rmat);
	cv::eigen2cv(t, Tmat);
	Rmat.convertTo(Rmat, CV_32FC1);
	Tmat.convertTo(Tmat, CV_32FC1);
}

void UVR_SLAM::FrameVertex::UpdateParam() {

	Eigen::Matrix4d DeltaParam = UVR_SLAM::MatrixOperator::EXP_Eigen(d);
	Eigen::Matrix3d DeltaR = DeltaParam.block(0, 0, 3, 3);
	Eigen::Vector3d DeltaP = DeltaParam.block(0, 3, 3, 1);

	/*if (bLeft)
		mpPose->LeftUpdate(DeltaR, DeltaP);
	else
		mpPose->RightUpdate(DeltaR, DeltaP);*/
	//param = DeltaP*param;
	if(!bFixed){
		R = DeltaR*R;
		t = DeltaR*t + DeltaP;
	}
	ResetHessian();
}
//연결된 포즈 정보를 업데이트 하기 위한 것
//포즈 이용 전에 항상 해줘야 함.

void UVR_SLAM::FrameVertex::Accept() {
	Hold = H;
	Bold = b;
	//subHold = subH.clone();
	//mpPose->Accept();
	Rold = R;
	Told = t;
}
void UVR_SLAM::FrameVertex::Reject() {
	b = Bold;
	H = Hold;
	//subH = subHold.clone();
	//mpPose->Reject();
	R = Rold;
	t = Told;
}
void* UVR_SLAM::FrameVertex::GetPointer() {
	return nullptr;
	//return mpFrame;
}

Eigen::Matrix3d UVR_SLAM::FrameVertex::GetRotationMatrix() {
	//return param(cv::Rect(0,0,3,3)).clone();
	return R;
}
Eigen::Vector3d UVR_SLAM::FrameVertex::GetTranslationMatrix() {
	//return param(cv::Rect(3,0,1,3)).clone();
	return t;
}

//UVR_SLAM::Pose* UVR_SLAM::FrameVertex::GetPose() {
//	return mpPose;
//}

UVR_SLAM::PoseOptimizationEdge::PoseOptimizationEdge(cv::Mat _Xw, int rsize) :Edge(rsize) {
	cv::cv2eigen(_Xw, Xw);
}

double UVR_SLAM::PoseOptimizationEdge::CalcError() {
	//vertex1과 vertex2의 param을 projection하고 에러 계산하고
	//코스트
	//휴버 코스트 펑션
	Eigen::Matrix3d R = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetRotationMatrix();
	Eigen::Vector3d t = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetTranslationMatrix();

	//std::cout << "R=" << R << std::endl << "t=" << t << std::endl;

	Xcam = R*Xw + t;
	Eigen::Vector3d p2D = K*Xcam;
	p2D(0) = p2D(0) / p2D(2);
	p2D(1) = p2D(1) / p2D(2);

	residual = p2D.head(2) - measurment;

	//std::cout << "Xcam = " << Xcam << std::endl << residual << std::endl;

	//basic chi2
	err = residual.dot(info*residual);
	if (mpRobustKernel) {
		Eigen::Vector3d rho;
		mpRobustKernel->robustify(err, rho);
		err = rho[0];
		w = rho[1];
	}

	bDepth = Xcam(2) > 0.0f;
	//std::cout << err<<R<<t<<Xw<<K<<p2D<<residual<< measurment <<info << std::endl;

	return err;
}

void UVR_SLAM::PoseOptimizationEdge::CalcJacobian() {
	invz = 1.0 / Xcam(2);
	Eigen::Matrix3d S = -1 * UVR_SLAM::MatrixOperator::Skew_Eigen(Xcam);
	Eigen::MatrixXd j = Eigen::MatrixXd::Zero(2, 3);

	double invz2 = invz*invz;
	j(0, 0) = fx*invz;
	j(0, 2) = -fx*Xcam(0)*invz2;
	j(1, 1) = fy*invz;
	j(1, 2) = -fy*Xcam(1)*invz2;

	Eigen::MatrixXd data = Eigen::MatrixXd::Zero(3, 6);
	Eigen::MatrixXd temp = Eigen::MatrixXd(3, 3);
	temp.setIdentity();
	data.block(0, 0, 3, 3) = temp;
	data.block(0, 3, 3, 3) = S;

	mvpVertices[0]->SetJacobian(j*data);

}
////////////////////////////////////////////////////////////////////////
UVR_SLAM::MapPointVertex::MapPointVertex() :Vertex() {
	param;
}
UVR_SLAM::MapPointVertex::MapPointVertex(cv::Mat _Xw, int idx, int _size) :Xw(_Xw),Vertex(idx, _size, false) {
	SetParam();
}
UVR_SLAM::MapPointVertex::~MapPointVertex(){}
void UVR_SLAM::MapPointVertex::SetParam(){
	Xw.convertTo(Xw, CV_64FC1);
	cv::cv2eigen(Xw, param);
}
void UVR_SLAM::MapPointVertex::RestoreData(){
	cv::eigen2cv(param, Xw);
	Xw.convertTo(Xw, CV_32FC1);
}
void UVR_SLAM::MapPointVertex::UpdateParam(){
	param += d;
}
void* UVR_SLAM::MapPointVertex::GetPointer() {
	return nullptr;
}

///////////////////////////////////////////////////////////////////////
bool UVR_SLAM::EdgePoseNMap::GetDepth() {
	return bDepth;
}

double UVR_SLAM::EdgePoseNMap::CalcError(){
	//vertex1과 vertex2의 param을 projection하고 에러 계산하고
	//코스트
	//휴버 코스트 펑션
	Eigen::Matrix3d R = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetRotationMatrix();
	Eigen::Vector3d t = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetTranslationMatrix();
	Xw = ((UVR_SLAM::MapPointVertex*)mvpVertices[1])->GetParam();
	Xcam = R*Xw + t;
	Eigen::Vector3d p2D = K*Xcam;
	p2D(0) = p2D(0) / p2D(2);
	p2D(1) = p2D(1) / p2D(2);

	residual = p2D.head(2) - measurment;

	//basic chi2
	err = residual.dot(info*residual);
	if (mpRobustKernel) {
		Eigen::Vector3d rho;
		mpRobustKernel->robustify(err, rho);
		err = rho[0];
		w = rho[1];
	}

	bDepth = Xcam(2) > 0.0f;
	//std::cout << err<<R<<t<<Xw<<K<<p2D<<residual<< measurment <<info << std::endl;

	return err;
}
void UVR_SLAM::EdgePoseNMap::CalcJacobian() {
	invz = 1.0 / Xcam(2);
	Eigen::Matrix3d S = -1 * UVR_SLAM::MatrixOperator::Skew_Eigen(Xcam);
	Eigen::MatrixXd j = Eigen::MatrixXd::Zero(2, 3);

	double invz2 = invz*invz;
	j(0, 0) = fx*invz;
	j(0, 2) = -fx*Xcam(0)*invz2;
	j(1, 1) = fy*invz;
	j(1, 2) = -fy*Xcam(1)*invz2;

	Eigen::MatrixXd data = Eigen::MatrixXd::Zero(3, 6);
	Eigen::MatrixXd temp = Eigen::MatrixXd(3, 3);
	temp.setIdentity();
	data.block(0, 0, 3, 3) = temp;
	data.block(0, 3, 3, 3) = S;
	
	mvpVertices[0]->SetJacobian(j*data);
	Eigen::Matrix3d R = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetRotationMatrix();
	mvpVertices[1]->SetJacobian(j*R);
}
void UVR_SLAM::EdgePoseNMap::SetHessian() {
	Eigen::MatrixXd I = w*info;
	mvpVertices[0]->SetHessian(I, residual);
	mvpVertices[1]->SetHessian(I, residual);
	if (mvpVertices[0]->GetFixed() || mvpVertices[1]->GetFixed())
		return;
	mvSubHs[0] = mvpVertices[0]->GetJacobian().transpose()*mvpVertices[1]->GetJacobian();
}

int UVR_SLAM::EdgePoseNMap::GetIndex() {
	return mvpVertices[0]->GetIndex();
}