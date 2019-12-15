#include <GroundPlaneInitialization.h>
#include <opencv2/core/eigen.hpp>

UVR_SLAM::PlaneInitVertex::PlaneInitVertex():Vertex() {}
UVR_SLAM::PlaneInitVertex::PlaneInitVertex(cv::Mat _P, int _size) : Vertex(0, _size, false), matPlane(_P){
	SetParam();
}
UVR_SLAM::PlaneInitVertex::~PlaneInitVertex() {}
void UVR_SLAM::PlaneInitVertex::SetParam() {
	matPlane.convertTo(matPlane, CV_64FC1);
	cv::cv2eigen(matPlane, param);
}

void UVR_SLAM::PlaneInitVertex::RestoreData() {
	cv::eigen2cv(param, matPlane);
	matPlane.convertTo(matPlane, CV_32FC1);
}

void* UVR_SLAM::PlaneInitVertex::GetPointer() {
	return nullptr;
	//return mpFrame;
}
void UVR_SLAM::PlaneInitVertex::UpdateParam() {
	param += d;
	Normalize();
	ResetHessian();
}
void UVR_SLAM::PlaneInitVertex::Normalize() {
	Eigen::Vector3d Normal = param.head(3);
	param /= Normal.norm();
}
////////////////////////////////////////////////////////////
//Edge
UVR_SLAM::PlaneInitEdge::PlaneInitEdge() :Edge() {}
UVR_SLAM::PlaneInitEdge::PlaneInitEdge(cv::Mat _X, cv::Mat _R, cv::Mat _t,int rsize) :Edge(rsize) {
	cv::cv2eigen(_R, R);
	cv::cv2eigen(_t, t);
	cv::cv2eigen(_X, Xplane);
	//Xcam = Eigen::Vector3d();
}
UVR_SLAM::PlaneInitEdge::~PlaneInitEdge() {}

double UVR_SLAM::PlaneInitEdge::CalcError() {
	Normal = mvpVertices[0]->GetParam().head(3);
	Dist   = mvpVertices[0]->GetParam()[3];
	tempDepth = 1.0 / Normal.dot(Xplane);
	Xdepth = -Dist*tempDepth;
	Xcam = R*Xdepth*Xplane + t;
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
	return err;
}

void UVR_SLAM::PlaneInitEdge::CalcJacobian() {
	Eigen::MatrixXd j = Eigen::MatrixXd::Zero(2, 3);
	invz = 1.0 / Xcam(2);
	double invz2 = invz*invz;
	j(0, 0) = fx*invz;
	j(0, 2) = -fx*Xcam(0)*invz2;
	j(1, 1) = fy*invz;
	j(1, 2) = -fy*Xcam(1)*invz2;

	j = j*R*Xplane;

	Eigen::Vector4d data = Eigen::Vector4d::Zero();
	double invDepth2 = tempDepth*tempDepth;
	Eigen::Vector3d  temp = Xplane*Dist*invDepth2;
	data.head(3) = temp;
	//data.block(0, 0, 3, 1) = temp;
	data[3] = -tempDepth;
	mvpVertices[0]->SetJacobian(j*data.transpose());
}

void UVR_SLAM::PlaneInitEdge::SetHessian() {
	Eigen::MatrixXd I = w*info;
	mvpVertices[0]->SetHessian(I, residual);
}
