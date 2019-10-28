//
// Created by UVR-KAIST on 2019-02-01.
//

#include <MatrixOperator.h>
#include <omp.h>


Eigen::Matrix3d UVR_SLAM::MatrixOperator::Skew_Eigen(Eigen::Vector3d v) {
	Eigen::Matrix3d skew;
	skew.setZero();
	//row, column
	skew(0, 1) = -v[2];
	skew(1, 0) = v[2];
	skew(0, 2) = v[1];
	skew(2, 0) = -v[1];
	skew(1, 2) = -v[0];
	skew(2, 1) = v[0];
	return skew;
}
Eigen::Matrix3d UVR_SLAM::MatrixOperator::EXP_Eigen(Eigen::Vector3d d) {
	Eigen::Matrix3d exp;
	exp.setIdentity();
	if (!d.isZero(1e-4)) {
		double theta = d.norm();
		Eigen::Matrix3d skew = Skew_Eigen(d);
		skew /= theta;
		Eigen::Matrix3d skew2 = skew*skew;
		exp += sin(theta)*skew + (1.0 - cos(theta))*skew2;
	}
	return exp;
}
Eigen::Matrix4d UVR_SLAM::MatrixOperator::EXP_Eigen(Eigen::VectorXd d) {
	Eigen::Vector3d v = d.head(3);
	Eigen::Vector3d w = d.tail(3);

	Eigen::Matrix3d R = EXP_Eigen(w);
	Eigen::Matrix4d P;
	P.setIdentity();

	double theta = w.norm();
	Eigen::Matrix3d skew = Skew_Eigen(w);
	Eigen::Matrix3d skew2 = skew*skew;

	Eigen::Matrix3d V;
	V.setIdentity();
	if (abs(theta)> 2.5e-4) {
		V += (1 - cos(theta))*skew / (theta*theta) + (theta - sin(theta)) / (theta*theta*theta)*skew2;
	}
	Eigen::Vector3d t = V*v;

	P.block(0, 0, 3, 3) = R;
	P.block(0, 3, 3, 1) = t;

	return P;
}
void UVR_SLAM::MatrixOperator::Mat2Eigen(cv::Mat src, Eigen::VectorXd& dst) {
	//type check필요
	//double || float
	src.convertTo(src, CV_64FC1);
	int size = MAX(src.rows, src.cols);
	dst = Eigen::VectorXd(size);
	for (int i = 0; i < size; i++) {
		dst(i) = src.at<double>(i);
	}
}
void UVR_SLAM::MatrixOperator::Mat2Eigen(cv::Mat src, Eigen::MatrixXd& dst) {
	src.convertTo(src, CV_64FC1);
	dst = Eigen::MatrixXd(src.rows, src.cols);
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			dst(y, x) = src.at<double>(y, x);
		}
	}
}




void UVR_SLAM::MatrixOperator::CholeskyDecomposition(const Mat A, Mat& U) {
	int i, j, k;
	unsigned int size = A.rows * A.cols;

	//U = A.clone();

	// Perform the Cholesky decomposition in place on the U matrix
	for (k = 0; k < U.rows; k++) {
		// Take the square root of the diagonal element
		U.at<float>(k, k) = sqrt(U.at<float>(k, k));
		if (U.at<float>(k, k) <= 0) {
			printf("Cholesky decomposition failed. \n");
		}

		// Division step
		//Parallelize this - seems like declaring private iterator
		//values slows it down....?
#pragma omp parallel for
		for (j = (k + 1); j < U.rows; j++)
			U.at<float>(k, j) /= U.at<float>(k, k);

#pragma omp parallel for private(i,j)
		for (i = (k + 1); i < U.rows; i++)
		{
			for (j = i; j < U.rows; j++)
			{
				U.at<float>(i, j) -= U.at<float>(k, i)*U.at<float>(k, j);
				//U.elements[i * U.num_rows + j] -= U.elements[k * U.num_rows + i] * U.elements[k * U.num_rows + j];
			}
		}
	}

	// As the final step, zero out the lower triangular portion of U
#pragma omp parallel for private(i,j)
	for (i = 0; i < U.rows; i++)
		for (j = 0; j < i; j++)
			U.at<float>(i, j) = 0.0;
}
cv::Mat UVR_SLAM::MatrixOperator::Lsolve(cv::Mat L, cv::Mat b) {
	cv::Mat X = cv::Mat::zeros(b.rows, 1, b.type());
	for (int y = 0; y < L.rows; y++) {
		float sum = b.at<float>(y);
#pragma omp parallel for reduction(+:sum)
		for (int x = 0; x < y; x++) {
			sum -= L.at<float>(y, x)*X.at<float>(x);
		}
		X.at<float>(y) = sum / L.at<float>(y, y);
	}
	return X;
}
cv::Mat UVR_SLAM::MatrixOperator::Usolve(cv::Mat U, cv::Mat b) {
	cv::Mat X = cv::Mat::zeros(b.rows, 1, b.type());
	for (int y = U.cols - 1; y >= 0; y--) {
		float sum = b.at<float>(y);
#pragma omp parallel for reduction(+:sum)
		for (int x = U.rows - 1; x > y; x--) {
			sum -= U.at<float>(y, x)*X.at<float>(x);
		}
		X.at<float>(y) = sum / U.at<float>(y, y);
	}
	return X;
}

//타입은 더블형
cv::Mat UVR_SLAM::MatrixOperator::Diag(cv::Mat src) {
	cv::Mat Hdiag = cv::Mat::zeros(src.size(), src.type());
	for (int i = 0; i < src.cols; i++) {
		Hdiag.at<double>(i, i) = src.at<double>(i, i);
	}
	return Hdiag;
}

cv::Mat UVR_SLAM::MatrixOperator::Jacobian(cv::Mat mat) {
	double theta = CalcDistance(mat);
	cv::Mat EYE = cv::Mat::eye(3, 3, CV_64FC1);
	if (theta < 2.5e-4) {
		return EYE;
	}
	mat = mat / theta;
	cv::Mat skewMat = GetSkewSymetricMatrix(mat);
	cv::Mat skewMat2 = skewMat*skewMat;
	return EYE - (1 - cos(theta)) / theta*skewMat + (theta - sin(theta)) / theta*skewMat2;


}
cv::Mat UVR_SLAM::MatrixOperator::InvJacobian(cv::Mat mat) {
	double theta = CalcDistance(mat);
	cv::Mat EYE = cv::Mat::eye(3, 3, CV_64FC1);
	if (theta < 2.5e-4) {
		return EYE;
	}
	mat = mat / theta;
	cv::Mat skewMat = GetSkewSymetricMatrix(mat);
	cv::Mat skewMat2 = skewMat*skewMat;
	double temp = 0.5*theta;
	return EYE + temp*skewMat + skewMat2 + temp / sin(theta)*(1 + cos(theta))*skewMat2;
}

cv::Mat UVR_SLAM::MatrixOperator::EXP(double wx, double wy, double wz) {
	cv::Mat rotationMat;
	if (wx == 0.0 && wy == 0.0 && wz == 0.0) {
		rotationMat = cv::Mat::eye(3, 3, CV_64FC1);
	}
	else {
		double theta = sqrt(wx*wx + wy*wy + wz*wz);
		//skew_mat = (cv::Mat_<float>(3, 3) << 0.0, -wz, wy, wz, 0.0, -wx, -wy, wx, 0.0);
		cv::Mat skew_mat = cv::Mat::zeros(3, 3, CV_64FC1);
		skew_mat.at<double>(0, 1) = -wz;
		skew_mat.at<double>(0, 2) = wy;
		skew_mat.at<double>(1, 0) = wz;
		skew_mat.at<double>(1, 2) = -wx;
		skew_mat.at<double>(2, 0) = -wy;
		skew_mat.at<double>(2, 1) = wx;
		skew_mat /= theta;
		cv::Mat s2 = skew_mat*skew_mat;
		cv::Mat e = cv::Mat::eye(3, 3, CV_64FC1);
		rotationMat = e + sin(theta)*skew_mat + (1.0 - cos(theta))*s2;
	}
	return rotationMat;
}
cv::Mat UVR_SLAM::MatrixOperator::EXPD(cv::Mat mat) {
	double theta = CalcDistance(mat);
	cv::Mat EYE = cv::Mat::eye(3, 3, CV_64FC1);
	if (theta < 2.5e-4) {
		return EYE;
	}
	mat = mat / theta;
	cv::Mat skewMat = GetSkewSymetricMatrix(mat);
	cv::Mat skewMat2 = skewMat*skewMat;
	return EYE + sin(theta)*skewMat + (1.0 - cos(theta))*skewMat2;
}

cv::Mat UVR_SLAM::MatrixOperator::EXP6(double vx, double vy, double vz, double wx, double wy, double wz) {
	cv::Mat T;
	cv::Mat v = cv::Mat::zeros(3, 1, CV_64FC1);
	v.at<double>(0) = vx;
	v.at<double>(1) = vy;
	v.at<double>(2) = vz;
	cv::Mat w = cv::Mat::zeros(3, 1, CV_64FC1);
	w.at<double>(0) = wx;
	w.at<double>(1) = wy;
	w.at<double>(2) = wz;

	cv::Mat R = EXP(wx, wy, wz);

	double theta = sqrt(wx*wx + wy*wy + wz*wz);
	cv::Mat skew_mat = GetSkewSymetricMatrix(w);
	cv::Mat s2 = skew_mat*skew_mat;

	cv::Mat V = cv::Mat::eye(3, 3, CV_64FC1);
	if (abs(theta)> 2.5e-4) {
		V += (1 - cos(theta))*skew_mat / (theta*theta) + (theta - sin(theta)) / (theta*theta*theta)*s2;
	}
	cv::Mat t = V*v;

	T = cv::Mat::zeros(3, 4, CV_64FC1);
	R.copyTo(T.colRange(0, 3));
	t.copyTo(T.col(3));
	return T;
}
cv::Mat UVR_SLAM::MatrixOperator::EXP6X44(double vx, double vy, double vz, double wx, double wy, double wz) {
	cv::Mat T;
	cv::Mat v = cv::Mat::zeros(3, 1, CV_64FC1);
	v.at<double>(0) = vx;
	v.at<double>(1) = vy;
	v.at<double>(2) = vz;
	cv::Mat w = cv::Mat::zeros(3, 1, CV_64FC1);
	w.at<double>(0) = wx;
	w.at<double>(1) = wy;
	w.at<double>(2) = wz;

	cv::Mat R = EXP(wx, wy, wz);

	double theta = sqrt(wx*wx + wy*wy + wz*wz);
	cv::Mat skew_mat = GetSkewSymetricMatrix(w);
	cv::Mat s2 = skew_mat*skew_mat;

	cv::Mat V = cv::Mat::eye(3, 3, CV_64FC1);
	if (abs(theta)> 2.5e-4) {
		V += (1 - cos(theta))*skew_mat / (theta*theta) + (theta - sin(theta)) / (theta*theta*theta)*s2;
	}
	cv::Mat t = V*v;

	T = cv::Mat::eye(4, 4, CV_64FC1);
	R.copyTo(T.colRange(0, 3).rowRange(0, 3));
	t.copyTo(T.col(3).rowRange(0, 3));
	return T;
}
cv::Mat UVR_SLAM::MatrixOperator::LOG(cv::Mat rmat) {
	cv::Mat res = cv::Mat::zeros(3, 1, CV_32FC1);
	float traceR = rmat.at<float>(0, 0) + rmat.at<float>(1, 1) + rmat.at<float>(2, 2);
	if (traceR == 3.0) {

	}
	else {
		float theta = acos((traceR - 1.0f) / 2.0);
		float temp = sin(theta);

		if (abs(theta)> 2.5e-4)
			return res;
		temp = theta / (2.0*temp);
		res.at<float>(0) = rmat.at<float>(2, 1) - rmat.at<float>(1, 2);
		res.at<float>(1) = rmat.at<float>(0, 2) - rmat.at<float>(2, 0);
		res.at<float>(2) = rmat.at<float>(1, 0) - rmat.at<float>(0, 1);
		res *= temp;
	}
	return res;
}
cv::Mat UVR_SLAM::MatrixOperator::LOGD(cv::Mat rmat) {
	cv::Mat res = cv::Mat::zeros(3, 1, CV_64FC1);
	//trace가 3인 경우
	//sin(theta)가 0이 되는 경우에 에러가 발생함.
	double traceR = rmat.at<double>(0, 0) + rmat.at<double>(1, 1) + rmat.at<double>(2, 2);
	if (abs(traceR - 3.0)< 2.5e-4) {

	}
	else if (abs(traceR - 1.0)<2.5e-4) {

	}
	else if (traceR > 3.0 || traceR < -1.0) {}

	else {
		double theta = acos((traceR - 1.0) / 2.0);
		double temp = sin(theta);

		if (abs(temp)< 2.5e-4)
			return res;
		temp = theta / (2.0*temp);
		res.at<double>(0) = rmat.at<double>(2, 1) - rmat.at<double>(1, 2);
		res.at<double>(1) = rmat.at<double>(0, 2) - rmat.at<double>(2, 0);
		res.at<double>(2) = rmat.at<double>(1, 0) - rmat.at<double>(0, 1);
		res *= temp;
	}
	return res;
}

//t의 타입은 double
//최적화와 F 찾을 때 사용됨. F 사용시 t는 float이기 때문에 double로 변화
cv::Mat UVR_SLAM::MatrixOperator::GetSkewSymetricMatrix(cv::Mat t) {
	cv::Mat res = cv::Mat::zeros(3, 3, CV_64FC1);
	res.at<double>(0, 1) = -t.at<double>(2);
	res.at<double>(1, 0) = t.at<double>(2);
	res.at<double>(0, 2) = t.at<double>(1);
	res.at<double>(2, 0) = -t.at<double>(1);
	res.at<double>(1, 2) = -t.at<double>(0);
	res.at<double>(2, 1) = t.at<double>(0);
	return res;
}

cv::Mat UVR_SLAM::MatrixOperator::GetSkewSymetricMatrix(double x, double y, double z) {
	cv::Mat res = cv::Mat::zeros(3, 3, CV_64FC1);
	res.at<double>(0, 1) = -z;
	res.at<double>(1, 0) = z;
	res.at<double>(0, 2) = y;
	res.at<double>(2, 0) = -y;
	res.at<double>(1, 2) = -x;
	res.at<double>(2, 1) = x;
	return res;
}

int UVR_SLAM::MatrixOperator::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
	const int *pa = a.ptr<int32_t>();
	const int *pb = b.ptr<int32_t>();

	int dist = 0;

	for (int i = 0; i<8; i++, pa++, pb++)
	{
		unsigned  int v = *pa ^ *pb;
		v = v - ((v >> 1) & 0x55555555);
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}

	return dist;
}

const float UVR_SLAM::MatrixOperator::deg2rad = (float)(CV_PI / 180.0);
const float UVR_SLAM::MatrixOperator::rad2deg = (float)(180.0 / CV_PI);