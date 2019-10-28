#include <PlaneBastedOptimization.h>


double UVR_SLAM::PlaneEdgeOnlyPose::CalcError() {
	Eigen::Matrix3d R = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetRotationMatrix();
	Eigen::Vector3d t = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetTranslationMatrix();
	Eigen::Matrix4d T;
	T.setIdentity();
	T.block(0, 0, 3, 3) = R;
	T.block(0, 3, 3, 1) = t;

	Eigen::Matrix4d Tinv;
	Tinv.setIdentity();
	Eigen::Matrix3d Rinv = R.transpose();
	Eigen::Vector3d tinv = -Rinv*t;
	Tinv.block(0, 0, 3, 3) = Rinv;
	Tinv.block(0, 3, 3, 1) = tinv;

	Pc = Tinv.transpose()*Pw;

	Xcam = Kinv*Ximg;
	double tempval = Pc.head(3).dot(Xcam)*mdDepth + Pc(3);
	residual = residual.setOnes()*tempval;

	//basic chi2
	err = residual.dot(info*residual);
	if (mpRobustKernel) {
		Eigen::Vector3d rho;
		mpRobustKernel->robustify(err, rho);
		err = rho[0];
		w = rho[1];
	}
}

void UVR_SLAM::PlaneEdgeOnlyPose::CalcJacobian() {

	Eigen::Vector3d NormalPw = Pw.head(3);
	Eigen::Matrix3d R = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetRotationMatrix();
	Eigen::Matrix3d Rinv = R.transpose();
	Eigen::MatrixXd J1 = Eigen::MatrixXd::Zero(1, 6);
	Eigen::VectorXd tmp1 = NormalPw.transpose()*Rinv;
	J1.block(0, 0, 1, 3) = tmp1;

	J1(0, 3) = Xcam(1)*NormalPw.dot(Rinv.col(2)) - Xcam(2)*NormalPw.dot(Rinv.col(1));
	J1(0, 4) = Xcam(2)*NormalPw.dot(Rinv.col(0)) - Xcam(0)*NormalPw.dot(Rinv.col(2));
	J1(0, 5) = Xcam(0)*NormalPw.dot(Rinv.col(1)) - Xcam(1)*NormalPw.dot(Rinv.col(0));

	mvpVertices[0]->SetJacobian(-J1);

	//Eigen::Vector3d NormalPc = Pc.head(3);
	//Eigen::MatrixXd J2 = NormalPc.transpose()*Xcam;
	//mvpVertices[1]->SetJacobian(J2);
}

double UVR_SLAM::PlaneEdgeOnlyPoseNMap::CalcError() {
	Eigen::Matrix3d R = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetRotationMatrix();
	Eigen::Vector3d t = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetTranslationMatrix();
	Eigen::Matrix4d T;
	T.setIdentity();
	T.block(0, 0, 3, 3) = R;
	T.block(0, 3, 3, 1) = t;

	Eigen::Matrix4d Tinv;
	Tinv.setIdentity();
	Eigen::Matrix3d Rinv = R.transpose();
	Eigen::Vector3d tinv = -Rinv*t;
	Tinv.block(0, 0, 3, 3) = Rinv;
	Tinv.block(0, 3, 3, 1) = tinv;

	Pc = Tinv.transpose()*Pw;

	double depth = ((UVR_SLAM::PlaneMapPointVertex*)mvpVertices[1])->GetParam()(0);

	Xcam = Kinv*Ximg;
	double tempval = Pc.head(3).dot(Xcam)*depth + Pc(3);
	residual = residual.setOnes()*tempval;

	//basic chi2
	err = residual.dot(info*residual);
	if (mpRobustKernel) {
		Eigen::Vector3d rho;
		mpRobustKernel->robustify(err, rho);
		err = rho[0];
		w = rho[1];
	}
}

void UVR_SLAM::PlaneEdgeOnlyPoseNMap::CalcJacobian() {
	Eigen::Vector3d NormalPc = Pc.head(3);
	Eigen::Vector3d NormalPw = Pw.head(3);

	//std::cout << "eigen test" << std::endl << NormalPc << std::endl << Pc << std::endl;

	Eigen::Matrix3d R = ((UVR_SLAM::FrameVertex*)mvpVertices[0])->GetRotationMatrix();
	Eigen::Matrix3d Rinv = R.transpose();
	Eigen::MatrixXd J1 = Eigen::MatrixXd::Zero(1, 6);
	Eigen::VectorXd tmp1 = NormalPw.transpose()*Rinv;
	J1.block(0, 0, 1, 3) = tmp1;

	J1(0, 3) = Xcam(1)*NormalPw.dot(Rinv.col(2)) - Xcam(2)*NormalPw.dot(Rinv.col(1));
	J1(0, 4) = Xcam(2)*NormalPw.dot(Rinv.col(0)) - Xcam(0)*NormalPw.dot(Rinv.col(2));
	J1(0, 5) = Xcam(0)*NormalPw.dot(Rinv.col(1)) - Xcam(1)*NormalPw.dot(Rinv.col(0));

	Eigen::MatrixXd J2 = NormalPc.transpose()*Xcam;

	mvpVertices[0]->SetJacobian(-J1);
	mvpVertices[1]->SetJacobian(J2);
}