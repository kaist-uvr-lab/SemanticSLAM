#include <DirectMethod.h>

namespace g2o {

	bool EdgeDirectXYZOnlyPose::read(std::istream& is){
		/*for (int i = 0; i<2; i++) {
			is >> _measurement[i];
		}
		for (int i = 0; i<2; i++)
			for (int j = i; j<2; j++) {
				is >> information()(i, j);
				if (i != j)
					information()(j, i) = information()(i, j);
			}*/
		return true;
	}
	bool EdgeDirectXYZOnlyPose::write(std::ostream& os) const {
		/*for (int i = 0; i<2; i++) {
			os << measurement()[i] << " ";
		}

		for (int i = 0; i<2; i++)
			for (int j = i; j<2; j++) {
				os << " " << information()(i, j);
			}*/
		return os.good();
	}
	void EdgeDirectXYZOnlyPose::computeError(){
		const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
		//Eigen::Matrix<double,1,1> obs(_measurement);
		Vector2d est = cam_project(v1->estimate().map(Xw));
		cv::Point2f pt(est[0], est[1]);
		Eigen::Matrix<double, 1, 1> val;
		
		/*if (pt.x >= w || pt.y >= h || pt.x < 0 || pt.y < 0){
			val(0) = -1000.0;
			_error = val;
			return;
		}*/
		val(0) = _measurement-(double)gra2.at<uchar>(pt);
		_error = val;
	}
	void EdgeDirectXYZOnlyPose::linearizeOplus(){
		VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
		Vector3d xyz_trans = vi->estimate().map(Xw);
		Vector2d est = cam_project(xyz_trans);
		double x = xyz_trans[0];
		double y = xyz_trans[1];
		double invz = 1.0 / xyz_trans[2];
		double invz_2 = invz*invz;
		
		cv::Point2f pt((float)est[0], (float)est[1]);
		
		Eigen::Matrix<double, 1, 2> temp2;
		temp2(0, 0) = dx.at<double>(pt);
		temp2(0, 1) = dy.at<double>(pt);
		//_jacobianOplusXi
		Eigen::Matrix<double, 2, 6> temp;
		
		temp(0, 0) = x*y*invz_2 *fx;
		temp(0, 1) = -(1 + (x*x*invz_2)) *fx;
		temp(0, 2) = y*invz *fx;
		temp(0, 3) = -invz *fx;
		temp(0, 4) = 0;
		temp(0, 5) = x*invz_2 *fx;

		temp(1, 0) = (1 + y*y*invz_2) *fy;
		temp(1, 1) = -x*y*invz_2 *fy;
		temp(1, 2) = -x*invz *fy;
		temp(1, 3) = 0;
		temp(1, 4) = -invz *fy;
		temp(1, 5) = y*invz_2 *fy;

		_jacobianOplusXi = temp2*temp;
	}
	Vector2d EdgeDirectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
		Vector2d proj;
		proj(0) = trans_xyz(0) / trans_xyz(2);
		proj(1) = trans_xyz(1) / trans_xyz(2);
		Vector2d res;
		res[0] = proj[0] * fx + cx;
		res[1] = proj[1] * fy + cy;
		return res;
	}
}