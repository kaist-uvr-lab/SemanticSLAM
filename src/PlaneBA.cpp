#include "PlaneBA.h"

namespace g2o{
//Plane Vertex
PlaneVertex::PlaneVertex() :BaseVertex<6, Vector6d>() {

}
bool PlaneVertex::read(std::istream& is) {
	Vector6d lv;
	for (int i = 0; i<6; i++)
		is >> _estimate[i];
	return true;
}
bool PlaneVertex::write(std::ostream& os) const {
	Vector6d lv = estimate();
	for (int i = 0; i< 6; i++) {
		os << lv[i] << " ";
	}
	return os.good();
}
//Edge
PlaneBAEdge::PlaneBAEdge() {

}
bool PlaneBAEdge::read(std::istream& is){
	return true;
}

bool PlaneBAEdge::write(std::ostream& os) const {
	return true;
}

void PlaneBAEdge::computeError(){
	const PlaneVertex* v1 = static_cast<const PlaneVertex*>(_vertices[0]);
	const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[1]);

	Vector6d param1 = v1->estimate();
	Vector3d param2 = v2->estimate();
	
	Vector3d normal = param1.head(3).normalized();
	double dist = param1[3]/param1.head(3).squaredNorm();

	_error[0] = normal.dot(param2) + dist;
}
void PlaneBAEdge::linearizeOplus(){
	const PlaneVertex* v1 = static_cast<const PlaneVertex*>(_vertices[0]);
	const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[1]);

	Vector6d param1 = v1->estimate(); //평면
	Vector3d param2 = v2->estimate(); //맵포인트

	_jacobianOplusXi = Vector6d::Zero();
	_jacobianOplusXi.head(3) = param2;
	_jacobianOplusXi[3] = 1.0;
	_jacobianOplusXj = param1.head(3).normalized();
}

PlaneCorrelationBAEdge::PlaneCorrelationBAEdge(){}
bool PlaneCorrelationBAEdge::read(std::istream& is){
	return true;
}
bool PlaneCorrelationBAEdge::write(std::ostream& os) const{
	return true;
}
void PlaneCorrelationBAEdge::computeError(){
	//type 1 = orthogonal. 수직이려면 0
	//type 2 = parallel. 수평이려면 1, 즉 1-val이 되어야 함.
	
	const PlaneVertex* v1 = static_cast<const PlaneVertex*>(_vertices[0]);
	const PlaneVertex* v2 = static_cast<const PlaneVertex*>(_vertices[1]);

	Vector3d normal1 = v1->estimate().head(3).normalized();
	Vector3d normal2 = v2->estimate().head(3).normalized();
	if(type == 1)
		_error[0] = abs(normal1.dot(normal2));
	else 
		_error[0] = 1.0-abs(normal1.dot(normal2));
}
void PlaneCorrelationBAEdge::linearizeOplus(){
	const PlaneVertex* v1 = static_cast<const PlaneVertex*>(_vertices[0]);
	const PlaneVertex* v2 = static_cast<const PlaneVertex*>(_vertices[1]);

	Vector3d normal1 = v1->estimate().head(3).normalized();
	Vector3d normal2 = v2->estimate().head(3).normalized();
	double temp = (normal1.dot(normal2));

	_jacobianOplusXi = Vector6d::Zero();
	_jacobianOplusXj = Vector6d::Zero();
	
	_jacobianOplusXi.head(3) = normal2*temp;
	_jacobianOplusXj.head(3) = normal1*temp;
	if (type == 2) {
		_jacobianOplusXi.head(3) *= -1.0;
		_jacobianOplusXj.head(3) *= -1.0;
	}
}

}//g2o