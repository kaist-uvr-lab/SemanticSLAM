# SemanticSLAM
1. 추가 라이브러리
  - OpenCV
  - Eigen
  - dbow3
  - rapidjson
  - happyhttp
2. 주의 사항
  - fbow의 windows.h에 의해서 opencv와 충돌이 발생함.
    - 이를 해결하기 위해 cv namespace를  이용하지 않거나, fbow가 opencv 보다 우선적으로 include 되어야 함
