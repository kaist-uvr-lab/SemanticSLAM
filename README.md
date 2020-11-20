# SemanticSLAM
1. 추가 라이브러리
  - OpenCV
  - GraphOptimizer
  - Eigen
  - fbow
  - rapidjson
  - happyhttp
2. 주의 사항
  - fbow의 windows.h에 의해서 opencv와 충돌이 발생함.
    - 이를 해결하기 위해 cv namespace를  이용하지 않거나, fbow가 opencv 보다 우선적으로 include 되어야 함
    
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2019-0-01270, WISE AR UI/UX Platform Development for Smartglasses)
