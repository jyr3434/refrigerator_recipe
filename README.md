﻿># 프로젝트 : 냉장고 레시피 추천
[[[ 추가 및 작업내용은 업데이트 예정 ]]]

>## 구현 기능 예정 리스트
  
 - 냉장고에 남아 있는 식재료를 토대로 가장 구현 가능한 요리 레시피를 추천받을수 있는 기능
 - 사용자가 원하는 카테고리 및 테마를 선택하여 레시피 추천
 - 냉장고 안에 있는 뒤 섞여 있는 식재료의 이미지만 으로도 각 식재료 판별을 한뒤 레시피 추천
	

>##	필요예상 작업 순서

 1. 레시피 정보가 있는 페이지 조사 및 체크
 2. 레시피 정보 크롤링 으로 데이터 구축 
	 - 멀티프로세싱을 활용하여 작업시간 단축을 목표
 3. 위 내용을 가지고 자연어 처리 및 식재료 키워드 추출
	 - 불필요한 단어 및 
 5.  추출된 키워드로 이미지 자료 크롤링
   
 6. 크롤링 된 이미지 데이터를 근거로 분류학습 -> 결과물 : 식재료 구별
   
 7. 절대적인 추천기준이 없음으로 몇가의 추천시스템 구축 및 사용자 의견 또한 데이터에 반영
   
 8. 웹페이지를 통한 최종 결과물 시뮬레이션 도출

> 통계작업 예정 부분
 [ 위 데이터중 자연어처리된 데이터를 이용 ]
	1. 분류분석
	2. 군집분석
	3. 요인분석
 

 > 포트폴리오 부분
  - 최종 웹 사이트를 구축 함으로써 기능과 스킬 부분에 있어 
	회사에 직접적인 어필이 가능할것으로 보여짐

 > 난관이 예상되는 부분
 - 식재료 종류 수량 및 표준화 처리 및 미처리에 의한 영향 여부
 - 각 식재료의 이미지 딥러닝에 있어 근거자료 양, 매트릭스 크기, 
 학습시간이 여부
