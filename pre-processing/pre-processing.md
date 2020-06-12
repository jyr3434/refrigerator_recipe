> # **recipe_preprocess_model.py**
***
 >## 사용된 라이브러리
    import pandas as pd  
    import cx_Oracle  
    import re  

  >## 사용된 메소드

   class RecipePreProcess(): 
    
     def concat_data(self,x,y):       
     def merge_data(self,x,y,on): 
     def save_data(self,filename,df):    
     def str_drop_na(self,df):      
	     - 데이터 결측치 행 제거   
     def filter_rawData_by_db_id(self,df): 
	     -
     def execute_to_clob(self,col,text):  
	     - 데이터 프레임을 오라클에 인서트 
     def select_all_db(self):  
	     -
     def comprehession_data(self,db,raw):  
	     -
     
       
  >## 목적
	  기존 확보한 데이터 에서 결측치를 없애고
	  최종 정리된 데이터를 오라클에 인서트 시키는 작업
	  (python query insert 시간이 오래걸리는 관계로 sqldeveloper에서 데이터 임포트를
	  통해 inset를 수행해도 된다.)
  >## 순서
> 1. catecory, detail 데이터를 recipe_id를 기반으로 merge한다.
> 2. [1]에서 '-'로 치환된 결측치를 제거한다.
> 3. oracle table에 [2]을 insert한다.
> 4. [3] bulk loading을 수행하여 결과 삽입도중 변한 값을 제외하여 유지한 데이터를 complete data로 선정한다.