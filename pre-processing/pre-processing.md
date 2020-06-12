
 >## 사용된 라이브러리
    import pandas as pd  
    import cx_Oracle  
    import re  

  >## 사용된 메소드

   class RecipePreProcess: 
    
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
     
       
