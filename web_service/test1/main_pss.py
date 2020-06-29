from flask import Flask, render_template, redirect, request, url_for, send_file, jsonify
import flask
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

posts = [
    {
        'author': {
            'username': 'test-user'
        },
        'title': '첫 번째 포스트',
        'content': '첫 번째 포스트 내용입니다.',
        'date_posted': datetime.strptime('2018-08-01', '%Y-%m-%d')
    },
    {
        'author': {
            'username': 'test-user'
        },
        'title': '두 번째 포스트',
        'content': '두 번째 포스트 내용입니다.',
        'date_posted': datetime.strptime('2018-08-03', '%Y-%m-%d')
    },
]
import os
app_root = os.path.abspath(os.path.dirname(__file__))

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', posts=posts)


@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/fileUpload', methods = ['GET', 'POST'])
# 이미지 파일 업로드 및 출력 시도
def upload_file():
   if request.method == 'POST':
      cat1 = request.form['cate1']
      cat2 = request.form['cate2']
      cat3 = request.form['cate3']
      cat4 = request.form['cate4']
      result = (cat1,cat2,cat3,cat4)
      print('resutl :',result)
      #저장할 경로 + 파일명
      print(app_root)
      # return 'uploads 디렉토리 -> 파일 업로드 성공!'
      filenames = []
      for file in request.files.getlist('file'):
          file.save(app_root + '/static/naver/' + file.filename)
          filenames.append(file.filename)
          print(app_root)
      return 'uploads 디렉토리 -> 파일 업로드 성공!' + render_template('about.html', label=filenames,result=None)

# def make_prediction():
#     if request.method == 'POST':
#
#         # 업로드 파일 처리 분기
#         file = request.files['image']
#         if not file: return render_template('about.html', label="No Files")
#         return render_template('about.html', label="yes Files")

# 드롭박스 값 출력 시도
@app.route('/layout', methods = ['GET', 'POST'])
def get_updated_settings():
    if request.method == 'GET':
       # good for debug, make sure args were sent
       print ( request.args )
       cat1 = request.args.get('cat1', 'default_if_none')
       cat2 = request.args.get('cat2', 'default_if_none')
       cat3 = request.args.get('cat3', 'default_if_none')
       cat4 = request.args.get('cat4', 'default_if_none')
       result = (cat1 + cat2 + cat3 + cat4)
       print(result)
       #output = {}
       # I have no idea what this returns...just doing a list generator here assuming we get a list of values
       # output['am_1'] = [x for x in db.hget(skey, '1am')]
       # output['am_2'] = [x for x in db.hget(skey, '1am')]
       return render_template('about.html', result=result, label="result")

if __name__ == '__main__':
    #서버 실행
   app.run(debug = True)