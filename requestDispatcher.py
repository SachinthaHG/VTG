from flask import Flask,request
from getClass import getClass

app = Flask(__name__)

@app.route('/dispatch',methods=['POST'])
def dispatch():
    contnt = request.form['desc']
    location = request.form['loc']
    a = getClass()
    test = a.imageSearch(contnt, location)
#     print test
    return test

if __name__ == '__main__':
    app.run(host='0.0.0.0')