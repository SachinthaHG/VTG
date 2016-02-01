from flask import Flask,request
from getClass import getClass
from Recommendations import Recommendations

app = Flask(__name__)

@app.route('/dispatch',methods=['POST'])
def dispatch():
    contnt = request.form['desc']
    location = request.form['loc']
    bearing = request.form['bearing']
    a = getClass()
    test = a.imageSearch(contnt, location, bearing)
    return test

@app.route('/query',methods=['POST'])
def query():
    a = Recommendations()
    test = a.sendDataToHomePage()
#     print test
    return test
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)