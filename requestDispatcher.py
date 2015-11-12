from flask import Flask,json,request
import getClass

app = Flask(__name__)

@app.route('/dispatch',methods=['POST'])
def dispatch():
    contnt = request.form['desc']
    return getClass.imageSearch(contnt)

if __name__ == '__main__':
    app.run(host='0.0.0.0')