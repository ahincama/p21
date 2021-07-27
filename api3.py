#!/usr/bin/python
from flask import Flask, request
from m09_model_P22 import predict_price

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def preciopredict():
    return {
         "result": predict_price(request.args.get('feat'))
        }, 200



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)

