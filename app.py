from flask import Flask, after_this_request, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def root():
    return jsonify({
        'text': 'Hello world!',
        'misc': 'other stuff'
    })

if __name__ == '__main__':
    app.run()
