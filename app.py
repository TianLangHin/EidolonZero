from flask import Flask, after_this_request, jsonify, request
from flask_cors import CORS

import api

app = Flask(__name__)
CORS(app)

@app.route('/')
def root():
    return jsonify({
        'text': 'Hello world!',
        'misc': 'other stuff'
    })

@app.route('/testlegalmoves')
def test_legal_moves():
    fen = request.args.get('fen')
    return jsonify({
        'moves': api.legal_chess_moves_from_fen(fen),
    })

if __name__ == '__main__':
    app.run()
