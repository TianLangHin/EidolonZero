from flask import Flask, after_this_request, jsonify, request
from flask_cors import CORS

import api
import boards

app = Flask(__name__)
CORS(app)

@app.route('/')
def root():
    return jsonify({
        'text': 'Hello world!',
        'misc': 'other stuff'
    })

@app.route('/test/legalmoves')
def test_legal_moves():
    fen = request.args.get('fen')
    return jsonify({
        'moves': api.legal_chess_moves_from_fen(fen),
    })

@app.route('/test/fowmovetensor')
def test_fow_move_tensor():
    fen = request.args.get('fen')
    return jsonify({'tensor': api.fow_chess_move_tensor(fen)})

@app.route('/test/sanity/position')
def test_position_tensor_sanity():
    fen = request.args.get('fen')
    return jsonify({'result': api.sanity_check_position_tensor(fen)})

if __name__ == '__main__':
    app.run()
