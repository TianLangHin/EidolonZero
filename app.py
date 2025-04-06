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

# Function stubs for mocking main workflow API
@app.route('/getfoggedstate/stub')
def stub_getfoggedstate():
    fen = request.args.get('fen')
    invert_move = request.args.get('invert')
    return jsonify(api.stub_getfoggedstate(fen, invert_move))

@app.route('/inference/stub')
def stub_inference():
    fen = request.args.get('fen')
    args_translate = {
        'wp': 'white_pawns',
        'bp': 'black_pawns',
        'wn': 'white_knights',
        'bn': 'black_knights',
        'wb': 'white_bishops',
        'bb': 'black_bishops',
        'wr': 'white_rooks',
        'br': 'black_rooks',
        'wq': 'white_queens',
        'bq': 'black_queens',
        'wk': 'white_kings',
        'bk': 'black_kings',
    }
    material_counter = boards.MaterialCounter(**{
        args_translate[pt]: request.args.get(pt) for pt in args_translate.keys()
    })
    return jsonify(api.stub_inference(fen, material_counter))

@app.route('/makemove/stub')
def stub_makemove():
    fen = request.args.get('fen')
    move = request.args.get('move')
    return jsonify(api.stub_makemove(fen, move))

if __name__ == '__main__':
    app.run()
