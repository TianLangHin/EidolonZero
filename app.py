from flask import Flask, jsonify, request
from flask_cors import CORS

import api
from architectures import ConvNet, PuctConfig, VAE
import boards
from evaluation import PlayConfig
import numpy as np
import os
import random
import torch

ARGS_TRANSLATE = {
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

SEED = 19937
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CONFIG = PlayConfig(
    puct_config=PuctConfig(
        c_puct=0.5,
        dirichlet_alpha=0.15,
        epsilon=0.25,
        move_limit=128,
    ),
    possibilities=1,
    simulations=200,
)

CONVNET = ConvNet()
VAE = VAE(512)
CONVNET.load_state_dict(torch.load(
    os.path.join(os.getcwd(), 'models', 'convnet-dirichletalpha0.15-12.pt'),
    weights_only=True
))
VAE.load_state_dict(torch.load(
    os.path.join(os.getcwd(), 'models', 'vae-dirichletalpha0.15-12.pt'),
    weights_only=True
))
CONVNET.eval()
VAE.eval()

EIDOLONZERO_MODEL = (CONVNET, VAE)

app = Flask(__name__)
CORS(app)

@app.route('/')
def root():
    return jsonify({
        'eidolon-zero': True
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

@app.route('/getfoggedstate')
@app.route('/getfoggedstate/stub')
def getfoggedstate():
    fen = request.args.get('fen')
    invert_move = request.args.get('invert')
    return jsonify(api.getfoggedstate(fen, invert_move))

@app.route('/inference/stub')
def stub_inference():
    fen = request.args.get('fen')
    material_counter = boards.MaterialCounter(**{
        ARGS_TRANSLATE[pt]: int(request.args.get(pt)) for pt in ARGS_TRANSLATE.keys()
    })
    return jsonify(api.stub_inference(fen, material_counter))

@app.route('/inference')
def inference():
    fen = request.args.get('fen')
    material_counter = boards.MaterialCounter(**{
        ARGS_TRANSLATE[pt]: int(request.args.get(pt)) for pt in ARGS_TRANSLATE.keys()
    })
    return jsonify(api.inference(fen, material_counter, CONFIG, EIDOLONZERO_MODEL))

@app.route('/makemove')
@app.route('/makemove/stub')
def makemove():
    fen = request.args.get('fen')
    move = request.args.get('move')
    return jsonify(api.makemove(fen, move))

if __name__ == '__main__':
    app.run()
