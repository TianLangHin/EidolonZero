"""
    #I'm thinking (miku miku) of making this using a VAE first then we can decide to make it using GAN and compare???


    #Need a softmax at the end for probability :thumbsup:

    Things to do:
        - Test the structure first***
        - Find a way to insert data for training/testing,
        - Construct training loop and needed elements    
        - Padding might not be needed because being in at the edge of the board does mean something
"""


"""
    Summary of VAE architecture (defogger):
        - Take in a 8x8x13 tensor, encode, get the latent sapce mean and the logvar of the latent space
        - Use these 2 values to get the latent vector
        - Decode using the latent vector

    Input and Output:
        - Input: Fogged board
        - Output: Defogged (predicted) board
    As long as fogged and defogged board has the same input shape, this architecture will eventually work without modification

    Note: Need to test different latent dim (start with either 32 or 64 and slowly increase it)
"""

from boards import MaterialCounter, tensor_to_position

import chess
import torch 
from torch import nn
import numpy as np

#Well I am assuming that we are using 8x8x13 for both full/partial state
"""
VAE class with structure and encode/decode function.
"""
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        #Encode component: I'm testing with 512 because what else :<
        self.fc_encodeLayer = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels= 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            #Might change the stride because th9is might take a long time with 128 and 256?            
            nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels= 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Conv2d(256, latent_dim, kernel_size=3, stride=1, padding=1)
        self.fc_logvar = nn.Conv2d(256, latent_dim, kernel_size=3, stride=1, padding=1)


        #can have output_padding for the conv layers, the stride must be bigger than it tho 
        #Decode part: Reconstructing the output:
        self.fc_decodeLayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels= 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            #Might change the stride because th9is might take a long time with 128 and 256?            
            nn.ConvTranspose2d(in_channels=256, out_channels= 128, kernel_size=3, stride=1, padding=1), #output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels= 64, kernel_size=3, stride=1, padding=1), #output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels= 13, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=0)
        )
        

    def encodeBoard(self, inputBoard):
        inputBoard = self.fc_encodeLayer(inputBoard)

        #Mean of the latent space
        latent_mean = self.fc_mu(inputBoard)
        logvar = self.fc_logvar(inputBoard)

        return latent_mean, logvar
    

    def reparameterise(self, mu, logvar):
        std_dev = torch.exp(0.5 * logvar)
        
        """
        A sample for randomness/noise for sampling process (`epsilon` bro)
         -> more diverse outputs + enhance backpropagation with the conv layers
         """
        eps = torch.randn_like(std_dev)

        #Return the actual latent vector 
        return mu + eps * std_dev
        

    def decodeBoard(self, latent_vector):
        board = self.fc_decodeLayer(latent_vector)
        return board
    

    def forward(self, inputBoard):
        latent_mean, logvar = self.encodeBoard(inputBoard)
        latent_vector = self.reparameterise(latent_mean, logvar)
        board = self.decodeBoard(latent_vector)

        return board, latent_mean, logvar
    

def most_likely_predicted_state(
    vae_output: torch.Tensor,
    original_board_tensor: torch.Tensor,
    material: MaterialCounter) -> chess.Board:

    # We take the original board as a template.
    # We even preserve P2 piece observations too, since they are definite.
    board_tensor = original_board_tensor[:,:,:]

    # We extract out the P2 pieces,
    # and only consider places that are not visible.
    p2_pieces = vae_output[6:12,:,:] * (1 - original_board_tensor[12])

    # We figure out which turn we have.
    if (original_board_tensor[13] == 0).all().item():
        piece_type_list = [
            material.black_pawns,
            material.black_knights,
            material.black_bishops,
            material.black_rooks,
            material.black_queens,
            material.black_kings,
        ]
    else:
        piece_type_list = [
            material.white_pawns,
            material.white_knights,
            material.white_bishops,
            material.white_rooks,
            material.white_queens,
            material.white_kings,
        ]

    # We start finding the most likely square for king, queen, rook, etc.
    # Each time we predict the presence of a piece at some square,
    # that square can no longer be used for predicting other types of pieces.
    for stack_index, piece_count in enumerate(reversed(piece_type_list)):
        indices = torch.topk(p2_pieces[stack_index].flatten(), piece_count).indices
        for i in indices:
            square_rank, square_file = divmod(i.item(), 8)
            board_tensor[11 - stack_index, square_rank, square_file] = 1
            p2_pieces[:, square_rank, square_file] = 0

    return tensor_to_position(board_tensor)

