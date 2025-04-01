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


import torch 
from torch import nn
import numpy as np
import random

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

#Well I am assuming that we are using 8x8x13 for both full/partial state
"""
VAE class with structure and encode/decode function.
"""
class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(VAE, self).__init__()

        #I dont need the input shape at all, just incase we go back to flattening
        self.input_shape = input_shape
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
            nn.Softmax()
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
        return mu + eps + std_dev
        

    def decodeBoard(self, latent_vector):
        board = self.fc_decodeLayer(latent_vector)
        return board
    

    def forward(self, inputBoard):
        latent_mean, logvar = self.encodeBoard(inputBoard)
        latent_vector = self.reparameterise(latent_mean, logvar)
        board = self.decodeBoard(latent_vector)

        return board, latent_mean, logvar
    


