from keras.models import load_model
import os
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense
from keras.layers.recurrent import LSTM
from IPython.display import Audio
from pipes import quote

import keras
print(keras.__version__)


def write_np_as_wav(X, sample_rate, file):
    # Converting the tensor back to it's original form
    Xnew = X * 32767.0
    Xnew = Xnew.astype('int16')
    # wav.write constructs the .wav file using the specified sample_rate and tensor
    wav.write(file, sample_rate, Xnew)
    return
def fft_blocks_to_time_blocks(blocks_ft_domain):
    # Time blocks initialized
    time_blocks = []
    for block in blocks_ft_domain:
        num_elems = block.shape[0] / 2
        # Extracts real part of the amplitude corresponding to the frequency
        real_chunk = block[0:num_elems]
        # Extracts imaginary part of the amplitude corresponding to the frequency
        imag_chunk = block[num_elems:]
        # Represents amplitude as a complex number corresponding to the frequency
        new_block = real_chunk + 1.0j * imag_chunk
        # Computes the one-dimensional discrete inverse Fourier Transform and returns the transformed
        # block from frequency domain to time domain
        time_block = np.fft.ifft(new_block)
        # Joins a sequence of blocks along time axis.
        time_blocks.append(time_block)
    return time_blocks
def convert_sample_blocks_to_np_audio(blocks):
    # Flattens the blocks into a single list
    song_np = np.concatenate(blocks)
    return song_np

model_path = 'model500.h5'
# returns a compiled model
# identical to the previous one
model = load_model(model_path)
x_data,y_data = np.load('YourMusicLibraryNP_xmerged.npy')
max_seq_len = 100
sample_frequency = 44100
block_size = 44100
# We take the first chunk of the training data itself for seed sequence.
tmp = np.random.randint(low = 0, high = x_data.shape[0])
seed_seq = x_data[tmp]
# Reshaping the sequence to feed to the RNN.
seed_seq = np.reshape(seed_seq, (1, seed_seq.shape[0], seed_seq.shape[1]))
# Generated song sequence is stored in output.
output = []

for it in range(max_seq_len):
    # Generates new value
    seedSeqNew = model.predict(seed_seq) 
    # Appends it to the output
    if it == 0:
        for i in range(seedSeqNew.shape[1]):
            output.append(seedSeqNew[0][i].copy())
    else:
        output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy()) 
    # newSeq contains the generated sequence.
    newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
    # Reshaping the new sequence for concatenation.
    newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
    # Appending the new sequence to the old sequence.
    seed_seq = np.concatenate((seed_seq, newSeq), axis=1)
    seed_seq = seed_seq[:,1:,:]


# The path for the generated song
song_path = 'generated_music.wav'
# Reversing the conversions
time_blocks = fft_blocks_to_time_blocks(output)
song = convert_sample_blocks_to_np_audio(time_blocks)
write_np_as_wav(song, sample_frequency, song_path)
