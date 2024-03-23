import librosa
import sys, os, os.path
from os.path import isfile, join
from pathlib import Path
import glob 
import csv 
import wave

wavnames = [] # list with all of the audiofile's names 
wavsamples = [] # list of lists with all of the audiofile's sample values 
wavsamplerates = [] # list with all of the audiofile's samplerates (default: 44,1 kHz)
path = '/Users/abc/Desktop/WAV_Folder' # folder with all the data to put inside the dataset
pathlist = Path(path).glob('**/*.wav')

def sampled_audiofile(audiofile):
    list_audiosamples_for_one_file = []
    y,sr = librosa.load(audiofile,sr=44100)
    list_audiosamples_for_one_file.append(y)
    return list_audiosamples_for_one_file

for path in pathlist:
    wavnames += pathlist
    path_in_str = str(path)
    wavdata = sampled_audiofile(path_in_str)
    wavsamples += wavdata
    with wave.open(path_in_str, "rb") as wave_file:
        samplerate = []
        value = wave_file.getframerate()
        samplerate.append(value)
        wavsamplerates += samplerate 

