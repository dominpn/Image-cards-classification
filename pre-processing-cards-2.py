from os import listdir
from shutil import copyfile
from random import seed
from random import random

figures = ['ace', 'jack', 'king', 'queen', 'nine', 'ten']

dataset_home = 'playing-card-ml'
for figure in figures:
    seed(1)
    val_ratio = 0.25
    src_directory = 'cards/'+figure+'/'
    for file in listdir(src_directory):
        src = src_directory + '/' + file
        dst_dir = 'train/'
        if random() < val_ratio:
            dst_dir = 'test/'
        dst = dataset_home + '/' + dst_dir + figure + '/' + file
        copyfile(src, dst)
