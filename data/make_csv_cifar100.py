import pandas as pd
import numpy as np
import os

import cv2
import shutil

#from tqdm import tqdm
# tqdm doesn't work well in colab.
# This is the solution:
# https://stackoverflow.com/questions/41707229/tqdm-printing-to-newline
import tqdm.notebook as tq
#for i in tq.tqdm(...):

import matplotlib.pyplot as plt

def unpickle(file):
    
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        
    return dict
  

def Load_the_train_data():
  
  path = './cifar100/train'

  train_dict = unpickle(path)

  path = './cifar100/meta'

  names_dict = unpickle(path)

  return train_dict, names_dict

if __name__ == '__main__':
  train_dict, names_dict = Load_the_train_data()
  fine_labels_list = train_dict[b'fine_labels']
  coarse_labels_list = train_dict[b'coarse_labels']

  fine_label_names_list = names_dict[b'fine_label_names']
  coarse_label_names_list = names_dict[b'coarse_label_names']

  print(len(fine_labels_list))
  print(len(coarse_labels_list))
  print(len(fine_label_names_list))
  print(len(coarse_label_names_list))

