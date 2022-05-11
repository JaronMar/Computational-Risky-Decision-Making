import math
import statistics
from itertools import *
import numpy as np
import re
from numba import numba
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    return K.mean(y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(
        tf.math.maximum(margin - y_pred, 0.0)
    ))


def flatten(listOfLists):
    return list(chain.from_iterable(listOfLists))

@numba.njit()
def squared_euclidean_distance(A,B,sqrt=False):
  dist=np.dot(A,B.T)

  TMP_A=np.empty(A.shape[0],dtype=A.dtype)
  for i in range(A.shape[0]):
    sum=0.
    for j in range(A.shape[1]):
      sum+=A[i,j]**2
    TMP_A[i]=sum

  TMP_B=np.empty(B.shape[0],dtype=A.dtype)
  for i in range(B.shape[0]):
    sum=0.
    for j in range(B.shape[1]):
      sum+=B[i,j]**2
    TMP_B[i]=sum

  if sqrt==True:
    for i in range(A.shape[0]):
      for j in range(B.shape[0]):
        dist[i,j]=np.sqrt(-2.*dist[i,j]+TMP_A[i]+TMP_B[j])
  else:
    for i in range(A.shape[0]):
      for j in range(B.shape[0]):
        dist[i,j]=-2.*dist[i,j]+TMP_A[i]+TMP_B[j]
  return dist

def average(input):
    return sum(input)/len(input) if len(input) > 0 else 0

def median(input):
    return statistics.median(input) if len(input) > 0 else 0

def standard_error(sample):
    return np.std(sample) / math.sqrt(len(sample))

def load_model(model_file):
    from tensorflow.keras.models import model_from_json
    from Cat2Vec.Attention import create_custom_objects
    json_file = open(model_file + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects=create_custom_objects())
    loaded_model.load_weights(model_file + ".h5")
    return loaded_model

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def preprocess_text(text, length_limit=0):
    from gensim.parsing.preprocessing import remove_stopwords

    special_words = re.compile(r"([a-z0-9_\-.@]+\.(com|edu|co\.uk))", flags=re.IGNORECASE)
    text = special_words.sub('', text)
    special_pattern = re.compile("[<!,'\"“@.,”?#…():‘’\[\]\|_]")
    text = special_pattern.sub('', text)
    special_pattern2 = re.compile("(-{2,}|\.{2,})")
    text = special_pattern2.sub('', text)
    special_with_space = re.compile("[\\]|[/>*]")
    text = special_with_space.sub(' ', text)
    text = remove_stopwords(text.lower())
    return [word for word in text.split() if len(word) >= length_limit]
