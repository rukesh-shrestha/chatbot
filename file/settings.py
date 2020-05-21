import os

#Retrieval based data
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATHH = os.path.join(ROOT,'../data/all_data.json')
CHECKPOINT_PATHH = os.path.join(ROOT,"../model/re_training/model.h5")
CHECKPOINT_DIRR = os.path.dirname(CHECKPOINT_PATHH)

#Generative based data
DATA_PATH = os.path.join(ROOT,'../data/file.csv')
CHECKPOINT_PATH = os.path.join(ROOT,"../model/training_2/cp.ckpt")
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
