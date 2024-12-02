import pickle
import pandas

def data_dict(file_path):
    file = open(file_path, 'rb')
    data_d = pickle.load(file)
    file.close()
    return data_d

