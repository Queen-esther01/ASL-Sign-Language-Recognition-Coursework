import pandas as pd
import numpy as np
import ast
from data_loader import DataLoader

class Preprocessor:
    ''' 
    '''

    def __init__(self, data):
        self.data = data

    def reshape_data(self):
        data = pd.read_csv(self.data)
        
        x = np.array(data['x'].apply(ast.literal_eval).tolist())
        y = np.array(data['y'].apply(ast.literal_eval).tolist())

        # concatenate x and y arrays -> [[x1, y1], [x2, y2], ...] to [[x1, y1, x2, y2, ...], [x1, y1, x2, y2, ...], ...]
        # shape should be (n_samples, 42) -> (n_samples, 21 * 2)
        training_data = np.concatenate((x, y), axis=1)
        return training_data

if __name__ == "__main__":
    dataset_path = 'data/clean_images/*'
    clean_dataset_path = 'data/clean_dataset/data.csv'
    classes = DataLoader(dataset_path).get_class_names()
    labels = DataLoader(dataset_path).load_dataset()
    preprocessor = Preprocessor(clean_dataset_path)
    training_data = preprocessor.reshape_data()
    print(training_data.shape)