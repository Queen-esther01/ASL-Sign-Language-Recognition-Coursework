import glob
import os
import uuid
import cv2 as cv
from pathlib import Path


class DataLoader:
    def __init__(self, path):
        self.dataset_path = path

    def parse_filename(self, file):
        '''Returns an class and number extracted from file name'''
        stemFilename = (Path(os.path.basename(file)).stem)
        filename = stemFilename.split('_')
        return filename[0], filename[2] #returns letter, number

    def load_dataset(self):
        '''returns all the labels for images'''
        images = []
        labels = []
        for file in sorted(glob.glob(self.dataset_path)):
            for item in sorted(glob.glob(file + '/*')):
                print(item.split('/')[-1])
                grayscale = cv.imread(item, cv.IMREAD_GRAYSCALE)
                if(grayscale.shape == (256, 256)): # Filter out images that are the wrong dimension.
                    label, _ = self.parse_filename(item)
                    labels.append(label)
                    images.append(grayscale)
        print(f'loaded {len(images)} images')
        print(f'image {images[0].ndim}, {images[0].shape}')
        print(f'labels {len(labels)}')
     

    def get_class_names(self):
        '''Return the classnames A - J'''
        classnames = []
        for file in sorted(glob.glob(self.dataset_path)):
            stemFilename = (Path(os.path.basename(file)).stem)
            classnames.append(stemFilename)
        return classnames

    def get_total_count(self):
        '''Return the total number of images in the dataset'''
        count = 0
        per_character = {}
        for file in sorted(glob.glob(self.dataset_path)):
            per = 0
            for item in sorted(glob.glob(file + '/*')):
                label, _ = self.parse_filename(item)
                count += 1
                if label in per_character:
                    per_character[label] += 1
                else:
                    per_character[label] = 1
        return count, per_character

if __name__ == '__main__':
    dataloader = DataLoader('data/CW2_dataset_final/*')
    dataloader.load_dataset()
    # print(dataloader.get_total_count())
    # print(dataloader.parse_filename('data/CW2_dataset_final/A/A_sample_1.jpg'))
    # dataloader.get_class_names()
