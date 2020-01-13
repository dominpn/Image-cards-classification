from os import makedirs
import csv
import cv2
import uuid


def prepare_data(data_type):
    dataset_home = 'playing-card-ml/'
    with open(f'playing-card/{data_type}_labels.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img = cv2.imread(f'playing-card/{data_type}/{row["filename"]}')
            crop_img = img[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])]
            path = f'{dataset_home}/{data_type}/{row["class"]}'
            makedirs(path, exist_ok=True)
            cv2.imwrite(f'{path}/{str(uuid.uuid4()).replace("-", "")}.jpg', crop_img)


prepare_data('train')
prepare_data('test')
