import pickle
import cv2
import os


def image_loader(dir):
    images = []
    entries = os.listdir(dir)
    names = []
    for entry in entries:
        # if entry != '.DS_Store':
        if entry[0] != '.':
            image = cv2.imread(dir + entry)
            # image = skimage.io.imread(dir + entry)
            images.append(image)
            names.append(entry)
    return images, names


def pkl_loader(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    return data


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def img_show(image, win_name='image'):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, image)
    cv2.waitKey(0)
