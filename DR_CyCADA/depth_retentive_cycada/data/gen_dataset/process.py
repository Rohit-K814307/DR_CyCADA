import torch
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm.auto import tqdm
from PIL import Image


def generate_dataset(type="dr_cycada",image_dir="./raw/input_images", dim_control=[]):
    file_list = os.listdir(image_dir)
    progress_bar = tqdm(file_list, desc="Processing files", unit="file")
    idx = 0

    mode="train"

    #setup directories
    # parent_path = ".."
    # os.makedirs(os.path.join(parent_path,"habitat_dataset_"+type,"testA","habitat"),0o666)
    # os.makedirs(os.path.join(parent_path,"habitat_dataset_"+type,"testB","habitat"),0o666)
    # os.makedirs(os.path.join(parent_path,"habitat_dataset_"+type,"trainA","habitat"),0o666)
    # os.makedirs(os.path.join(parent_path,"habitat_dataset_"+type,"trainB","habitat"),0o666)

    for file in progress_bar:
         #inputs
        image = cv2.imread(image_dir + "/" + file)

        #transformations
        t1 = cv2.convertScaleAbs(image, alpha=dim_control[0],beta=dim_control[1])
        
        cv2.imshow("dimmed", t1)
        cv2.waitKey(1)

        cv2.imwrite("../habitat_dataset_"+type+"/"+mode+"A/habitat/"+str(idx)+".png",image)
        cv2.imwrite("../habitat_dataset_"+type+"/"+mode+"B/habitat/"+str(idx)+".png",t1)

        if idx >=398:
            mode = "test"
            idx=-1

        idx += 1

        progress_bar.set_postfix(file=file)
        progress_bar.update(1)


if __name__ == "__main__":
    generate_dataset(type="dr_cycada",dim_control=[0.25,0.2])