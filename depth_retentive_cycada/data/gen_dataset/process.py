import torch
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt

def center_crop(img, dim):
    """Returns center cropped image

    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]  #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
  
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]

    return crop_img

def generate_dataset(type="dr_cycada",image_dir="./raw/input_images", depth_dir="./raw/depth_maps", dim_control=[]):
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
        fnm = "depth_" + file.split(".")[0] + ".txt"
        depth = np.genfromtxt(depth_dir + "/" + fnm, delimiter=" ")

        image = center_crop(cv2.imread(image_dir + "/" + file), (128,128))
        depth = center_crop(depth, (128,128))
        depth = depth / np.max(depth)

        


        #transformations
        t1 = cv2.convertScaleAbs(image, alpha=dim_control[0],beta=dim_control[1])
        
        cv2.imshow("target", t1)
        cv2.imshow("source", image)
        cv2.imshow("depth", depth)

        cv2.waitKey(3)

        cv2.imwrite("../habitat_dataset_"+type+"/"+mode+"A/habitat/"+str(idx)+".png",image)
        cv2.imwrite("../habitat_dataset_"+type+"/"+mode+"B/habitat/"+str(idx)+".png",t1)
        np.savetxt("../habitat_dataset_"+type+"/"+mode+"_depth_A/habitat/"+str(idx)+".txt",depth)

        if idx >=398:
            mode = "test"
            idx=-1

        idx += 1

        progress_bar.set_postfix(file=file)
        progress_bar.update(1)


if __name__ == "__main__":
    generate_dataset(type="dr_cycada",dim_control=[0.15,0.01])