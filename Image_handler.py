#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


###### read an image ####
def read(img_path):
   return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


##### save an image ###
def save(img, img_path="output.jpeg"):
   cv2.imwrite(img_path, img)


### get a list of all the images in a folder ####
# returns (list of images, list of names)
def list_images(folder_path):
    result = []

    # get list of names of all the images in the folder
    file_names = os.listdir(folder_path)

    # read each image as numpy aaray
    for img_name in file_names:
        # full path of the img = folder_path + image name
        image_path = os.path.join(folder_path, img_name)

        # Read the image and add it to the list
        img = read(image_path)
        if img is not None:
            result.append(read(image_path))

    return (result, file_names)


import os
import glob

#delete all files in a folder
def delete_all_files(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


# Display image
def display(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Remove axes
    plt.show()

#compress image
  # reduce img from size (r1, c1) to (r2, c2)
  # every (r1/r2)th row and every (c1/c2)th vol of the original img shld be copied to the compressed img
def compress_img(img, r1, c1, r2, c2):
    compressed_img = img[::r1 // r2, ::c1 // c2]
    return compressed_img


# (h_old, w_old) to (h_new, w_new)
def crop(img, r_start, num_rows, c_start, num_cols):
    cropped_img = img[r_start:r_start + num_rows, c_start: c_start + num_cols :]
    return cropped_img


#create a composite image from a list of images
# images = list of images
def get_composite(images, weights=[], sum_only=True):
    # Ensure all images have the same shape
    image_shape = images[0].shape
    for img in images:
        if img.shape != image_shape:
            raise ValueError("All images must have the same shape")

    # Initialize an array to store the composite image
    composite_image = np.zeros_like(images[0], dtype=np.float32)

    n = len(images)
    # if weights value has not been passed
    if len(weights)==0:
        weights = [1 for _ in range(n)]

    if len(weights) != n:
        print("No of images and weights should be same")
        return -1

    # Sum the pixel values from all images
    for i in range(n):
        img = images[i]
        w = weights[i]
        composite_image = composite_image + w*img.astype(np.float32)
        np.clip(composite_image, 0, 255)

    if sum_only == False:
        composite_image = (composite_image / len(images)) #.astype(np.uint8)
    
    return composite_image.astype(np.uint8)

# gaussian blurr
def blurr(img, kernel_shape=(3,3)):
    return cv2.GaussianBlur(img, ksize=kernel_shape, sigmaX=0)