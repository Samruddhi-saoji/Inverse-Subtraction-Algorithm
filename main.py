#import libraries
import cv2
from Image_handler import read, save, get_composite
from CNR_calculator import calculate_cnr 
from Inverse_subtraction_algorithm import inverse_subtraction

# read the noisy image as numpy array
img = read("path to image") # grayscale 

# extract features from the image
texture = inverse_subtraction(img, box=(2,5), alpha=2) 
save(texture, "Texture.jpg")

# denoise the input image # Gaussian blur
d = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=0)
save(d, "Denoised.jpg")

# add texture to the denoised image
dt = get_composite([d, texture], [1, 1], sum_only=False)
save(dt, "Denoised_texture.jpg")

# compare the CNR of the 2 images
cnr_d = calculate_cnr(d)
cnr_dt = calculate_cnr(dt)

print("CNR value of denoised image = ", cnr_d)
print("CNR value when texture is added to the denoised image = ", cnr_dt)