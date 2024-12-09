#import libraries
import cv2
import numpy as np

# Non Local Means denoising
def anlm(img, h=10, window=20, patch=5):
    result = cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=patch, searchWindowSize=window)

    return result

# Extract Inverse image from input image
# hyper parameters
    # img : input image
    # box : (w, h) for patch-wise processing
    # alpha : lower alpha value retains more features, but may result in noise being retained too. Suggested value is 2
def get_inverse(img, box=(2,6), alpha=2):
    # shape of input image
    rows, cols = img.shape

    #the result array  #same shape as input
    result = img.copy()

    # get width and height of the patch (box)
    box_width, box_height = box[1], box[0]

    global_avg = np.mean(img) 

    # process the image one patch at a time
    for y in range(0, rows, box_height):
        for x in range(0, cols, box_width):
            # Extract the patch
            patch = img[y:y+box_height, x:x+box_width]

            # Calculate the average patch value
            local_avg = np.mean(patch[:,:])

            # change value of all pixels in the patch a/c to the equation
            # new_val = old_val * (global_mean/local_mean)^alpha
            patch[:,:] = patch[:,:]*(global_avg/local_avg)**alpha

            # update the patch in result image
            result[y:y+box_height, x:x+box_width] = patch

    np.clip(result, 0, 255)
    return result


# extract features (texture) from noisy image
# hyper parameters
    # img : input image
    # box : (w, h) for patch-wise processing
    # alpha : lower alpha value retains more features, but may result in noise being retained too. Suggested value is 2
def inverse_subtraction(img, box=(2,6), alpha=2):
    d = anlm(img) # prelimnary denoising
    inverse = get_inverse(d, box=box, alpha=alpha)
    result = img.copy()

    rows, cols = img.shape
    for r in range(rows):
        for c in range(cols):
            new_val = int(img[r][c]) - int(inverse[r][c])

            #clip
            new_val = min(new_val, 255)
            new_val = max(new_val, 0)
            result[r][c] = new_val


    return result