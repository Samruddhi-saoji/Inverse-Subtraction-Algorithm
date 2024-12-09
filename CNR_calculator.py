#import libraries
import cv2
import numpy as np

#### CNR value to evaluate results ####
# manually select the background and foreground from the image
def get_bg_and_fg(image):
    # Foreground selection
    print("Select the Region of Interest (ROI).")
    roi_bbox = cv2.selectROI("Select ROI", image, showCrosshair=True)
    fg = image[int(roi_bbox[1]):int(roi_bbox[1]+roi_bbox[3]), int(roi_bbox[0]):int(roi_bbox[0]+roi_bbox[2])]
    cv2.destroyWindow("Select ROI")

    # Background selection
    print("Select the background region.")
    bkg_bbox = cv2.selectROI("Select Background", image, showCrosshair=True)
    bg = image[int(bkg_bbox[1]):int(bkg_bbox[1]+bkg_bbox[3]), int(bkg_bbox[0]):int(bkg_bbox[0]+bkg_bbox[2])]
    cv2.destroyWindow("Select Background")

    return bg, fg


def calculate_cnr(image):
    # get background and foreground (roi)
    # manual selection by user
    bg, roi = get_bg_and_fg(image)

    # Calculate the needed stats
    mean_roi = np.mean(roi)
    mean_bg = np.mean(bg)
    std_bg = np.std(bg)

    # Calculate the CNR
    cnr = np.abs(mean_roi - mean_bg) / std_bg

    return cnr
