# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2

# neighbors=8, line -labels = measure.label(thresh, neighbours = 8 , background=0)


# load the image,  
image = cv2.imread('sample.jpg', 1)

# convert it to grayscale, and blur it
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# threshold the image to reveal light regions in the blurred image
thresh = cv2.threshold(gray_image, 225, 255, cv2.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

# perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
labels = measure.label(thresh,  background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):

	# if this is the background label, ignore it
    if label == 0:
        continue

	# otherwise, construct the label mask and count the number of pixels 
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)

	# if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
    if numPixels > 300:
        mask = cv2.add(mask, labelMask)
	
# find the contours in the mask, then sort them from left to right
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours.sort_contours(contours)[0]

contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# loop over the contours

# Initialize lists to store centroid coordinates and area
centroids = []
areas = []

# Loop over the contours
for i, contour in enumerate(contours):
    
    area = cv2.contourArea(contour)

    # Calculate the area of the contour
    M = cv2.moments(contour)
    cX = float(M["m10"] / M["m00"])
    cY = float(M["m01"] / M["m00"])
    centroid = (cX, cY)
    

    # Draw the bright spot on the image
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    # cv2.circle(image, centroid, 10, (0, 0, 255), -1)


    # Append centroid coordinates and area to the respective lists
    centroids.append(centroid)
    areas.append(area)

# Save the output image as a PNG file
cv2.imwrite("led_detection_results_sample.png", image)

# Open a text file for writing
with open("led_detection_results_sample.txt", "w") as file:
    # Write the number of LEDs detected to the file
    # file.write(f"No. of LEDs detected: {a}\n")
    file.write(f"No. of LEDs detected: {len(contours)}\n")
    # Loop over the contours
    
        # Write centroid coordinates and area for each LED to the file
    file.write(f"Centroid #{i + 1}: {centroid}\nArea #{i + 1}: {area}\n")
# Close the text file
file.close()
