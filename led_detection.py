# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2

# load the image,  
image = cv2.imread('led.jpg', 1)

# convert it to grayscale, and blur it
grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# threshold the image to reveal light regions in the blurred image
threshhold = cv2.threshold(grayscale_image,255,255,cv2.THRESH_BINARY)[1]


# perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
threshhold = cv2.erode(threshhold,None,iterations=2)
threshhold = cv2.dilate(threshhold,None,iterations=4)

# perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
labels = measure.label(threshhold, background = 0 )
mask = np.zeros(threshhold.shape, dtype= "uint8")

# loop over the unique components
for label in np.unique(labels):

	# if this is the background label, ignore it
    if label == 0:
        continue

	# otherwise, construct the label mask and count the number of pixels 
    labelmask = np.zeros(threshhold.shape, dtype="uint8")
    labelmask[labels == label] = 255
    
    numberPixels = cv2.countNonZero(labelmask)
	# if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
    if (numberPixels > 300 ):
        mask = cv2.add(mask, labelmask)
        
 
 
# find the contours in the mask, then sort them from left to right
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])



# loop over the contours

# Initialize lists to store centroid coordinates and area
centroids = []
areas = []

# Loop over the contours

for i, contour in enumerate(contours):


    # Calculate the area of the contour
    M = cv2.moments(contour)
    centriodX = float(M["m10"] / M["m00"])
    centriodY = float(M["m01"] / M["m00"])
    centroid = (centriodX,centriodY)
    
    
    # Draw the bright spot on the image


    # Append centroid coordinates and area to the respective lists

# Save the output image as a PNG file
cv2.imwrite("led_detection_results.png", image)

# Open a text file for writing
with open("led_detection_results.txt", "w") as file:
    # Write the number of LEDs detected to the file
    file.write(f"No. of LEDs detected: {a}\n")
    # Loop over the contours
    
        # Write centroid coordinates and area for each LED to the file
    file.write(f"Centroid #{i + 1}: {centroid}\nArea #{i + 1}: {area}\n")
# Close the text file
file.close()
