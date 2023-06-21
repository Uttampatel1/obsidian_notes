```
# import the necessary packages
import cv2
import os
import pytesseract
from PIL import Image
import re

# set the path of the input image
image_path = "img/3.png"

# set the preprocessor to use
pre_processor = "thresh" # can be "thresh" or "blur"

# load the image
image = cv2.imread(image_path)

# Ask user to select ROI
roi = cv2.selectROI(image)

# Crop the image

image = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply pre-processing
if pre_processor == "thresh":
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif pre_processor == "blur":
gray = cv2.medianBlur(gray, 3)

# apply image denoising
gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

# apply image smoothing
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# # apply adaptive thresholding
# gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# apply morphological transformations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# add image to memory
filename = "{}.jpg".format(os.getpid())

cv2.imwrite(filename, gray)

# extract text from cropped image
text = pytesseract.image_to_string(gray, lang="eng", config="--psm 6")

# remove the temporary file
os.remove(filename)

cv2.imshow("Cropped Image", gray)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# print the extracted text
print(text)
```


reference:
- https://nanonets.com/blog/ocr-with-tesseract/#:~:text=Pytesseract%20or%20Python%2Dtesseract%20is,image%20to%20text%20use%20cases.
- https://www.geeksforgeeks.org/reading-text-from-the-image-using-tesseract/
- https://docs.openvino.ai/2022.1/notebooks/405-paddle-ocr-webcam-with-output.html#:~:text=PaddleOCR%20is%20an%20ultra%2Dlight,model%20(9.4M)%E2%80%9D.