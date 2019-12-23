# Import the various libraries
import numpy as np
import imutils
import dlib
import cv2

# This list specifies how each points are connected to form Delaunay triangles. Obtained experimentally.
# for e.g., [1,36,41] means the landmark point numbers 2,37 and 42 are triangulated.
triangleList = [[1, 36, 41], [36, 1, 0], [1, 71, 0], [71, 1, 2], [0, 71, 17], [31, 2, 29], [2, 31, 3], [71, 68, 17], [10, 54, 11], [54, 10, 55], [71, 3, 73], [3, 71, 2], [2, 1, 41], [27, 21, 22], [21, 27, 39], [73, 3, 4], [33, 50, 32], [50, 33, 51], [73, 5, 6], [5, 73, 4], [4, 3, 48], [31, 29, 30], [73, 7, 74], [7, 73, 6], [5, 4, 48], [58, 62, 57], [62, 58, 61], [6, 5, 59], [40, 29, 41], [29, 40, 28], [74, 7, 8], [7, 6, 57], [44, 25, 45], [25, 44, 24], [74, 8, 9], [8, 7, 56], [74, 11, 75], [11, 74, 10], [9, 8, 56], [75, 11, 12], [9, 10, 74], [10, 9, 55], [42, 29, 28], [29, 42, 35], [30, 34, 33], [34, 30, 35], [72, 75, 13], [12, 11, 54], [40, 37, 38], [37, 40, 41], [13, 75, 12], [13, 12, 54], [37, 18, 19], [18, 37, 36], [72, 13, 14], [14, 13, 54], [37, 19, 20], [70, 72, 16], [14, 15, 72], [15, 14, 45], [36, 17, 18], [17, 36, 0], [16, 72, 15], [16, 15, 26], [69, 24, 23], [24, 69, 25], [17, 68, 18], [42, 22, 23], [22, 42, 27], [18, 68, 19], [27, 42, 28], [69, 20, 68], [20, 69, 23], [23, 22, 20], [20, 19, 68], [20, 21, 38], [21, 20, 22], [29, 35, 30], [42, 23, 43], [46, 44, 45], [44, 46, 47], [45, 14, 46], [23, 24, 43], [70, 25, 69], [25, 70, 26], [26, 15, 45], [25, 26, 45], [16, 26, 70], [29, 2, 41], [27, 28, 39], [49, 32, 50], [32, 49, 31], [34, 52, 33], [52, 34, 53], [32, 31, 30], [48, 3, 31], [6, 59, 58], [32, 30, 33], [57, 62, 63], [63, 53, 55], [53, 63, 65], [54, 46, 14], [46, 54, 35], [34, 35, 53], [44, 47, 43], [37, 20, 38], [40, 38, 39], [36, 37, 41], [28, 40, 39], [38, 21, 39], [35, 42, 47], [42, 43, 47], [43, 24, 44], [46, 35, 47], [48, 31, 49], [5, 48, 59], [48, 49, 60], [59, 67, 61], [67, 59, 49], [49, 50, 67], [51, 33, 52], [7, 57, 56], [50, 51, 61], [63, 55, 56], [51, 52, 65], [54, 55, 64], [52, 53, 65], [53, 35, 64], [64, 35, 54], [9, 56, 55], [55, 53, 64], [62, 65, 63], [65, 62, 51], [61, 58, 59], [56, 57, 63], [57, 6, 58], [49, 59, 60], [59, 48, 60], [50, 61, 67], [61, 51, 62]]
# This constant specifies the impact of Actor points on the Portrait image i.e., portrait movement is a fraction of movement of actor
k = 0.3

# Convert the detected landmark points to a Numpy array
# Based on imutils.shape_to_np() function by Adrian Rosebrock
def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# In addition to landmark points, a box bounding the landmark points is also required for better warping to include hair and neck. 
# This function adds the corners of bounding box to the landmark points array
def addBoxPoints(imageCopy,arrayToAdd):
    # determines the bounding box for all landmark points
    x,y,w,h = cv2.boundingRect(np.float32(arrayToAdd))
    # Since several parts of face are not detected, this offsets the rectangle to include head, ears and neck
    headTop = 80
    headRight = 50
    x1 = x-int(headRight/2)
    y1 = y-headTop
    x2 = x + w + int(headRight/2)
    y2 = y + h + int(headTop/2)
    midx = int((x1+x2)/2)
    midy = int((y1+y2)/2)
    # Creates the offseted points of rectangle
    rectPoints = [(x1,y1),(midx,y1),(x2,y1),(x1,midy),(x2,midy),(x1,y2),(midx,y2),(x2,y2)]
    # appends the offseted points to the landmark array
    for pt in rectPoints:
        arrayToAdd.append(list(pt))
        
# Using the list of Delaunay triangle points, all the landmark points are triangulated
# This function takes three landmark points at a time, triangulates and finds the coordinates of the triangle.
# The list of all triangle points is returned
def drawDelaunay(img, delaunay_color, landmarkArray) :
    trianglePoints=[]
    # triangle list holds the indices of all landmark array points that are to be triangulated
    # for e.g., [1,36,41] means the landmark point numbers 2,37 and 42 are triangulated.
    for t in triangleList :
        # Find the coordinates of triangle corresponding to each landmarkArray index
        pt1 = tuple(landmarkArray[t[0]])
        pt2 = tuple(landmarkArray[t[1]])
        pt3 = tuple(landmarkArray[t[2]])
        # draw the triangulations on the copy of image passed
        cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
        trianglePoints.append([pt1,pt2,pt3])
    # return all the triangulated coordinate points
    return trianglePoints

# Warps a given triangle from source to triangle at destination
# Since cv2.warpAffine() does not allow triangles to be warped properly, a bounding box for triangle is used instead to warp
# and then extract the triangular region.
def warpTriangle(src, dest, t1, t2) :
    # Creates a bounding box for src and dest image
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    srcRect = []
    destRect = []
    # offsets each triangle points as they'll be cropped at the end
    for i in range(0, 3):
        srcRect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        destRect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
    # create a mask for cropping the warped image
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(destRect), (1.0, 1.0, 1.0), 16, 0)
    # crop a subregion of the src image for warping
    srcCropped = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # Apply affine transform from src triangle to dest triangle on the cropped sub-image
    warpImage = affineTransform(srcCropped, srcRect, destRect, (r2[2], r2[3]))
    # Apply mask to dest image and place the warped part over the dest image
    dest[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dest[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + warpImage * mask

# The input source image is warped such that source triangle is affine transformed to the points of destination triangle
# Bilinear interpolation is used for intensities
def affineTransform(src, srcTri, dstTri, size) :
    warp = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    dst = cv2.warpAffine( src, warp, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    # returns the warped image
    return dst

# Face-detector instantiation
detector = dlib.get_frontal_face_detector()
# Predictor instantiation
# The input file contains the pre-trained network for detecting the 68 landmark points
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Retrieve the Mona Lisa Portrait Image
portraitImage = cv2.imread("Mona_Lisa.jpg")
# Crop a section of image
portraitImage = portraitImage[200:2200,750:2100]
# Resize the image for lesser resolution (faster computation)
portraitImage = cv2.resize(portraitImage,(300,400))
# Blur the image to decrease the details
portraitImage = cv2.blur(portraitImage, (3,3))
# Keep a copy of original image
origPortrait = portraitImage.copy()
# Convert to grayscale
grayPortrait = cv2.cvtColor(portraitImage, cv2.COLOR_BGR2GRAY)
# Apply face-detection on gray image and find the box for the detected face part
rects = detector(grayPortrait, 1)
# Find the landmark points for the region detected by face-detection
facepPts = predictor(grayPortrait, rects[0])
# convert the detected array to a numpy array
facepPts = shape_to_np(facepPts)

portraitLandmarkArr = [] # Holds the landmark points of portrait image
# Create a white background image to show the landmark points of portrait
portraitLandmark = np.zeros((grayPortrait.shape[0], grayPortrait.shape[1],3), dtype="uint8")
portraitLandmark.fill(255)
# Append all the landmark points detected to the portrait landmark array
for (x, y) in facepPts:
    portraitLandmarkArr.append([x,y])
# Adds additional bounding box points to landmark array
addBoxPoints(portraitImage.copy(),portraitLandmarkArr)
# Triangulates the portrait image using landmark points
prevTrianglePts = drawDelaunay(portraitImage.copy(), (255, 255, 255), portraitLandmarkArr)
# Open the web-cam
cap= cv2.VideoCapture(0)
# Initial Flag. To skip first step in warping as no movement in actor image is achieved at this point
initFlag = 0

# Until the web-cam is opened
while(cap.isOpened()):
    # Read the captured frame
    ret, frame = cap.read()
    # Return from loop if no frame is detected
    if ret == False:
        print('returned')
        break
    # Resize the frame for lesser computations
    frame = cv2.resize(frame, (500,400))
    # convert to grayscale image
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create a white background image to show the landmark points of actor
    actorLandmarkImg = np.zeros((grayFrame.shape[0], grayFrame.shape[1],3), dtype="uint8")
    actorLandmarkImg.fill(255)
    # Apply face-detection on gray actor image and find the box for the detected face part
    rects = detector(grayFrame, 1)
    # if no face is detected, skip the loop
    if len(rects) == 0:
        continue
    # Find the landmark points for the region detected by face-detection
    facePts = predictor(grayFrame, rects[0])
    # convert the detected array to a numpy array
    facePts = shape_to_np(facePts)
    actorLandmarkArr = [] # Holds the landmark points of actor image
    # create a copy of the portrait landmark image
    copyPortraitLandmark = portraitLandmark.copy()
    # Append all the landmark points detected to the actor landmark array
    for (x, y) in facePts:
        actorLandmarkArr.append([x,y])
    # Adds additional bounding box points to landmark array
    addBoxPoints(frame.copy(),actorLandmarkArr)
    
    # for all the landmark points
    for i in range(0,len(portraitLandmarkArr)):
        # Moves the portrait landmark points a fraction (k) of the actor landmark points movement
        if initFlag == 1: # Skips first step as no movement will be achieved
            portraitLandmarkArr[i][0] += int(k*(actorLandmarkArr[i][0] - prevActorLandmarkArr[i][0]))
            portraitLandmarkArr[i][1] += int(k*(actorLandmarkArr[i][1] - prevActorLandmarkArr[i][1]))
        # plot the updated landmark Points on corresponding images
        cv2.circle(frame, tuple(actorLandmarkArr[i]), 1, (0, 0, 255), -1)
        cv2.circle(actorLandmarkImg, tuple(actorLandmarkArr[i]), 1, (0, 0, 255), -1)
        cv2.circle(copyPortraitLandmark, tuple(portraitLandmarkArr[i]), 1, (0, 0, 255), -1)
        
    prevActorLandmarkArr = actorLandmarkArr # holds previous actor landmark locations
    # Skip displaying images if it is the first loop and set the intial flag to 1.
    if initFlag == 0:
        initFlag = 1
        continue 
        
    copyPortraitTriangles = portraitImage.copy()
    # Triangulates the portrait image using landmark points. This is a new triangulation to be warped
    newTrianglePts = drawDelaunay(copyPortraitTriangles, (255, 255, 255),portraitLandmarkArr)
    prevPortraitImg = portraitImage.copy()
    # Warp the previous portrait image from each old to new triangular point coordinates
    for t1,t2 in zip(prevTrianglePts,newTrianglePts):
        warpTriangle(prevPortraitImg, portraitImage, t1, t2)
    prevTrianglePts = newTrianglePts # store the list of old triangular points
    
    # Display all the images in sequence to form a video
    cv2.imshow("Actor Video", frame)
    cv2.imshow("Actor Landmark Image",actorLandmarkImg)
    cv2.imshow("Portrait Landmark Image",copyPortraitLandmark)
    cv2.imshow("Delaunay Triangulation of Portrait Image",copyPortraitTriangles)
    cv2.imshow("Final Portrait Video",portraitImage)
    
    # Run the loop till space key is hit
    keyPress = cv2.waitKey(5)&0xFF
    if keyPress == 32:
        # Release all the frames and web-cam when space key is hit
        cap.release()
        cv2.destroyAllWindows()
        break

