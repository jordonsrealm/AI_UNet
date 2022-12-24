import cv2
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist


def midpoint(pt_a, pt_b):
    return (pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5
# Load in image, convert to gray scale, and Otsu's threshold


def cca_analysis(orig_image, predict_image, erode_iteration, open_iteration):
    kernel1 = (np.ones((5, 5), dtype=np.float32))
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                 [-1, -1, -1]])
    image = predict_image
    image2 = orig_image
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1, iterations=open_iteration)
    image = cv2.filter2D(image, -1, kernel_sharpening)
    image = cv2.erode(image, kernel1, iterations = erode_iteration)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    labels = cv2.connectedComponents(thresh, connectivity=8)[1]
    a = np.unique(labels)
    count2 = 0
    for label in a:
        if label == 0:
            continue
    
        # Create a mask
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255
        # Find contours and determine contour area
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        c_area = cv2.contourArea(contours)
        # threshold for tooth count
        if c_area > 2000:
            count2 += 1
        
        (x, y), radius = cv2.minEnclosingCircle(contours)
        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")    
        box = perspective.order_points(box)
        color1 = (list(np.random.choice(range(150), size=3)))  
        color = [int(color1[0]), int(color1[1]), int(color1[2])]
        cv2.drawContours(image2, [box.astype("int")], 0, color, 2)
        (tl, tr, br, bl) = box
        
        (top_left_top_right_x, top_left_top_right_y) = midpoint(tl, tr)
        (bottom_left_bottom_right_x, bottom_left_bottom_right_y) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-right and bottom-right
        (top_left_bottom_left_x, top_left_bottom_left_y) = midpoint(tl, bl)
        (top_right_bottom_right_x, top_right_bottom_right_y) = midpoint(tr, br)
        # draw the midpoints on the image
        cv2.circle(image2, (int(top_left_top_right_x), int(top_left_top_right_y)), 5, (255, 0, 0), -1)
        cv2.circle(image2, (int(bottom_left_bottom_right_x), int(bottom_left_bottom_right_y)), 5, (255, 0, 0), -1)
        cv2.circle(image2, (int(top_left_bottom_left_x), int(top_left_bottom_left_y)), 5, (255, 0, 0), -1)
        cv2.circle(image2, (int(top_right_bottom_right_x), int(top_right_bottom_right_y)), 5, (255, 0, 0), -1)
        cv2.line(image2, (int(top_left_top_right_x), int(top_left_top_right_y)),
                 (int(bottom_left_bottom_right_x), int(bottom_left_bottom_right_y)), color, 2)
        cv2.line(image2, (int(top_left_bottom_left_x), int(top_left_bottom_left_y)),
                 (int(top_right_bottom_right_x), int(top_right_bottom_right_y)), color, 2)
        distance_a = dist.euclidean((top_left_top_right_x, top_left_top_right_y), (bottom_left_bottom_right_x,
                                                                                   bottom_left_bottom_right_y))
        distance_b = dist.euclidean((top_left_bottom_left_x, top_left_bottom_left_y), (top_right_bottom_right_x,
                                                                                       top_right_bottom_right_y))

        pixels_per_metric = 1
        dim_a = distance_a * pixels_per_metric
        dim_b = distance_b * pixels_per_metric
        cv2.putText(image2, "{:.1f}pixel".format(dim_a), (int(top_left_top_right_x - 15),
                                                          int(top_left_top_right_y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, color, 2)
        cv2.putText(image2, "{:.1f}pixel".format(dim_b), (int(top_right_bottom_right_x + 10),
                                                          int(top_right_bottom_right_y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, color, 2)
        cv2.putText(image2, "{:.1f}".format(label), (int(top_left_top_right_x - 35), int(top_left_top_right_y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 2)
    teeth_count = count2
    return image2, teeth_count
