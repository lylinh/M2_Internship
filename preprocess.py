import tifffile as tiff
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
from sklearn.cluster import DBSCAN

mband_path = 'Multi_spectral_20220913\\Multi_4Cm_Ortho_20220913.tif'

def is_close(point1, point2, gap):
    return np.linalg.norm(np.array(point1) - np.array(point2)) < gap

def calculate_angle(line):
    x1, y1, x2, y2 = line[0]
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def group_by_angle(lines, angle_threshold=5):
    angle_clusters = []
    for line in lines:
        angle = calculate_angle(line)
        added = False
        for cluster in angle_clusters:
            if abs(calculate_angle(cluster[0]) - angle) < angle_threshold:
                cluster.append(line)
                added = True
                break
        if not added:
            angle_clusters.append([line])
    return angle_clusters

def find_longest_line(cluster):
    points = set()
    for line in cluster:
        x1, y1, x2, y2 = line[0]
        points.add((x1, y1))
        points.add((x2, y2))
    
    max_dist = 0
    longest_line = None
    for (p1, p2) in combinations(points, 2):
        dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        if dist > max_dist:
            max_dist = dist
            longest_line = [p1[0], p1[1], p2[0], p2[1]]
    
    return np.array([longest_line], dtype=np.int32)

def lines_intersect(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return False  # Lines are parallel and do not intersect
    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return min(x1, x2) <= intersect_x <= max(x1, x2) and min(x3, x4) <= intersect_x <= max(x3, x4) and \
           min(y1, y2) <= intersect_y <= max(y1, y2) and min(y3, y4) <= intersect_y <= max(y3, y4)

def merge_lines_by_angle(lines, gap=0, angle_threshold=1):
    if len(lines) <= 1:
        return lines  # Return the single line or an empty list

    # Group lines by angle
    angle_clusters = group_by_angle(lines, angle_threshold)

    # Find the cluster with the most lines
    largest_cluster = max(angle_clusters, key=len)

    return largest_cluster

def find_intersection(horizontal_line, vertical_line):
    x1, y1, x2, y2 = horizontal_line[0]
    x3, y3, x4, y4 = vertical_line[0]
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # Lines are parallel and do not intersect
    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return (int(intersect_x), int(intersect_y))

if os.path.exists(mband_path):
    try:
        tif_image = tiff.imread(mband_path)
        if tif_image is not None and tif_image.ndim == 3 and tif_image.shape[2] >= 5:
            channel_5 = tif_image[:, :, 4]
            if channel_5.dtype != np.uint8:
                channel_5 = cv2.normalize(channel_5, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            x_start, x_end = 750*2, 1750*2
            y_start, y_end = 600*2, 1900*2
            cropped_channel_5 = channel_5[y_start:y_end, x_start:x_end]

            target_color = 7
            mask = cv2.inRange(cropped_channel_5, target_color, target_color)

            scale_percent = 50
            width = int(mask.shape[1] * scale_percent / 100)
            height = int(mask.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)

            blurred_mask = cv2.GaussianBlur(resized_mask, (5, 5), 1.1)
            edges = cv2.Canny(blurred_mask, 50, 100)

            kernel = np.ones((5, 5), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)

            lines = cv2.HoughLinesP(eroded_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

            horizontal_lines = []
            vertical_lines = []
            if lines is not None:
                for line in lines:
                    if len(line[0]) == 4:  # Ensure the line has 4 values
                        x1, y1, x2, y2 = line[0]
                        if abs(y2 - y1) < abs(x2 - x1):
                            horizontal_lines.append(line)
                        else:
                            vertical_lines.append(line)

            horizontal_clusters = merge_lines_by_angle(horizontal_lines, gap=50, angle_threshold=10)
            vertical_clusters = merge_lines_by_angle(vertical_lines, gap=50, angle_threshold=10)

            intersection_points = []
            for h_line in horizontal_lines:
                for v_line in vertical_lines:
                    if lines_intersect(h_line, v_line):
                        intersection_points.append(find_intersection(h_line, v_line))

            line_image = np.copy(resized_mask)
            line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)

            for i, line in enumerate(horizontal_lines):
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(line_image, f'{i}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            for i, line in enumerate(vertical_lines):
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # cv2.putText(line_image, f'{i}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            
            print(len(intersection_points))
            for point in intersection_points:
                cv2.circle(line_image, point, 5, (0, 0, 255), -1)

            plt.figure(figsize=(24, 16), num=2)
            plt.subplot(1, 2, 1)
            plt.title('Original Resized Channel 5')
            plt.imshow(resized_mask, cmap='gray')

            plt.subplot(1, 2, 2)
            plt.title('Hough Lines and Intersections')
            plt.imshow(line_image)

            plt.show()
        else:
            print("Error: The image does not have at least five channels or could not be read.")
    except Exception as e:
        print(f"Error reading the TIFF file: {e}")
else:
    print(f"Error: File '{mband_path}' does not exist.")

import gc
gc.collect()

