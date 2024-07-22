import tifffile as tiff
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
from itertools import permutations
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, KDTree
from sklearn.cluster import KMeans
import cv2
from itertools import combinations
from sklearn.cluster import DBSCAN
import os
import tifffile as tiff



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

def calculate_area(points):
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]
    return 0.5 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - y1*x2 - y2*x3 - y3*x4 - y4*x1)


def calculate_angle_between_lines(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def is_square(points, angle_tolerance=10):
    angles = []
    for i in range(4):
        angles.append(calculate_angle_between_lines(points[i], points[(i + 1) % 4], points[(i + 2) % 4]))
    return all(90 - angle_tolerance <= angle <= 90 + angle_tolerance for angle in angles)

def calculate_center(cluster):
    return np.mean(cluster, axis=0).astype(int)

# Define a function to calculate the angle between two points relative to a center
def calculate_angle(center, point):
    # print(np.arctan2(point[1] - center[1], point[0] - center[0]))
    return np.arctan2(point[1] - center[1], point[0] - center[0])

# Define a function to check if two angles are approximately the same
def angles_approx_equal(angle1, angle2, tolerance=np.pi/180):  # 10 degrees tolerance
    # print("Diff:", np.abs(angle1 - angle2), tolerance)
    return np.abs(angle1 - angle2) < tolerance


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

            intersection_points = []
            for h_line in horizontal_lines:
                for v_line in vertical_lines:
                    if lines_intersect(h_line, v_line):
                        intersection_points.append(find_intersection(h_line, v_line))

            # Clustering intersection points
            intersection_points = np.array(intersection_points)
            db = DBSCAN(eps=10, min_samples=4).fit(intersection_points)
            labels_point = db.labels_
            unique_labels = set(labels_point)
            intersection_points_clusters = [intersection_points[labels_point == k] for k in unique_labels if k != -1]

            # Highlight clusters
            line_image = np.copy(resized_mask)
            line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
            
            cluster_centers = []
            for cluster in intersection_points_clusters:
                center = calculate_center(cluster)
                cluster_centers.append(center)
                cv2.circle(line_image, tuple(center), 5, (0, 255, 255), -1)  # Draw center in yellow

            # Convert cluster_centers to numpy array
            cluster_centers = np.array(cluster_centers)

            # Perform Delaunay triangulation
            tri = Delaunay(cluster_centers)


            # Plot the triangles and squares
            plt.figure(figsize=(24, 16), num=2)
            plt.subplot(1, 2, 1)
            plt.title('Original Resized Channel 5')
            plt.imshow(resized_mask, cmap='gray')

            plt.subplot(1, 2, 2)
            plt.title('Triangles with Common Edge and Squares Highlighted')
            plt.imshow(line_image)

            for idx, triangle in enumerate(tri.simplices):
                plt.plot(cluster_centers[triangle, 0], cluster_centers[triangle, 1], 'b-', lw=1)
                # Calculate centroid of the triangle to annotate
                centroid = np.mean(cluster_centers[triangle], axis=0)
                plt.text(centroid[0], centroid[1], str(idx), fontsize=12, ha='center', va='center', color='r')

            squares = []
            square_count = 0

            # Check if triangles with indices 14 and 23 form a square
            found_square = False
            triangles_to_remove = set()

            # Find and draw all squares formed by triangles with a common edge
            for i, triangle1 in enumerate(tri.simplices):
                for j, triangle2 in enumerate(tri.simplices):
                    if i < j:  # Ensure we're not checking the same pair twice
                        # Find common vertices between triangle1 and triangle2
                        common_vertices = set(triangle1).intersection(set(triangle2))

                        if len(common_vertices) == 2:  # Triangles share an edge
                            # Get the two common vertices and find the other two for potential square
                            common_vertices = list(common_vertices)
                            triangle1_vertices = list(set(triangle1) - set(common_vertices))
                            triangle2_vertices = list(set(triangle2) - set(common_vertices))
                            
                            square_vertices = triangle1_vertices + [common_vertices[0]] + triangle2_vertices + [common_vertices[1]]
                            
                            if is_square(cluster_centers[square_vertices]):
                                # Draw lines for the square
                                squares.append(square_vertices)
                                triangles_to_remove.update([i, j])

            # Find sets of three triangles that share edges and form squares
            for i, triangle1 in enumerate(tri.simplices):
                if i in triangles_to_remove:
                    continue
                for j, triangle2 in enumerate(tri.simplices):
                    if i != j:
                        common_vertices_1_2 = set(triangle1).intersection(set(triangle2))
                        if len(common_vertices_1_2) == 2:
                            for k, triangle3 in enumerate(tri.simplices):
                                if k != i and k != j:
                                    common_vertices_1_3 = set(triangle1).intersection(set(triangle3))
                                    if len(common_vertices_1_3) == 2:
                                        common_point = set(common_vertices_1_2).intersection(set(common_vertices_1_3))
                                        
                                        tri2_verticle = list(set(triangle2) - set(common_vertices_1_2))
                                        tri3_verticle = list(set(triangle3) - set(common_vertices_1_3))
                                        tri31_verticle = list(set(common_vertices_1_3) - set(common_point))
                                        tri21_verticle = list(set(common_vertices_1_2) - set(common_point))

                                        square_vertices = tri2_verticle + tri3_verticle + tri31_verticle + tri21_verticle
                                        
                                        if is_square(cluster_centers[np.array(square_vertices)]):
                                            squares.append(square_vertices)
                                            triangles_to_remove.update([i, j, k])
                                            square_count += 1

            # Calculate the area of each square and perform clustering
            square_areas = [calculate_area(cluster_centers[np.array(square)]) for square in squares]
            square_areas = np.array(square_areas).reshape(-1, 1)

            # Perform KMeans clustering on square areas
            n_clusters = 3  # Change this value based on how many clusters you want
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(square_areas)
            labels = kmeans.labels_

            # Plot squares with different colors based on clusters
            colors = [ 'm', 'y', 'k', 'r', 'g', 'b', 'c']
            cluster_counts = np.bincount(labels)
            most_common_cluster = np.argmax(cluster_counts)

            # Select squares in the most common cluster
            selected_squares = [square for idx, square in enumerate(squares) if labels[idx] == most_common_cluster]

            for square in selected_squares:
                for l in range(4):
                    plt.plot([cluster_centers[square[l], 0], cluster_centers[square[(l + 1) % 4], 0]],
                                                            [cluster_centers[square[l], 1], cluster_centers[square[(l + 1) % 4], 1]], 'g-', lw=2)
            

            # Modify squares to align with cluster centers
            square_after_modified = []
            for square in selected_squares:
                modified_square = []
                center = calculate_center([cluster_centers[idx] for idx in square])
                for index in square:
                    point = cluster_centers[index]
                    # Find the cluster this point belongs to
                    cluster = intersection_points_clusters[index]
                    # Find the center of the cluster
                    cluster_center = calculate_center(cluster)
                    # Calculate the angle between the square center and the current point

                    center_angle = calculate_angle(center, point)
                    
                    # Initialize the closest point and minimum distance
                    closest_point = point
                    min_distance = np.linalg.norm(closest_point - center)
                    
                    for candidate_point in cluster:
                        distance = np.linalg.norm(candidate_point - center)

                        candidate_angle = calculate_angle(center, candidate_point)
                        
                        # Check if the candidate point direction matches the basic corner direction
                        if angles_approx_equal(center_angle, candidate_angle) and distance < min_distance :
                            min_distance = distance
                            closest_point = candidate_point

                    modified_square.append(closest_point)
                square_after_modified.append(modified_square)

             # Draw the modified squares on the line_image
            for i, square in enumerate(square_after_modified):
                center = calculate_center(square)
                cv2.putText(line_image, str(i), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                for l in range(4):
                    pt1 = tuple(square[l])
                    pt2 = tuple(square[(l + 1) % 4])
                    cv2.line(line_image, square[l], square[(l + 1) % 4], (0, 255, 0), 2)  # Draw in green

            # Draw indices on the image
            # for idx, center in enumerate(cluster_centers):
            #     cv2.putText(line_image, str(idx), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


            # Plot the modified squares
            plt.figure(figsize=(24, 16))
            plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
            plt.title('Modified Squares Drawn on Image')
            plt.show()

        else:
            print("Error: The image does not have at least five channels or could not be read.")
    except Exception as e:
        print(f"Error reading the TIFF file: {e}")
else:
    print(f"Error: File '{mband_path}' does not exist.")

import gc
gc.collect()