import tensorflow as tf
import numpy as np

MazeSolve = input("Path to maze jpg")

# Load the image of the maze
maze_image = tf.io.read_file(MazeSolve)
maze_image = tf.io.decode_image(maze_image, channels=1)

# Convert the image to a tensor and get the shape
maze = tf.cast(maze_image, tf.float32)
rows, cols = tf.shape(maze)[0], tf.shape(maze)[1]

# Threshold the image to segment the walls
_, maze = tf.threshold(maze, 127, 255, tf.THRESH_BINARY)

# Define the starting and ending points
start = (0, 0)
end = (rows-1, cols-1)

# Create a list to store the path
path = []

# Use A* algorithm to find the shortest path
def astar(maze, start, end):
    heap = []
    heappush(heap, (0, start))
    visited = set()
    while heap:
        # Get the node with the lowest f value
        (f, node) = heappop(heap)
        # Check if we have reached the end
        if node == end:
            while node != start:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return path
        # Mark the node as visited
        visited.add(node)
        # Generate the neighbors of the current node
        for neighbor in get_neighbors(maze, node):
            if neighbor in visited:
                continue
            # Calculate the cost of moving to the neighbor
            g = cost[node] + 1
            h = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
            f = g + h
            # Update the cost and came_from values for the neighbor
            if neighbor not in heap or f < cost[neighbor]:
                cost[neighbor] = f
                came_from[neighbor] = node
                heappush(heap, (f, neighbor))

# Function to get the neighbors of a node
def get_neighbors(maze, node):
    neighbors = []
    x, y = node
    if x > 0 and maze[x-1, y] != 0:
        neighbors.append((x-1, y))
    if x < rows-1 and maze[x+1, y] != 0:
        neighbors.append((x+1, y))
    if y > 0 and maze[x, y-1] != 0:
        neighbors.append((x, y-1))
    if y < cols-1 and maze[x, y+1] != 0:
        neighbors.append((x, y+1))
    return neighbors

# Call the astar function
astar(maze, start, end)

# Reverse the path to start from the beginning
path.reverse()

# Draw the path on the original image using OpenCV
for i in range(len(path)):
    x, y = path[i]
    maze_image = cv2.circle(maze_image, (y, x), 2, (0, 0, 255), -1)

# Display the image with the path
cv2.imshow("Maze", maze_image)
cv2.waitKey(0)
