# This file's goal is to make use of functions to return:
# - the starting and ending points of the various fringes from the example
#   background image
# - the starting and ending points of the various fringes from the example image
# - the delta_x of the example background image and example image



from PIL import Image
import numpy as np



def find_transitions(image_path):
    # Load the BMP image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Get the dimensions of the image
    height, width = image_np.shape

    # Choose the middle row for analysis
    middle_row = image_np[height // 2, :]

    # Initialize the dictionary to store transitions
    transitions = {
        "transitions": []
    }

    # Iterate through the middle row to find transitions from white to black
    in_black = False
    start = None

    for i in range(1, len(middle_row)):
        if middle_row[i - 1] == 255 and middle_row[i] == 0:
            if not in_black:
                start = i
                in_black = True
        elif middle_row[i - 1] == 0 and middle_row[i] == 255:
            if in_black:
                end = i - 1
                transitions["transitions"].append((start, end))
                in_black = False

    # Handle case where the fringe goes till the end of the row
    if in_black:
        end = len(middle_row) - 1
        transitions["transitions"].append((start, end))

    return transitions



def find_delta(dict1, dict2):
    # ARGS: dict1, dict2; two dictionaries who contain the starting and ending
    # point of each fringe within the given image
    # RETURNS: A 2D matrix (list) which contains the integer difference between
    # each fringe in the background and actual image (gas or plasma)

    # Extract the transitions lists from both dictionaries
    transitions1 = dict1['transitions']
    transitions2 = dict2['transitions']

    # Ensure both lists have the same length
    if len(transitions1) != len(transitions2):
        raise ValueError("Both transition lists must have the same length")

    # Calculate the difference of the first entries
    delta = [transitions1[i][0] - transitions2[i][0] for i in range(len(transitions1))]

    return delta


# Example dictionaries
background_fringes = find_transitions('assets/gas_example_background_image.bmp')
actual_fringes = find_transitions('assets/gas_example_image.bmp')



# Using the find_delta function
delta = find_delta(background_fringes, actual_fringes)
print(delta)