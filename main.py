# This file's goal is to make use of functions to return:
# - the starting and ending points of the various fringes from the example
#   background image
# - the starting and ending points of the various fringes from the example image
# - the delta_x of the example background image and example image



from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def num_of_fringes(image_path):
    """
        ARGS:
            image_path: a string which shows the RELATIVE path to the file
                        you want to reference. Do not use absolute paths, 
                        as those are not compatible across systems and are 
                        not good practice.
        
        RETURNS: 
            An integer representing the number of fringes within the
            given BMP file
    """
    num_of_fringes = 0 # Init. the num of fringes to 0

    # Load the BMP image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Get the dimensions of the image
    height, width = image_np.shape

    # Check if the image has at least 10 rows
    if height < 10:
        raise ValueError("The image must have at least 10 rows")

    # Choose the 10th row for analysis (index 9 since indexing starts at 0)
    row = image_np[9, :]

    # Initialize variables to find transitions
    in_black = False

    for i in range(1, len(row)):
        if row[i - 1] == 255 and row[i] == 0:
            if not in_black:
                in_black = True
        elif row[i - 1] == 0 and row[i] == 255:
            if in_black:
                num_of_fringes += 1
                in_black = False

    # Handle case where the fringe goes till the end of the row
    if in_black:
        num_of_fringes += 1

    return num_of_fringes



def find_transitions(image_path, x):
    """
        ARGS:
            image_path: a string which shows the RELATIVE path to the file
                        you want to reference. Do not use absolute paths,
                        as those are not compatible across systems and are
                        not good practice.
            x: an integer which denoted how many samples across the y-axis
                that you would like to take. The more you take, the more
                data will be put into the equations and graphs that will be
                produced from this computation.
        RETURNS:
            A 2D matrix (list of lists) which contains the integer starting
            point on the x-axis of the BMP file of each fringe within the
            image. Each index represents the sample number, which each entry
            within each index represents the starting point of each fringe
            on the x-axis. The number of entries within each index of the
            matrix represents the # of fringes within the image. If you'd
            like to quickly ascertain the # of fringes within the image, I
            would suggest to use the "num_of_fringes()" function (if it
            exists within your version of this project).
    """

    # Load the BMP image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Get the dimensions of the image
    height, width = image_np.shape

    # Calculate the step size for sampling rows
    step = height // x

    # Initialize the list to store the starting points of fringes
    fringe_starts = [[] for _ in range(x)]

    for sample_num in range(x):
        # Determine the row index to sample
        row_index = sample_num * step
        row = image_np[row_index, :]

        # Initialize variables to find transitions
        in_black = False
        start = None

        for i in range(1, len(row)):
            if row[i - 1] == 255 and row[i] == 0:
                if not in_black:
                    start = i
                    in_black = True
            elif row[i - 1] == 0 and row[i] == 255:
                if in_black:
                    end = i - 1
                    fringe_starts[sample_num].append(start)
                    in_black = False

        # Handle case where the fringe goes till the end of the row
        if in_black:
            fringe_starts[sample_num].append(start)

    return fringe_starts


def find_delta(list1, list2):
    """
    ARGS:
        list1: List[List[int]]; a list of lists containing the starting points of each fringe in the first set of samples
        list2: List[List[int]]; a list of lists containing the starting points of each fringe in the second set of samples
    RETURNS:
        A 2D matrix (list of lists) which contains the integer difference between each fringe in the two given lists
    """

    # Ensure both lists have the same number of samples
    if len(list1) != len(list2):
        raise ValueError("Both input lists must have the same number of samples")

    # Initialize the delta matrix
    delta_matrix = []

    # Iterate through each sample
    for i in range(len(list1)):
        # Get the starting points for the current sample from both lists
        sample1 = list1[i]
        sample2 = list2[i]

        # Ensure both samples have the same number of fringes
        if len(sample1) != len(sample2):
            raise ValueError("Sample {} in both lists must have the same number of fringes".format(i))

        # Calculate the differences between corresponding fringes
        delta_sample = [sample1[j] - sample2[j] for j in range(len(sample1))]

        # Add the calculated differences to the delta matrix
        delta_matrix.append(delta_sample)

    return delta_matrix



def plot_delta_heatmap(delta_matrix):
    """
    ARGS:
        delta_matrix: A 2D list (list of lists) containing the delta values
                      which represent the differences between fringes.

    RETURNS:
        None. This function displays a heatmap plot of the delta values.
    """
    # Convert the delta matrix to a numpy array
    data_np = np.array(delta_matrix)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data as a heatmap
    cax = ax.imshow(data_np, cmap='viridis', aspect='auto')

    # Add colorbar to the plot
    cbar = fig.colorbar(cax)

    # Set the labels and title
    ax.set_xlabel('Index')
    ax.set_ylabel('Sample')
    ax.set_title('Delta Values Heatmap')

    # Display the plot
    plt.show()


# Example dictionaries
background_fringes = find_transitions('assets/gas_example_background_image.bmp', num_of_fringes('assets/gas_example_background_image.bmp'))
actual_fringes = find_transitions('assets/gas_example_image.bmp', num_of_fringes('assets/gas_example_image.bmp'))


for i in background_fringes:
    print(i)

print("")

for i in actual_fringes:
    print(i)

print("")

# Using the find_delta function
delta = find_delta(background_fringes, actual_fringes)

for i in delta:
    print(i)

plot_delta_heatmap(delta)