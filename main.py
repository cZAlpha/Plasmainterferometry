from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math



def convert_to_black_and_white(image_path):
    """
    Convert the input BMP image to black and white by setting all gray pixels
    to white (255, 255, 255), while leaving black (0, 0, 0) and white (255, 255, 255)
    pixels unchanged.

    Args:
    - image_path (str): Path to the BMP image file.

    Returns:
    - None. Saves the modified image as a new BMP file.
    """
    # Open the BMP image
    image = Image.open(image_path)

    # Convert the image to RGB mode (to access RGB values)
    image_rgb = image.convert('RGB')

    # Get image dimensions
    width, height = image_rgb.size

    # Create a new image to store the modified pixels
    new_image = Image.new('RGB', (width, height))

    # Iterate through each pixel
    for x in range(width):
        for y in range(height):
            # Get RGB values of the pixel
            r, g, b = image_rgb.getpixel((x, y))

            # Check if the pixel is gray (not pure black or white)
            if not (r == 0 and g == 0 and b == 0) and not (r == 255 and g == 255 and b == 255):
                # Set gray pixel to white
                new_image.putpixel((x, y), (255, 255, 255))
            else:
                # Copy pixel as is (black or white)
                new_image.putpixel((x, y), (r, g, b))

    # Save the new image
    new_image.save(image_path)
    print("'", image_path, "'", " has been modified to remove all grey pixels.", sep='')



def print_rgb_values(image_path):
    # Used for debugging, has no usage in the actual purpose of the project
    """
    ARGS:
        image_path: a string which shows the RELATIVE path to the file
                    you want to reference. Do not use absolute paths,
                    as those are not compatible across systems and are
                    not good practice.
    """
    # Load the BMP image
    image = Image.open(image_path)  # Load original image
    image_np = np.array(image)  # Convert the image to a numpy array

    # Get the dimensions of the image
    height, width = image_np.shape[:2]  # Get height and width only

    # Check if the image has at least 10 rows
    if height < 10:
        raise ValueError("The image must have at least 10 rows")

    for i in range(width):
        # Get grayscale value for each pixel
        pixel_value = image_np[9, i]

        # Print pixel value
        print(f"Pixel ({i}, 9) - Value: {pixel_value}")



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

    convert_to_black_and_white(image_path)

    num_of_fringes = 0  # Init. the num of fringes to 0

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

    for pixel_value in row:
        if pixel_value != 255:
            if not in_black:
                num_of_fringes += 1
                in_black = True
        else:
            in_black = False

    # Handle case where the fringe goes till the end of the row
    if in_black:
        num_of_fringes += 1

    if num_of_fringes == 0:
        raise ValueError("The image does not contain any fringes.")

    return num_of_fringes



def find_fringes(image_path, slices = 1):
    """
        PURPOSE:
            The purpose of this function is to take the slightly altered given image and
            find the location of the given fringes in the x axis of the image, and return
            a matrix of values whose row # corresponds to the x row of the image and whose
            values within the row represent the given x coordinate starting point of each
            fringe within that given x row of the image. To be clear, you can change the
            second parameter to take that # of vertical slices of the image if you would
            like to conserve processing power for testing or otherwise, at the very severe
            cost of accuracy/precision of your data.
        ARGS:
            image_path: a string which shows the RELATIVE path to the file
                        you want to reference. Do not use absolute paths,
                        as those are not compatible across systems and are
                        not good practice.
            slices: an integer which represents the number of samples the algorithm
                will take.
        RETURNS:
            A 2D matrix (list of lists) which contains the integer starting
            point on the x-axis of the BMP file of each fringe within the
            image. Each index represents the sample number, where each entry
            within each index represents the starting point of each fringe
            on the x-axis. The number of entries within each index of the
            matrix represents the # of fringes within the image. If you'd
            like to quickly ascertain the # of fringes within the image, I
            would suggest using the "num_of_fringes()" function (if it
            exists within your version of this project).
    """

    # Processes the given image to ensure that there are no gray pixels (I'm too lazy to program threshold values and
    # don't view it as particularly necessary for the way I am going about this)
    convert_to_black_and_white(image_path)

    # Load the BMP image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Get the dimensions of the image
    height, width = image_np.shape

    # x is an integer which represents the # of fringes within the image, this
    # keeps the number of samples simple and makes the graph look more symmetrical
    # if you'd like to take more samples, change this value!
    # IMPORTANT: if you change x's value, you MUST ensure that the image is ONLY
    # black and white with NO GREY PIXELS, otherwise the code will likely not work!
    if slices <= 0: # lower bound case
        raise ValueError("The number of samples (x) must be greater than 0.")

    if slices >= 100: # upper bound case (after testing this is around where the algorithm shits itself)
        raise ValueError("The number of samples (x) must be less than 100. Try to not input an argument for this value "
                         "unless testing.")

    if ( slices == 1): # this just covers the default case by ensuring it is the same as the # of fringes
        x = height



    # Calculate the step size for sampling rows
    step = height // slices

    # Initialize the list to store the starting points of fringes
    fringe_starts = [[] for _ in range(slices)]

    for sample_num in range(slices):
        # Determine the row index to sample
        row_index = sample_num * step
        row = image_np[row_index, :]

        # Initialize variables to find transitions
        in_black = False
        start = None

        for i in range(len(row)):
            pixel_value = row[i]

            if pixel_value != 255:
                if not in_black:
                    start = i
                    in_black = True
            else:
                if in_black:
                    fringe_starts[sample_num].append(start)
                    in_black = False

        # Handle case where the fringe goes till the end of the row
        if in_black:
            fringe_starts[sample_num].append(start)

    # TESTING ONLY
    print(image_path)
    for row in fringe_starts:
        print(row)

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

        # Convert delta_sample to a NumPy array and multiply by 2 * pi
        delta_sample = np.array(delta_sample) * (2 * np.pi)

        # Convert the result back to a list and add to the delta matrix
        delta_matrix.append(delta_sample.tolist())

    for i in delta_matrix:
        print(i)

    return delta_matrix



def expand_data(matrix, x): # DO NOT USE, DEPRICATED
    """
    Expands the amount of data within a given list of lists (matrix) by adding new entries
    at the midpoint between each pair of existing entries in each row, targeting 'x' entries per row.

    Args:
    - matrix (list of lists): 2D list containing the original data.
    - x (int): Target number of entries per row after doubling.

    Returns:
    - list of lists: A new 2D list with doubled data.
    """

    if x <= len(matrix[0]):
        raise ValueError("Target number of entries (x) must be greater than the current number of columns in the matrix")

    new_matrix = []

    for row in matrix:
        new_row = []
        step_size = (len(row) - 1) / (x - 1)  # Calculate the step size

        for j in range(x):
            if j == 0:
                new_row.append(row[0])
            elif j == x - 1:
                new_row.append(row[-1])
            else:
                # Calculate the index in the original row
                index = (j - 1) * step_size
                lower_index = int(index)
                upper_index = lower_index + 1

                # Calculate the interpolated value
                weight = index - lower_index
                interpolated_value = (1 - weight) * row[lower_index] + weight * row[upper_index]
                interpolated_value = math.floor(interpolated_value)  # Floor the interpolated value
                new_row.append(interpolated_value)

        new_matrix.append(new_row)

    return new_matrix



def plot_delta_heatmap(delta_matrix, plot_title):
    """
    ARGS:
        delta_matrix: A 2D list (list of lists) containing the delta values
                      which represent the differences between fringes.
        plot_title: a string which represents the title of the plot.

    RETURNS:
        None. This function displays a heatmap plot of the delta values.
    """
    # Convert the delta matrix to a numpy array
    data_np = np.array(delta_matrix)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data as a heatmap with the 'plasma' colormap
    cax = ax.imshow(data_np, cmap='jet_r', aspect='auto')

    # Add colorbar to the plot
    cbar = fig.colorbar(cax)

    # Set label for the colorbar
    cbar.set_label('Delta Values (px)', rotation=90, labelpad=15)

    # Set ticks and labels for the colorbar
    cbar.ax.invert_yaxis()  # Invert the colorbar so that 0 is at the bottom

    # Set the labels and title
    ax.set_xlabel('Fringe #')
    ax.set_ylabel('Sample #')
    ax.set_title(plot_title)

    # Set the window title (backend-specific method)
    fig.canvas.manager.set_window_title('Delta Heatmap')

    # Display the plot
    plt.show()






# START - Examples
#
# PLASMA DATA
# background_fringes = find_transitions('assets/ExampleImages/plasma_example_background_image.bmp')
# actual_fringes = find_transitions('assets/ExampleImages/plasma_example_image.bmp')
# delta = find_delta(background_fringes, actual_fringes)
# delta = expand_data(delta, 1000)
# plot_delta_heatmap(delta, "Plasma Fringe Δx")

# GAS DATA
# background_fringes = find_transitions('assets/ExampleImages/gas_example_background_image.bmp')
# actual_fringes = find_transitions('assets/ExampleImages/gas_example_image.bmp')
# delta = find_delta(background_fringes, actual_fringes)
# delta = expand_data(delta, 1000)
# plot_delta_heatmap(delta, "Gas Fringe Δx")

# STOP - Examples



# START - 6/21/24 Data

# GAS DATA
background_fringes = find_fringes('assets/6_21/700psiCFAirbkg.bmp', 26) # REMOVE 26 AFTER TESTING
actual_fringes = find_fringes('assets/6_21/700psiCFAir.bmp', 26) # REMOVE 26 AFTER TESTING
delta = find_delta(background_fringes, actual_fringes)
# delta = expand_data(delta, 1000)
plot_delta_heatmap(delta, "6/21 Gas Fringe Δx")

# STOP  - 6/21/24 Data
