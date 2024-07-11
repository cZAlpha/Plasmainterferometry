from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D  # Used in IDW optimized interpolation function





def convert_to_black_and_white(image_path):
    """
        Purpose:
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
    print("\n", "'", image_path, "'", " has been modified to remove all grey pixels.", sep='')



def minimize_fringe_width(image_path):
    """
        Purpose:
            Convert the fringes in the given image to 1px thick, works on both plasma and bkg imgs
        Args:
        - image_path (str): Path to the BMP image file.

        Returns:
        - None. Saves the modified image as a new BMP file.
    """
    try:
        # Open the image
        image = Image.open(image_path)
        pixels = image.load()

        # Get image dimensions
        width, height = image.size

        for y in range(height):
            x = 0
            while x < width:
                # Check for the start of a group of black pixels
                if pixels[x, y] == (0, 0, 0):
                    start = x
                    while x < width and pixels[x, y] == (0, 0, 0):
                        x += 1
                    end = x

                    # Check if the group width is between 2 and 15 pixels
                    group_width = end - start
                    if 2 <= group_width <= 20:
                        # Set all pixels in the group to white except the leftmost one
                        for i in range(start + 1, end):
                            pixels[i, y] = (255, 255, 255)
                else:
                    x += 1

        # Save the modified image
        image.save(image_path)
        print("\n", "'", image_path, "'", " has ben modified to minimize fringe width",sep="")

    except Exception as e:
        print(f"An error occurred: {e}")



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



def find_fringes(image_path, slices=1):
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

    # Load the BMP image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Get the dimensions of the image
    height, width = image_np.shape

    # Validate slices parameter
    if slices <= 0:
        raise ValueError("The number of samples (slices) must be greater than 0.")
    if slices >= 100:
        raise ValueError("The number of samples (slices) must be less than 100. Try to not input an argument for this value unless testing.")
    if slices == 1:
        slices = height

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

    return fringe_starts



def find_fringes_from_PIL_img(new_img, slices=1):
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
        new_img: a PIL image object that represents the image.
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

    # Convert the image to grayscale
    gray_img = new_img.convert('L')

    # Convert the image to a numpy array
    image_np = np.array(gray_img)

    # Get the dimensions of the image
    height, width = image_np.shape

    # Validate slices parameter
    if slices <= 0:
        raise ValueError("The number of samples (slices) must be greater than 0.")
    if slices >= 100:
        raise ValueError("The number of samples (slices) must be less than 100. Try to not input an argument for this value unless testing.")
    if slices == 1:
        slices = height

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

    return fringe_starts



def actual_to_bkg(image_path, top_pixels=20, bottom_pixels=20):
    if top_pixels % 2 != 0:
        print("ERROR: TOP PIXELS MUST BE EVEN")
    if bottom_pixels % 2 != 0:
        print("ERROR: BOTTOM PIXELS MUST BE EVEN")

    # Load the image
    img = Image.open(image_path)
    width, height = img.size

    # Crop the top and bottom parts
    top_part = img.crop((0, 0, width, top_pixels))
    bottom_part = img.crop((0, height - bottom_pixels, width, height))

    # Create a new image with the same dimensions as the original
    new_img = Image.new("RGB", (width, height), (255, 255, 255))  # white background

    # Paste the top and bottom parts into the new image
    new_img.paste(top_part, (0, 0))
    new_img.paste(bottom_part, (0, height - bottom_pixels))

    # Find location of the top and bottom fringe pixels, store them
    fringe_location_matrix = find_fringes_from_PIL_img(new_img)

    # Filter out empty rows (middle of the image)
    filtered_fringe_starts = [row for row in fringe_location_matrix if row]

    # Keep only the middle two rows
    if len(filtered_fringe_starts) >= 2:
        mid_index = len(filtered_fringe_starts) // 2
        middle_two_rows = filtered_fringe_starts[mid_index - 1:mid_index + 1]
    else:
        middle_two_rows = filtered_fringe_starts

    y_loc_of_top_vals = top_pixels - 1
    y_loc_of_bot_vals = height - bottom_pixels

    tuple_location_matrix = [[], []]

    # Iterate through top_vals and bot_vals simultaneously
    for x_val_top, x_val_bot in zip(middle_two_rows[0], middle_two_rows[1]):
        # Append tuples to result matrix
        tuple_location_matrix[0].append((x_val_top, y_loc_of_top_vals))
        tuple_location_matrix[1].append((x_val_bot, y_loc_of_bot_vals))

    # Draw lines on the new_img based on tuple_location_matrix
    draw = ImageDraw.Draw(new_img)
    for i in range(len(tuple_location_matrix[0])):
        x_top, y_top = tuple_location_matrix[0][i]
        x_bot, y_bot = tuple_location_matrix[1][i]
        draw.line((x_top, y_top, x_bot, y_bot), fill="black", width=1)

    # Save the new image with the suffix "_bkg"
    new_image_path = image_path.replace(".", "_bkg.")
    new_img.save(new_image_path)

    # Print to the console that the new image has been made
    print("\n", "A new background image has been made for: ","'", image_path, "'", ", named: ", "'", new_image_path, "'",sep="")

    return new_img, new_image_path



def process_image(image_path):
    """
        Purpose:
            Processes the given BMP image file by first converting it to only black and white
            pixels and then thins the fringes out to 1px wide by only keeping the leftmost pixel.
        Args:
        - image_path (str): Path to the BMP image file.

        Returns:
        - Saves the modified image as a new BMP file.
        - Returns the background image filepath of the inputted image
    """
    convert_to_black_and_white(image_path)  # converts image to only black and white pixels
    minimize_fringe_width(image_path)       # thins the fringes out to 1px using leftmost px
    bkg_img_filepath = actual_to_bkg(image_path)               # Creates a background image from the image itself

    return bkg_img_filepath[1]



def fill_in_zeros(delta_matrix, fringe_starts, image_path):
    """
    PURPOSE:
        Fill in the entries of delta_matrix into return_matrix at positions specified by fringe_starts.
    ARGS:
        delta_matrix: The matrix whose entries need to be placed into return_matrix.
        fringe_starts: A matrix specifying positions to place entries from delta_matrix within return_matrix.
        image_path: Relative path to the BMP file. Used solely for the dimensions of the image.
    RETURNS:
        A modified return_matrix after placing delta_matrix entries at specified positions.
    """

    # Ensure delta_matrix and fringe_starts are numpy arrays
    delta_matrix = np.array(delta_matrix)
    fringe_starts = np.array(fringe_starts)

    # Load the BMP image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Get the dimensions of the image
    height, width = image_np.shape

    # Creating a matrix filled with 0.0
    return_matrix = np.zeros((height, width))

    # Ensure fringe_starts is within bounds
    max_x = return_matrix.shape[1] - 1  # Max x-coordinate index in return_matrix

    # Iterate over each row in fringe_starts and place delta_matrix entries
    for row_idx in range(len(fringe_starts)):
        for i, x_position in enumerate(fringe_starts[row_idx]):
            if x_position <= max_x:
                return_matrix[row_idx, int(x_position)] = delta_matrix[row_idx, i]


    return return_matrix



def linearly_interpolate_matrix(matrix):
    """
    Interpolates the data within each row of the matrix by smoothing out the zeros between non-zero values.

    Args:
        matrix: A 2D numpy array where rows contain data to be interpolated.

    Returns:
        A 2D numpy array with interpolated values.
    """
    # Ensure the matrix is a numpy array
    matrix = np.array(matrix)

    # Get the dimensions of the matrix
    rows, cols = matrix.shape

    # Initialize the output interpolated matrix
    interpolated_matrix = np.zeros((rows, cols))

    # Iterate over each row in the matrix
    for row_idx in range(rows):
        row = matrix[row_idx]

        # Find the indices of non-zero values
        non_zero_indices = np.nonzero(row)[0]

        # If there are less than 2 non-zero values, no interpolation is needed
        if len(non_zero_indices) < 2:
            interpolated_matrix[row_idx] = row
            continue

        # Iterate over each segment between non-zero values and interpolate
        for i in range(len(non_zero_indices) - 1):
            start_idx = non_zero_indices[i]
            end_idx = non_zero_indices[i + 1]

            start_value = row[start_idx]
            end_value = row[end_idx]

            # Linear interpolation between the two non-zero values
            interpolated_matrix[row_idx, start_idx:end_idx + 1] = np.linspace(start_value, end_value,
                                                                              end_idx - start_idx + 1)

        # Fill in values outside the non-zero range with linear interpolation towards zero
        if non_zero_indices[0] > 0:
            interpolated_matrix[row_idx, :non_zero_indices[0]] = np.linspace(0, row[non_zero_indices[0]],
                                                                             non_zero_indices[0])

        if non_zero_indices[-1] < cols - 1:
            interpolated_matrix[row_idx, non_zero_indices[-1]:] = np.linspace(row[non_zero_indices[-1]], 0,
                                                                              cols - non_zero_indices[-1])

    return interpolated_matrix



def interpolate_matrix_idw_optimized(matrix, radius=500):
    """
    Optimized IDW interpolation using KD-tree for nearest neighbor search.

    Args:
        matrix: A 2D numpy array where rows contain data to be interpolated.
        radius: Search radius for nearest neighbors (optional, default is 500, but this will take ages to compute btw).

    Returns:
        A 2D numpy array with interpolated values using optimized IDW interpolation.
    """
    # Ensure the matrix is a numpy array
    matrix = np.array(matrix)

    # Get the dimensions of the matrix
    rows, cols = matrix.shape

    # Initialize the output interpolated matrix
    interpolated_matrix = np.copy(matrix)

    # Flatten matrix coordinates for KD-tree
    points = np.array([[i, j] for i in range(rows) for j in range(cols) if matrix[i, j] != 0])
    values = np.array([matrix[i, j] for i in range(rows) for j in range(cols) if matrix[i, j] != 0])

    # Create KD-tree for fast nearest neighbor search
    tree = KDTree(points)

    # Iterate over each cell in the matrix
    count = 0
    for i in range(rows):
        for j in range(cols):
            count+= 1
            if matrix[i, j] == 0:
                # Query KD-tree for nearest neighbors within radius
                dist, idx = tree.query([i, j], k=min(len(points), radius))

                # Calculate weights using inverse distances
                weights = np.array([1 / d if d != 0 else 1 for d in dist])

                # Perform IDW interpolation
                interpolated_value = np.sum(weights * values[idx]) / np.sum(weights)

                # Update interpolated matrix
                interpolated_matrix[i, j] = interpolated_value

    return interpolated_matrix



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

        # Calculate the differences between corresponding fringes
        delta_sample = []
        for j in range(len(sample1)):

            delta_sample.append(sample1[j] - sample2[j])

        # Convert delta_sample to a NumPy array
        delta_sample = np.array(delta_sample)

        # Multiply by 2 * pi to make it radial
        # delta_sample = (delta_sample) * (2 * np.pi)

        # Take the absolute value of the delta_sample
        delta_sample = np.abs(delta_sample)

        # Convert the result back to a list and add to the delta matrix
        delta_matrix.append(delta_sample.tolist())

    return delta_matrix



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
    cax = ax.imshow(data_np, cmap='jet', aspect='auto')

    # Add colorbar to the plot
    cbar = fig.colorbar(cax)

    # Set label for the colorbar
    cbar.set_label('Delta Values (rads)', rotation=90, labelpad=15)

    # Set ticks and labels for the colorbar
    cbar.ax.invert_yaxis()  # Invert the colorbar so that 0 is at the bottom

    # Set the labels and title
    ax.set_xlabel('X-Axis (px)')
    ax.set_ylabel('Y-Axis (px)')
    ax.set_title(plot_title)

    # Set the window title (backend-specific method)
    fig.canvas.manager.set_window_title('Delta Heatmap')

    # Display the plot
    plt.show()



def analyze_image(act_img_path, bool_phaseShift_plot, bool_onaxis_density_plot, twoD_density_mapping, plotname="Output Results"):
    # Function that fully processes a given background and actual image of plasma and/or gas
    # and graphs the results as well. Does not return anything, it is void.

    # If the user specifies to analyze the inputted image in any way, do so
    if ( bool_phaseShift_plot or bool_onaxis_density_plot or twoD_density_mapping ):
        # Printing to the console that analysis will be performed
        print("\n", "Image: ", "'", act_img_path, "'", " will now be analyzed.", sep="")

        # Process both images
        bkg_img_filepath = process_image(act_img_path)

        # Find the fringe locations for both images
        actual_fringes = find_fringes(act_img_path)
        background_fringes = find_fringes(bkg_img_filepath)

        # Find the delta x of the fringes from each image
        delta = find_delta(background_fringes, actual_fringes)

        # Intermediately finalize the delta matrix by scaling it back to the original image size for accuracy
        intermediate_matrix = fill_in_zeros(delta, actual_fringes, act_img_path)

        # Finalize the matrix by interpolating the data across the x-axis
        final_matrix = linearly_interpolate_matrix(intermediate_matrix)

        # If the user wants the phase shift plot, plot it and display it
        if ( bool_phaseShift_plot ):
            # Plot the final phase shift matrix for visualization
            plot_delta_heatmap(final_matrix, plotname)

        # If the user wants the on-axis density plot, plot it and display it
        if ( bool_onaxis_density_plot ):
            print("\n", "The on-axis density plot functionality of the program has not been completed yet, apologies.", sep="")

        # If the user wants the 2D density mapping, plot it and display it
        if (twoD_density_mapping):
            print("\n", "The 2D density mapping functionality of the program has not been completed yet, apologies.", sep="")

        print("\n", "Analysis Has Concluded.", sep="")

    else:
        print("\n", "No analysis has been conducted, as you did not specify to display any analysis results.", sep="")
