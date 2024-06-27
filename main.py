from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



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
    print("'", image_path, "'", " has been modified to remove all grey pixels.", sep='')



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
                    if 2 <= group_width <= 15:
                        # Set all pixels in the group to white except the leftmost one
                        for i in range(start + 1, end):
                            pixels[i, y] = (255, 255, 255)
                else:
                    x += 1

        # Save the modified image
        image.save(image_path)
        print(f"Minimized fringe image overwritten onto the original image path.")

    except Exception as e:
        print(f"An error occurred: {e}")



def process_image(image_path):
    """
        Purpose:
            Processes the given BMP image file by first converting it to only black and white
            pixels and then thins the fringes out to 1px wide by only keeping the leftmost pixel.
        Args:
        - image_path (str): Path to the BMP image file.

        Returns:
        - None. Saves the modified image as a new BMP file.
    """
    convert_to_black_and_white(image_path)  # converts image to only black and white pixels
    minimize_fringe_width(image_path)       # thins the fringes out to 1px using leftmost px



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

    # print("")
    # for row in fringe_starts:
    #     print(row)

    return fringe_starts



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

    # Print shapes for debugging
    # print("Shapes - delta_matrix:", delta_matrix.shape, "fringe_starts:", fringe_starts.shape, "return_matrix:",
    #       return_matrix.shape)

    # Iterate over each row in fringe_starts and place delta_matrix entries
    for row_idx in range(len(fringe_starts)):
        for i, x_position in enumerate(fringe_starts[row_idx]):
            if x_position <= max_x:
                return_matrix[row_idx, int(x_position)] = delta_matrix[row_idx, i]

    return return_matrix



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
            # if sample1[j] != 0 and sample2[j] != 0 and sample1[j] == sample2[j]:
            #     delta_sample.append(400)
            # else:
            delta_sample.append(sample1[j] - sample2[j])

        # Convert delta_sample to a NumPy array
        delta_sample = np.array(delta_sample)

        # Multiply by 2 * pi to make it radial
        # delta_sample = (delta_sample) * (2 * np.pi)

        # Take the absolute value of the delta_sample
        delta_sample = np.abs(delta_sample)

        # Convert the result back to a list and add to the delta matrix
        delta_matrix.append(delta_sample.tolist())

    # print("\nDelta")
    # for row in delta_matrix:
    #     print(row)

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
    cbar.set_label('Delta Values (px)', rotation=90, labelpad=15)

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



def analyze_images(bkg_img_path, act_img_path, plotname="Output Results"):
    # Function that fully processes a given background and actual image of plasma and/or gas
    # and graphs the results as well.

    # Process both images
    process_image(bkg_img_path)
    process_image(act_img_path)

    # Find the fringe locations for both images
    actual_fringes = find_fringes(actual_image_path)
    background_fringes = find_fringes(background_image_path)

    # Find the delta x of the fringes from each image
    delta = find_delta(background_fringes, actual_fringes)

    # Finalize the delta matrix by scaling it back to the original image size for accuracy
    final_matrix = fill_in_zeros(delta, actual_fringes, actual_image_path)

    # Plot the final matrix for visualization
    plot_delta_heatmap(final_matrix, plotname)




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
    # Plasma
actual_image_path = "assets/6_21/700psiCFAir.bmp"
background_image_path = "assets/6_21/700psiCFAirbkg.bmp"
analyze_images(actual_image_path, background_image_path, "6/21 Plasma Fringe Δx")
# STOP  - 6/21/24 Data


# START - TESTING
    # Gas
actual_image_path = "assets/ExampleImages/gas_example_image.bmp"
background_image_path = "assets/ExampleImages/gas_example_background_image.bmp"
analyze_images(actual_image_path, background_image_path, "Gas Example Fringe Δx")
    # Plasma
actual_image_path = "assets/ExampleImages/plasma_example_image.bmp"
background_image_path = "assets/ExampleImages/plasma_example_background_image.bmp"
analyze_images(actual_image_path, background_image_path, "Plasma Example Fringe Δx")
# STOP  - TESTING
