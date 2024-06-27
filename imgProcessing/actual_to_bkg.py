from PIL import Image, ImageDraw
import numpy as np



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



def find_fringes(new_img, slices=1):
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




# process_image("/Users/nbklaus21/PycharmProjects/Plasmainterferometry/Plasmainterferometry/imgProcessing/test_assets/gas_example_background_image.bmp")
# process_image("/Users/nbklaus21/PycharmProjects/Plasmainterferometry/Plasmainterferometry/imgProcessing/test_assets/gas_example_image.bmp")
# process_image("/Users/nbklaus21/PycharmProjects/Plasmainterferometry/Plasmainterferometry/imgProcessing/test_assets/plasma_example_background_image.bmp")
# process_image("/Users/nbklaus21/PycharmProjects/Plasmainterferometry/Plasmainterferometry/imgProcessing/test_assets/plasma_example_image.bmp")





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
    fringe_location_matrix = find_fringes(new_img)

    # Filter out empty rows (middle of the image)
    filtered_fringe_starts = [row for row in fringe_location_matrix if row]

    # Printing to debug
    print(f"Number of non-empty fringe rows: {len(filtered_fringe_starts)}")
    for row in filtered_fringe_starts:
        print(row)

    # Keep only the middle two rows
    if len(filtered_fringe_starts) >= 2:
        mid_index = len(filtered_fringe_starts) // 2
        middle_two_rows = filtered_fringe_starts[mid_index - 1:mid_index + 1]
    else:
        middle_two_rows = filtered_fringe_starts

    # Printing to debug
    print(f"Number of middle rows: {len(middle_two_rows)}")
    for row in middle_two_rows:
        print(row)

    y_loc_of_top_vals = top_pixels - 1
    y_loc_of_bot_vals = height - bottom_pixels

    tuple_location_matrix = [[], []]

    # Iterate through top_vals and bot_vals simultaneously
    for x_val_top, x_val_bot in zip(middle_two_rows[0], middle_two_rows[1]):
        # Append tuples to result matrix
        tuple_location_matrix[0].append((x_val_top, y_loc_of_top_vals))
        tuple_location_matrix[1].append((x_val_bot, y_loc_of_bot_vals))

    print("Tuple Location Matrix:")
    for row in tuple_location_matrix:
        print(row)

    # Draw lines on the new_img based on tuple_location_matrix
    draw = ImageDraw.Draw(new_img)
    for i in range(len(tuple_location_matrix[0])):
        x_top, y_top = tuple_location_matrix[0][i]
        x_bot, y_bot = tuple_location_matrix[1][i]
        draw.line((x_top, y_top, x_bot, y_bot), fill="black", width=1)

    return new_img


result_img = actual_to_bkg("/Users/nbklaus21/PycharmProjects/Plasmainterferometry/Plasmainterferometry/imgProcessing/test_assets/gas_example_image.bmp")

result_img.show()