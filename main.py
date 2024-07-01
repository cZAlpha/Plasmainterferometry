from mainFunctions import *  # Import all analysis functions
from ui import *             # Import all user interface functions

#
# Welcome!
#
# To analyze an image, follow the readMe directions for image preprocessing, then
# replace the 'img_to_analyze' variable's string value to the relative path to
# the image you'd like to analyze.
#
# After that, simply run this file and you will have the plot of the phase shift
# displayed by the MatPlotLib library!
#
# To save your results, simply hit the save button on the plot window that pops up.
#

ui_responses = mainUI()  # This will open the UI and return the dictionary containing the results
analyze_image(ui_responses['Image Path'], ui_responses['Phase Shift'], ui_responses['On-axis Density Plot'], ui_responses['2D Density Mapping'])  # This line will analyze the image specified by the UI w/ the wanted plots
