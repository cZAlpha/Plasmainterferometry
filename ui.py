import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Global variables for the main application window and widgets
app = None
label = None
uploadButton = None
confirmButton = None
rejectButton = None
checkboxes_frame = None  # Frame to hold checkboxes
img_path = None  # Variable to store the image path

# Variables to track checkbox states and store results
checkbox_vars = {}
results = {}  # Global variable to store results



# Function that resizes the app window to original size for importing file button
def resize_small():
    app.geometry("300x150")

# Function that resizes the app window to medium size
def resize_medium():
    app.geometry("800x600")


# Function to clear all buttons and labels
def clearAll():
    global app, uploadButton, confirmButton, rejectButton, checkboxes_frame, title_label

    # Destroy all buttons and checkboxes
    if uploadButton:
        uploadButton.destroy()
    if confirmButton:
        confirmButton.destroy()
    if rejectButton:
        rejectButton.destroy()
    if checkboxes_frame:
        checkboxes_frame.destroy()
    if title_label:
        title_label.destroy()



# Function to store the dictionary of checkbox states including image path
def updateResults():
    global results
    checkbox_states = {text: var.get() for text, var in checkbox_vars.items()}
    checkbox_states['Image Path'] = img_path  # Add image path to the dictionary
    results = checkbox_states



# Function that creates the finish button that closes the UI application
def createFinishButton():
    global app

    # Create and place the "Finish" button
    finishButton = tk.Button(app, text="Finish", font=("Arial", 16, "bold"), bg="lightcoral",
                             fg="black", padx=20, pady=10, relief="raised",
                             borderwidth=2, command=app.quit)
    finishButton.pack(pady=20)



# Function to confirm if the displayed image is correct
def confirmImage(is_correct, path):
    global app, uploadButton, confirmButton, rejectButton, checkboxes_frame, checkbox_vars, img_path, results, title_label

    # Store the image path
    img_path = path

    # Destroy all buttons and checkboxes
    clearAll()

    # If image is correct, display checkboxes
    if is_correct:
        # Create and populate checkbox_vars
        checkbox_vars = {
            "Phase Shift": tk.BooleanVar(app),
            "On-axis Density Plot": tk.BooleanVar(app),
            "2D Density Mapping": tk.BooleanVar(app)
        }

        # Create a frame to hold the checkboxes
        checkboxes_frame = tk.Frame(app, bg="grey6")
        checkboxes_frame.pack(side=tk.BOTTOM, pady=20)

        # Create and place the label above the checkboxes
        title_label = tk.Label(checkboxes_frame, text="Which Plots Would You Like?", bg="grey6", font=("Arial", 20, "bold"))
        title_label.pack(pady=10)

        # Create and place checkboxes
        for text, var in checkbox_vars.items():
            checkbox = tk.Checkbutton(checkboxes_frame, text=text, variable=var, bg="grey6", font=("Arial", 14))
            checkbox.pack(anchor="w")

        # Create and place the "Done" button
        doneButton = tk.Button(checkboxes_frame,
                               text="Done",
                               font=("Arial", 14, "bold"),
                               bg="lightgreen",
                               fg="black",
                               padx=10, pady=10,
                               relief="raised",
                               borderwidth=2,
                               command=lambda: [clearAll(), updateResults(), createFinishButton(),
                                                resize_medium() ])
        doneButton.pack(pady=10)
    else:
        # Clear the displayed image
        label.config(image='')

        # Re-create the original upload button
        uploadButton = tk.Button(app, text="Import Image",
                                 command=imageUploader,
                                 font=("Arial", 20, "bold"),
                                 bg="lightgreen",
                                 fg="black",
                                 padx=20, pady=10,
                                 relief="raised",
                                 borderwidth=2)
        uploadButton.pack(side=tk.BOTTOM, pady=20)



# Function that facilitates the uploading of images
def imageUploader():
    global app, uploadButton, confirmButton, rejectButton, title_label

    # Resize window
    app.geometry("800x800")

    fileTypes = [("BMP files", "*.bmp")]
    path = filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if path:
        img = Image.open(path)
        img = img.resize((500, 500))
        pic = ImageTk.PhotoImage(img)

        # Resize the app window in order to fit picture and button
        label.config(image=pic)
        label.image = pic

        # Remove the upload button after selecting an image
        if uploadButton:
            uploadButton.destroy()

        # Create the label for image confirmation
        title_label = tk.Label(app, text="Is This The Correct Image?", pady=0,
                               bg="grey6", font=("Arial", 20, "bold"))
        title_label.pack(pady=(10, 0))  # Adjust the padding here

        # Create confirm and reject buttons
        rejectButton = tk.Button(app, text="No",
                             font=("Arial", 15, "bold"),
                             bg="lightgreen",
                             fg="black",
                             padx=20, pady=10,
                             relief="raised",
                             borderwidth=2,
                             command=lambda: (confirmImage(False, path), resize_small() ))
        rejectButton.pack(side=tk.BOTTOM, pady=(0, 20))

        confirmButton = tk.Button(app, text="Yes",
                                  font=("Arial", 15, "bold"),
                                  bg="lightgreen",
                                  fg="black",
                                  padx=20, pady=10,
                                  relief="raised",
                                  borderwidth=2,
                                  command=lambda: confirmImage(True, path))
        confirmButton.pack(side=tk.BOTTOM, pady=(0, 10))
    else:
        print("Please choose a file!")



# Parent / container function for the UI
def ui():
    global app, label, uploadButton

    # Define the tkinter object
    app = tk.Tk()

    # Set the title and basic size of the app
    app.title("Plasmainterferemetry Plasma & Gas Density Diagnosis Program")
    app.geometry("300x150")

    # Set the background color of the main window to grey
    app.configure(bg="grey6")

    # Set the application icon
    # Load an image file
    icon = Image.open('assets/uiAssets/plasmaInterferometry_UI_Icon.png')  # Make sure to use the correct path to your .png file
    app_icon = ImageTk.PhotoImage(icon)  # Create PhotoImage object
    app.iconphoto(False, app_icon)  # Set the icon for the app

    # Label to display images
    label = tk.Label(app, bg="grey6")
    label.pack(pady=10)

    # Define and style the upload button
    uploadButton = tk.Button(app, text="Import Image", command=imageUploader,
                             font=("Arial", 20, "bold"), bg="darkgoldenrod2",
                             fg="black", padx=20, pady=10, relief="raised",
                             borderwidth=2)
    uploadButton.pack(side=tk.TOP, pady=0)



# Actual container function for the UI
def mainUI():
    global results

    # Run the UI
    ui()

    # Start the Tkinter event loop
    app.mainloop()

    # Return the results after the event loop ends
    return results
