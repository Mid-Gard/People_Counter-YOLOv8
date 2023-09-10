import tkinter as tk
from tkinter import ttk

def on_dropdown_select(event):
    selected_option.set(dropdown.get())
    update_video_label()

def update_video_label():
    selected_option_text = selected_option.get()
    video_label.config(text=f"Selected Option: {selected_option_text}")

def button_clicked(button_text):
    update_video_label()
    video_label.config(text=f"Button '{button_text}' clicked!")

# Create the main window
root = tk.Tk()
root.title("People Detector - Saro Farm")

# Maximize the window
root.state('zoomed')

# Create a border between the two columns
separator = ttk.Separator(root, orient="vertical")
separator.grid(row=0, column=0, sticky="ns")

# Configure the main window background color
root.configure(bg="lightgray")

# Column 1: Divide into two rows
frame_column1 = tk.Frame(root, padx=20, pady=20, bg="#f0f0f0")
frame_column1.grid(row=0, column=0, sticky="nsew")

# Row 1: Dropdown, Buttons
frame_row1 = tk.Frame(frame_column1, bg="#f0f0f0")
frame_row1.grid(row=0, column=0, sticky="nsew")

# Dropdown
dropdown_label = ttk.Label(frame_row1, text="Select Option:", background="#f0f0f0", font=("Helvetica", 12))
dropdown_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
options = ["Option 1", "Option 2", "Option 3", "Option 4"]
selected_option = tk.StringVar()
dropdown = ttk.Combobox(frame_row1, values=options, textvariable=selected_option, font=("Helvetica", 12))
dropdown.grid(row=0, column=1, padx=10, pady=10, sticky="w")
dropdown.bind("<<ComboboxSelected>>", on_dropdown_select)

# Buttons with colors and rounded borders
button1 = ttk.Button(frame_row1, text="Button 1", style="CoolButton.TButton", command=lambda: button_clicked("Button 1"))
button1.grid(row=1, column=0, padx=10, pady=10)
button2 = ttk.Button(frame_row1, text="Button 2", style="CoolButton.TButton", command=lambda: button_clicked("Button 2"))
button2.grid(row=1, column=1, padx=10, pady=10)
button3 = ttk.Button(frame_row1, text="Button 3", style="CoolButton.TButton", command=lambda: button_clicked("Button 3"))
button3.grid(row=2, column=0, padx=10, pady=10)
button4 = ttk.Button(frame_row1, text="Button 4", style="CoolButton.TButton", command=lambda: button_clicked("Button 4"))
button4.grid(row=2, column=1, padx=10, pady=10)

# Row 2: Reserved Space with border
frame_row2 = tk.Frame(frame_column1, bg="#f0f0f0")
frame_row2.grid(row=1, column=0, sticky="nsew")

# Reserved Space with rounded border
reserved_space = ttk.Label(frame_row2, text="Reserved Space", font=("Helvetica", 12, "italic"), relief="ridge", borderwidth=2)
reserved_space.grid(row=0, column=0, padx=10, pady=10, sticky="w")

# Column 2: Video
frame_column2 = tk.Frame(root, bg="black")
frame_column2.grid(row=0, column=1, sticky="nsew")

# Create a placeholder video (you can replace this with your video player widget)
video_label = ttk.Label(frame_column2, text="Video Placeholder", font=("Helvetica", 14, "bold"), foreground="white", background="black")
video_label.pack(fill="both", expand=True)

# Configure row and column weights to make the layout expand properly
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)

# Define a style for the buttons
style = ttk.Style()
style.configure("CoolButton.TButton", foreground="white", background="#007acc", padding=10, font=("Helvetica", 12), borderwidth=0, relief="solid")

# Start the GUI main loop
root.mainloop()
