import cv2
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from PIL import Image, ImageTk


def log_event(event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event_log.insert(tk.END, f"{timestamp} - {event}\n")
    event_log.see(tk.END)


def detect_motion():
    global prev_frame, background_model

    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a running average to the frame to create a background model
    if background_model is None:
        background_model = gray.copy().astype('float32')
    cv2.accumulateWeighted(gray, background_model, 0.01)

    # Compute the absolute difference between the current frame and the background
    diff = cv2.absdiff(gray, cv2.convertScaleAbs(background_model))

    # Threshold the difference to highlight moving objects
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours of the motion
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            log_event("Motion detected")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_label.config(text=f"Last Update: {timestamp}")

    # Display the result
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Repeat the detection every 100 milliseconds
    video_label.after(100, detect_motion)


def quit_app():
    cap.release()
    root.destroy()


# Create the main window
root = tk.Tk()
root.title("Motion Detection App")

# Open the camera
cap = cv2.VideoCapture(0)
_, prev_frame = cap.read()
background_model = None

# Create labels and buttons
video_label = tk.Label(root)
video_label.pack(padx=10, pady=10)

time_label = tk.Label(root, text="Last Update: -", font=("Helvetica", 10))
time_label.pack(pady=5)

event_log = tk.Text(root, height=10, width=50)
event_log.pack(padx=10, pady=10)

quit_button = ttk.Button(root, text="Quit", command=quit_app)
quit_button.pack(pady=10)

# Start motion detection
detect_motion()

# Start the Tkinter event loop
root.mainloop()
