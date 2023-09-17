import cv2
import numpy as np
import datetime
import os

# Create a directory to save recorded videos if it doesn't exist
if not os.path.exists("recorded_videos"):
    os.mkdir("recorded_videos")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

out_cartoon = None  # Initialize as None
cartoon_video_filename = None


# Function to apply cartoon effect to a frame
def cartoonize(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


recording = False
start_time = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Apply cartoon effect to the frame
    cartoon_frame = cartoonize(frame)

    # Display both the real video frame and the cartoon frame side by side with increased width
    merged_frame = np.hstack((frame, cartoon_frame))

    if recording:
        if out_cartoon is None:
            start_time = datetime.datetime.now()
            current_time = start_time
            cartoon_video_filename = f"recorded_videos/cartoon_video_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
            fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            out_cartoon = cv2.VideoWriter(cartoon_video_filename, fourcc_mp4, 20.0, (1280, 720))
        out_cartoon.write(cartoon_frame)

        # Calculate and display recording time
        recording_time = (datetime.datetime.now() - start_time).total_seconds()
        cv2.putText(merged_frame, f"Recording: {recording_time:.2f} seconds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
    else:
        if out_cartoon is not None:
            out_cartoon.release()
            video_length = (datetime.datetime.now() - start_time).total_seconds()
            print(
                f"Saved recorded cartoon video '{cartoon_video_filename}' with a length of {video_length:.2f} seconds.")
            out_cartoon = None

    cv2.imshow('Real and Cartoon Video', merged_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if not recording:
            print("Started recording.")
            recording = True
            start_time = datetime.datetime.now()
    elif key == ord('d'):
        if recording:
            print("Stopped recording.")
            recording = False
    elif key == ord('q'):
        if recording:
            print("Stopped recording.")
            recording = False
        break  # Close all operations

# Release resources
cap.release()
if out_cartoon is not None:
    out_cartoon.release()
cv2.destroyAllWindows()
