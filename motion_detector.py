"""
This app detects moving object in front of computer webcam.
It also writes the rectangle on a moving object.
It records the times when the movement starts and the times when the movement ends.
It also saves the video with the movement to 'output.avi'.
"""

from datetime import datetime
import cv2
import pandas

# A reference frame - static background
first_frame = None
frame_number = 0
# A list of status of moving objects, 0 = no moving, 1 = moving
status_list = [None, None] # we put None so we don't get error 'index out of bounds'
# The list of times when movement was detected
times = []
# DataFrame where time of movements will be stored
df = pandas.DataFrame(columns=['Start', 'End'])
# Definition of video
video = cv2.VideoCapture(0) #index of camera - I have only one, so index is 0
# Definition of the codec and filename for saving the video
path = 'output.avi'
frame_width = int(video.get(3))
frame_height = int(video.get(4))
out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:

    # Get video frames
    check, frame = video.read() # Use print(frame) to se the pixels of the frame

    # Define status of moving objects,  0 = no moving, 1 = moving
    status = 0

    # Skip first 100 frames so camera adjusts
    frame_number = frame_number + 1
    if frame_number < 100:
        continue # continue to the beginning of the loop

    # Change from color to gray - its better to use grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blurs the image thus removing the noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Save reference frame - static background
    if first_frame is None:
        first_frame = gray
        continue # continue to the beginning of the loop

    # Compare frame with the reference frame
    delta_frame = cv2.absdiff(first_frame, gray)

    # Set a threshold to the delta_frame to detect only big difference (> 50 pixels)
    # 50 = threshold limit, 255 = assigned value over treshold limit
    # cv2.threshold returns a tuple, we only need the second value which is frame
    threshold_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]

    # Smooth the threshold frame
    threshold_frame = cv2.dilate(threshold_frame, None, iterations = 2)  # to smooth the frame

    # Find contours on the threshold frame
    (contours, _) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # _ because it is not important for us

    # Write a rectangle around the contour
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 3) # writes directly on frame

    # In case we find any contours on the frame, that means we have a movement
    if len(contours) != 0:
        status = 1
        out.write(frame)
    # Write status in the status_list
    status_list.append(status)
    # Keep only last two items in the list so we don't get memory problems
    status_list = status_list[-2:]
    # Define the start/end of the movement
    if status_list[-1] == 1 and status_list[-2] == 0: # movement started
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1: # movement ended
        times.append(datetime.now())

    # Show the frames
    cv2.imshow('Gray frame', gray)
    cv2.imshow('Delta frame', delta_frame)
    cv2.imshow('Threshold frame', threshold_frame)
    cv2.imshow('Color frame', frame)

    # It waits for 1ms; in case we press q the loop breaks and the script stops
    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now()) #in case if we turn off camera with moving object
        break

# From times list write to df and save it to .csv file
for i in range(0, len(times), 2):
    df = df.append({'Start': times[i], 'End': times[i+1]}, ignore_index = True)
df.to_csv('Times.csv')

video.release()
out.release()
cv2.destroyAllWindows()
