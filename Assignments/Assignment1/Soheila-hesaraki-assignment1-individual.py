import cv2
import numpy as np
from Image_processing_functions import *
from scipy.signal import convolve2d
from sklearn.metrics import mean_squared_error

path = "/esat/audioslave/shesarak/computer-vision/individual-assignment/"
input_video_file = path + "my_video.mp4"
output_video_file = path + "soheila_hesaraki.mp4"


# helper function to change what I do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


# OpenCV video objects to work with
cap = cv2.VideoCapture(input_video_file)
fps = int(round(cap.get(5)))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fource = cv2.VideoWriter_fourcc(*'mp4v')    # saving output video as .mp4
out = cv2.VideoWriter(output_video_file, fource, fps, (frame_width, frame_height))

# Define the size and position of the black rectangle to write my text on
rect_height = 80
rect_y = frame_height - rect_height
rect_color = (0, 0, 0)  # black color in BGR format

# Define the font and parameters for the subtitles
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # white color in BGR format
thickness = 2

fire_wheel_cap = cv2.VideoCapture("/esat/audioslave/shesarak/computer-vision/individual-assignment_/fire_wheel_video.mp4")
# cap.set(cv2.CAP_PROP_POS_FRAMES, int(51 * fps))
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.rectangle(frame, (0, rect_y), (frame_width, frame_height), rect_color, -1)

        if cv2.waitKey(28) & 0xFF == ord('q'):
            break
        cv2.imwrite(path + "frame.jpg", frame)
        if between(cap, 0, 1000):
            font_color = (255, 255, 0)  # yellow
            # Write the subtitles on the black rectangle
            cv2.putText(frame, "In Gray scale", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        elif between(cap, 1000, 2000):
            cv2.putText(frame, " Standard open CV color (BGR)", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)

        elif between(cap, 2000, 3000):
            cv2.putText(frame, "Gray scale again!", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        elif between(cap, 3000, 4000):
            cv2.putText(frame, " BGR agian!", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)

        elif between(cap, 4000, 5000):
            cv2.putText(frame, "Gray scale again!", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        elif between(cap, 5000, 6500):
            cv2.putText(frame, "Gaussian smoothing, kernel 5*5", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

        elif between(cap, 6500, 8000):
            frame = cv2.GaussianBlur(frame, (25, 25), 0)
            cv2.putText(frame, "More blur! kernel 25*25", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)

        elif between(cap, 8000, 10000):
            cv2.putText(frame, "Bilateral smoothing, kernel 15*15", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)
            frame = cv2.bilateralFilter(frame, 15, 75, 75)

        elif between(cap, 10000, 13000):
            font_scale = 0.70
            thickness = 1
            cv2.putText(frame, "Bilateral smoothing, kernel 35*35, blurs the edges more", (5, frame_height - 50),
                        font, font_scale, font_color, thickness)
            frame = cv2.bilateralFilter(frame, 35, 135, 135)

        elif between(cap, 15000, 17500):
            lower_RGB = np.array((140, 30, 25))
            upper_RGB = np.array((255, 90, 95))

            frame = grab_in_rgb(frame, lower_RGB, upper_RGB)
            font_color = (255, 255, 0)
            cv2.putText(frame, "Grab in RGB color space", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)

        elif between(cap, 17500, 20000):
            frame = grab_red_object_in_hsv(frame)

            font_color = (255, 255, 0)
            cv2.putText(frame, "Grab in HSV color space", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)

        elif between(cap, 20000, 22000):
            frame = sobel_horizontal_edge(frame)
            cv2.putText(frame, "Sobel filter for Horizontal edges", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)

        elif between(cap, 22000, 24000):
            frame = sobel_vertical_edges(frame)
            cv2.putText(frame, "Sobel filter for Vertical edges", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)

        elif between(cap, 24000, 25000):
            frame = sobel_combined_edges(frame)
            cv2.putText(frame, "Sobel filter for both!", (50, frame_height - 50),
                        font, font_scale, font_color, thickness)

        elif between(cap, 25000, 27000):
            frame = hugh(frame, step=0, MinDist=np.linspace(5, 150, 5),
                         MinRadius=np.around(np.linspace(10, 25, 5)).astype(int), Param2=np.linspace(10, 65, 5),
                         MaxRadius=np.around(np.linspace(600, 50, 5)).astype(int))

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 220), (frame_width, frame_height), (255, 255, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            font_scale, thickness = 0.7, 1
            font_color = (0, 0, 0)

            cv2.putText(frame, "The cv2.HoughCircles() function uses the Hough Transform to", (5, frame_height - 200),
                        font, font_scale, font_color, thickness)
            cv2.putText(frame, "detect circles, with parameters like maxRadius, minDist,", (5, frame_height - 150),
                        font, font_scale, font_color, thickness)
            cv2.putText(frame, "MinRadius, and param2 . I have increased all", (5, frame_height - 100),
                        font, font_scale, font_color, thickness)
            cv2.putText(frame, "parameters except maxRadius to improve detection", (5, frame_height - 50),
                        font, font_scale, font_color, thickness)

        if between(cap, 27000, 29000):
            frame = hugh(frame, step=1, MinDist=np.linspace(5, 150, 5),
                         MinRadius=np.around(np.linspace(10, 25, 5)).astype(int), Param2=np.linspace(10, 65, 5),
                         MaxRadius=np.around(np.linspace(600, 80, 5)).astype(int))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 120), (frame_width, frame_height), (255, 255, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            font_scale, thickness = 0.7, 1
            font_color = (0, 0, 0)

            cv2.putText(frame, "maxRadius specifies the maximum radius of the circles to", (5, frame_height - 100),
                        font, font_scale, font_color, thickness)
            cv2.putText(frame, "detect. I changed from 600 to 80", (5, frame_height - 50),
                        font, font_scale, font_color, thickness)

        if between(cap, 29000, 31000):
            frame = hugh(frame, step=2, MinDist=np.linspace(5, 150, 5),
                         MinRadius=np.around(np.linspace(10, 25, 5)).astype(int), Param2=np.linspace(10, 65, 5),
                         MaxRadius=np.around(np.linspace(600, 80, 5)).astype(int))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 220), (frame_width, frame_height), (255, 255, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            font_scale, thickness = 0.7, 1
            font_color = (0, 0, 0)

            cv2.putText(frame, "MinDist specifies the minimum distance between the centers", (5, frame_height - 200),
                        font, font_scale, font_color, thickness)
            cv2.putText(frame, "of detected circles. If multiple circles are detected within", (5, frame_height - 150),
                        font, font_scale, font_color, thickness)
            cv2.putText(frame, "this distance, only the circle with the highest accumulator", (5, frame_height - 100),
                        font, font_scale, font_color, thickness)
            cv2.putText(frame, "value is returned. I changed from 5 to 150.", (5, frame_height - 50),
                        font, font_scale, font_color, thickness)

        if between(cap, 31000, 33000):
            frame = hugh(frame, step=3, MinDist=np.linspace(5, 150, 5),
                         MinRadius=np.around(np.linspace(10, 25, 5)).astype(int), Param2=np.linspace(10, 65, 5),
                         MaxRadius=np.around(np.linspace(600, 80, 5)).astype(int))

            font_scale, thickness = 0.7, 1
            cv2.putText(frame, "I increased minRadius from 10 to 25, where minRadius specifies the minimum radius.", (50, frame_height - 50), font, font_scale, font_color, thickness)

        if between(cap, 33000, 36000):
            frame = hugh(frame, step=4, MinDist=np.linspace(5, 150, 5),
                         MinRadius=np.around(np.linspace(10, 25, 5)).astype(int), Param2=np.linspace(10, 65, 5),
                         MaxRadius=np.around(np.linspace(600, 80, 5)).astype(int))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 170), (frame_width, frame_height), (255, 255, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            font_scale, thickness = 0.7, 1
            font_color = (0, 0, 0)

            cv2.putText(frame, "param2 is the accumulator threshold for the circle centers", (5, frame_height - 150),
                        font, font_scale, font_color, thickness)
            cv2.putText(frame, "during detection. High values result in fewer detections. Low", (5, frame_height - 100),
                        font, font_scale, font_color, thickness)
            cv2.putText(frame, "values may lead to false detections. I changed from 10 to 65.", (5, frame_height - 50),
                        font, font_scale, font_color, thickness)

        if between(cap, 36500, 39000):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 70), (frame_width, frame_height), (255, 255, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            font_scale, thickness = 0.7, 1
            font_color = (0, 0, 0)
            cv2.putText(frame, "Detecting object with template matching", (5, frame_height - 50), font, font_scale,
                        font_color, thickness)
            draw_boundry(frame, path + 'template.jpg', threshold=0)

        if between(cap, 39000, 41000):
            frame = gray_scale_map(frame, path + 'template.jpg')

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 120), (frame_width, frame_height), (255, 255, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            font_scale, thickness = 0.7, 1
            font_color = (0, 0, 0)

            cv2.putText(frame, "This gray scale map shows the probability of the presence", (5, frame_height - 100),
                        font, font_scale, font_color, thickness)
            cv2.putText(frame, "of a given object in the video frame. brighter = more likely.",
                        (5, frame_height - 50), font, font_scale, font_color, thickness)

        if between(cap, 41000, 42000):
            continue

        if between(cap, 44000, 47000):
            template_folder = "/esat/audioslave/shesarak/computer-vision/individual-assignment/template_folder/"
            draw_boundry(frame, template_folder + "template1.jpg", threshold=0.8)
            font_scale, thickness = 0.7, 1
            cv2.putText(frame, "Object detection for various orientations and motions", (50, frame_height - 50), font,
                        font_scale, font_color, thickness)

        if between(cap, 47000, 48000):
            frame = bright(frame, -100)
            cv2.putText(frame, "Make it darker!", (50, frame_height - 50), font,
                        font_scale, font_color, thickness)

        if between(cap, 48000, 49000):
            frame = sepia(frame)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 60), (frame_width, frame_height), (255, 255, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            font_scale, thickness = 0.7, 1
            font_color = (0, 0, 0)

            cv2.putText(frame, "Sepia filter adds a warm brown effect!", (5, frame_height - 50),
                        font, font_scale, font_color, thickness)

        if between(cap, 49000, 50000):
            frame = sharpen(frame)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 60), (frame_width, frame_height), (255, 255, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            font_scale, thickness = 0.7, 1
            font_color = (0, 0, 0)

            cv2.putText(frame, "Sharp effect!", (5, frame_height - 50),
                        font, font_scale, font_color, thickness)

        if between(cap, 50000, 51000):
            frame = pencil_sketch_col(frame)
            cv2.putText(frame, "Pencil sketch effect!", (50, frame_height - 50), font,
                        font_scale, font_color, thickness)

        if between(cap, 51000, 52000):
            frame = invert(frame)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 60), (frame_width, frame_height), (255, 255, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            font_scale, thickness = 0.7, 1
            font_color = (0, 0, 0)

            cv2.putText(frame, "Inverting colors!", (5, frame_height - 50),
                        font, font_scale, font_color, thickness)

        if between(cap, 51000, 53000):
            frame = pencil_sketch_grey(frame)
            font_scale, thickness = 0.7, 2
            cv2.putText(frame, "Pencil sketch with gray effect!", (50, frame_height - 50), font, font_scale, font_color, thickness)

        if between(cap, 54000, 55000):
            front_wheel = cv2.imread(path + "front_wheel.jpg")
            front_center, front_radius = sift_circular_detection(frame, front_wheel)
            img_with_circle = cv2.circle(frame, front_center, radius=front_radius-1, color=(0, 255, 0), thickness=2)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 60), (frame_width, frame_height), (255, 255, 255), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            font_scale, thickness = 0.7, 1
            font_color = (0, 0, 0)

            cv2.putText(frame, "Circle detection with SIFT!", (5, frame_height - 50),
                        font, font_scale, font_color, thickness)

        if between(cap, 55500, 59000):
            frame_copy = frame.copy()
            front_wheel = cv2.imread(path + "front_wheel.jpg")
            front_center, front_radius = sift_circular_detection(frame, front_wheel)

            ret_w, wheel_img = fire_wheel_cap.read()
            if ret_w:
                object_to_detect = crop_image(wheel_img, 254, 64, 722, 504)

                cropped_wheel_img = circular_crop(wheel_img, object_to_detect)

                resized_wheel = circular_resize(wheel_img, cropped_wheel_img, front_radius)

                frame = replace_circular_images(frame, front_wheel, resized_wheel)

                frame = remove_black_parts(frame, frame_copy)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, frame_height - 120), (frame_width, frame_height), (255, 255, 255), -1)
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                font_scale, thickness = 0.7, 2
                font_color = (0, 0, 0)

                cv2.putText(frame, "Mixing two videos! Replacing the motorbike's wheel with", (5, frame_height - 100),
                            font, font_scale, font_color, thickness)
                cv2.putText(frame, "a moving fire wheel!",
                            (5, frame_height - 50), font, font_scale, font_color, thickness)

        if between(cap, 61000, 63000):
            continue

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        out.write(frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture and writing object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
