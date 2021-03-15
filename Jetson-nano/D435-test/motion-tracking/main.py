
import pyrealsense2 as rs
import numpy as np
import datetime
import imutils
import cv2

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor_id = %d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def take_picture(num):
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=2))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        result = True
        # Window
        while (cv2.getWindowProperty("CSI Camera", 0) >= 0) & result:
            ret_val, img = cap.read()
            #cv2.imshow("CSI Camera", img)
            name = str(num) + ".jpg"
            cv2.imwrite(name,img)
            result = False
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        firstFrame = None
        count = 0
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            fr = imutils.resize(color_image, width=500)
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if firstFrame is None:
                firstFrame = gray
                continue

            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            for c in cnts:
                if cv2.contourArea(c) < 1000:
                    continue

                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(fr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"
                cv2.putText(fr, "Room Status: {}".format(text), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(fr, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                            (10, fr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            cv2.imshow("Security Feed", fr)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Frame Delta", frameDelta)

            keyCode = cv2.waitKey(30)
            if keyCode == 27:
                break
            elif keyCode == ord('s'):
                print("something")
                count = count + 1
                take_picture(count)
        cv2.destroyAllWindows()

    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    main()