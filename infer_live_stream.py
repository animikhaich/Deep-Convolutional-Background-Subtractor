import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import numpy as np
import cv2


model = tf.keras.models.load_model("weights.h5")

target_shape = (160, 160)


def letterbox_image(image, size):
    ih, iw = image.shape[:2]
    w, h = size
    scale = min(w / iw, h / ih)

    # Get the new scaled width and height
    nw = int(scale * iw)
    nh = int(scale * ih)

    # Resize Image based on it's initial size maintaining original aspect ratio
    if nw > iw or nh > ih:
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    else:
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    # Create a blank template image
    new_image = np.zeros((h, w, 3), np.uint8)

    # Calculate the offsets
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    new_image[dy : dy + nh, dx : dx + nw] = image

    return new_image


def preprocess(frame, letterbox=True):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if letterbox:
        frame = letterbox_image(frame, target_shape)
    else:
        frame = cv2.resize(frame, target_shape)
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame


def extract_largest_contour(image):
    """Detect and Extract the Largest Contour from the mask"""
    # Find the contours in the mask
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return image

    # Find the largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    contour_sizes = sorted(contour_sizes, key=lambda x: x[0], reverse=True)
    largest_contour = contour_sizes[0][1]

    # Draw the largest contour in the image
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    return mask


def predict(image, threshold=0.5, letterbox=True):
    # Keep a copy of the original image
    bgr_image = image.copy()

    # Original Image Dims
    h, w = image.shape[:2]

    # Preprocess and Infer
    image = preprocess(image, letterbox)
    pred = np.squeeze(model.predict(image))

    # Save Pred Image
    pred_image = (np.dstack([pred.copy()] * 3) * 255).astype(np.uint8)
    pred_image = cv2.resize(pred_image, (w, h), interpolation=cv2.INTER_CUBIC)

    # Threshold the prediction image
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1

    # Convert to 8 bit image
    pred = (pred * 255).astype(np.uint8)

    # Close the holes
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, np.ones([5, 5]), iterations=2)

    # Extract Largest Contour
    pred = extract_largest_contour(pred)

    # Convert Grayscale to BGR
    pred_mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    # Apply Mask to a colored image
    merged_image = np.squeeze(cv2.bitwise_and(bgr_image, bgr_image, mask=pred_mask))

    # Convert to RGB
    pred_mask = np.dstack([pred_mask] * 3)

    return pred_image, pred_mask, merged_image


cap = cv2.VideoCapture("http://192.168.0.118:4747/video")

# class CustomVideoWriter:
#     def __init__(self, filename):
#         self.filename = filename

#     def init_video_writer(frame):


writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 960))

while True:
    ret, frame = cap.read()

    if frame is None:
        break

    pred_image, pred_mask, merged_image = predict(frame, threshold=0.5, letterbox=True)

    row_1 = np.hstack((frame, pred_image))
    row_2 = np.hstack((pred_mask, merged_image))

    frame = np.vstack((row_1, row_2))

    cv2.imshow("Result", frame)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
writer.release()
