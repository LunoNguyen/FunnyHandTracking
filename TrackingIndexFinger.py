import cv2
from HandTrackingModule import HandDetector
import time
import numpy as np

def init_image():
    imgs = []
    image_paths = [
        'Images/67.jpg',
        'Images/aha-aha-monkey.png',
        'Images/images.jpg'
    ]

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not load image: {path}")
            # Create a blank red image as fallback
            img = np.zeros((80, 80, 3), dtype=np.uint8)
            img[:] = (0, 0, 255)
        else:
            # Resize properly using cv2
            img = cv2.resize(img, (150, 150))
        imgs.append(img)
    return imgs

def overlay_image(frame, overlay, x, y):
    h, w = overlay.shape[:2]

    # Prevent going out of frame bounds
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame

    # Handle 4-channel (RGBA) vs 3-channel images
    if overlay.shape[2] == 4:  # PNG with transparency
        overlay_rgb = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])

        roi = frame[y:y + h, x:x + w]
        foreground = cv2.multiply(overlay_rgb.astype(float), alpha)
        background = cv2.multiply(roi.astype(float), 1.0 - alpha)
        blended = cv2.add(foreground, background).astype(np.uint8)
        frame[y:y + h, x:x + w] = blended
    else:  # Regular image (no transparency)
        frame[y:y + h, x:x + w] = overlay

    return frame

def draw(frame, imgs, id, cx, cy):
    if id == 8:
        if ((cx >= 100 and cx <= 175) and (cy >= 300 and cy <= 365)):
            frame = overlay_image(frame, imgs[1], 400, 255)
        elif ((cx >= 250 and cx <= 300) and (cy >= 370 and cy <= 450)):
            frame = overlay_image(frame, imgs[2], 400, 255)
    return frame

def main():
    pTime = 0
    cTime = 0
    imgs = init_image()

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, frame = cap.read()

        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0:
            print(lmList[8])
            frame = draw(frame, imgs, lmList[8][0], lmList[8][1], lmList[8][2])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, "FPS: " + str(int(fps)), (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

