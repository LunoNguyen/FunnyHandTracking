import cv2
import mediapipe as mp
import time
import numpy as np

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.imgs = []
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
            self.imgs.append(img)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def overlay_image(self, frame, overlay, x, y):
        """Overlay an image (with or without alpha) at position (x, y)"""
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

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append([id, cx, cy])

                if draw:
                    if id == 8:
                        if ((cx >= 100 and cx <= 175) and (cy >= 300 and cy <= 365)):
                            frame = self.overlay_image(frame, self.imgs[1], 400, 255)
                        if ((cx >= 250 and cx <= 300) and (cy >= 370 and cy <= 450)):
                            frame = self.overlay_image(frame, self.imgs[2], 400, 255)
        return lmList

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, frame = cap.read()

        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=True)
        if len(lmList) != 0:
            print(lmList[8])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, "FPS: " + str(int(fps)), (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)








if __name__ == "__main__":
    main()