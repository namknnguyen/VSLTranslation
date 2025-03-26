import cv2
import numpy as np
import pickle
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import os

class SignLanguageInference:
    def __init__(self, model_path='sign_language_model.pkl', font_path='custom_font.ttf'):
        # Load the trained model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # MediaPipe hands setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # Load custom font
        self.font_path = font_path
        if os.path.exists(self.font_path):
            try:
                self.font = ImageFont.truetype(self.font_path, 64)  # Increased font size to 64
            except Exception as e:
                print(f"Warning: Could not load custom font. Using default font. Error: {e}")
                self.font = None
        else:
            print("Warning: Custom font not found. Using default font.")
            self.font = None

    def overlay_text(self, frame, text, position, font_color=(0, 255, 0)):
        """Overlay text on the frame using Pillow for custom fonts."""
        # Convert OpenCV image (BGR) to PIL image (RGB)
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # Use the custom font if available, otherwise use default OpenCV font
        if self.font:
            draw.text(position, text, font=self.font, fill=font_color)
        else:
            # Convert back to OpenCV format and use OpenCV text rendering
            frame = cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)

        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    def predict(self, image, hands):
        """Predict the sign from a given image."""
        # Convert to RGB as MediaPipe expects RGB images
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            # Flatten the hand landmarks into a single vector
            hand_landmarks = results.multi_hand_landmarks[0]  # For single hand detection
            hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] 
                                      for landmark in hand_landmarks.landmark]).flatten())

            # Make prediction using the model
            prediction = self.model.predict([hand_row])
            return prediction[0]  # Return predicted label
        return None

    def start_inference(self):
        """Start the inference process."""
        cap = cv2.VideoCapture(0)  # Open the webcam

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Perform inference on the frame
                predicted_label = self.predict(frame, hands)

                # Overlay the predicted label on the frame
                if predicted_label is not None:
                    frame = self.overlay_text(frame, f"Predicted Sign: {predicted_label}", (10, 70))

                # Process the image and draw landmarks
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Display the image
                cv2.imshow('Sign Language Inference', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on pressing 'q'
                    break

        cap.release()
        cv2.destroyAllWindows()

# Run the inference
inference = SignLanguageInference()
inference.start_inference()
