import mediapipe as mp
import cv2
import csv
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

class SignLanguageTrainer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.csv_path = 'sign_language_data.csv'
        self.model_path = 'sign_language_model.pkl'
        
    def setup_csv(self):
        """Initialize CSV with headers"""
        num_coords = 21  # MediaPipe Hands has 21 landmarks
        landmarks = ['class']
        for val in range(1, num_coords + 1):
            landmarks += [f'x{val}', f'y{val}', f'z{val}']
        
        try:
            with open(self.csv_path, mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)
            return True
        except Exception as e:
            print(f"Error creating CSV file: {e}")
            return False

    def collect_training_data(self, sign_label):
        """Collect training data for a specific sign label"""
        if not os.path.exists(self.csv_path):
            print("Training file not found. Please set up a new training file first.")
            return

        cap = cv2.VideoCapture(0)  # Default camera
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        with self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
            start_time = time.time()  # Start timer
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                elapsed_time = time.time() - start_time  # Calculate elapsed time
                if elapsed_time > 3:  # Stop after 3 seconds
                    print("Data collection complete.")
                    break
                
                # Process the image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                results = hands.process(image)
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                        
                        # Extract hand landmarks
                        hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] 
                                                  for landmark in hand_landmarks.landmark]).flatten())
                        
                        # Add label to the row
                        row = [sign_label] + hand_row
                        
                        # Save to CSV
                        with open(self.csv_path, mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(row)
                        
                # Show the frame
                cv2.imshow('Sign Language Data Collection', image)
                cv2.waitKey(10)  # Small delay to refresh frames

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Prevent window hanging

    def train_model(self):
        """Train the model using collected data"""
        if not os.path.exists(self.csv_path):
            print("No training data found. Please collect training data first.")
            return
            
        try:
            df = pd.read_csv(self.csv_path)
            if len(df) == 0:
                print("Training file is empty. Please collect training data first.")
                return

            X = df.drop('class', axis=1)
            y = df['class']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            pipeline = make_pipeline(StandardScaler(), LogisticRegression())
            model = pipeline.fit(X_train, y_train)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model, f)
            
            accuracy = model.score(X_test, y_test)
            print(f"Model accuracy: {accuracy:.2%}")
            print(f"Model saved as {self.model_path}")

        except Exception as e:
            print(f"Error during training: {e}")

def main():
    trainer = SignLanguageTrainer()
    
    while True:
        print("\nSign Language Trainer Menu:")
        print("1. Setup new training file")
        print("2. Collect training data")
        print("3. Train model")
        print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                if trainer.setup_csv():
                    print("New training file created successfully!")
                
            elif choice == '2':
                label = input("Enter the label for this sign (e.g., 'A', 'B', 'C'): ").strip()
                if label:
                    print(f"Collecting training data for label '{label}'... This will run for 3 seconds.")
                    time.sleep(1)  # Add 1 second delay after label input
                    trainer.collect_training_data(label)
                else:
                    print("Label cannot be empty. Please try again.")
            
            elif choice == '3':
                print("Training model...")
                trainer.train_model()
            
            elif choice == '4':
                print("Exiting...")
                break
            
            else:
                print("Invalid choice! Please enter a number between 1 and 4.")
                
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

main()
