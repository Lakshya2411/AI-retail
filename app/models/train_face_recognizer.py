import os
import joblib
import cv2
import numpy as np

def generate_synthetic_face(customer_id, variation):
    # Create a 100x100 grayscale image
    img = np.zeros((100, 100), dtype=np.uint8)
    
    # Draw a face silhouette
    cv2.circle(img, (50, 50), 40, 200, -1)
    
    # Draw eyes based on customer ID to make the faces distinct
    if customer_id == 1: # Alice
        cv2.circle(img, (35, 40), 5, 50, -1) # left eye
        cv2.circle(img, (65, 40), 5, 50, -1) # right eye
        cv2.ellipse(img, (50, 65), (15, 10), 0, 0, 180, 50, -1) # smile
    elif customer_id == 2: # Bob
        cv2.rectangle(img, (30, 35), (40, 45), 50, -1) # left square eye
        cv2.rectangle(img, (60, 35), (70, 45), 50, -1) # right square eye
        cv2.line(img, (35, 65), (65, 65), 50, 3) # straight mouth
    else: # Charlie
        cv2.circle(img, (35, 40), 7, 80, -1) # left eye
        cv2.circle(img, (65, 40), 7, 80, -1) # right eye
        cv2.ellipse(img, (50, 60), (10, 15), 0, 0, 360, 50, -1) # open mouth
        
    # Add random noise variation to simulate different capture conditions
    np.random.seed(customer_id * 100 + variation)
    noise = np.random.randint(-15, 15, (100, 100)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Blur slightly to simulate camera lens
    img_noisy = cv2.GaussianBlur(img_noisy, (3, 3), 0)
    
    return img_noisy

def train_face_recognition():
    print("Generating synthetic face dataset for Alice, Bob, and Charlie...")
    
    faces = []
    labels = []
    
    # Customer DB mapping
    customer_db = {
        1: {"name": "Alice", "loyalty_points": 120, "last_visit": "2026-08-01 14:30"},
        2: {"name": "Bob", "loyalty_points": 350, "last_visit": "2026-07-30 11:15"},
        3: {"name": "Charlie", "loyalty_points": 50, "last_visit": "2026-08-02 09:00"}
    }
    
    # Create 15 samples per customer with different noise variations
    for cid in customer_db.keys():
        for var in range(15):
            face_img = generate_synthetic_face(cid, var)
            faces.append(face_img)
            labels.append(cid)
            
    print(f"Generated {len(faces)} face samples. Training LBPH Face Recognizer...")
    
    # Create and train LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(labels))
    
    # Save the model
    os.makedirs("app/models", exist_ok=True)
    xml_path = "app/models/face_recognizer.xml"
    db_path = "app/models/face_db.pkl"
    
    print(f"Saving face recognizer xml to {xml_path}...")
    recognizer.write(xml_path)
    
    print(f"Saving face database pkl to {db_path}...")
    joblib.dump(customer_db, db_path)
    
    print("Face recognition model training complete!")

if __name__ == "__main__":
    train_face_recognition()
