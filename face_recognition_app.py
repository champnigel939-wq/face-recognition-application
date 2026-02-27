"""
Professional Face Recognition Application with Database Integration
===================================================================

A real-time face recognition system with persistent storage of:
- Face encodings in SQLite database
- Recognition history and statistics
- Unknown face tracking
- Person metadata

Author: Computer Vision Engineer
Date: 2026
"""

import cv2
import numpy as np
import face_recognition
import os
from pathlib import Path
from datetime import datetime
import sys
import time

# Import configuration and database
try:
    from config import *
    from database import FaceDatabase
except ImportError as e:
    print(f"❌ Error: Missing required module: {e}")
    sys.exit(1)


# ============================================================================
# GLOBAL DATABASE INSTANCE
# ============================================================================
db = None  # Will be initialized in main()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def log_message(message, level="INFO"):
    """
    Print formatted log messages to console.
    
    Args:
        message (str): Message to display
        level (str): Log level - INFO, SUCCESS, WARNING, ERROR
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if level == "SUCCESS":
        print(f"✅ [{timestamp}] {message}")
    elif level == "WARNING":
        print(f"⚠️  [{timestamp}] {message}")
    elif level == "ERROR":
        print(f"❌ [{timestamp}] {message}")
    else:
        print(f"ℹ️  [{timestamp}] {message}")

def load_and_encode_known_faces():
    """
    Load all images from known_faces directory and encode them.
    Uses database for persistent storage if enabled.
    
    Returns:
        tuple: (known_encodings, known_labels)
    """
    
    known_encodings = []
    known_labels = []
    
    log_message(f"Loading known faces from {KNOWN_FACES_DIR}...", "INFO")
    
    if not KNOWN_FACES_DIR.exists():
        log_message(f"Directory {KNOWN_FACES_DIR} does not exist.", "WARNING")
        return [], []
    
    person_folders = [f for f in KNOWN_FACES_DIR.iterdir() if f.is_dir()]
    
    if not person_folders:
        log_message(
            f"No person folders found in {KNOWN_FACES_DIR}",
            "WARNING"
        )
        return [], []
    
    total_images_processed = 0
    total_faces_encoded = 0
    
    for person_folder in person_folders:
        person_name = person_folder.name
        person_images = list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.png"))
        
        if not person_images:
            log_message(f"No images found for {person_name}.", "WARNING")
            continue
        
        person_encodings_count = 0
        
        for image_path in person_images:
            try:
                total_images_processed += 1
                
                image = cv2.imread(str(image_path))
                if image is None:
                    log_message(f"Failed to load image: {image_path}", "WARNING")
                    continue
                
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(
                    rgb_image,
                    model=FACE_DETECTION_MODEL
                )
                
                if not face_locations:
                    log_message(f"No face detected in {image_path}", "WARNING")
                    continue
                
                face_encoding = face_recognition.face_encodings(
                    rgb_image,
                    face_locations
                )[0]
                
                known_encodings.append(face_encoding)
                known_labels.append(person_name)
                person_encodings_count += 1
                total_faces_encoded += 1
                
                # Store in database if enabled
                if USE_DATABASE and db:
                    db.add_face_encoding(person_name, face_encoding, str(image_path))
                
                if DEBUG_MODE or SHOW_ENCODING_MESSAGES:
                    log_message(
                        f"Encoded face for {person_name}: {image_path.name}",
                        "SUCCESS"
                    )
                
            except Exception as e:
                log_message(f"Error processing {image_path}: {str(e)}", "ERROR")
                continue
        
        if person_encodings_count > 0:
            log_message(
                f"✓ {person_name}: {person_encodings_count} faces encoded",
                "SUCCESS"
            )
    
    log_message(
        f"Total: {total_images_processed} images processed, "
        f"{total_faces_encoded} faces encoded",
        "INFO"
    )
    
    return known_encodings, known_labels

def load_encodings_from_database():
    """
    Load face encodings from database instead of files.
    More efficient if database is already populated.
    
    Returns:
        tuple: (known_encodings, known_labels)
    """
    
    if not USE_DATABASE or not db:
        return [], []
    
    log_message("Loading encodings from database...", "INFO")
    
    encodings, labels = db.get_all_encodings_with_labels()
    
    log_message(f"Loaded {len(encodings)} encodings from database", "SUCCESS")
    
    return encodings, labels

def recognize_face(face_encoding, known_encodings, known_labels):
    """
    Recognize a face by comparing its encoding with known encodings.
    
    Args:
        face_encoding: The encoding of the face to recognize
        known_encodings: List of known face encodings
        known_labels: List of corresponding person names
    
    Returns:
        tuple: (name, confidence)
    """
    
    if not known_encodings:
        return "Unknown", 0.0
    
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(distances)
    best_distance = distances[best_match_index]
    confidence = 1 - best_distance
    
    if best_distance <= (1 - RECOGNITION_THRESHOLD):
        name = known_labels[best_match_index]
        return name, confidence
    
    return "Unknown", confidence

def save_unknown_face(frame, face_location):
    """
    Save an unknown face to disk and database.
    
    Args:
        frame: The video frame containing the face
        face_location: Tuple of (top, right, bottom, left) coordinates
    """
    
    if not AUTO_SAVE_UNKNOWN:
        return
    
    try:
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = UNKNOWN_FACES_DIR / f"unknown_{timestamp}{SAVE_IMAGE_FORMAT}"
        
        cv2.imwrite(str(filename), face_image)
        
        if DEBUG_MODE:
            log_message(f"Saved unknown face: {filename.name}", "SUCCESS")
        
        # Store in database if enabled
        if USE_DATABASE and db and SAVE_UNKNOWN_FACE_IMAGES:
            # Extract face encoding
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                [face_location]
            )
            if face_encodings:
                db.add_unknown_face(face_encodings[0], str(filename))
            
    except Exception as e:
        log_message(f"Error saving unknown face: {str(e)}", "ERROR")

def initialize_camera():
    """
    Initialize and test the camera/webcam.
    
    Returns:
        cv2.VideoCapture: Camera object, or None if failed
    """
    
    log_message(f"Initializing camera (index: {CAMERA_INDEX})...", "INFO")
    
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            log_message(
                f"Failed to open camera at index {CAMERA_INDEX}.",
                "ERROR"
            )
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        log_message("Camera initialized successfully", "SUCCESS")
        return cap
        
    except Exception as e:
        log_message(f"Error initializing camera: {str(e)}", "ERROR")
        return None

def process_frame(frame, known_encodings, known_labels):
    """
    Process a single frame: detect and recognize faces.
    
    Args:
        frame: Video frame (numpy array)
        known_encodings: List of known face encodings
        known_labels: List of person names
    
    Returns:
        tuple: (processed_frame, faces_data)
    """
    
    faces_data = []
    
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=FRAME_RESIZE_FACTOR,
        fy=FRAME_RESIZE_FACTOR
    )
    
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(
        rgb_small_frame,
        model=FACE_DETECTION_MODEL
    )
    
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        name, confidence = recognize_face(face_encoding, known_encodings, known_labels)
        
        faces_data.append({
            'name': name,
            'confidence': confidence,
            'location': face_location
        })
        
        # Log to database if enabled
        if USE_DATABASE and db and LOG_RECOGNITIONS:
            db.log_recognition(name, confidence)
        
        # Save unknown faces
        if name == "Unknown":
            save_unknown_face(frame, face_location)
    
    processed_frame = draw_face_labels(frame, faces_data)
    
    return processed_frame, faces_data

def draw_face_labels(frame, faces_data):
    """Draw rectangles and labels on detected faces."""
    
    output_frame = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    
    for face in faces_data:
        top = int(face['location'][0] / FRAME_RESIZE_FACTOR)
        right = int(face['location'][1] / FRAME_RESIZE_FACTOR)
        bottom = int(face['location'][2] / FRAME_RESIZE_FACTOR)
        left = int(face['location'][3] / FRAME_RESIZE_FACTOR)
        
        if face['name'] == "Unknown":
            color = UNKNOWN_RECTANGLE_COLOR
            label = "Unknown"
        else:
            color = RECTANGLE_COLOR
            label = f"{face['name']} ({face['confidence']:.2f})"
        
        cv2.rectangle(
            output_frame,
            (left, top),
            (right, bottom),
            color,
            RECTANGLE_THICKNESS
        )
        
        label_y = top - 10 if top > 30 else bottom + 20
        cv2.putText(
            output_frame,
            label,
            (left, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            color,
            FONT_THICKNESS
        )
    
    return output_frame

def display_info_panel(frame, fps, known_count, unknown_count):
    """Display information panel on the frame."""
    
    output_frame = frame.copy()
    height, width = frame.shape[:2]
    
    panel_height = 100
    overlay = output_frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    output_frame = cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0)
    
    if SHOW_FPS:
        info_text = [
            f"FPS: {fps:.1f}",
            f"Known: {known_count} | Unknown: {unknown_count}",
            "Press 'Q' to quit | 'S' to save | 'H' for help"
        ]
    else:
        info_text = [
            f"Known: {known_count} | Unknown: {unknown_count}",
            "Press 'Q' to quit | 'S' to save | 'H' for help"
        ]
    
    y_offset = 25
    for text in info_text:
        cv2.putText(
            output_frame,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        y_offset += 25
    
    return output_frame

def print_help():
    """Print keyboard shortcuts and help information."""
    print("\n" + "="*70)
    print(" KEYBOARD SHORTCUTS ")
    print("="*70)
    print("Q        - Quit application")
    print("S        - Save current frame")
    print("H        - Print this help message")
    print("D        - Display database statistics")
    print("="*70 + "\n")

def print_database_stats():
    """Print database statistics."""
    if not USE_DATABASE or not db:
        log_message("Database is not enabled", "WARNING")
        return
    
    print("\n" + "="*70)
    print(" DATABASE STATISTICS ")
    print="="*70)
    
    summary = db.get_database_summary()
    print(f"Total Persons:        {summary.get('total_persons', 0)}")
    print(f"Total Encodings:      {summary.get('total_encodings', 0)}")
    print(f"Total Recognitions:   {summary.get('total_recognitions', 0)}")
    print(f"Unknown Faces:        {summary.get('unknown_faces', 0)}")
    print(f"Database File:        {summary.get('database_file', 'N/A')}")
    print("="*70 + "\n")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application loop."""
    
    global db
    
    print("\n" + "="*70)
    print(" FACE RECOGNITION APPLICATION WITH DATABASE ")
    print("="*70 + "\n")
    
    # Initialize database if enabled
    if USE_DATABASE:
        try:
            db = FaceDatabase(str(DATABASE_PATH))
            log_message("Database initialized successfully", "SUCCESS")
        except Exception as e:
            log_message(f"Error initializing database: {e}", "ERROR")
            db = None
    
    # Load face encodings
    known_encodings, known_labels = load_and_encode_known_faces()
    
    # Initialize camera
    cap = initialize_camera()
    if cap is None:
        log_message("Failed to initialize camera.", "ERROR")
        return
    
    log_message("Starting face recognition loop. Press 'Q' to quit.", "INFO")
    print("-" * 70 + "\n")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                log_message("Failed to read frame from camera", "ERROR")
                break
            
            frame_count += 1
            
            if frame_count % FRAME_SKIP != 0:
                continue
            
            processed_frame, faces_data = process_frame(
                frame,
                known_encodings,
                known_labels
            )
            
            known_count = sum(1 for f in faces_data if f['name'] != "Unknown")
            unknown_count = len(faces_data) - known_count
            
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps = 30 / elapsed
                fps_start_time = time.time()
            
            final_frame = display_info_panel(
                processed_frame,
                fps,
                known_count,
                unknown_count
            )
            
            cv2.imshow("Face Recognition - Press 'Q' to quit", final_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                log_message("Quit signal received.", "INFO")
                break
            elif key == ord('s') or key == ord('S'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}.jpg"
                cv2.imwrite(filename, final_frame)
                log_message(f"Frame saved: {filename}", "SUCCESS")
            elif key == ord('h') or key == ord('H'):
                print_help()
            elif key == ord('d') or key == ord('D'):
                print_database_stats()
            
    except KeyboardInterrupt:
        log_message("Application interrupted by user.", "WARNING")
    
    except Exception as e:
        log_message(f"Unexpected error: {str(e)}", "ERROR")
    
    finally:
        log_message("Cleaning up resources...", "INFO")
        cap.release()
        cv2.destroyAllWindows()
        
        if db:
            db.close()
        
        log_message("Application closed successfully.", "SUCCESS")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()