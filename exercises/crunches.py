import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from utils import calculate_angle

# --- Global Audio Engine Setup ---
audio_engine = pyttsx3.init()
last_feedback_spoken = ""
last_spoken_time = 0

def say_feedback(text):
    """Speaks the feedback text if it's new and hasn't been spoken recently."""
    global last_feedback_spoken, last_spoken_time
    current_time = time.time()
    if text and text != last_feedback_spoken and current_time - last_spoken_time > 2:
        last_feedback_spoken = text
        last_spoken_time = current_time
        audio_engine.say(text)
        audio_engine.runAndWait()

# --- Main Exercise Function ---
def run_crunches():
    """
    Final corrected version with a transparent status box and properly scaled skeleton.
    """
    cap = cv2.VideoCapture(0)
    
    WINDOW_NAME = 'POSECOUNTER'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    # --- State and Counter Variables ---
    rep_count, stage, feedback = 0, "DOWN", "Starting Up"
    
    # --- Tempo and Symmetry Variables ---
    down_start_time, up_start_time = time.time(), None
    SYMMETRY_THRESHOLD = 20
    
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    ) as pose:
        
        say_feedback("Starting crunches. Get in position.")

        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            
            # For performance, we process a smaller image
            process_w, process_h = 640, 480
            resized_for_processing = cv2.resize(image, (process_w, process_h))
            image_rgb = cv2.cvtColor(resized_for_processing, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image_rgb)
            
            # --- MODIFIED: Create a semi-transparent overlay ---
            overlay = image.copy()
            alpha = 0.6  # Transparency factor
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1) # Draw black box on the overlay
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0) # Blend overlay with the original image

            # Initialize metrics for the data panel
            abdomen_angle, knee_angle = 0, 0
            hands_behind_head = False
            posture_feedback = "No Person Detected"
            feedback_color = (0, 0, 255)

            if results.pose_landmarks:
                # --- BUG FIX: Scale landmarks from the small processed image to the large display image ---
                # We create a new PoseLandmarks object to hold the scaled landmarks
                scaled_landmarks = mp_pose.PoseLandmarks()
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    # To scale, we multiply the normalized coords by the large window's width/height
                    # The drawing utility will then use these pixel values correctly.
                    # This is a bit of a workaround as the drawing utility expects a specific object.
                    scaled_landmarks.landmark.add(
                        x=landmark.x * w, y=landmark.y * h, z=landmark.z, visibility=landmark.visibility
                    )
                
                # Draw the scaled skeleton
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks, # Pass original for connections
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                try:
                    # Use the UN SCALED landmarks for calculations (they are normalized 0-1)
                    landmarks = results.pose_landmarks.landmark
                    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                    r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                    r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

                    # All calculations remain the same
                    abdomen_angle = calculate_angle([l_shoulder.x, l_shoulder.y], [l_hip.x, l_hip.y], [l_knee.x, l_knee.y])
                    knee_angle = calculate_angle([l_hip.x, l_hip.y], [l_knee.x, l_knee.y], [l_ankle.x, l_ankle.y])
                    
                    wrist_ear_dist_l = np.linalg.norm(np.array([l_wrist.x, l_wrist.y]) - np.array([l_ear.x, l_ear.y]))
                    wrist_ear_dist_r = np.linalg.norm(np.array([r_wrist.x, r_wrist.y]) - np.array([r_ear.x, r_ear.y]))
                    hands_behind_head = (wrist_ear_dist_l < 0.15) or (wrist_ear_dist_r < 0.15)

                    crunch_angle_l = abdomen_angle
                    crunch_angle_r = calculate_angle([r_shoulder.x, r_shoulder.y], [r_hip.x, r_hip.y], [r_knee.x, r_knee.y])
                    
                    if crunch_angle_l and crunch_angle_r and abs(crunch_angle_l - crunch_angle_r) > SYMMETRY_THRESHOLD:
                        feedback = "Keep Shoulders Level"
                        posture_feedback = "Posture: Keep body aligned!"
                        feedback_color = (0, 0, 255)
                    elif not hands_behind_head:
                        feedback = "Hands Behind Head!"
                        posture_feedback = "Posture: Place hands behind your head!"
                        feedback_color = (0, 0, 255)
                    else:
                        feedback_color = (0, 255, 0)
                        posture_feedback = "Posture: Good!"
                        if crunch_angle_l > 160:
                            if stage == "UP": down_start_time = time.time()
                            stage = "DOWN"
                            feedback = "Ready"
                        elif crunch_angle_l < 130 and stage == "DOWN":
                            if time.time() - down_start_time > 0.2:
                                up_start_time = time.time()
                                stage = "UP"
                                rep_count += 1
                                say_feedback(str(rep_count))
                                feedback = "UP"
                            else: feedback = "Too Fast!"
                        elif stage == "UP" and up_start_time and time.time() - up_start_time > 0.5:
                            feedback = "Go Down Slowly"
                    
                    # Angles visualization using unscaled (normalized) coordinates
                    angles_to_display = {"L_ELBOW": (l_shoulder, l_elbow, l_wrist), "L_SHOULDER": (l_hip, l_shoulder, l_elbow), "L_HIP": (l_shoulder, l_hip, l_knee), "L_KNEE": (l_hip, l_knee, l_ankle)}
                    for name, (p1, p2, p3) in angles_to_display.items():
                        if all(p.visibility > 0.6 for p in [p1, p2, p3]):
                            angle = calculate_angle([p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y])
                            text_pos = (int(p2.x * w), int(p2.y * h))
                            cv2.putText(image, f"{int(angle)}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                            cv2.putText(image, f"{int(angle)}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                except Exception as e:
                    feedback = "Make sure body is fully visible"
            
            say_feedback(feedback)

            # --- Display Text on top of the semi-transparent overlay ---
            cv2.putText(image, f"Sit-ups: {rep_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, posture_feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2, cv2.LINE_AA)
            cv2.putText(image, f"Abdomen Angle: {int(abdomen_angle)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Knee Angle: {int(knee_angle)}", (w - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Hands Behind Head: {hands_behind_head}", (w - 350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Feedback: {feedback}", (w - 350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, image)

            if cv2.waitKey(5) & 0xFF == ord('q'): break
                
    cap.release()
    cv2.destroyAllWindows()
    say_feedback(f"Set complete. You did {rep_count} reps.")
    return rep_count