import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from utils import calculate_angle # We will draw angles manually, so visualize_angle is not needed

# --- Global Audio Engine Setup ---
audio_engine = pyttsx3.init()
last_feedback_spoken = ""
last_spoken_time = 0

def say_feedback(text):
    """Speaks the feedback text if it's new and hasn't been spoken recently."""
    global last_feedback_spoken, last_spoken_time
    current_time = time.time()
    if text != last_feedback_spoken and current_time - last_spoken_time > 2: # Reduced delay slightly
        last_feedback_spoken = text
        last_spoken_time = current_time
        audio_engine.say(text)
        audio_engine.runAndWait()

# --- Main Exercise Function ---
def run_crunches():
    """
    Combines advanced logic (tempo, symmetry, audio) with advanced visualization (data panel, on-joint angles).
    """
    cap = cv2.VideoCapture(0)
    # Add these two lines in both exercise files
    WINDOW_NAME = 'POSECOUNTER'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720) # Width, Height
    
    # --- State and Counter Variables from existing code ---
    rep_count = 0
    stage = "DOWN"
    feedback = "Starting Up"
    
    # --- Tempo and Symmetry Variables from existing code ---
    down_start_time = time.time()
    up_start_time = None
    SYMMETRY_THRESHOLD = 20  # Max degrees of difference between left/right side
    
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

            # --- PERFORMANCE OPTIMIZATION from existing code ---
            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            small_frame = cv2.resize(image, (w // 2, h // 2))
            image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image_rgb)
            
            # --- Initialize NEW metrics for the data panel ---
            abdomen_angle, knee_angle, elbow_knee_dist = 0, 0, 0
            hands_behind_head = False
            posture_feedback = "No Person Detected"
            feedback_color = (0, 0, 255) # Red for bad posture

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                vis_threshold = 0.6

                # Draw the basic skeleton from existing code
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                try:
                    # Get all necessary landmark objects
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

                    # --- 1. Calculate Custom Metrics for the NEW Data Panel ---
                    abdomen_angle = calculate_angle([l_shoulder.x, l_shoulder.y], [l_hip.x, l_hip.y], [l_knee.x, l_knee.y])
                    knee_angle = calculate_angle([l_hip.x, l_hip.y], [l_knee.x, l_knee.y], [l_ankle.x, l_ankle.y])
                    elbow_knee_dist = np.linalg.norm(np.array([l_elbow.x, l_elbow.y]) - np.array([l_knee.x, l_knee.y]))
                    
                    # Use wrist-ear distance for a more robust check
                    wrist_ear_dist_l = np.linalg.norm(np.array([l_wrist.x, l_wrist.y]) - np.array([l_ear.x, l_ear.y]))
                    wrist_ear_dist_r = np.linalg.norm(np.array([r_wrist.x, r_wrist.y]) - np.array([r_ear.x, r_ear.y]))
                    hands_behind_head = (wrist_ear_dist_l < 0.15) or (wrist_ear_dist_r < 0.15)

                    # --- 2. EXISTING strict logic for counting and feedback ---
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
                        feedback_color = (0, 255, 0) # Green for good posture
                        posture_feedback = "Posture: Good!"
                        # Rep Counting & Tempo Logic
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
                        elif stage == "UP" and time.time() - up_start_time > 0.5:
                            feedback = "Go Down Slowly"
                    
                    # --- 3. NEW VISUALIZATION: Angles on Joints ---
                    angles_to_display = {"L_ELBOW": (l_shoulder, l_elbow, l_wrist), "L_SHOULDER": (l_hip, l_shoulder, l_elbow), "L_HIP": (l_shoulder, l_hip, l_knee), "L_KNEE": (l_hip, l_knee, l_ankle)}
                    for name, (p1, p2, p3) in angles_to_display.items():
                        if all(p.visibility > vis_threshold for p in [p1, p2, p3]):
                            angle = calculate_angle([p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y])
                            text_pos = (int(p2.x * w), int(p2.y * h))
                            cv2.putText(image, f"{int(angle)}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                            cv2.putText(image, f"{int(angle)}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                except Exception as e:
                    feedback = "Make sure body is fully visible"
            else:
                feedback = "No Person Detected"
            
            # --- Say feedback aloud (using existing audio logic) ---
            say_feedback(feedback)

            # --- 4. NEW VISUALIZATION: Data Panel ---
            cv2.rectangle(image, (0, 0), (w, 150), (0, 0, 0), -1)
            cv2.putText(image, f"Sit-ups: {rep_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, posture_feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2, cv2.LINE_AA)
            cv2.putText(image, f"Abdomen Angle: {int(abdomen_angle)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Knee Angle: {int(knee_angle)}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Hands Behind Head: {hands_behind_head}", (w - 350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Feedback: {feedback}", (w - 350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('WINDOW_NAME', image)

            if cv2.waitKey(5) & 0xFF == ord('q'): break
                
    cap.release()
    cv2.destroyAllWindows()
    say_feedback(f"Set complete. You did {rep_count} reps.")
    return rep_count