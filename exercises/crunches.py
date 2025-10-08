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
    Final corrected version with a clean text layout and transparent background.
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
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # --- Text background has been removed ---

            abdomen_angle, knee_angle = 0, 0
            hands_behind_head = False
            posture_feedback = "No Person Detected"
            feedback_color = (0, 0, 255)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                connections = mp_pose.POSE_CONNECTIONS
                if connections:
                    for connection in connections:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        
                        if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
                            start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                            end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                            cv2.line(image, start_point, end_point, (255, 255, 255), 2)
                
                for landmark in landmarks:
                     if landmark.visibility > 0.5:
                         center = (int(landmark.x * w), int(landmark.y * h))
                         cv2.circle(image, center, 5, (255, 0, 0), -1)

                try:
                    # --- All workout logic is unchanged ---
                    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                    r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]

                    abdomen_angle = calculate_angle([l_shoulder.x, l_shoulder.y], [l_hip.x, l_hip.y], [l_knee.x, l_knee.y])
                    knee_angle = calculate_angle([l_hip.x, l_hip.y], [l_knee.x, l_knee.y], [l_ankle.x, l_ankle.y])
                    
                    wrist_ear_dist = np.linalg.norm(np.array([l_wrist.x, l_wrist.y]) - np.array([l_ear.x, l_ear.y]))
                    hands_behind_head = wrist_ear_dist < 0.15

                    crunch_angle_l = abdomen_angle
                    crunch_angle_r = calculate_angle([r_shoulder.x, r_shoulder.y], [r_hip.x, r_hip.y], [r_knee.x, r_knee.y]) if all(lm.visibility > 0.5 for lm in [r_shoulder, r_hip, r_knee]) else None
                    
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
                except Exception as e:
                    feedback = "Make sure body is fully visible"
            
            say_feedback(feedback)

            # --- MODIFIED: Cleaned up text layout with smaller fonts ---
            
            font_size_large = 0.9
            font_size_small = 0.7
            font_color = (255, 255, 255)
            outline_color = (0, 0, 0)
            
            # Helper to draw text with outline
            def draw_text(text, position, font_scale, text_color):
                cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, 3, cv2.LINE_AA)
                cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

            # Column 1
            draw_text(f"REP: {rep_count}", (10, 30), font_size_large, font_color)
            draw_text(posture_feedback, (10, 60), font_size_small, feedback_color)
            draw_text(f"Feedback: {feedback}", (10, 85), font_size_small, (0, 255, 255))

            # Column 2
            col2_x = w - 300
            draw_text(f"Abdomen Angle: {int(abdomen_angle)}", (col2_x, 30), font_size_small, font_color)
            draw_text(f"Knee Angle: {int(knee_angle)}", (col2_x, 55), font_size_small, font_color)
            draw_text(f"Hands Behind Head: {hands_behind_head}", (col2_x, 80), font_size_small, font_color)

            cv2.imshow(WINDOW_NAME, image)

            if cv2.waitKey(5) & 0xFF == ord('q'): break
                
    cap.release()
    cv2.destroyAllWindows()
    say_feedback(f"Set complete. You did {rep_count} reps.")
    return rep_count