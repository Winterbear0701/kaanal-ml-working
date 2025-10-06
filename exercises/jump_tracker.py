import cv2
import mediapipe as mp
import time
from utils import calculate_angle, visualize_angle # Assuming utils.py is correctly set up

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def run_jump_tracker():
    """
    Vertical jump tracker with user height input for accurate height calculation in CM.
    Includes robust ground calibration and maintains all visualizations.
    """
    cap = cv2.VideoCapture(0)
    # Add these two lines in both exercise files
    WINDOW_NAME = 'POSECOUNTER'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720) # Width, Height
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return []

    # --- 1. Get user height input ---
    user_height_cm = 0
    while True:
        try:
            height_input = input("Please enter your actual height in CM: ")
            user_height_cm = float(height_input)
            if user_height_cm <= 0:
                print("Height must be a positive number. Please try again.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a number for your height.")

    # --- State variables ---
    jump_count = 0
    jump_stage = 'DOWN' # Can be 'DOWN' (on ground), 'JUMPING_UP', 'AIRBORNE', 'LANDING'
    jump_heights_cm = []
    
    # Calibration variables
    initial_hip_y = None # Y-coordinate of hip when standing straight for calibration
    hip_pixel_height_at_calibration = None # Pixel height of hip when user_height_cm is known
    
    calibration_done = False
    
    print("\nStarting Vertical Jump Tracker...")
    print("Please stand straight and fully upright for 5 seconds for calibration.")
    print("Ensure your whole body, especially your hips and ankles, are visible.")
    calibration_start_time = time.time()
    
    # Initialize for visualization even before person detection
    knee_angle_l, knee_angle_r = 0, 0
    hip_angle_l, hip_angle_r = 0, 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # --- Initialize default values for angles and feedback for current frame ---
            current_jump_height_cm = 0.0
            display_feedback = "No person detected"
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                vis_threshold = 0.6

                # --- Draw the skeleton ---
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                try:
                    # Get relevant landmark coordinates
                    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    
                    # Use a stable point, like the mid-hip, for vertical movement tracking
                    mid_hip_y = (l_hip.y + r_hip.y) / 2
                    
                    # --- Calibration Phase ---
                    if not calibration_done:
                        display_feedback = "Calibrating: Stand straight!"
                        # Take average hip position over a few seconds when user is asked to stand straight
                        if time.time() - calibration_start_time < 5:
                            if initial_hip_y is None:
                                initial_hip_y = mid_hip_y
                            else:
                                initial_hip_y = (initial_hip_y + mid_hip_y) / 2 # Smoothen
                            hip_pixel_height_at_calibration = h * (1 - initial_hip_y) # Pixel height from bottom
                        else:
                            calibration_done = True
                            print(f"Calibration complete! Initial Hip Y (normalized): {initial_hip_y:.2f}")
                            print(f"Hip Pixel Height at Calibration (from bottom): {hip_pixel_height_at_calibration:.2f}")
                            display_feedback = "Calibration complete. Start jumping!"
                            
                    # --- Jump Tracking Logic (after calibration) ---
                    else:
                        # Angle calculations for visualization
                        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                        # Calculate and visualize knee angles
                        if all(p.visibility > vis_threshold for p in [l_hip, l_knee, l_ankle]):
                            knee_angle_l = calculate_angle([l_hip.x, l_hip.y], [l_knee.x, l_knee.y], [l_ankle.x, l_ankle.y])
                            visualize_angle(image, [l_hip.x, l_hip.y], [l_knee.x, l_knee.y], [l_ankle.x, l_ankle.y], int(knee_angle_l))
                        
                        if all(p.visibility > vis_threshold for p in [r_hip, r_knee, r_ankle]):
                            knee_angle_r = calculate_angle([r_hip.x, r_hip.y], [r_knee.x, r_knee.y], [r_ankle.x, r_ankle.y])
                            visualize_angle(image, [r_hip.x, r_hip.y], [r_knee.x, r_knee.y], [r_ankle.x, r_ankle.y], int(knee_angle_r))

                        # Calculate and visualize hip angles
                        if all(p.visibility > vis_threshold for p in [l_shoulder, l_hip, l_knee]):
                            hip_angle_l = calculate_angle([l_shoulder.x, l_shoulder.y], [l_hip.x, l_hip.y], [l_knee.x, l_knee.y])
                            visualize_angle(image, [l_shoulder.x, l_shoulder.y], [l_hip.x, l_hip.y], [l_knee.x, l_knee.y], int(hip_angle_l))
                        
                        if all(p.visibility > vis_threshold for p in [r_shoulder, r_hip, r_knee]):
                            hip_angle_r = calculate_angle([r_shoulder.x, r_shoulder.y], [r_hip.x, r_hip.y], [r_knee.x, r_knee.y])
                            visualize_angle(image, [r_shoulder.x, r_shoulder.y], [r_hip.x, r_hip.y], [r_knee.x, r_knee.y], int(hip_angle_r))
                            
                        # --- Jump Counting Logic ---
                        current_hip_pixel_height = h * (1 - mid_hip_y) # Current pixel height of hip from bottom
                        
                        if jump_stage == 'DOWN':
                            display_feedback = "Get Ready!"
                            # When hip moves significantly below initial (squatting)
                            if (knee_angle_l < 160 or knee_angle_r < 160) and (hip_angle_l < 160 or hip_angle_r < 160): # Squat detection
                                if current_hip_pixel_height < hip_pixel_height_at_calibration * 0.9: # If hip goes down
                                    jump_stage = 'PREPARING_JUMP'
                                    peak_hip_y_during_jump = mid_hip_y # Reset peak
                                    
                        elif jump_stage == 'PREPARING_JUMP':
                            display_feedback = "Push Up!"
                            if mid_hip_y < peak_hip_y_during_jump: # Hip is moving upwards
                                peak_hip_y_during_jump = mid_hip_y # Update peak of upward movement
                                
                            # If hip goes above calibration height, user is airborne
                            if current_hip_pixel_height > hip_pixel_height_at_calibration * 1.05: # 5% above initial height
                                jump_stage = 'AIRBORNE'
                                display_feedback = "AIRBORNE!"
                                
                            # If hip goes back down without jumping
                            elif current_hip_pixel_height < hip_pixel_height_at_calibration * 0.8 and (knee_angle_l > 170 and knee_angle_r > 170):
                                jump_stage = 'DOWN' # Canceled jump
                                display_feedback = "Ready!"

                        elif jump_stage == 'AIRBORNE':
                            display_feedback = "Landing..."
                            if mid_hip_y < peak_hip_y_during_jump: # Continue tracking peak in air
                                peak_hip_y_during_jump = mid_hip_y
                            
                            # If hip returns near ground level, jump is complete
                            if current_hip_pixel_height < hip_pixel_height_at_calibration * 1.05: # Back near calibrated height
                                jump_count += 1
                                # Convert pixel difference to CM using calibrated scale
                                pixel_jump_distance = hip_pixel_height_at_calibration - (h * (1 - peak_hip_y_during_jump)) # Distance from calibrated hip height to peak hip height
                                
                                # Scale: (User_height_cm / hip_pixel_height_at_calibration) gives cm_per_pixel factor
                                # This is a simplified scale. A more accurate one would use actual limb lengths.
                                if hip_pixel_height_at_calibration > 0:
                                     cm_per_pixel = user_height_cm / hip_pixel_height_at_calibration
                                     current_jump_height_cm = pixel_jump_distance * cm_per_pixel
                                else:
                                    current_jump_height_cm = 0 # Fallback if calibration failed
                                
                                jump_heights_cm.append(round(current_jump_height_cm, 1))
                                jump_stage = 'LANDING'
                                display_feedback = f"JUMP! {int(current_jump_height_cm)} cm"
                                
                        elif jump_stage == 'LANDING':
                            display_feedback = "Jump complete! Reset."
                            if knee_angle_l > 170 and knee_angle_r > 170 and current_hip_pixel_height >= hip_pixel_height_at_calibration * 0.95:
                                jump_stage = 'DOWN' # Back to standing
                                display_feedback = "Ready!"

                except Exception as e:
                    display_feedback = f"Pose not clear: {e}"
                    knee_angle_l, knee_angle_r = 0, 0 # Reset angles if error

            # --- Status Box Visualization ---
            cv2.rectangle(image, (0, 0), (w, 100), (20, 20, 20), -1)
            cv2.putText(image, 'REPS', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, str(jump_count), (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'STATUS', (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, display_feedback, (120, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if "Ready" in display_feedback or "JUMP" in display_feedback else (255, 255, 0), 2, cv2.LINE_AA)
            
            if jump_heights_cm:
                cv2.putText(image, 'LAST JUMP (cm)', (w - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, f"{jump_heights_cm[-1]:.1f}", (w - 320, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, 'LAST JUMP (cm)', (w - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, "N/A", (w - 320, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'L-KNEE', (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"{int(knee_angle_l)}", (w - 130, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'R-KNEE', (w - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"{int(knee_angle_r)}", (w - 30, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('WINDOW_NAME', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return jump_heights_cm