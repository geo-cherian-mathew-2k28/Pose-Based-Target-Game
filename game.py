import cv2
import numpy as np
import time
import math
import rando

LW_LOWER = np.array([0, 150, 150])    
LW_UPPER = np.array([10, 255, 255])  
LW_COLOR_BGD = (0, 0, 255)

RW_LOWER = np.array([100, 150, 50])
RW_UPPER = np.array([130, 255, 255])
RW_COLOR_BGD = (255, 0, 0) 

SH_LOWER = np.array([40, 50, 50])    
SH_UPPER = np.array([80, 255, 255])
SH_COLOR_BGD = (0, 255, 0) 
EXTENDED_ARM_THRESHOLD = 250

score = 0
game_state = "START"
targets = [] 
TARGET_SPEED = 2
SPAWN_RATE = 50 
current_frame_count = 0
font = cv2.FONT_HERSHEY_SIMPLEX


def get_marker_position(frame, lower_hsv, upper_hsv):
    """
    Detects and returns the center coordinates of the largest blob of the target color.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            M = cv2.moments(largest_contour)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            return (center_x, center_y)
            
    return None

def check_arm_extension(shoulder_pos, wrist_pos, threshold):
    """
    Checks if the arm is extended by calculating the distance between shoulder and wrist.
    """
    if shoulder_pos and wrist_pos:
        distance = math.dist(shoulder_pos, wrist_pos)
        return distance > threshold
    return False

def spawn_target(frame_w):
    """
    Spawns a new target at the top of the screen with a random required pose.
    """
    pose_options = ["LEFT_ARM_UP", "RIGHT_ARM_UP", "BOTH_ARMS_UP"]
    targets.append([random.randint(50, frame_w - 50), 0, random.choice(pose_options), False])

def update_game(frame_w, frame_h, ls_pos, rs_pos, lw_pos, rw_pos):
    """
    Moves targets and checks for hits/misses based on player pose.
    """
    global score
    
    is_left_extended = check_arm_extension(ls_pos, lw_pos, EXTENDED_ARM_THRESHOLD)
    is_right_extended = check_arm_extension(rs_pos, rw_pos, EXTENDED_ARM_THRESHOLD)
    
    is_centered = (abs(ls_pos[0] - rs_pos[0]) < frame_w // 4) if ls_pos and rs_pos else False

    targets_to_keep = []
    for target in targets:
        target[1] += TARGET_SPEED 
        
        target_x, target_y, required_pose, is_hit = target
        
        HIT_ZONE_Y = frame_h * 0.8
        
        if target_y >= HIT_ZONE_Y and not is_hit:
            
            pose_match = False
            if required_pose == "LEFT_ARM_UP" and is_left_extended and not is_right_extended:
                pose_match = True
            elif required_pose == "RIGHT_ARM_UP" and is_right_extended and not is_left_extended:
                pose_match = True
            elif required_pose == "BOTH_ARMS_UP" and is_right_extended and is_left_extended:
                pose_match = True

            if pose_match and is_centered:
                score += 10
                target[3] = True 
    
        if target_y < frame_h:
            targets_to_keep.append(target)
        elif not target[3]: 
            score = max(0, score - 5) 

    targets[:] = targets_to_keep


def draw_game_elements(frame):
    """
    Draws the game targets, markers, and score.
    """
    h, w, _ = frame.shape
    
    HIT_ZONE_Y = int(h * 0.8)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, HIT_ZONE_Y), (w, HIT_ZONE_Y + 50), (0, 0, 150), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, "POSE MATCH ZONE", (w // 2 - 100, HIT_ZONE_Y + 30), 
                font, 0.6, (255, 255, 255), 2)
            
    for target in targets:
        target_x, target_y, required_pose, is_hit = target
        target_center = (target_x, int(target_y))
        
        color = (0, 255, 0) if required_pose == "LEFT_ARM_UP" else \
                (255, 0, 0) if required_pose == "RIGHT_ARM_UP" else \
                (0, 255, 255)
     
        cv2.circle(frame, target_center, 20, color, -1)
        
        text = required_pose.split('_')[0] 
        cv2.putText(frame, text, (target_x - 15, target_y + 5), 
                    font, 0.5, (0, 0, 0), 2)
    
        if is_hit:
            cv2.putText(frame, "HIT!", target_center, 
                        font, 1.0, (255, 255, 255), 3)

    cv2.putText(frame, f"SCORE: {score}", (20, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Hold the required pose in the yellow zone!", (20, 60), 
                font, 0.5, (255, 255, 0), 1)
def run_pose_game():
    global current_frame_count, score, game_state, targets
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read webcam feed.")
        cap.release()
        return

    h, w, _ = frame.shape
    game_state = "RUNNING"

    print("Non-Contact Pose Game Initialized.")
    print("Place colored markers on your shoulders and wrists.")

    while cap.isOpened() and game_state == "RUNNING":
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) 
        frame_copy = frame.copy()
        shoulder_pos = get_marker_position(frame_copy, SH_LOWER, SH_UPPER)
    
        ls_pos = (shoulder_pos[0] - 50, shoulder_pos[1]) if shoulder_pos else None
        rs_pos = (shoulder_pos[0] + 50, shoulder_pos[1]) if shoulder_pos else None
    
        lw_pos = get_marker_position(frame_copy, LW_LOWER, LW_UPPER)
        rw_pos = get_marker_position(frame_copy, RW_LOWER, RW_UPPER)

        if ls_pos: cv2.circle(frame, ls_pos, 10, SH_COLOR_BGD, -1)
        if rs_pos: cv2.circle(frame, rs_pos, 10, SH_COLOR_BGD, -1)
        if lw_pos: cv2.circle(frame, lw_pos, 10, LW_COLOR_BGD, -1)
        if rw_pos: cv2.circle(frame, rw_pos, 10, RW_COLOR_BGD, -1)

        if ls_pos and lw_pos: cv2.line(frame, ls_pos, lw_pos, (255, 255, 255), 3)
        if rs_pos and rw_pos: cv2.line(frame, rs_pos, rw_pos, (255, 255, 255), 3)

        if current_frame_count % SPAWN_RATE == 0:
            spawn_target(w)
            
        update_game(w, h, ls_pos, rs_pos, lw_pos, rw_pos)

        draw_game_elements(frame)
        
        cv2.imshow('Pose-Based Target Game (Press Q to Exit)', frame)

        current_frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            game_state = "QUIT"
    if game_state == "RUNNING" or game_state == "QUIT":
     
        final_frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(final_frame, "GAME OVER", (w // 2 - 150, h // 2 - 50), font, 2, (0, 0, 255), 3)
        cv2.putText(final_frame, f"FINAL SCORE: {score}", (w // 2 - 150, h // 2 + 50), font, 1.5, (0, 255, 0), 2)
        cv2.putText(final_frame, "Press Q to close...", (w // 2 - 100, h // 2 + 100), font, 0.7, (255, 255, 255), 1)
        
        cv2.imshow('Pose-Based Target Game (Press Q to Exit)', final_frame)
        cv2.waitKey(0) 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_pose_game()
