<h1>Pose-Based Target Game (Non-Contact CV Controller)</h1>

<h2>Project Overview</h2>  

  This project transforms the user's body into a non-contact controller for a fast-paced reaction game, relying solely on real-time Computer Vision (CV) without any external sensors or deep learning models.

  The goal is to match your physical pose (Left Arm Up, Right Arm Up, or Both Arms Up) with a target as it passes through a score zone.

Key CV and Programming Skills Demonstrated:

  Marker-Based Pose Estimation: Uses simple color markers (tape, colored objects) on the body (shoulders and wrists) to simulate key skeletal joints.

  Real-Time State Machine: Implements game logic, target spawning, scoring, and pose checking in a continuous loop.

  Robust Image Processing: Utilizes cv2.inRange (HSV Color Filtering) and cv2.findContours to accurately locate and track user markers in varying lighting conditions.

  Human-Computer Interaction (HCI): Creates a unique, physical interaction system for fitness, rehabilitation, or entertainment applications.

Setup and Requirements

  This project requires only fundamental Python libraries and is guaranteed to work on Python 3.13.

Dependencies:

    pip install opencv-python numpy


Markers Needed: You must use 4 markers (e.g., colored stickers, tape, or objects) on your body:

    Shoulders (2): Use the color defined by SH_LOWER/SH_UPPER (Default: GREEN).

    Left Wrist (1): Use the color defined by LW_LOWER/LW_UPPER (Default: RED).

    Right Wrist (1): Use the color defined by RW_LOWER/RW_UPPER (Default: BLUE).

How to Play

Start the Game:

    python pose_game.py


  Calibrate Colors: Adjust the LW_LOWER, LW_UPPER, RW_LOWER, RW_UPPER, and SH_LOWER, SH_UPPER constants in the Python file until the colored circles accurately track your markers on the screen.

  Action: Falling targets will display a required pose (e.g., LEFT). When the target enters the POSE MATCH ZONE (the colored band at the bottom), extend the corresponding arm(s).

  LEFT: Extend only the left arm.

  RIGHT: Extend only the right arm.

  BOTH: Extend both arms.

Pose Detection Logic

  The system determines the pose by applying simple geometric rules to the tracked marker positions:

Pose Condition

  Rule Implemented in Code (check_arm_extension)

Arm is Extended

  Calculate the distance between the Shoulder Marker and the Wrist Marker (math.dist). If this distance is greater than the EXTENDED_ARM_THRESHOLD (250 pixels), the arm is considered up.

Centered

  Assumes the player is facing the camera if the shoulder markers are roughly aligned horizontally.

Scoring

  A hit is registered only if the Required Pose is held while the target is inside the POSE MATCH ZONE.
