import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import ssl
import urllib.request

class PitcherAnalysis:
    def __init__(self, offline_mode=False):
        """
        Initialize the PitcherAnalysis class
        
        Args:
            offline_mode: If True, uses a workaround for SSL certificate issues
        """
        # If offline mode is enabled, apply SSL context fix
        if offline_mode:
            self._fix_ssl_certificates()
            
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        try:
            print("Initializing MediaPipe Pose model...")
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Use medium model for better compatibility
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Pose model initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe Pose: {e}")
            print("Try running with offline_mode=True or download the model manually")
            raise
        
        # For tracking key angles during pitching motion
        self.shoulder_angles = []
        self.elbow_angles = []
        self.knee_angles = []
        self.frame_numbers = []
    
    def _fix_ssl_certificates(self):
        """Apply workaround for SSL certificate issues on macOS."""
        print("Applying SSL certificate fix for macOS...")
        # Create unverified SSL context
        ssl._create_default_https_context = ssl._create_unverified_context
        print("SSL certificate verification disabled")
    
    def process_video(self, video_path, output_path=None, display=True, save_data=True):
        """
        Process baseball pitching video with MediaPipe Pose.
        
        Args:
            video_path: Path to the input video (.mp4, .mov, etc.)
            output_path: Path to save the processed video (if None, no video is saved)
            display: Whether to display the processed frames
            save_data: Whether to save joint angle data
        """
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return None, None, None, None
            
        # Print video file information
        file_ext = os.path.splitext(video_path)[1].lower()
        print(f"Processing video: {video_path} (Format: {file_ext})")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Couldn't open video file {video_path}")
            print("Possible solutions:")
            print("1. Make sure the file exists and path is correct")
            print("2. Install additional codecs (like ffmpeg) if needed")
            print("3. Try converting the video to a different format")
            return None, None, None, None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} at {fps} FPS, {frame_count} frames")
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            # Determine output format based on extension
            out_ext = os.path.splitext(output_path)[1].lower()
            
            # Select appropriate codec based on output format
            if out_ext == '.mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4
            elif out_ext == '.mov':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # MOV
            elif out_ext == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI
            else:
                # Default to MP4
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output video will be saved to: {output_path}")
        
        # Reset tracking data
        self.shoulder_angles = []
        self.elbow_angles = []
        self.knee_angles = []
        self.frame_numbers = []
        
        frame_idx = 0
        start_time = time.time()
        processing_fps = 0
        
        # Process the video frame by frame
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # For faster processing, optionally resize the image
            # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                
            # Convert the image to RGB (MediaPipe requires RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            try:
                results = self.pose.process(image_rgb)
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                frame_idx += 1
                continue
            
            # Draw pose landmarks on the image
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Calculate and track key angles
                self.track_pitching_angles(results.pose_landmarks, frame_idx)
                
                # Calculate processing speed
                if frame_idx > 0:
                    elapsed_time = time.time() - start_time
                    processing_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
                
                # Add info to frame
                cv2.putText(image, f"Frame: {frame_idx} | FPS: {processing_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the image if requested
            if display:
                cv2.imshow('Baseball Pitcher Analysis', image)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Press 'Esc' to exit
                    break
                elif key == 32:  # Press 'Space' to pause/resume
                    cv2.waitKey(0)
            
            # Write the frame if saving
            if writer:
                writer.write(image)
            
            frame_idx += 1
            # Print progress every 30 frames
            if frame_idx % 30 == 0:
                progress = frame_idx/frame_count*100 if frame_count > 0 else 0
                print(f"Processing frame {frame_idx}/{frame_count} ({progress:.1f}%) | FPS: {processing_fps:.1f}")
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Plot the angle data
        if save_data and self.frame_numbers:
            self.plot_angle_data()
            
        print(f"Processed {frame_idx} frames in {time.time() - start_time:.2f} seconds")
        return self.shoulder_angles, self.elbow_angles, self.knee_angles, self.frame_numbers
    
    def track_pitching_angles(self, landmarks, frame_idx):
        """Calculate key angles for pitching mechanics analysis."""
        # Get landmark coordinates
        lm = landmarks.landmark
        
        # Calculate right elbow angle (assuming right-handed pitcher)
        right_shoulder = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        right_elbow = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
        right_wrist = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
        
        # Calculate right shoulder angle
        right_hip = np.array([lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        
        # Calculate lead knee angle (left knee for right-handed pitcher)
        left_hip = np.array([lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y])
        left_knee = np.array([lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y])
        left_ankle = np.array([lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
        
        # Calculate angles
        elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        shoulder_angle = self.calculate_angle(right_hip, right_shoulder, right_elbow)
        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        
        # Store the angles
        self.elbow_angles.append(elbow_angle)
        self.shoulder_angles.append(shoulder_angle)
        self.knee_angles.append(knee_angle)
        self.frame_numbers.append(frame_idx)
    
    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points (in degrees)."""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure the value is in range
        
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        
        return angle
    
    def plot_angle_data(self):
        """Plot the tracked angles over frames."""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.frame_numbers, self.shoulder_angles, 'b-')
        plt.title('Shoulder Angle Over Pitching Motion')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(self.frame_numbers, self.elbow_angles, 'r-')
        plt.title('Elbow Angle Over Pitching Motion')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(self.frame_numbers, self.knee_angles, 'g-')
        plt.title('Lead Knee Angle Over Pitching Motion')
        plt.xlabel('Frame Number')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('pitcher_joint_angles.png')
        print("Joint angle plot saved to 'pitcher_joint_angles.png'")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Use offline mode to fix SSL certificate issues on macOS
    analyzer = PitcherAnalysis(offline_mode=True)
    
    # Replace with your video path - can be .MP4, .MOV, or other formats
    video_path = "pitcher_video.mov"
    
    # Output path automatically uses same format as input
    file_name, file_ext = os.path.splitext(video_path)
    output_path = f"{file_name}_analyzed{file_ext}"
    
    # Process the video
    analyzer.process_video(video_path, output_path=output_path, display=True)
