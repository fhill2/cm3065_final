import cv2
import numpy as np
import collections # For a deque to store previous frames

class CarDetectorAndTracker:
    def __init__(self, video_path, history=20, varThreshold=25, detectShadows=True,
                 min_contour_area=500, max_contour_area=50000,
                 frame_diff_threshold=25, blur_kernel_size=(5, 5)):
        """
        Initializes the car detector and tracker.

        Args:
            video_path (str): Path to the input video file.
            history (int): Number of frames for MOG2 background history.
            varThreshold (int): Threshold on the squared Mahalanobis distance to decide if a pixel is foreground.
            detectShadows (bool): Whether to detect shadows in MOG2.
            min_contour_area (int): Minimum contour area to consider an object a car.
            max_contour_area (int): Maximum contour area to consider an object a car.
            frame_diff_threshold (int): Threshold for binary conversion in frame differencing.
            blur_kernel_size (tuple): Kernel size for Gaussian blur.
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error: Could not open video file at {video_path}")

        # Background Subtractor (MOG2 is generally good for traffic)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=history,
                                                      varThreshold=varThreshold,
                                                      detectShadows=detectShadows)

        # Frame Differencing
        # Use a deque to store previous frames for differencing
        self.prev_frames = collections.deque(maxlen=2) # Stores current and previous for diff
        self.frame_diff_threshold = frame_diff_threshold
        self.blur_kernel_size = blur_kernel_size

        # Contour filtering parameters
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area

        # Simple tracking (placeholder for more advanced tracking)
        self.tracked_cars = {} # Stores {car_id: (x, y, w, h)}
        self.next_car_id = 0

    def preprocess_frame(self, frame):
        """Applies grayscale and Gaussian blur."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        return blurred

    def get_frame_difference_mask(self, current_frame_gray_blurred):
        """Calculates frame difference mask."""
        if len(self.prev_frames) < 2:
            return None # Not enough frames yet for differencing

        # Get the second to last frame (previous frame)
        prev_frame_gray_blurred = self.prev_frames[0]

        # Compute absolute difference between current and previous frame
        diff = cv2.absdiff(prev_frame_gray_blurred, current_frame_gray_blurred)

        # Apply threshold to get a binary mask
        _, thresh = cv2.threshold(diff, self.frame_diff_threshold, 255, cv2.THRESH_BINARY)
        return thresh

    def process_frame(self, frame):
        """Processes a single frame for car detection."""
        original_frame = frame.copy()
        processed_frame = self.preprocess_frame(original_frame)

        # 1. Background Subtraction Mask
        fgmask_bg_sub = self.fgbg.apply(processed_frame)

        # 2. Frame Differencing Mask
        fgmask_frame_diff = self.get_frame_difference_mask(processed_frame)

        # Update previous frames for next iteration of frame differencing
        self.prev_frames.append(processed_frame)

        combined_mask = None
        if fgmask_frame_diff is not None:
            # Combine the masks using OR operation
            # This means if EITHER background subtraction OR frame differencing detects motion,
            # it's considered foreground.
            combined_mask = cv2.bitwise_or(fgmask_bg_sub, fgmask_frame_diff)
        else:
            combined_mask = fgmask_bg_sub # Use only BG subtraction if not enough frames for diff

        if combined_mask is None:
            return original_frame # No mask to process yet

        # 3. Morphological Operations to clean the mask
        # Kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)

        # Apply opening (erosion followed by dilation) to remove small noise
        mask_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        # Apply closing (dilation followed by erosion) to fill small holes
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=3)

        # 4. Find Contours
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_frame_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # 5. Filter contours by area to ignore small noise and too large (e.g., entire background change)
            if self.min_contour_area < area < self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Optional: Further filter by aspect ratio for car-like shapes (e.g., width > height)
                # You'll need to tune this based on your camera angle and car sizes
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 3.0: # Typical cars are wider than tall, or slightly squarish
                    cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    current_frame_detections.append((x, y, w, h))

        # Simple Tracking (replace with more robust algo for complex scenarios)
        # This is a very basic centroid tracking. For true tracking, use Kalman filters or DeepSORT.
        self.update_simple_tracking(current_frame_detections, original_frame)

        # Display masks (for debugging)
        # cv2.imshow('BG Subtraction Mask', fgmask_bg_sub)
        # if fgmask_frame_diff is not None:
        #     cv2.imshow('Frame Difference Mask', fgmask_frame_diff)
        # cv2.imshow('Combined & Cleaned Mask', mask_closed)

        return original_frame

    def update_simple_tracking(self, current_detections, frame):
        """
        A very basic centroid-based tracking.
        For demonstration; not suitable for robust multi-object tracking.
        """
        if not self.tracked_cars:
            for x, y, w, h in current_detections:
                self.tracked_cars[self.next_car_id] = {'bbox': (x, y, w, h), 'center': (x + w // 2, y + h // 2)}
                self.next_car_id += 1
            return

        new_tracked_cars = {}
        matched_detection_indices = set()

        for car_id, car_info in self.tracked_cars.items():
            old_center = car_info['center']
            best_match_idx = -1
            min_dist = float('inf')

            for i, (x, y, w, h) in enumerate(current_detections):
                if i in matched_detection_indices:
                    continue

                new_center = (x + w // 2, y + h // 2)
                dist = np.sqrt((new_center[0] - old_center[0])**2 + (new_center[1] - old_center[1])**2)

                # Define a maximum distance for association (tune this)
                if dist < min_dist and dist < 100: # Max 100 pixels movement
                    min_dist = dist
                    best_match_idx = i

            if best_match_idx != -1:
                x, y, w, h = current_detections[best_match_idx]
                new_tracked_cars[car_id] = {'bbox': (x, y, w, h), 'center': (x + w // 2, y + h // 2)}
                matched_detection_indices.add(best_match_idx)
            # Else: car disappeared (for simplicity, we just drop it)

        # Add new detections that didn't match any existing track
        for i, (x, y, w, h) in enumerate(current_detections):
            if i not in matched_detection_indices:
                new_tracked_cars[self.next_car_id] = {'bbox': (x, y, w, h), 'center': (x + w // 2, y + h // 2)}
                self.next_car_id += 1

        self.tracked_cars = new_tracked_cars

        # Draw tracked IDs
        for car_id, car_info in self.tracked_cars.items():
            x, y, w, h = car_info['bbox']
            # Draw bounding box (already done by the detection part, but we can redraw with ID)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue for tracked
            cv2.putText(frame, f'ID: {car_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    def run(self):
        """Runs the car detection and tracking process on the video."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('Car Detection & Tracking', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- IMPORTANT ---
    # 1. Replace 'traffic_video.mp4' with the actual path to your video file.
    #    You'll need a video of a traffic intersection for best results.
    #    Example video: Search YouTube for "traffic intersection video for computer vision"
    #    and download one (e.g., using youtube-dl) or record your own.
    video_file_path = './Traffic_Laramie_1.mp4'

    # Create a dummy video file if it doesn't exist (for basic testing)
    try:
        cap_test = cv2.VideoCapture(video_file_path)
        if not cap_test.isOpened():
            raise FileNotFoundError
        cap_test.release()
    except FileNotFoundError:
        print(f"'{video_file_path}' not found.")
        print("Please provide a valid video path for the car detection to work.")
        print("You can create a simple dummy video or point to an existing one.")
        # As creating a useful dummy video is complex, we'll just exit here if not found.
        exit()


    # Initialize and run the detector
    detector = CarDetectorAndTracker(
        video_path=video_file_path,
        history=500, # Increased history for MOG2 to learn background better
        varThreshold=50, # Higher threshold might reduce noise
        detectShadows=True,
        min_contour_area=800, # Adjust these based on the size of cars in your video
        max_contour_area=60000,
        frame_diff_threshold=30, # Slightly higher threshold for diff
        blur_kernel_size=(7, 7) # Larger blur kernel for smoother results
    )
    detector.run()
