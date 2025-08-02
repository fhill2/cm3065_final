
import cv2
import numpy as np
import collections

# --- Car Class ---
# This class represents a single tracked car, encapsulating its state.
class Car:
    """
    A class to represent a single tracked car object.
    It stores its ID, position, and turn status based on zone entry.
    """
    def __init__(self, object_id, bbox):
        self.id = object_id
        self.bbox = bbox
        x, y, w, h = bbox
        self.center = (x + w // 2, y + h // 2)
        self.turned_left = False
        self.counted = False # Flag to ensure the car is counted only once
        self.has_entered_source_box = False # Tracks if the car has ever been in the source box
        self.has_entered_target_box = False # New flag to track if the car has ever been in the target box
        self.frames_without_detection = 0 # Counter for frames where the car is not detected
        self.is_stationary = False
        self.stationary_frames = 0
        self.last_positions = collections.deque(maxlen=10) # Stores last few positions to check for stationarity

    def update_position(self, bbox):
        """Updates the car's bounding box, center, and position history."""
        self.bbox = bbox
        x, y, w, h = bbox
        self.center = (x + w // 2, y + h // 2)
        self.last_positions.append(self.center)

# --- Video Processor Class ---
# This class encapsulates all the video processing logic.
class VideoProcessor:
    """
    Main class for processing a video to detect, track, and count stationary cars.
    It uses frame differencing against a static background and a simple centroid-based
    tracking algorithm with a stationarity check.
    """
    # --- Hardcoded Parameters (as class attributes) ---
    MIN_CAR_CONTOUR_AREA = 800
    MAX_CAR_CONTOUR_AREA = 60000
    GAUSSIAN_BLUR_KERNEL_SIZE = (7, 7)
    DETECTION_ZONE_Y_SPLIT_RATIO = 0.5
    CAR_ASPECT_RATIO_MIN = 0.8
    CAR_ASPECT_RATIO_MAX = 3.0
    MORPHOLOGICAL_KERNEL_SIZE = (5, 5)
    MORPHOLOGICAL_OPEN_ITERATIONS = 2
    MORPHOLOGICAL_CLOSE_ITERATIONS = 3
    TRACKING_MAX_DISTANCE_PIXELS = 100
    MAX_FRAMES_WITHOUT_DETECTION = 30
    STATIONARY_POSITION_THRESHOLD = 5 # In pixels
    STATIONARY_FRAME_COUNT = 10 # Number of frames to check for stationarity
    BOUNDING_BOX_SHRINKAGE_FACTOR = 0.2
    MIN_IOU_THRESHOLD = 0.2 # Minimum Intersection over Union for a match

    # --- Junction Box Definitions (as class attributes) ---
    # Ratios are relative to the overall video frame dimensions
    # Source Box: Bottom-right, where cars start their turn
    SOURCE_BOX_X_START_RATIO = 0.25
    SOURCE_BOX_X_END_RATIO = 0.45
    SOURCE_BOX_Y_START_RATIO = 0.75
    SOURCE_BOX_Y_END_RATIO = 0.95

    # Target Box: Top-left, where cars complete their turn
    TARGET_BOX_X_START_RATIO = 0.8
    TARGET_BOX_X_END_RATIO = 0.95
    TARGET_BOX_Y_START_RATIO = 0.5
    TARGET_BOX_Y_END_RATIO = 0.6
    
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        assert self.video.isOpened(), "Error: Could not open video file."

        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Capture the first frame as a static background
        ret, self.background_frame = self.video.read()
        assert ret, "Error: Could not read first frame for background."
        self.background_frame = self._preprocess_frame(self.background_frame)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind the video

        # Define the detection zone (ROI) coordinates
        self.detect_start_y = int(self.frame_height * self.DETECTION_ZONE_Y_SPLIT_RATIO)
        self.detect_end_y = self.frame_height
        self.detect_start_x = 0
        self.detect_end_x = self.frame_width

        # Tracking and Counting state
        self.tracked_cars = {} # Stores {object_id: Car_object}
        self.next_car_id = 0
        self.cars_turned_left_count = 0

    def _preprocess_frame(self, input_frame):
        """Converts frame to grayscale and applies a Gaussian blur."""
        gray_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, self.GAUSSIAN_BLUR_KERNEL_SIZE, 0)
        return blurred_frame

    def _detect_stationary_cars(self, current_frame):
        """
        Uses frame differencing against a static background to find objects that have appeared.
        Creates a smaller bounding box for each detection to help with tracking.
        """
        # Preprocess the current frame
        preprocessed_frame = self._preprocess_frame(current_frame)

        # Compute the absolute difference between the current frame and the background
        frame_diff = cv2.absdiff(self.background_frame, preprocessed_frame)

        # Apply a threshold to get a binary image of the differences
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the noise
        morph_kernel = np.ones(self.MORPHOLOGICAL_KERNEL_SIZE, np.uint8)
        mask_after_opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel, iterations=self.MORPHOLOGICAL_OPEN_ITERATIONS)
        cleaned_mask = cv2.morphologyEx(mask_after_opening, cv2.MORPH_CLOSE, morph_kernel, iterations=self.MORPHOLOGICAL_CLOSE_ITERATIONS)

        # Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.MIN_CAR_CONTOUR_AREA < area < self.MAX_CAR_CONTOUR_AREA:
                # Get the original bounding box
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if self.CAR_ASPECT_RATIO_MIN < aspect_ratio < self.CAR_ASPECT_RATIO_MAX:
                    # Create a new, smaller bounding box for tracking
                    shrink_amount_w = int(w * self.BOUNDING_BOX_SHRINKAGE_FACTOR)
                    shrink_amount_h = int(h * self.BOUNDING_BOX_SHRINKAGE_FACTOR)
                    new_x = x + shrink_amount_w // 2
                    new_y = y + shrink_amount_h // 2
                    new_w = w - shrink_amount_w
                    new_h = h - shrink_amount_h
                    
                    filtered_bounding_boxes.append((new_x, new_y, new_w, new_h))
        
        return filtered_bounding_boxes, cleaned_mask
        
    def _calculate_iou(self, box1, box2):
        """Calculates the Intersection over Union (IoU) of two bounding boxes."""
        # Unpack the boxes
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Determine the coordinates of the intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # The area of each box
        box1_area = w1 * h1
        box2_area = w2 * h2

        # The union area is the sum of the areas minus the intersection area
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _update_car_tracking(self, current_detections):
        """
        Associates new detections with existing tracked cars based on proximity and IoU.
        Creates new Car objects for new detections and removes old ones.
        Also checks for stationarity.
        """
        newly_tracked_cars = {}
        matched_detection_indices = set()
        
        # Step 1: Match existing cars with current detections
        for car_id, car in self.tracked_cars.items():
            best_match_index = -1
            max_iou = 0.0
            min_distance = float('inf')

            for i, detection_bbox in enumerate(current_detections):
                if i in matched_detection_indices:
                    continue

                # Calculate IoU and distance
                iou = self._calculate_iou(car.bbox, detection_bbox)
                det_center = (detection_bbox[0] + detection_bbox[2] // 2, detection_bbox[1] + detection_bbox[3] // 2)
                distance = np.sqrt(
                    (det_center[0] - car.center[0])**2 + (det_center[1] - car.center[1])**2
                )
                
                # Check if this is a better match based on IoU and distance
                if iou > max_iou and iou > self.MIN_IOU_THRESHOLD and distance < self.TRACKING_MAX_DISTANCE_PIXELS:
                    max_iou = iou
                    best_match_index = i
                elif iou > 0 and iou <= self.MIN_IOU_THRESHOLD and distance < min_distance and distance < self.TRACKING_MAX_DISTANCE_PIXELS:
                    # Fallback to distance if IoU is too low but there's a small overlap and the distance is small
                    min_distance = distance
                    best_match_index = i
            
            if best_match_index != -1:
                # Update the existing car's position with the best matching detection
                car.update_position(current_detections[best_match_index])
                car.frames_without_detection = 0  # Reset the counter
                newly_tracked_cars[car_id] = car
                matched_detection_indices.add(best_match_index)
            else:
                # Car was not found in this frame, increment counter
                car.frames_without_detection += 1
                # Keep the car if it hasn't been "lost" for too many frames
                if car.frames_without_detection < self.MAX_FRAMES_WITHOUT_DETECTION:
                    newly_tracked_cars[car_id] = car

        # Step 2: Add any unmatched detections as new cars
        for i, bbox in enumerate(current_detections):
            if i not in matched_detection_indices:
                new_car = Car(self.next_car_id, bbox)
                newly_tracked_cars[self.next_car_id] = new_car
                self.next_car_id += 1
        
        self.tracked_cars = newly_tracked_cars

        # Step 3: Check for stationarity of all tracked cars
        for car in self.tracked_cars.values():
            if len(car.last_positions) == car.last_positions.maxlen:
                first_pos = car.last_positions[0]
                last_pos = car.last_positions[-1]
                distance = np.sqrt((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)
                if distance < self.STATIONARY_POSITION_THRESHOLD:
                    car.stationary_frames += 1
                    if car.stationary_frames >= self.STATIONARY_FRAME_COUNT:
                        car.is_stationary = True
                else:
                    car.stationary_frames = 0
                    car.is_stationary = False


    def _detect_and_count_left_turns(self):
        """
        Detects left turns based on whether a car passes through two specific boxes.
        """
        # Define source and target box coordinates based on frame dimensions
        source_box_x_start = int(self.frame_width * self.SOURCE_BOX_X_START_RATIO)
        source_box_x_end = int(self.frame_width * self.SOURCE_BOX_X_END_RATIO)
        source_box_y_start = int(self.frame_height * self.SOURCE_BOX_Y_START_RATIO)
        source_box_y_end = int(self.frame_height * self.SOURCE_BOX_Y_END_RATIO)

        target_box_x_start = int(self.frame_width * self.TARGET_BOX_X_START_RATIO)
        target_box_x_end = int(self.frame_width * self.TARGET_BOX_X_END_RATIO)
        target_box_y_start = int(self.frame_height * self.TARGET_BOX_Y_START_RATIO)
        target_box_y_end = int(self.frame_height * self.TARGET_BOX_Y_END_RATIO)

        for car in self.tracked_cars.values():
            # Skip cars that have already been counted or are not stationary
            if car.counted:
                continue
            
            car_center = car.center

            # Check if the stationary car is in the source box and set the flag permanently
            if car.is_stationary and not car.has_entered_source_box:
                is_in_source_box_now = (source_box_x_start < car_center[0] < source_box_x_end and
                                        source_box_y_start < car_center[1] < source_box_y_end)
                if is_in_source_box_now:
                    car.has_entered_source_box = True

            # Check if the stationary car is currently in the target box
            is_in_target_box_now = (target_box_x_start < car_center[0] < target_box_x_end and
                                    target_box_y_start < car_center[1] < target_box_y_end)
            
            # If the car is currently in the target box and is stationary, set the persistent flag
            if car.is_stationary and is_in_target_box_now:
                car.has_entered_target_box = True

            # Left turn is detected if the car has entered both the source and target boxes while stationary
            if car.has_entered_source_box and car.has_entered_target_box:
                car.turned_left = True
                if not car.counted:
                    self.cars_turned_left_count += 1
                    car.counted = True
                    print(f"Car ID {car.id} detected turning left! Total left turns: {self.cars_turned_left_count}")


    def _draw_on_frame(self, display_frame):
        """
        Draws bounding boxes, IDs, the two junction boxes, and the total count on the frame.
        """
        # Draw the main detection zone line
        cv2.line(display_frame, (self.detect_start_x, self.detect_start_y),
                 (self.detect_end_x, self.detect_start_y), (0, 0, 255), 2)

        # Draw the source box
        source_box_x_start = int(self.frame_width * self.SOURCE_BOX_X_START_RATIO)
        source_box_x_end = int(self.frame_width * self.SOURCE_BOX_X_END_RATIO)
        source_box_y_start = int(self.frame_height * self.SOURCE_BOX_Y_START_RATIO)
        source_box_y_end = int(self.frame_height * self.SOURCE_BOX_Y_END_RATIO)
        cv2.rectangle(display_frame, (source_box_x_start, source_box_y_start),
                      (source_box_x_end, source_box_y_end), (255, 165, 0), 2) # Orange
        cv2.putText(display_frame, 'Source Box', (source_box_x_start, source_box_y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

        # Draw the target box
        target_box_x_start = int(self.frame_width * self.TARGET_BOX_X_START_RATIO)
        target_box_x_end = int(self.frame_width * self.TARGET_BOX_X_END_RATIO)
        target_box_y_start = int(self.frame_height * self.TARGET_BOX_Y_START_RATIO)
        target_box_y_end = int(self.frame_height * self.TARGET_BOX_Y_END_RATIO)
        cv2.rectangle(display_frame, (target_box_x_start, target_box_y_start),
                      (target_box_x_end, target_box_y_end), (0, 255, 255), 2) # Yellow
        cv2.putText(display_frame, 'Target Box', (target_box_x_start, target_box_y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw details for each tracked car
        for car in self.tracked_cars.values():
            x, y, w, h = car.bbox
            color = (0, 255, 0) # Green for regular cars
            if car.is_stationary:
                color = (255, 0, 0) # Blue for stationary cars
            if car.turned_left:
                color = (0, 0, 255) # Red for cars that have turned left
            
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, f'ID: {car.id}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the total left turn count
        cv2.putText(display_frame, f'Left Turns: {self.cars_turned_left_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    def process_single_frame(self, current_frame):
        """
        Processes a single video frame to detect cars, update tracking,
        and perform left-turn detection. This method is the core pipeline.
        """
        display_frame = current_frame.copy()

        # Find car-like contours and their bounding boxes using a new detection method
        detected_bounding_boxes, _ = self._detect_stationary_cars(current_frame)
        
        # Update the simple object tracking with the new detections
        self._update_car_tracking(detected_bounding_boxes)

        # Analyze car trajectories to detect and count left turns
        self._detect_and_count_left_turns()

        # Draw all visual elements on the display frame
        self._draw_on_frame(display_frame)

        return display_frame

    def run(self):
        """Main function to run the video processing loop."""
        while True:
            retrieved_successfully, current_frame = self.video.read()

            if not retrieved_successfully:
                print("End of video stream or error reading frame.")
                break

            processed_frame = self.process_single_frame(current_frame)
            cv2.imshow('Car Detection & Tracking (Stationary)', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()


def main():
    """Entry point of the program."""
    processor = VideoProcessor('Traffic_Laramie_1.mp4')
    processor.run()

if __name__ == "__main__":
    main()


