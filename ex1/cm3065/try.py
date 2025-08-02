
import math
import cv2
import numpy as np

class Car:
    """
    A class to represent a single tracked vehicle.
    """
    def __init__(self, car_id, x, y, w, h):
        self.car_id = car_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cx = int(x + w / 2)
        self.cy = int(y + h / 2)
        self.inside_box1 = False  # To track if the car has entered the first detection box
        self.inside_box2 = False  # To track if the car has entered the second detection box

class CarTracker:
    """
    A simple centroid-based tracker for assigning unique IDs to cars
    and maintaining their positions across multiple frames.

    This class works by calculating the Euclidean distance between the
    centroids of new detections and the centroids of currently tracked objects.
    If a new detection is close enough to an existing object, it is
    considered the same object, and its position is updated. Otherwise,
    a new ID is assigned.
    """

    def __init__(self, max_distance):
        """
        Initializes the tracker.

        Args:
            max_distance (int): The maximum Euclidean distance a new centroid
                                can be from an existing one to be considered
                                the same object.
        """
        # Dictionary to store the Car objects of currently tracked objects.
        # Format: {object_id: Car_object}
        self.tracked_objects = {}
        
        # A counter for assigning new unique IDs.
        self.next_object_id = 0
        
        # The maximum distance to consider a match.
        self.max_distance = max_distance

    def update(self, detections):
        """
        Takes a list of new detections (bounding boxes) and updates the
        list of tracked objects. The method updates the self.tracked_objects
        dictionary in place and does not return any value.

        Args:
            detections (list): A list of new bounding boxes in the format
                               [(x, y, w, h), ...].
        """
        matched_ids = []
        current_tracked_objects_copy = self.tracked_objects.copy()

        for new_bbox in detections:
            x, y, w, h = new_bbox
            new_cx = int(x + w / 2)
            new_cy = int(y + h / 2)
            
            is_new_object = True
            min_dist = float('inf')
            closest_id = -1

            for object_id, car_obj in current_tracked_objects_copy.items():
                distance = math.hypot(new_cx - car_obj.cx, new_cy - car_obj.cy)
                
                if distance < min_dist:
                    min_dist = distance
                    closest_id = object_id
            
            if min_dist < self.max_distance:
                # Update the existing Car object with new detection data
                car_obj = self.tracked_objects[closest_id]
                car_obj.x, car_obj.y, car_obj.w, car_obj.h = x, y, w, h
                car_obj.cx, car_obj.cy = new_cx, new_cy
                matched_ids.append(closest_id)
                is_new_object = False
                
                if closest_id in current_tracked_objects_copy:
                    del current_tracked_objects_copy[closest_id]

            if is_new_object:
                new_id = self.next_object_id
                new_car = Car(new_id, x, y, w, h)
                self.tracked_objects[new_id] = new_car
                self.next_object_id += 1
                matched_ids.append(new_id)

        objects_to_remove = [
            obj_id for obj_id in self.tracked_objects if obj_id not in matched_ids
        ]
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]

def region_of_interest(frame):
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (0, height // 2), (width, height), 255, -1)
    mask = cv2.bitwise_and(frame, frame, mask=mask)
    return mask

def main():
    cap = cv2.VideoCapture("Traffic_Laramie_1.mp4")
    object_detector = cv2.createBackgroundSubtractorMOG2()

    tracker = CarTracker(max_distance=120)
    
    # Define the two boxes for left turn detection
    box1_coords = (400, 450, 550, 600)
    box2_coords = (700, 320, 1040, 425)

    left_turn_counter = 0
    
    # Get video properties for final calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate video duration upfront
    if fps > 0:
        total_seconds = frame_count / fps
    else:
        total_seconds = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = region_of_interest(frame)
        
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3500:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append((x, y, w, h))
        
        # Update the tracker with the new detections, but no return value
        tracker.update(detections)
        
        # Draw the left turn detection boxes
        x1, y1, x2, y2 = box1_coords
        cv2.rectangle(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(roi, "Box 1", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        x1, y1, x2, y2 = box2_coords
        cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(roi, "Box 2", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Iterate through the stored Car objects
        for car_id, car in tracker.tracked_objects.items():
            # Check for left turn conditions
            if not car.inside_box1 and box1_coords[0] < car.cx < box1_coords[2] and box1_coords[1] < car.cy < box1_coords[3]:
                print(f"{car_id} Entered Box1")
                car.inside_box1 = True

            if car.inside_box1 and box2_coords[0] < car.cx < box2_coords[2] and box2_coords[1] < car.cy < box2_coords[3]:
                left_turn_counter += 1
                print(f"Car ID {car_id} made a left turn!")
                # Reset the flag to avoid double-counting
                car.inside_box1 = False
            
            # Draw the bounding box and ID
            cv2.rectangle(roi, (car.x, car.y), (car.x + car.w, car.y + car.h), (0, 255, 0), 3)
            cv2.putText(roi, str(car.car_id), (car.x, car.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Draw a circle at the centroid of the car
            cv2.circle(roi, (car.cx, car.cy), 5, (139, 0, 139), -1)
        
        # Display the left turn counter in real-time
        cv2.putText(roi, f"Left Turns: {left_turn_counter}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Roi", roi)

        key = cv2.waitKey(2)
        if key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    # Final calculation and output after playback ends
    print("-" * 50)
    print("Video Analysis Summary")
    print("-" * 50)
    print(f"Total Left Turns: {left_turn_counter}")
    if total_seconds > 0:
        cars_per_minute = (left_turn_counter / total_seconds) * 60
        print(f"Total Video Duration: {total_seconds:.2f} seconds")
        print(f"Left Turns Per Minute: {cars_per_minute:.2f}")
    else:
        print("Video duration is zero, cannot calculate turns per minute.")
    print("-" * 50)

    # Create a blank image to display the summary in a new window
    summary_img = np.zeros((200, 500, 3), dtype=np.uint8)
    cv2.putText(summary_img, "Video Analysis Summary", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(summary_img, f"Total Left Turns: {left_turn_counter}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if total_seconds > 0:
        cv2.putText(summary_img, f"Total Video Duration: {total_seconds:.2f} s", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(summary_img, f"Left Turns Per Minute: {cars_per_minute:.2f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(summary_img, "Cannot calculate turns per minute.", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display the summary window
    cv2.imshow("Analysis Summary", summary_img)
    key = cv2.waitKey(0)
    if key == 27:
        exit()

if __name__ == "__main__":
    main()

