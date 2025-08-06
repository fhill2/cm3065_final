
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
    subtractor = cv2.createBackgroundSubtractorMOG2()

    tracker = CarTracker(max_distance=120)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = region_of_interest(frame)
        
        mask = subtractor.apply(roi)
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

        # Iterate through the stored Car objects
        for _, car in tracker.tracked_objects.items():
            # Draw the bounding box and ID
            cv2.rectangle(roi, (car.x, car.y), (car.x + car.w, car.y + car.h), (0, 255, 0), 3)
            cv2.putText(roi, str(car.car_id), (car.x, car.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Draw a circle at the centroid of the car
            cv2.circle(roi, (car.cx, car.cy), 5, (139, 0, 139), -1)
        

        cv2.imshow("Mask", mask)
        cv2.imshow("Roi", roi)

        key = cv2.waitKey(2)
        if key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    key = cv2.waitKey(0)
    if key == 27:
        exit()

if __name__ == "__main__":
    main()

