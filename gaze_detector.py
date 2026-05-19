import cv2
import mediapipe as mp
import numpy as np
import time

class GazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark Indices
        self.LEFT_EYE_INNER = 362
        self.LEFT_EYE_OUTER = 263
        self.RIGHT_EYE_INNER = 133
        self.RIGHT_EYE_OUTER = 33
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [468, 469, 470, 471]
        
        # Identity Verification
        self.registered_signature = None
        self.SIGNATURE_THRESHOLD = 0.15 # Sensitivity for face matching

    def get_face_signature(self, landmarks):
        """Creates a relative distance signature of the face to identify the person."""
        # Use key landmarks: Nose tip (1), Left eye (33), Right eye (263), Mouth center (13)
        p1 = np.array([landmarks[1].x, landmarks[1].y])
        p2 = np.array([landmarks[33].x, landmarks[33].y])
        p3 = np.array([landmarks[263].x, landmarks[263].y])
        p4 = np.array([landmarks[13].x, landmarks[13].y])
        
        # Calculate distances relative to eye-to-eye distance (to be scale invariant)
        eye_dist = np.linalg.norm(p2 - p3)
        if eye_dist == 0: return None
        
        sig = [
            np.linalg.norm(p1 - p2) / eye_dist,
            np.linalg.norm(p1 - p3) / eye_dist,
            np.linalg.norm(p1 - p4) / eye_dist,
            np.linalg.norm(p2 - p4) / eye_dist,
            np.linalg.norm(p3 - p4) / eye_dist
        ]
        return np.array(sig)

    def register_candidate(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            self.registered_signature = self.get_face_signature(results.multi_face_landmarks[0].landmark)
            return True
        return False

    def get_gaze_ratio(self, iris_center, eye_outer, eye_inner):
        dist_total = np.linalg.norm(eye_outer - eye_inner)
        dist_iris = np.linalg.norm(iris_center - eye_outer)
        return dist_iris / dist_total if dist_total != 0 else 0.5

    def detect(self, frame):
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        status = "CENTER"
        multi_person = False
        identity_match = True
        
        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)
            if num_faces > 1:
                multi_person = True
                
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Identity Check
            if self.registered_signature is not None:
                current_sig = self.get_face_signature(landmarks)
                if current_sig is not None:
                    diff = np.linalg.norm(self.registered_signature - current_sig)
                    if diff > self.SIGNATURE_THRESHOLD:
                        identity_match = False

            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                for p in landmarks
            ])

            # Iris Centers
            (l_cx, l_cy), _ = cv2.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
            (r_cx, r_cy), _ = cv2.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])
            
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            left_ratio = self.get_gaze_ratio(center_left, mesh_points[self.LEFT_EYE_OUTER], mesh_points[self.LEFT_EYE_INNER])
            right_ratio = self.get_gaze_ratio(center_right, mesh_points[self.RIGHT_EYE_OUTER], mesh_points[self.RIGHT_EYE_INNER])
            
            avg_ratio = (left_ratio + right_ratio) / 2

            if avg_ratio <= 0.42:
                status = "RIGHT"
            elif avg_ratio >= 0.58:
                status = "LEFT"
            else:
                status = "CENTER"
                
            return status, multi_person, identity_match
        
        return "AWAY", False, False


if __name__ == "__main__":
    detector = GazeDetector()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        status, mp_flag = detector.detect(frame)
        
        color = (0, 255, 0) if status == "CENTER" else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if mp_flag:
            cv2.putText(frame, "MULTIPLE PEOPLE DETECTED!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow("Proctor Test", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()
