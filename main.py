import cv2
import time
import logging
from gaze_detector import GazeDetector
from window_monitor import WindowMonitor

# --- CONFIG ---
THRESHOLD_AWAY_SECONDS = 3  # Flag if looking away for 3+ seconds

# --- LOGGING ---
logging.basicConfig(filename="AI_Proctor_Guard/logs/proctor_report.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    detector = GazeDetector()
    monitor = WindowMonitor()
    
    cap = cv2.VideoCapture(0)
    away_start_time = None
    registered = False
    
    print("="*50)
    print("AI PROCTOR GUARD: INITIALIZING...")
    print("Step 1: Position your face clearly in the camera.")
    print("Step 2: Press 'R' to REGISTER as the authorized candidate.")
    print("="*50)
    
    logging.info("--- SESSION STARTED ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        if not registered:
            cv2.putText(display_frame, "PRESS 'R' TO REGISTER CANDIDATE", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("AI Proctor Guard - Registration", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                if detector.register_candidate(frame):
                    registered = True
                    print("Candidate Registered Successfully!")
                    cv2.destroyWindow("AI Proctor Guard - Registration")
                else:
                    print("Failed to detect face. Try again.")
            elif key == 27: break
            continue

        # 1. Camera Monitoring
        gaze_status, multi_person, id_match = detector.detect(frame)
        
        # UI Overlays
        status_color = (0, 255, 0)
        
        # Check Identity
        if not id_match:
            msg = "ALERT: IDENTITY MISMATCH! UNKNOWN PERSON DETECTED."
            cv2.putText(display_frame, msg, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            logging.warning(msg)
            status_color = (0, 0, 255)

        # Check Gaze
        if gaze_status != "CENTER":
            if away_start_time is None:
                away_start_time = time.time()
            else:
                elapsed = time.time() - away_start_time
                if elapsed > THRESHOLD_AWAY_SECONDS:
                    msg = f"SUSPICIOUS: Looking {gaze_status} for {elapsed:.1f}s"
                    cv2.putText(display_frame, "ALERT: LOOKING AWAY!", (30, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    logging.warning(msg)
                    status_color = (0, 0, 255)
        else:
            away_start_time = None

        if multi_person:
            msg = "ALERT: MULTIPLE PEOPLE DETECTED!"
            cv2.putText(display_frame, msg, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            logging.warning(msg)
            status_color = (0, 0, 255)

        # 2. System Monitoring (Window Switching)
        window_title, is_suspicious = monitor.check_active_window()
        if window_title:
            if is_suspicious:
                msg = f"SUSPICIOUS WINDOW: {window_title}"
                logging.warning(msg)
                print(f"!!! {msg} !!!")

        # Status HUD
        cv2.rectangle(display_frame, (10, 10), (250, 100), (0,0,0), -1)
        cv2.putText(display_frame, f"Gaze: {gaze_status}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"ID Match: {'YES' if id_match else 'NO'}", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if id_match else (0, 0, 255), 1)
        cv2.putText(display_frame, f"People: {'Multiple' if multi_person else 'Single'}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("AI Proctor Guard - Active Monitoring", display_frame)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("--- SESSION ENDED ---")
    print("Monitoring ended. Check AI_Proctor_Guard/logs/proctor_report.log for details.")

if __name__ == "__main__":
    main()
