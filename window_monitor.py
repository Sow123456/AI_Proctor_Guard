import pygetwindow as gw
import time
import logging

class WindowMonitor:
    def __init__(self, log_file="AI_Proctor_Guard/logs/system_activity.log"):
        self.active_window = ""
        self.suspicious_apps = [
            "ChatGPT", "Claude", "Gemini", "Perplexity", 
            "Chrome", "Edge", "Firefox", "Opera", "Brave",
            "Discord", "Telegram", "WhatsApp", "Slack",
            "Notepad", "Word", "Excel", "Calculator",
            "Python", "VS Code", "Sublime", "Atom"
        ]
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(message)s')

    def check_active_window(self):
        try:
            current_window = gw.getActiveWindow()
            if current_window:
                title = current_window.title
                if title != self.active_window:
                    self.active_window = title
                    is_suspicious = any(app.lower() in title.lower() for app in self.suspicious_apps)
                    
                    message = f"ACTIVE WINDOW CHANGE: {title}"
                    if is_suspicious:
                        message += " [SUSPICIOUS]"
                    
                    print(message)
                    logging.info(message)
                    return title, is_suspicious
        except Exception as e:
            pass
        return None, False

if __name__ == "__main__":
    monitor = WindowMonitor()
    print("Monitoring system windows... Press Ctrl+C to stop.")
    while True:
        monitor.check_active_window()
        time.sleep(1)
