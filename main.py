import time
import csv
import os
import psutil
import threading
import pandas as pd
import win32gui               
import win32process 
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from PyQt6.QtWidgets import QFileDialog
from datetime import datetime         
from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox,
    QTableWidget, QTableWidgetItem, QLineEdit, QHBoxLayout, QListWidget
)

import sys
import json
from sentence_transformers import SentenceTransformer, util

# File paths
LOG_FILE = "screen_activity_log.csv"
GOALS_FILE = "user_goals.json"
ANALYZED_LOG_FILE = "analyzed_screen_activity_log.csv"
XGBOOST_RESULT_FILE = "xgboost_result.csv"
XGBOOST_MODEL_FILE = "xgboost_model.json"

# Activity tracking globals
current_window = None
current_app = None
current_url = None
start_timestamp = None
activity_data = []
tracking = False

# Load NLP model
model = SentenceTransformer("all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = 0.25

def init_csv():
    if not os.path.exists(LOG_FILE):
        headers = ["Start Timestamp", "End Timestamp", "Application", "Window Title", "URL", "Time Spent (s)"]
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(headers)

def get_active_process_name():
    try:
        hwnd = win32gui.GetForegroundWindow()
        if hwnd:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)
            return process.name()
        return "Unknown"
    except Exception:
        return "Unknown"

def get_active_window_title():
    try:
        hwnd = win32gui.GetForegroundWindow()
        if hwnd:
            return win32gui.GetWindowText(hwnd)
        return "Unknown"
    except Exception:
        return "Unknown"

def track_activity():
    global current_window, current_app, current_url, start_timestamp, activity_data, tracking
    app_mapping = {"Code.exe": "VS Code", "chrome.exe": "Google Chrome", "msedge.exe": "Microsoft Edge"}

    while tracking:
        try:
            window_title = get_active_window_title()
            exe_name = get_active_process_name()
            app_name = app_mapping.get(exe_name, exe_name)
            url = "N/A"
            end_timestamp = datetime.now()

            if window_title != current_window or app_name != current_app:
                if current_window and current_app and start_timestamp:
                    time_spent = (end_timestamp - start_timestamp).total_seconds()
                    activity_data.append([
                        start_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        end_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        current_app, current_window, current_url, round(time_spent, 2)
                    ])
                current_window, current_app, current_url, start_timestamp = window_title, app_name, url, end_timestamp

            time.sleep(1)
        except Exception as e:
            print(f"Error tracking activity: {e}")
            time.sleep(1)

def save_logs():
    global activity_data
    while tracking:
        if activity_data:
            try:
                df = pd.DataFrame(activity_data, columns=["Start Timestamp", "End Timestamp", "Application", "Window Title", "URL", "Time Spent (s)"])
                df.to_csv(LOG_FILE, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)
                activity_data = []
            except Exception as e:
                print(f"Error writing to CSV: {e}")
        time.sleep(1)

def run_semantic_matching():
    if not os.path.exists(LOG_FILE) or not os.path.exists(GOALS_FILE):
        return

    log_df = pd.read_csv(LOG_FILE)
    with open(GOALS_FILE, "r", encoding='utf-8') as f:
        goals_data = json.load(f)

    log_df["Matched Goal"] = ""
    log_df["Similarity Score"] = 0.0

    for index, row in log_df.iterrows():
        activity_text = f"{row['Application']} {row['Window Title']}"
        activity_embedding = model.encode(activity_text, convert_to_tensor=True)

        max_score = 0
        matched_goal = "No Match"

        for goal_entry in goals_data:
            goal = goal_entry["goal"]
            keywords = goal_entry["keywords"]
            goal_text = goal + " " + " ".join(keywords)
            goal_embedding = model.encode(goal_text, convert_to_tensor=True)
            similarity = util.cos_sim(activity_embedding, goal_embedding).item()

            if similarity > max_score:
                max_score = similarity
                matched_goal = goal

        log_df.at[index, "Matched Goal"] = matched_goal if max_score >= SIMILARITY_THRESHOLD else "No Match"
        log_df.at[index, "Similarity Score"] = round(max_score, 3)

    log_df["User Corrected Goal"] = log_df["Matched Goal"]
    log_df.to_csv(ANALYZED_LOG_FILE, index=False)
    print("Semantic matching complete.")

def save_goals_to_file(goals):
    with open(GOALS_FILE, "w", encoding="utf-8") as f:
        json.dump(goals, f, indent=4)

def load_goals_from_file():
    if os.path.exists(GOALS_FILE):
        with open(GOALS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

class ScreenMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Screen Monitor")
        self.setGeometry(200, 200, 500, 500)

        layout = QVBoxLayout()
        self.label = QLabel("Status: Monitoring Not Started")
        layout.addWidget(self.label)

        self.goal_input = QLineEdit()
        self.goal_input.setPlaceholderText("Enter study goal")
        layout.addWidget(self.goal_input)

        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("Enter keywords (comma separated)")
        layout.addWidget(self.keyword_input)

        btn_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Goal")
        self.save_button.clicked.connect(self.save_goal)
        btn_layout.addWidget(self.save_button)

        self.delete_button = QPushButton("Delete Goal")
        self.delete_button.clicked.connect(self.delete_goal)
        btn_layout.addWidget(self.delete_button)
        layout.addLayout(btn_layout)

        self.goal_list = QListWidget()
        layout.addWidget(self.goal_list)
        self.load_goals()

        self.start_button = QPushButton("Start Monitoring")
        self.start_button.clicked.connect(self.start_tracking)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Monitoring")
        self.stop_button.clicked.connect(self.stop_tracking)
        layout.addWidget(self.stop_button)

        self.view_button = QPushButton("View Logs")
        self.view_button.clicked.connect(self.view_logs)
        layout.addWidget(self.view_button)

        self.view_analyzed_button = QPushButton("View Predictions of NLP")
        self.view_analyzed_button.clicked.connect(self.view_analyzed_logs)
        layout.addWidget(self.view_analyzed_button)

        self.setLayout(layout)
        self.goals = load_goals_from_file()

        self.predict_button = QPushButton("View Predictions of XGBoost")
        self.predict_button.clicked.connect(self.view_predictions)
        layout.addWidget(self.predict_button)

    def save_goal(self):
        goal = self.goal_input.text().strip()
        keywords = [kw.strip() for kw in self.keyword_input.text().strip().split(",") if kw.strip()]
        if goal and keywords:
            self.goals.append({"goal": goal, "keywords": keywords})
            self.goal_list.addItem(f"{goal} - {keywords}")
            save_goals_to_file(self.goals)
            self.goal_input.clear()
            self.keyword_input.clear()
            QMessageBox.information(self, "Saved", f"Goal '{goal}' saved.")
        else:
            QMessageBox.warning(self, "Invalid", "Please enter both goal and keywords.")

    def delete_goal(self):
        selected_items = self.goal_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a goal to delete.")
            return
        for item in selected_items:
            index = self.goal_list.row(item)
            self.goal_list.takeItem(index)
            del self.goals[index]
        save_goals_to_file(self.goals)
        QMessageBox.information(self, "Deleted", "Goal(s) deleted.")

    def load_goals(self):
        self.goal_list.clear()
        for goal in load_goals_from_file():
            self.goal_list.addItem(f"{goal['goal']} - {goal['keywords']}")

    def start_tracking(self):
        global tracking
        if not tracking:
            tracking = True
            threading.Thread(target=track_activity, daemon=True).start()
            threading.Thread(target=save_logs, daemon=True).start()
            self.label.setText("Status: Monitoring Active")
            QMessageBox.information(self, "Started", "Screen monitoring started.")

    def stop_tracking(self):
        global tracking
        if tracking:
            tracking = False
            self.label.setText("Status: Monitoring Stopped")
            run_semantic_matching()
            QMessageBox.information(self, "Stopped", "Monitoring stopped.\nSemantic analysis complete.")

    def view_logs(self):
        self.log_viewer = LogViewer()
        self.log_viewer.show()

    def view_analyzed_logs(self):
        if not os.path.exists(ANALYZED_LOG_FILE):
            QMessageBox.warning(self, "Not Found", "Analyzed logs not found. Please stop monitoring first.")
            return
        self.analyzed_log_viewer = AnalyzedLogEditor()
        self.analyzed_log_viewer.show()
    def view_predictions(self):
        if not os.path.exists(XGBOOST_RESULT_FILE):
            run_xgboost_prediction()
        self.pred_viewer = PredictionViewer()
        self.pred_viewer.show()

class LogViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Activity Log Viewer")
        self.setGeometry(250, 250, 800, 400)
        layout = QVBoxLayout()
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.load_logs()

    def load_logs(self):
        if not os.path.exists(LOG_FILE):
            QMessageBox.critical(self, "Error", "Log file not found!")
            return
        df = pd.read_csv(LOG_FILE)
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)
        for row_idx, row_data in df.iterrows():
            for col_idx, col_data in enumerate(row_data):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))

class AnalyzedLogEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edit Analyzed Log")
        self.setGeometry(300, 300, 1000, 500)
        layout = QVBoxLayout()
        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.save_button = QPushButton("Save Corrections")
        self.save_button.clicked.connect(self.save_corrections)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.load_logs()

    def load_logs(self):
        if not os.path.exists(ANALYZED_LOG_FILE):
            QMessageBox.critical(self, "Error", "Analyzed log file not found!")
            return
        self.df = pd.read_csv(ANALYZED_LOG_FILE)
        self.table.setRowCount(len(self.df))
        self.table.setColumnCount(len(self.df.columns))
        self.table.setHorizontalHeaderLabels(self.df.columns)

        for row_idx, row_data in self.df.iterrows():
            for col_idx, col_data in enumerate(row_data):
                item = QTableWidgetItem(str(col_data))
                if self.df.columns[col_idx] == "User Corrected Goal":
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row_idx, col_idx, item)

    def save_corrections(self):
        for row in range(self.table.rowCount()):
            self.df.at[row, "User Corrected Goal"] = self.table.item(row, self.df.columns.get_loc("User Corrected Goal")).text()
        self.df.to_csv(ANALYZED_LOG_FILE, index=False)
        QMessageBox.information(self, "Saved", "Corrections saved successfully!")
def run_xgboost_prediction():
    if not os.path.exists(ANALYZED_LOG_FILE) or not os.path.exists(XGBOOST_MODEL_FILE):
        print("Missing model or analyzed log file.")
        return

    df = pd.read_csv(ANALYZED_LOG_FILE)
    required_cols = ["Application", "Window Title", "User Corrected Goal"]
    if not all(col in df.columns for col in required_cols):
        print("Missing columns in analyzed log.")
        return

    # Convert text features into basic numeric representations (very simplified)
    df["Combined Text"] = df["Application"] + " " + df["Window Title"]
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["User Corrected Goal"])
    
    # Prepare features (this is a simplified placeholder for actual NLP embeddings)
    df["Text Length"] = df["Combined Text"].apply(len)
    df["Num Words"] = df["Combined Text"].apply(lambda x: len(x.split()))
    X = df[["Text Length", "Num Words"]]

    model = xgb.XGBClassifier()
    model.load_model(XGBOOST_MODEL_FILE)
    
    df["Predicted Goal"] = le.inverse_transform(model.predict(X))
    df.to_csv(XGBOOST_RESULT_FILE, index=False)
    print("Prediction complete and saved to xgboost_result.csv.")

class PredictionViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prediction Results")
        self.setGeometry(300, 300, 1000, 500)
        layout = QVBoxLayout()
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.load_predictions()

    def load_predictions(self):
        if not os.path.exists(XGBOOST_RESULT_FILE):
            QMessageBox.critical(self, "Error", "Prediction file not found!")
            return
        df = pd.read_csv(XGBOOST_RESULT_FILE)
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)
        for row_idx, row_data in df.iterrows():
            for col_idx, col_data in enumerate(row_data):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))

if __name__ == "__main__":
    init_csv()
    app = QApplication(sys.argv)
    window = ScreenMonitorApp()
    window.show()
    sys.exit(app.exec())