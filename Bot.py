import webbrowser
import time
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import speech_recognition as sr
import cv2
import os
from datetime import datetime
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QSpinBox, QRadioButton, 
                             QButtonGroup, QTextEdit, QProgressBar, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPixmap
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QCheckBox

# Initialize neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate training data based on common patterns
X_train = np.array([
    [1, 1],    # low duration, few views
    [2, 5],    # low duration, moderate views
    [5, 10],   # moderate duration, moderate views
    [10, 20],  # high duration, high views
    [0.5, 3],  # very low duration, few views
    [15, 50],  # high duration, many views
])

y_train = np.array([[0], [0], [1], [1], [0], [1]])

# Train the neural network
print("Training neural network...")
model.fit(X_train, y_train, epochs=50, verbose=0)
print("Neural network trained!")

# Voice recognition setup
def get_voice_input(prompt):
    """Capture voice input from user"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(prompt)
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio. Please try again.")
            return None
        except sr.RequestError:
            print("Could not request results from the speech recognition service.")
            return None

# Video generation function
def generate_video(output_filename, duration_seconds, fps=30):
    """Generate a video file with timestamps"""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    total_frames = int(duration_seconds * fps)
    print(f"Generating video: {output_filename}")
    
    for frame_num in range(total_frames):
        # Create a frame with background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (100, 100, 100)  # Gray background
        
        # Add timestamp text
        timestamp = frame_num / fps
        text = f"Frame: {frame_num} | Time: {timestamp:.2f}s"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        
        if (frame_num + 1) % (fps * 5) == 0:
            print(f"  Progress: {frame_num + 1}/{total_frames} frames")
    
    out.release()
    print(f"Video saved: {output_filename}")
    return output_filename

# Bot worker thread
class BotWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, url, views, interval, generate_vid, video_duration):
        super().__init__()
        self.url = url
        self.views = views
        self.interval = interval
        self.generate_vid = generate_vid
        self.video_duration = video_duration
    
    def run(self):
        if self.generate_vid:
            self.progress_signal.emit("🎬 Generating video...")
            video_filename = f"bot_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            generate_video(video_filename, self.video_duration)
            self.progress_signal.emit(f"✓ Video generated: {video_filename}")
        
        self.progress_signal.emit(f"🚀 Starting bot with {self.views} views...\n")
        
        for i in range(self.views):
            webbrowser.open_new(self.url)
            self.progress_signal.emit(f"✓ View {i+1}/{self.views} completed at {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(self.interval)
        
        self.progress_signal.emit("\n✓ Bot completed successfully!")
        self.finished_signal.emit()

# Futuristic GUI Class
class BotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        # Main window setup
        self.setWindowTitle("🤖 Advanced Bot Control System v2.0")
        self.setGeometry(100, 100, 900, 800)
        self.setStyleSheet(self.get_futuristic_style())
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("⚡ BOT CONTROL INTERFACE ⚡")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00ff00; text-shadow: 0 0 10px #00ff00;")
        main_layout.addWidget(title)
        
        # Input section
        input_layout = QVBoxLayout()
        
        # URL Input
        url_label = QLabel("🔗 TARGET URL:")
        url_label.setStyleSheet("color: #00ffff; font-weight: bold;")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter target URL...")
        self.url_input.setStyleSheet("background-color: #1a1a1a; color: #00ff00; border: 2px solid #00ff00; padding: 8px; border-radius: 5px;")
        input_layout.addWidget(url_label)
        input_layout.addWidget(self.url_input)
        
        # Duration and Views in a row
        row1_layout = QHBoxLayout()
        
        duration_label = QLabel("⏱️ INTERVAL (seconds):")
        duration_label.setStyleSheet("color: #00ffff; font-weight: bold;")
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setValue(5)
        self.duration_spin.setStyleSheet("background-color: #1a1a1a; color: #00ff00; border: 2px solid #00ff00; padding: 5px;")
        row1_layout.addWidget(duration_label)
        row1_layout.addWidget(self.duration_spin)
        
        views_label = QLabel("👁️ VIEWS:")
        views_label.setStyleSheet("color: #00ffff; font-weight: bold;")
        self.views_spin = QSpinBox()
        self.views_spin.setRange(1, 1000)
        self.views_spin.setValue(10)
        self.views_spin.setStyleSheet("background-color: #1a1a1a; color: #00ff00; border: 2px solid #00ff00; padding: 5px;")
        row1_layout.addWidget(views_label)
        row1_layout.addWidget(self.views_spin)
        
        input_layout.addLayout(row1_layout)
        
        # Video generation section
        video_layout = QHBoxLayout()
        self.video_checkbox = QCheckBox("🎬 Generate Video")
        self.video_checkbox.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.video_checkbox.toggled.connect(self.on_video_toggle)
        
        video_duration_label = QLabel("Duration (sec):")
        video_duration_label.setStyleSheet("color: #00ffff;")
        self.video_duration_spin = QSpinBox()
        self.video_duration_spin.setRange(1, 300)
        self.video_duration_spin.setValue(5)
        self.video_duration_spin.setEnabled(False)
        self.video_duration_spin.setStyleSheet("background-color: #1a1a1a; color: #00ff00; border: 2px solid #00ff00; padding: 5px;")
        
        video_layout.addWidget(self.video_checkbox)
        video_layout.addWidget(video_duration_label)
        video_layout.addWidget(self.video_duration_spin)
        video_layout.addStretch()
        
        input_layout.addLayout(video_layout)
        
        main_layout.addLayout(input_layout)
        
        # AI Prediction section
        pred_layout = QVBoxLayout()
        pred_label = QLabel("🤖 AI CONFIDENCE ANALYSIS")
        pred_label.setStyleSheet("color: #ff00ff; font-weight: bold; font-size: 12px;")
        
        self.prediction_output = QLabel("Ready for analysis...")
        self.prediction_output.setStyleSheet("color: #ffff00; background-color: #1a1a1a; padding: 10px; border: 2px dashed #ff00ff; border-radius: 5px;")
        self.prediction_output.setWordWrap(True)
        
        pred_layout.addWidget(pred_label)
        pred_layout.addWidget(self.prediction_output)
        main_layout.addLayout(pred_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        analyze_btn = QPushButton("🔬 ANALYZE")
        analyze_btn.setStyleSheet(self.get_button_style("#ff6600"))
        analyze_btn.clicked.connect(self.analyze_parameters)
        button_layout.addWidget(analyze_btn)
        
        start_btn = QPushButton("▶️ INITIATE BOT")
        start_btn.setStyleSheet(self.get_button_style("#00ff00"))
        start_btn.clicked.connect(self.start_bot)
        button_layout.addWidget(start_btn)
        
        reset_btn = QPushButton("🔄 RESET")
        reset_btn.setStyleSheet(self.get_button_style("#ff0000"))
        reset_btn.clicked.connect(self.reset_form)
        button_layout.addWidget(reset_btn)
        
        main_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: 2px solid #00ff00;
                border-radius: 5px;
                text-align: center;
                color: #00ff00;
            }
            QProgressBar::chunk {
                background-color: #00ff00;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # Output log
        log_label = QLabel("📊 SYSTEM LOG:")
        log_label.setStyleSheet("color: #00ffff; font-weight: bold; font-size: 11px;")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #0a0a0a; color: #00ff00; border: 2px solid #00ff00; border-radius: 5px; font-family: Courier;")
        self.log_output.setMaximumHeight(250)
        
        main_layout.addWidget(log_label)
        main_layout.addWidget(self.log_output)
        
        main_layout.addStretch()
    
    def get_futuristic_style(self):
        return """
        QMainWindow {
            background-color: #0d0d0d;
        }
        QLabel {
            color: #00ff00;
        }
        QLineEdit, QSpinBox {
            background-color: #1a1a1a;
            color: #00ff00;
            border: 2px solid #00ff00;
            border-radius: 5px;
            padding: 5px;
            font-weight: bold;
        }
        QCheckBox {
            color: #00ff00;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
        """
    
    def get_button_style(self, color):
        return f"""
        QPushButton {{
            background-color: {color};
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            font-weight: bold;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background-color: {self.lighten_color(color)};
            box-shadow: 0 0 20px {color};
        }}
        QPushButton:pressed {{
            background-color: {self.darken_color(color)};
        }}
        """
    
    def lighten_color(self, color):
        return color.replace("0", "5")[:7]
    
    def darken_color(self, color):
        return color.replace("f", "a")[:7]
    
    def on_video_toggle(self):
        self.video_duration_spin.setEnabled(self.video_checkbox.isChecked())
    
    def analyze_parameters(self):
        try:
            duration = self.duration_spin.value()
            views = self.views_spin.value()
            
            prediction = model.predict(np.array([[duration, views]]), verbose=0)[0][0]
            confidence = prediction * 100
            
            if prediction > 0.5:
                status = "✓ OPTIMAL PARAMETERS"
                color = "#00ff00"
            else:
                status = "⚠️ SUBOPTIMAL PARAMETERS"
                color = "#ffff00"
            
            result = f"[ANALYSIS REPORT]\nConfidence Score: {confidence:.1f}%\nStatus: {status}\nRecommendation: {'Good for execution' if prediction > 0.5 else 'Consider adjusting values'}"
            self.prediction_output.setText(result)
            self.prediction_output.setStyleSheet(f"color: {color}; background-color: #1a1a1a; padding: 10px; border: 2px dashed #ff00ff; border-radius: 5px;")
            
            self.log_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔬 Analysis: Confidence {confidence:.1f}%")
        except Exception as e:
            self.log_output.append(f"[ERROR] {str(e)}")
    
    def start_bot(self):
        url = self.url_input.text().strip()
        if not url:
            self.log_output.append("[ERROR] URL is required!")
            return
        
        duration = self.duration_spin.value()
        views = self.views_spin.value()
        generate_vid = self.video_checkbox.isChecked()
        video_duration = self.video_duration_spin.value() if generate_vid else 0
        
        self.log_output.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚀 INITIATING BOT SEQUENCE...")
        self.worker = BotWorker(url, views, duration, generate_vid, video_duration)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()
    
    def on_progress(self, message):
        self.log_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def on_finished(self):
        self.log_output.append("[SYSTEM] Bot sequence completed!")
    
    def reset_form(self):
        self.url_input.clear()
        self.duration_spin.setValue(5)
        self.views_spin.setValue(10)
        self.video_checkbox.setChecked(False)
        self.video_duration_spin.setValue(5)
        self.prediction_output.setText("Ready for analysis...")
        self.log_output.clear()

# Main execution
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = BotGUI()
    gui.show()
    sys.exit(app.exec_())