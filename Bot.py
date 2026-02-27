import webbrowser
import time
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import speech_recognition as sr

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

# Input method selection
print("Choose input method:")
print("1. Voice input")
print("2. Text input")
choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    # Voice input mode
    print("\n--- Voice Input Mode ---")
    url = get_voice_input("Please say the URL:")
    while not url:
        url = get_voice_input("Please say the URL again:")
    
    duration_text = get_voice_input("Please say the time interval in seconds:")
    while not duration_text:
        duration_text = get_voice_input("Please say the time interval again:")
    duration = int(''.join(filter(str.isdigit, duration_text.split()[0])))
    
    views_text = get_voice_input("Please say how many views:")
    while not views_text:
        views_text = get_voice_input("Please say the number of views again:")
    x = int(''.join(filter(str.isdigit, views_text.split()[0])))
else:
    # Traditional text input mode
    print("\n--- Text Input Mode ---")
    url = input("url: ")
    duration = int(input(" Time space: "))
    x = int(input(" how many views: "))

# Use neural network to predict optimal behavior
prediction = model.predict(np.array([[duration, x]]), verbose=0)[0][0]
print(f"AI Confidence Score: {prediction:.2f}")

if prediction > 0.5:
    print("AI suggests: Good parameters for bot activity")
else:
    print("AI suggests: Consider adjusting parameters")

print(f"Starting bot with {x} views at {duration} second intervals...\n")

for i in range(x):
    webbrowser.open_new(url)
    time.sleep(duration)
    print(f"View {i+1}/{x} completed")

print("Bot completed!")