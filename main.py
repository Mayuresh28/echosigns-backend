from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
import base64
import mediapipe as mp

# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load models
try:
    # Only load the custom model
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM
    
    # Custom LSTM layer that ignores the 'time_major' parameter
    class CustomLSTM(LSTM):
        def __init__(self, *args, **kwargs):
            # Remove 'time_major' from kwargs if present
            if 'time_major' in kwargs:
                kwargs.pop('time_major')
            super().__init__(*args, **kwargs)
    
    # Use the custom LSTM when loading the models
    custom_objects = {'LSTM': CustomLSTM}
    
    custom_model = load_model('models/custom_sign_model.h5', custom_objects=custom_objects)
    print("Custom model loaded successfully!")
except Exception as e:
    print(f"Error loading custom model: {e}")
    custom_model = tf.keras.Sequential()

# Define actions for each model
custom_actions = np.array([
    'Hello',
    'How are you',
    'A partner who helps me talk freely anywhere',
    'Good morning',
    'This is Echosigns'
])

# Global variable to store final detected text (used by both modes)
final_detected = ""

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Store sequences and last prediction time for each peer
peer_sequences = {}
peer_last_pred_time = {}
THROTTLE_INTERVAL = 0.5  # seconds between predictions per peer

def process_frame_for_prediction(frame):
    """Process a frame and return keypoints for prediction"""
    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        # Extract hand landmarks
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))
        
        # Combine landmarks
        keypoints = np.concatenate([lh.flatten(), rh.flatten()])
        return keypoints
    except Exception as e:
        print(f"[Backend] Error in process_frame_for_prediction: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def predict_sign(peer_id, keypoints):
    """Make prediction for a peer's keypoints"""
    try:
        if keypoints is None:
            return None, None
            
        seq = peer_sequences.get(peer_id, [])
        seq.append(keypoints)
        seq = seq[-30:]  # Keep last 30 frames
        peer_sequences[peer_id] = seq
        
        if len(seq) == 30:
            input_arr = np.expand_dims(seq, axis=0)
            input_arr = np.array(input_arr, dtype=np.float32)
            res = custom_model.predict(input_arr)[0]
            action = custom_actions[np.argmax(res)]
            confidence = np.max(res)
            return action, confidence
        return None, None
    except Exception as e:
        print(f"[Backend] Error in predict_sign for peer {peer_id}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/isl')
def isl():
    return render_template('isl.html')

@app.route('/custom')
def custom():
    return render_template('custom.html')

@app.route('/get_text')
def get_text():
    global final_detected
    return jsonify({"text": final_detected})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global final_detected
    final_detected = ""
    return jsonify({"message": "Text cleared"})

@socketio.on('connect')
def handle_connect():
    print('[Socket.IO] Client connected')
    emit('connected', {'message': 'Connected to backend'})

@socketio.on('disconnect')
def handle_disconnect():
    print('[Socket.IO] Client disconnected')

@socketio.on('join_room')
def handle_join_room(data):
    room = data.get('room', 'unknown')
    print(f'[Socket.IO] User joined room: {room}')
    emit('backend_message', {'message': f'Hello from backend for room no: {room}'})

@socketio.on('camera_on')
def handle_camera_on(data):
    name = data.get('name', 'Unknown')
    print(f'[Socket.IO] {name} turned on the camera')
    emit('camera_ack', {'message': f'Camera is ON for {name}'})

@socketio.on('process_sign')
def handle_process_sign(data):
    """Handle sign language processing for remote users"""
    peer_id = data.get('peerId')
    frame_data = data.get('frame')
    
    print(f'[Backend] Received process_sign request from peer: {peer_id}')
    
    if not peer_id or not frame_data:
        print(f'[Backend] Missing peer_id or frame_data for peer: {peer_id}')
        return
    
    # Throttle predictions
    now = time.time()
    last_time = peer_last_pred_time.get(peer_id, 0)
    if now - last_time < THROTTLE_INTERVAL:
        print(f'[Backend] Throttling prediction for peer: {peer_id}')
        return
    peer_last_pred_time[peer_id] = now
    
    try:
        # Decode base64 frame
        print(f'[Backend] Decoding frame for peer: {peer_id}')
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        
        if frame is None:
            print(f'[Backend] Failed to decode frame for peer: {peer_id}')
            return
        
        print(f'[Backend] Processing frame for peer: {peer_id}')
        # Process frame and get keypoints
        keypoints = process_frame_for_prediction(frame)
        
        # Make prediction
        action, confidence = predict_sign(peer_id, keypoints)
        
        if action and confidence:
            print(f'[Backend] Prediction for peer {peer_id}: {action} (confidence: {confidence:.2f})')
            # Send prediction back to frontend with peer ID
            emit('sign_prediction', {
                'peerId': peer_id,
                'prediction': {
                    'action': str(action),
                    'confidence': float(confidence)
                }
            })
        else:
            print(f'[Backend] No prediction made for peer: {peer_id} (sequence length: {len(peer_sequences.get(peer_id, []))})')
            
    except Exception as e:
        print(f"[Backend] Error processing sign for peer {peer_id}: {str(e)}")
        import traceback
        print(traceback.format_exc())

@socketio.on('video_frame')
def handle_video_frame(data):
    """Handle video frames from local user (keeping for backward compatibility)"""
    name = data.get('name', 'Unknown')
    b64 = data.get('data')
    if not b64:
        return
    
    now = time.time()
    last_time = peer_last_pred_time.get(name, 0)
    if now - last_time < THROTTLE_INTERVAL:
        return
    peer_last_pred_time[name] = now
    
    try:
        jpg_bytes = base64.b64decode(b64)
        jpg_np = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_np, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # Process frame and get keypoints
        keypoints = process_frame_for_prediction(frame)
        
        # Make prediction
        action, confidence = predict_sign(name, keypoints)
        
        if action and confidence:
            print(f'[Prediction] {name}: {action} (confidence: {confidence:.2f})')
            emit('prediction', {
                'name': name,
                'action': str(action),
                'confidence': float(confidence)
            })
            
    except Exception as e:
        print(f"Error processing video frame for {name}: {e}")

if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)