import mediapipe as mp
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

sequence = []

def extract_landmarks(results):
    try:
        landmarks = np.array(
            [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    except:
        landmarks = np.zeros(132)
    return landmarks

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global sequence
    frame = request.files['frame'].read()
    frame = np.frombuffer(frame, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    keypoints = extract_landmarks(results)

    print(np.array(sequence).shape)

    if len(sequence) < 30:
        sequence.append(keypoints)

    if len(sequence) == 30:
        return jsonify({'sequence': np.expand_dims(sequence, axis=0).tolist()})
    else:
        return jsonify({'message': 'Keypoints collected but sequence length is less than 30'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
