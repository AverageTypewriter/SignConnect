from flask import Flask, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import pickle
from cv2 import imshow, waitKey, VideoCapture, destroyAllWindows
import joblib
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


app = Flask(__name__)


actions = np.array(['hello', 'thanks', 'iloveyou', 'eat'])
model = load_model('action.h5')


import os

def load_alphabet_model():
    try:
        
        try:
            model_dict = joblib.load(open('model.joblib', 'rb'))
            return model_dict['model'], None
        except:

            try:
                with open('model.p', 'rb') as f:
                    model_dict = pickle.load(f)
                return model_dict['model'], None
            except Exception as e:
                return None, f"Failed to load both joblib and pickle models: {str(e)}"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"




mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_camera():
    sequence = []
    sentence = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:30]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if res[np.argmax(res)] > threshold:
                    action = actions[np.argmax(res)]
                    if len(sentence) == 0 or action != sentence[-1]:
                        sentence.append(action)

            if len(sentence) > 5:
                sentence = sentence[-5:]

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Sign Language Recognizer', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return "Camera closed. Go back to <a href='/'>Home</a>"


@app.route('/start_alphabet')
def start_alphabet_recognition():
    
    try:
        model_dict = joblib.load('model.joblib')
        model = model_dict if hasattr(model_dict, 'predict') else model_dict['model']
    except Exception as e:
        return f"""
        <h2>Error loading model</h2>
        <p>{str(e)}</p>
        <a href='/'>Return Home</a>
        """

    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    
    labels_dict = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
        8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
        15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W',
        22: 'X', 23: 'Y'
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Webcam not accessible"

    try:
        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            if not ret:
                break

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Use only the first hand

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for point in hand_landmarks.landmark:
                    x = point.x
                    y = point.y
                    x_.append(x)
                    y_.append(y)

                for point in hand_landmarks.landmark:
                    data_aux.append(point.x - min(x_))
                    data_aux.append(point.y - min(y_))


                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_char = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_char, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow('Alphabet Recognition', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    return "Alphabet recognition closed. <a href='/'>Return Home</a>"

if __name__ == '__main__':
    app.run(debug=True)
