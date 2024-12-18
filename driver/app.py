from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import imutils
import face_recognition
import os

app = Flask(__name__)

# Load known faces
known_face_encodings = []
known_face_data = {}  # Store user information with their face encoding

# Directory containing user images and their data
image_dir = r"D:\MAJOR_PROJECT\driver\driver\major_face"
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        user_name = filename.split('.')[0]

        # Add the encoding and user data
        known_face_encodings.append(encoding)
        known_face_data[user_name] = {
            "name": user_name,
            "license": f"DL{str(len(known_face_data)).zfill(2)}",
            "vehicle": "Car Model XYZ"
        }

current_user = {"status": "No Face Detected", "user_info": None, "image_path": None}
output_dir = "static/detected_faces"  # Directory to save detected face images
os.makedirs(output_dir, exist_ok=True)

def generate_frames():
    global current_user
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=800)
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if len(face_locations) == 0:
            # No face detected
            current_user = {"status": "No Face Detected", "user_info": "New User", "image_path": None}
        else:
            detected = False
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = list(known_face_data.keys())[first_match_index]
                    detected = True

                    # Save the face image
                    face_image = frame[top:bottom, left:right]
                    face_image_path = os.path.join(output_dir, f"{name}.jpg")
                    cv2.imwrite(face_image_path, face_image)

                    # Update current user information
                    current_user = {
                        "status": "Face Recognized",
                        "user_info": known_face_data[name],
                        "image_path": f"/{face_image_path}"
                    }
                    break

            if not detected:
                current_user = {"status": "Unknown Face", "user_info": "New User", "image_path": None}

        # Draw boxes and labels around faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = list(known_face_data.keys())[first_match_index]
            else:
                name = "New User"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/user_info')
def user_info():
    """API endpoint to get the current user information."""
    return jsonify(current_user)


@app.route('/static/detected_faces/<filename>')
def get_face_image(filename):
    """Serve the saved face images."""
    return send_from_directory(output_dir, filename)


if __name__ == '__main__':
    app.run(debug=True)
