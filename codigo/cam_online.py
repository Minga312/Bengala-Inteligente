from flask import Flask, Response
import cv2

app = Flask(__name__)

def generate_frames(cam):
    camera = cv2.VideoCapture(cam)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/cam/<int:cam_number>')
def video_feed(cam_number):
    return Response(generate_frames(cam_number), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)