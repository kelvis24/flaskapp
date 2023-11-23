# from flask import Flask, request, jsonify, render_template
# import cv2
# import cvzone
# import math
# from ultralytics import YOLO
# import numpy as np
# import base64
# import io
# import matplotlib.pyplot as plt


# app = Flask(__name__)

# model = YOLO("ppe.pt")

# classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
#               'Safety Vest', 'machinery', 'vehicle']
# myColor = (0, 0, 255)

# @app.route('/')
# def home():
#     # Create a simple HTML page with an explanation of what Safety Scan does
#     return render_template('/index.html')


# @app.route('/application')
# def application():
#     # Create a simple HTML page with an explanation of what Safety Scan does
#     return render_template('/application.html')

# @app.route('/detect_objects', methods=['POST'])
# def detect_objects():
#     try:
#         # Get the uploaded image from the request
#         image_data = request.files['image']
#         image_bytes = image_data.read()
#         nparr = np.fromstring(image_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # Perform object detection on the uploaded image
#         results = model(img)

#         # Process the detection results
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Bounding Box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 # Confidence and Class Name
#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 cls = int(box.cls[0])
#                 currentClass = classNames[cls]

#                 if conf > 0.5:
#                     if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
#                         myColor = (0, 0, 255)
#                     elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
#                         myColor = (0, 255, 0)
#                     else:
#                         myColor = (255, 0, 0)

#                     cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
#                                        (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
#                                        colorT=(255, 255, 255), colorR=myColor, offset=5)
#                     cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

#         # Convert the processed image to base64
#         _, buffer = cv2.imencode('.jpg', img)
#         processed_image_base64 = base64.b64encode(buffer).decode()

#         return jsonify({"processed_image": processed_image_base64})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)


from flask import Flask, request, jsonify, render_template
import cv2
import cvzone
import math
from ultralytics import YOLO
import numpy as np
import base64
import os  # Import the os module
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

model = YOLO("safetyscanmodel/src/ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)

@app.route('/')
def home():
    # Create a simple HTML page with an explanation of what Safety Scan does
    return render_template('/index.html')

@app.route('/application')
def application():
    # Create a simple HTML page with an explanation of what Safety Scan does
    return render_template('/application.html')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        # Get the uploaded image from the request
        # image_data = request.files['image']
        # image_bytes = image_data.read()
        # nparr = np.frombuffer(image_bytes, np.uint8)  # Use frombuffer instead of fromstring
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_data = request.files['image']

        image_bytes = image_data.read()
        nparr = np.frombuffer(image_bytes, np.uint8)  # Use frombuffer instead of fromstring
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        

        # Perform object detection on the uploaded image
        results = model(img)

        # Process the detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confidence and Class Name
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > 0.5:
                    if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                        myColor = (0, 0, 255)
                    elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                        myColor = (0, 255, 0)
                    else:
                        myColor = (255, 0, 0)

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        # Save the processed image to a file
        processed_image_path = 'processed_image.jpg'
        cv2.imwrite(processed_image_path, img)

        # Convert the processed image to base64
        with open(processed_image_path, "rb") as image_file:
            processed_image_base64 = base64.b64encode(image_file.read()).decode()

        # Remove the temporary processed image file
        os.remove(processed_image_path)

        return jsonify({"processed_image": processed_image_base64})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
