from flask import Flask, abort, request, jsonify
import base64
import os
from datetime import datetime
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import tensorflow as tf
from flask_cors import CORS
from tensorflow.keras.models import load_model  # Import load_model function
from tensorflow.keras.preprocessing.image import load_img  # Import load_img function
import base64
import tensorflow as tf
import time  # Import the time module
import firebase_admin
from firebase_admin import credentials, storage
from PIL import Image
import io
import numpy as np  
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import cv2
import cvzone
import math
from ultralytics import YOLO
import numpy as np
import base64
import os  # Import the os module
from flask_cors import CORS
import requests
import json
from firebase_admin import credentials, initialize_app, storage
import os
import json
import shutil


app = Flask(__name__)
CORS(app)

# model = YOLO("/Users/elviskimara/Downloads/docker-lamba-aws/image/src/ppe.pt")
model = YOLO("app/src/ppe.pt")


classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)


data_dir = 'app/src/database_directory'
models_dir = 'app/src/models'

project_id = "safetyscan-8a191"
# Initialize Firebase Admin SDK with your credentials
cred = credentials.Certificate("app/src/safetyscan-8a191-firebase-adminsdk-rsrsi-85c1eee585.json")
firebase_admin.initialize_app(cred, {'storageBucket': f'{project_id}.appspot.com'})


@app.route('/')
def home():
    # Create a simple HTML page with an explanation of what Safety Scan does
    return render_template('/index.html')

@app.route('/application')
def application():
    # Create a simple HTML page with an explanation of what Safety Scan does
    return render_template('/application.html')


# Function to generate a unique ID based on the current time
def generate_unique_id():
    current_time = datetime.now()
    unique_id = current_time.strftime("%Y%m%d%H%M%S")
    return unique_id

# Function to generate a timestamp for the processed image
def generate_timestamp():
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    return timestamp


@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.json
    print("In add user")
    print(data['name'])
    print(data['organizationId'])
    try:
        # Access 'name' and 'base64' directly
        append_to_database(data['name'], data['base64'], data['organizationId'])
        print("HERE")
        train_model()  # Retrain the model whenever any new image is added
        return jsonify({'message': 'User added and model retrained'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/trigger_train', methods=['GET'])
def trigger_training():
    try:
        train_model()  # Manually trigger the training
        return jsonify({'message': 'Model training triggered and completed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def append_to_database(name, base64_image, organizationId):
    print("In append to database")
    new_image_dir = os.path.join(data_dir, name)
    os.makedirs(new_image_dir, exist_ok=True)
    print("done1")

    try:
        # Encode the base64 data as bytes
        image_bytes = base64.b64decode(base64_image)
        print("done2")

        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'{timestamp}.jpg'
        print("done3")

        # Save the image to the user's directory
        image_path = os.path.join(new_image_dir, filename)
        with open(image_path, 'wb') as file:
            file.write(image_bytes)

        # Save the name and organizationId to a JSON file
        info = {
            "name": name,
            "organizationId": organizationId,
            "image_path": image_path
        }
        info_filename = f'{timestamp}_info.json'
        info_path = os.path.join(new_image_dir, info_filename)
        with open(info_path, 'w') as info_file:
            json.dump(info, info_file)

        print("Image and information saved successfully")
    except Exception as e:
        print(f"Error decoding base64 data: {e}")

def get_organization_by_name(name):
    try:
        # Search for the information file based on the provided name
        user_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        for user_dir in user_dirs:
            info_files = [f for f in os.listdir(os.path.join(data_dir, user_dir)) if f.endswith('_info.json')]
            for info_file in info_files:
                with open(os.path.join(data_dir, user_dir, info_file), 'r') as info_file:
                    info = json.load(info_file)
                    if info["name"] == name:
                        return info["organizationId"]
    except Exception as e:
        print(f"Error in get_organization_by_name: {e}")
    return None


def train_model():
    try:
        print("Here 0")
        # Create an ImageDataGenerator for data augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)  # 20% of the data will be used for validation
        print("Here 1")

        # Load and preprocess the dataset from the database directory
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(160, 160),
            batch_size=32,
            class_mode='sparse',
            subset='training')
        print("Here 2")

        val_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(160, 160),
            batch_size=32,
            class_mode='sparse',
            subset='validation')
        
        print("Here 3")

        # Load a pre-trained MobileNetV2 model
        base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                                    include_top=False,
                                                    weights='imagenet')
        base_model.trainable = False  # Freeze the base model
        print("Here 4")

        # Add a custom classification head
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(train_generator.class_indices), activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        print("Here 5")

        # Train the model
        model.fit(train_generator, epochs=10, validation_data=val_generator)
        print("Here 6")

        # Save the model with a timestamp
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model.save(os.path.join(models_dir, f'my_facerecognition_model_{timestamp}'))

        # Keep only the 5 most recent models
        all_models = sorted(glob.glob(os.path.join(models_dir, 'my_facerecognition_model_*')), key=lambda x: int(x.split('_')[-1]), reverse=True)
        for i, old_model in enumerate(all_models[5:], start=1):
            try:
                shutil.rmtree(old_model)
                print(f"Deleted older model {i}: {old_model}")
            except OSError as e:
                print(f"Error deleting older model {i}: {old_model} - {e}")

                    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_image', methods=['POST'])
def test_image():
    try:
        data = request.json
        base64_image = data['base64']

        # Load the most recent model
        most_recent_model = get_most_recent_model()

        if most_recent_model is None:
            # return jsonify({'error': 'No models available'}), 500
            abort(500, description='No models available')
            
        # Extract the model name from the model file path
        model_name = os.path.basename(most_recent_model)
        print(model_name)

        # Load the model
        model = load_model(most_recent_model)
        
        print("done")

        # Preprocess the image
        image = preprocess_image(base64_image)
        print("done 2")

        # Predict the name
        prediction = predict_name( model, image)
        print("done 3")

        # Create a response JSON including the model name
        response_data = {
            'name': prediction,
            'model_used': model_name  # Include the model name in the response
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_most_recent_model():
    try:
        all_models = sorted(glob.glob(os.path.join(models_dir, 'my_facerecognition_model_*')))
        
        if not all_models:
            return None

        most_recent_model = max(all_models, key=os.path.getctime)
        return most_recent_model
    except Exception as e:
        print(f"Error getting most recent model: {str(e)}")
        return None


def preprocess_image(base64_image):
    try:
        # Decode the base64 data as bytes
        image_bytes = base64.b64decode(base64_image)

        # Open the image using PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Resize the image to the target size (160x160)
        image = image.resize((160, 160))

        # Convert the image to a NumPy array (if needed)
        image_array = np.array(image)

        # Perform any additional preprocessing (e.g., scaling to [0, 1])
        image_array = image_array / 255.0  # Scale to [0, 1] range

        return image_array

    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


def predict_name( model, image):
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)  # 20% of the data will be used for validation

    # Load and preprocess the dataset from the database directory
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(160, 160),
        batch_size=32,
        class_mode='sparse',
        subset='training')

    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    class_index = tf.argmax(prediction, axis=1)

    print(int(class_index[0]))
    predicted_label = np.argmax(prediction)
    predicted_name = list(train_generator.class_indices.keys())[predicted_label]

    return predicted_name


# Initialize a counter for consecutive safety violations
consecutive_safety_violations = 0

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        global consecutive_safety_violations

        # Get the uploaded image from the request
        image_data = request.files['image']
        location = request.form['location']
        date = request.form['date']
        time = request.form['time']
        
        print(location, date, time)
        
        image_bytes = image_data.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection on the uploaded image
        results = model(img)  # Perform object detection and assign results

        # Initialize violation details as a list
        violation_details = []

        # Define colors for different classes
        class_colors = {
            'NO-Hardhat': (255, 0, 0),  # Red for NO-Hardhat
            'NO-Safety Vest': (0, 0, 255),  # Blue for NO-Safety Vest
            'NO-Mask': (0, 255, 0),  # Green for NO-Mask
            'Hardhat': (0, 255, 255),  # Yellow for Hardhat
            'Safety Vest': (255, 0, 255),  # Purple for Safety Vest
            'Mask': (255, 255, 0)  # Cyan for Mask
        }

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
                        violation_details.append(currentClass)

                    myColor = class_colors.get(currentClass, (0, 0, 0))  # Default to black if class not found
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        original_image = img.copy()
        # Resize the image to (160, 160) for Face Recognition
        img_resized = cv2.resize(img, (160, 160))
        
        # Save the processed image to a file with a timestamp
        timestamp = generate_timestamp()
        processed_image_path = f'processed_image_{timestamp}.jpg'
        cv2.imwrite(processed_image_path, original_image)

        # Convert the processed image to base64
        with open(processed_image_path, "rb") as image_file:
            processed_image_base64 = base64.b64encode(image_file.read()).decode()

        # Load the most recent model
        most_recent_model = get_most_recent_model()

        if most_recent_model is None:
            # Handle the case when no models are available
            return jsonify({
                "violation_type": "No Violation",
                "violation_details": [],
                "name_prediction": None,
                "processed_image": processed_image_base64,
                "id": generate_unique_id(),
                "alarm": "off",
                "location": location,
                "date": date,
                "time": time,
                "organizationId": None
            })

        # Extract the model name from the model file path
        model_name = os.path.basename(most_recent_model)
        print(model_name)

        # Load the model
        name_model = load_model(most_recent_model)

        # Predict the name using the Face Recognition model
        name_prediction = predict_name(name_model, img_resized)
        
        organizationId = get_organization_by_name(name_prediction)

        # Determine the violation type based on violation details
        violation_type = "Safety Violation" if violation_details else "No Violation"

        # Check if the violation type is Safety Violation and increment the counter
        if violation_type == "Safety Violation":
            consecutive_safety_violations += 1
        else:
            consecutive_safety_violations = 0  # Reset the counter if no safety violation

        # Generate a unique ID based on the current time
        unique_id = generate_unique_id()

        # Create the response JSON
        response_data = {
            "violation_type": violation_type,
            "violation_details": violation_details,
            "name_prediction": name_prediction,
            "processed_image": processed_image_path,
            "id": unique_id,
            "alarm": "off",
            "location": location,
            "date": date,
            "time": time,
            "organizationId": organizationId
        }

        # Check if there have been 10 consecutive safety violations
        if consecutive_safety_violations >= 10:
            response_data["alarm"] = "on"
            consecutive_safety_violations = 0  # Reset the counter
            push_to_DB(response_data)

        print(response_data)

        # Remove the temporary processed image file
        os.remove(processed_image_path)
        
        response_data["processed_image"] = processed_image_base64
       
        return jsonify(response_data)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500



@app.route('/detect', methods=['POST'])
def detect():
    try:
        global consecutive_safety_violations

        # Get the uploaded image from the request
        image_data = request.files['image']
        location = request.form['location']
        date = request.form['date']
        time = request.form['time']
        
        print(location, date, time)
        
        image_bytes = image_data.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection on the uploaded image
        results = model(img)  # Perform object detection and assign results

        # Initialize violation details as a list
        violation_details = []

        # Define colors for different classes
        class_colors = {
            'NO-Hardhat': (255, 0, 0),  # Red for NO-Hardhat
            'NO-Safety Vest': (0, 0, 255),  # Blue for NO-Safety Vest
            'NO-Mask': (0, 255, 0),  # Green for NO-Mask
            'Hardhat': (0, 255, 255),  # Yellow for Hardhat
            'Safety Vest': (255, 0, 255),  # Purple for Safety Vest
            'Mask': (255, 255, 0)  # Cyan for Mask
        }

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
                        violation_details.append(currentClass)

                    myColor = class_colors.get(currentClass, (0, 0, 0))  # Default to black if class not found
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        original_image = img.copy()
        # Resize the image to (160, 160) for Face Recognition
        img_resized = cv2.resize(img, (160, 160))
        
        # Save the processed image to a file with a timestamp
        timestamp = generate_timestamp()
        processed_image_path = f'processed_image_{timestamp}.jpg'
        cv2.imwrite(processed_image_path, original_image)

        # Convert the processed image to base64
        with open(processed_image_path, "rb") as image_file:
            processed_image_base64 = base64.b64encode(image_file.read()).decode()

        # Determine the violation type based on violation details
        violation_type = "Safety Violation" if violation_details else "No Violation"

        # Check if the violation type is Safety Violation and increment the counter
        if violation_type == "Safety Violation":
            consecutive_safety_violations += 1
        else:
            consecutive_safety_violations = 0  # Reset the counter if no safety violation

        # Generate a unique ID based on the current time
        unique_id = generate_unique_id()

        # Create the response JSON
        response_data = {
            "violation_type": violation_type,
            "violation_details": violation_details,
            "processed_image": processed_image_path,
            "id": unique_id,
            "alarm": "off",
            "location": location,
            "date": date,
            "time": time,
        }

        # Check if there have been 10 consecutive safety violations
        if consecutive_safety_violations >= 10:
            response_data["alarm"] = "on"
            consecutive_safety_violations = 0  # Reset the counter
            push_to_DB(response_data)

        print(response_data)

        # Remove the temporary processed image file
        os.remove(processed_image_path)
        
        response_data["processed_image"] = processed_image_base64
       
        return jsonify(response_data)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


def upload_image_to_firebase(image_path):
    try:
        # Get the filename from the image_path
        filename = os.path.basename(image_path)

        # Get a reference to the Firebase Storage bucket
        bucket = storage.bucket()

        # Upload the image file to Firebase Storage with the same filename
        blob = bucket.blob(filename)
        blob.upload_from_filename(image_path)

        # Make the uploaded image public (optional)
        blob.make_public()

        # Get the public URL of the uploaded image
        image_url = blob.public_url

        return image_url
    except Exception as e:
        print(f"Error uploading image: {str(e)}")
        return None

def push_to_DB(data):
    try:
        print("pushing to db")
        # Replace these values with your Firestore project details
        project_id = "safetyscan-8a191"
        collection_id = "violations"
        api_key = "YOUR_API_KEY"  # Replace with your Firestore API key

        # URL for the Firestore REST API
        url_base = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/{collection_id}"

        # Prepare the data to be posted to Firestore
        firestore_data = {
            "fields": {
                "violation_type": {"stringValue": data["violation_type"]},
                "violation_details": {"arrayValue": {"values": [{"stringValue": detail} for detail in data["violation_details"]]}},
                "name_prediction": {"stringValue": data["name_prediction"]},
                "processed_image": {"stringValue": upload_image_to_firebase(data["processed_image"])},  # Upload and get the URL here
                "id": {"stringValue": data["id"]},
                "location": {"stringValue": data["location"]},
                "date": {"stringValue": data["date"]},
                "time": {"stringValue": data["time"]},
                "organizationId": {"stringValue": data["organizationId"]}
                # Add more fields as needed
            }
        }

        # Convert the data to JSON
        firestore_data_json = json.dumps(firestore_data)

        # Make a PATCH request to Firestore to post the data
        document_url = f"{url_base}/{data['id']}?key={api_key}"
        response = requests.patch(document_url, data=firestore_data_json, headers={"Content-Type": "application/json"})

        # Check the response
        if response.status_code == 200:
            print(f"Data posted to Firestore successfully.")
        else:
            print(f"Failed to post data to Firestore. Status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
