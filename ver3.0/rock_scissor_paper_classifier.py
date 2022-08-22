#import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import numpy as np
import argparse
import cv2

#parser = argparse.ArgumentParser(description='Rock, paper and scissors')
#parser.add_argument('--model_path', type=str, help='Specify the model path', required=True)

#args = parser.parse_args()
#model_path = args.model_path

labels = ['Rock', 'Paper', 'Scissors']

tflite_model_file = 'converted_model.tflite'
with open(tflite_model_file, 'rb') as fid:
    tflite_model = fid.read()

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# webcam
cam = cv2.VideoCapture(0)
while True:
    ret, img = cam.read()
    original_img = img

    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(150, 150), interpolation=cv2.INTER_AREA)
    img = np.array(img, dtype=np.float32)
    img = img / 255.

    # Add a batch dimension
    input_data = np.expand_dims(img, axis=0)

    # Point the data to be used for testing and run the interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Obtain results and print the predicted category
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(predictions)
    
    classify_result = labels[predicted_label]
    original_img = cv2.putText(original_img, classify_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    
    cv2.imshow('Webcam View', original_img)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
