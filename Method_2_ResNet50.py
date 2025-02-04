import cv2
import numpy as np
from keras.models import load_model
from pyzbar.pyzbar import decode

def detect_barcode(image, model):
    if image is None:
        print("Error: Image not found or unable to load.")
        return None, None

    resized_image = cv2.resize(image, (224, 224))
    scaled_image = resized_image / 255

    pred = model.predict(np.array([scaled_image]))
    xmin, ymin, xmax, ymax = pred[0][0], pred[0][1], pred[0][2], pred[0][3]

    xmin = int(xmin * image.shape[1])
    ymin = int(ymin * image.shape[0])
    xmax = int(xmax * image.shape[1])
    ymax = int(ymax * image.shape[0])

    return (xmin, ymin, xmax, ymax), image[ymin:ymax, xmin:xmax]

def scanner(image_path, model_path):
    image = cv2.imread(image_path)
    model = load_model(model_path)

    bbox, roi = detect_barcode(image, model)

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        decoded_objects = decode(roi)
        cv2.imshow("Detected Barcode", image)
        cv2.imshow("ROI", roi)

        if decoded_objects:
            print("Barcode detected:")
            for obj in decoded_objects:
                print("Type:", obj.type)
                print("Data:", obj.data.decode("utf-8"))
        else:
            print("No barcode detected.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No barcode region detected.")

# Example usage
image_path = 'images/example_barcode5.png'
model_path = 'models/ResNet50/model_50_resnet.keras'
scanner(image_path, model_path)
