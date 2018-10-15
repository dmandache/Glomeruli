import json

from keras.models import load_model, model_from_json
from Models.Metrics import precision, recall, sensitivity, specificity, f1_score

MODEL_PATH = '/Users/diana/Desktop/Glomeruli InceptionV3/model.hdf5'

# load model
model = load_model(MODEL_PATH, custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                               'specificity': specificity, 'f1_score': f1_score})

# try to remove custom objects
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# serialize model to JSON
model_json = model.to_json()

print(model_json)

with open("model.json", "w") as json_file:
    json.dump(model_json, json_file)

# serialize weights to HDF5
model.save_weights("model-weights.h5")

print("Saved model to disk")


with open("model.json", "r") as json_file:
    json_string = json.load(json_file.read())
    model = model_from_json(json_string)
