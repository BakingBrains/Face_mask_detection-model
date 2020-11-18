from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model('maskvsnomask.h5')

img_path = 'Masks Dataset/Validation/Non Mask/real_00001.jpg'
img = image.load_img(img_path, target_size=(150,150))

img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_batch)
print(prediction)
