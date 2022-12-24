import matplotlib.pyplot as plt
from keras.models import load_model
from images_prepare import *
from CCA_Analysis import *

# X, X_sizes = pre_images((512, 512), 'C:/Users/jordon.tijerina.VYNEDENTAL/Pictures/online_panoramic_xrays/')
X, X_sizes = pre_images((512, 512), 'DentalPanoramicXrays/Images/')
X = np.float32(X/255)
x_test = X[:, :, :, :]

loadedModel = load_model("model.h5")
loadedModel.summary()

predict_img = loadedModel.predict(x_test)

predict = predict_img[0, :, :, 0]
plt.imsave("predicted.png", predict)

# Plotting - RESULT Example with CCA_Analysis
img = X[0, :, :, 0]
# load image (mask was saved by matplotlib.pyplot)
predicted = cv2.imread("predicted.png")
predicted = cv2.resize(predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
cca_result, teeth_count = cca_analysis(img, predicted, 3, 2)
plt.imshow(cca_result)
plt.show()
print("Teeth found: " + str(teeth_count))
