import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class KNNClassifier:
    def __init__(self, k=3, image_size=(32,32)):
        self.k = k
        self.img_size = image_size
        self.X_train = []
        self.Y_train = []

    def extract_class(self, file_name):
      file_name = file_name.split('_')
      return file_name[0].lower()

    def load_img(self, img_path):
      img = Image.open(img_path).convert('L')
      img = img.resize(self.img_size)
      img_array = np.array(img)/255.0
      return img_array.flatten()
      
    def fit(self, trainFolder):
      for img in os.listdir(trainFolder):
        img_path = os.path.join(trainFolder, img)
        img_data = self.load_img(img_path)

        self.X_train.append(img_data)
        self.Y_train.append(self.extract_class(img))

      self.X_train = np.array(self.X_train)
      self.Y_train = np.array(self.Y_train)
      print(f"Loaded {len(self.X_train)} training images")
      print("Detected classes:", np.unique(self.Y_train))
      
    def majorityVote(self, labels):
      votes = {}
      for label in labels:
        votes[label] = votes.get(label, 0) + 1
      return max(votes, key=votes.get)

    def predict(self, test_img):
      test_img = self.load_img(test_img)

      # Display test image using matplotlib
      plt.imshow(test_img.reshape(self.img_size), cmap='gray')
      plt.axis('off')
      plt.show()

      distances = np.sqrt(np.sum((test_img - self.X_train)**2, axis=1))

      # Get k nearest neighbors
      k_indices = np.argsort(distances)[:self.k]
      k_nearest_labels = self.Y_train[k_indices]


  # Show nearest neighbors
      print("\nTop matches from training set:")
      plt.figure(figsize=(15, 3))
      for i, idx in enumerate(k_indices):
            img = self.X_train[idx].reshape(self.img_size)
            plt.subplot(1, self.k, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Class: {self.Y_train[idx]}\nDist: {distances[idx]:.2f}")
            plt.axis('off')
      plt.show()
        
        # Get prediction
      prediction = self.majorityVote(k_nearest_labels)
      return prediction

    
