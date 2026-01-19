# performance differences between the three models

- Model 1 uses an 1d array as an input with no processing more information allows the odel to achieve higher accuracy. the model converges in 10 epochs
- Model 2 uses an input with edge detection processing done which allows for lesser information as the image is only in black and white and is bitwise.the model has a slightly lower accuracy than model 1
- Model 3 uses hog feature extraction and reduces the image to 9x9 with lesser data as an input the the model has the lowest accuracy. 

# Feature Dimensionality Reduction:
- The features progressively reduce from Model 1 → Model 2 → Model 3, as follows:

- Model 1: High-dimensional raw pixel values (e.g., a 28x28 image = 784 features).
- Model 2: Reduced-dimensional edge features, focusing only on structural outlines and is bitwise.
- Model 3: Further reduction with HOG (or similar), which encodes gradients and their spatial relationships into a compact representation (e.g., HOG on 28x28 could results in 81 features).
 
more information allows for better model training and as The features progressively reduce from Model 1 → Model 2 → Model 3
so the models perform progressively worse.