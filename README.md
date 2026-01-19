# Computer Vision and Deep Learning Portfolio

This repository contains my implementations for the Spring 2025 Computer Vision course. The projects range from fundamental image processing to state-of-the-art transformer architectures.

---

### Folder 1: Image Processing and MLP Foundations
This folder focuses on digital image processing (DIP) and basic neural network feature engineering.
* **Cloverleaf Analysis:** Detecting circular structures in aerial imagery using custom preprocessing and radii measurement.
* **Document Line Segmentation:** Implementing rectangular and polygonal boundary detection for OCR in historical documents.
* **Feature-based MLP:** Comparing classification performance using raw pixels, Canny edges, and HOG features.

### Folder 2: ResNet and Network Visualization
Deepening understanding of CNN architectures and model interpretability.
* **ResNet18 Optimization:** Adapting ResNet18 for low-resolution (36x36) custom datasets by modifying initial convolutional layers.
* **Explainable AI:** Generating Saliency Maps to visualize image regions most influential for classification decisions.
* **Adversarial Attacks:** Implementing Gaussian noise and optimization-based methods to induce model misclassification.
* **Neural Style Transfer:** Using VGG19 and L-BFGS optimization to merge content and style representations.

### Folder 3: Object Detection (Faster R-CNN)
Advanced detection pipelines for standard and specialized use cases.
* **Oriented Bounding Boxes:** Extending Faster R-CNN to predict rotated boxes via direct angle regression and multi-bin classification.
* **Fruit Detection:** Converting segmentation masks to bounding boxes and training a ResNet-34 backed Faster R-CNN.
* **Human Part Detection:** Fine-tuning architectures such as Faster R-CNN for hierarchical body part recognition.

### Folder 4: Semantic Segmentation (FCN and U-Net)
Pixel-level classification using encoder-decoder architectures.
* **FCN Variants:** Comparative study of FCN-32s, FCN-16s, and FCN-8s with frozen versus fine-tuned VGG backbones.
* **U-Net Family:** Implementing Vanilla U-Net, Residual U-Net, and Gated Attention U-Net to analyze the impact of skip connections and attention gates.



### Folder 5: Transformers and Multi-Modal Learning
Exploring modern shifts in vision research.
* **Vision Transformer (ViT):** Training ViT from scratch on CIFAR-10, exploring patch sizes and various positional embedding types.
* **Differential ViT:** Implementing a Multi-Head Differential Attention mechanism to amplify context and cancel out noise.
* **CLIP and Multi-Modality:** Performing zero-shot inference with OpenAI’s CLIP and analyzing memory efficiencies using FP16 precision.

### Folder NeuralTraining: Biologically-Inspired Neural Network Model
* Developed and simulated biologically-plausible neuron models from scratch using Python and NumPy.
* Engineered a custom neural network architecture that replaced standard neurons with these complex biological
models. and evaluated its performance in Reinforcement Learning (RL) tasks



