# -Urban-Area-Detection-from-Satellite-Images
Here's a detailed breakdown of the entire project workflow,

---

### **1. Project Setup & Environment Preparation**
**Purpose**: Prepare Google Colab environment for deep learning tasks  
**Key Actions**:
- Installed necessary libraries: TensorFlow (deep learning), Matplotlib (visualization), scikit-learn (metrics)
- Verified GPU availability (Colab's free GPU acceleration)
- Set random seeds for reproducibility

---

### **2. Dataset Acquisition & Preparation**
**Purpose**: Obtain and organize satellite imagery data  
**Key Actions**:
- Downloaded **EuroSAT dataset** (contains 27,000 Sentinel-2 satellite images across 10 classes)
- Organized data into urban/non-urban categories:
  - **Urban classes**: Highway, Industrial, Residential
  - **Non-urban classes**: All others (e.g., Forest, SeaLake, Farmland)
- Created a Pandas DataFrame with filepaths and binary labels (0=Non-urban, 1=Urban)

---

### **3. Data Splitting & Stratification**
**Purpose**: Create balanced training/validation/test sets  
**Key Actions**:
- Split data into:
  - **Training set**: 80% of data
  - **Validation set**: 10% (used for hyperparameter tuning)
  - **Test set**: 10% (final evaluation)
- Used stratified splitting to maintain class balance across all sets

---

### **4. Data Preprocessing Pipeline**
**Purpose**: Prepare images for model consumption  
**Key Actions**:
- **Image loading**: Read images from disk using TensorFlow's `tf.data` API
- **Resizing**: Scaled images to 64x64 pixels (balance between speed and detail)
- **Normalization**: Applied EfficientNet-specific preprocessing (zero-centering)
- **Data Augmentation** (training only):
  - Random horizontal/vertical flips
  - Random rotations (±20 degrees)
  - Random zoom (20%)
- Created efficient data pipelines with batching (32 images/batch) and prefetching

---

### **5. Model Architecture Design**
**Purpose**: Build a deep learning model for binary classification  
**Key Actions**:
- Used **EfficientNetB0** as a frozen feature extractor (transfer learning)
- Added custom classification head:
  - Global Average Pooling → 128-unit dense layer → Dropout (20%) → Sigmoid output
- Total parameters: ~4.2M (only 164k trainable)
- Compiled with:
  - Loss: Binary crossentropy
  - Optimizer: Adam
  - Metrics: Accuracy, AUC

---

### **6. Model Training**
**Purpose**: Teach the model to distinguish urban/non-urban areas  
**Key Actions**:
- Trained for 20 epochs with early stopping (patience=5)
- Tracked training/validation accuracy and loss
- Used GPU acceleration for faster training (~10-15 minutes on Colab)

---

### **7. Model Evaluation**
**Purpose**: Assess performance on unseen data  
**Key Actions**:
- **Test set evaluation**:
  - Accuracy: ~0.96%
  - AUC: ~0.99%
- **Confusion Matrix**:
  - True Positive/Negative rates
  - Visualized with Seaborn
- **Classification Report**:
  - Precision, Recall, F1-score for both classes
- **ROC Curve**:
  - Demonstrated excellent discrimination (AUC ~0.96)

---

### **8. Prediction Visualization**
**Purpose**: Understand model behavior qualitatively  
**Key Actions**:
- Sampled 25 random test images
- Displayed:
  - Original satellite imagery
  - True labels vs predictions
  - Prediction confidence scores
  - Color-coded correctness (green=correct, red=error)
- Revealed model's strengths/weaknesses in real-world scenarios

---

### **9. Key Insights & Takeaways**
1. **Transfer Learning Power**: EfficientNetB0's pre-trained features worked well despite domain shift (natural images → satellite)
2. **Data Augmentation**: Helped model generalize better to rotations/flips common in satellite imagery
3. **Urban Class Challenges**: Some confusion between Residential areas and similar-looking non-urban classes
4. **Scalability**: The 64x64 input size allows fast inference while maintaining performance

---
