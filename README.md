# Human-Action-Recognition
1. Dataset Description and Preprocessing Steps:
Dataset Used:
The project utilized the Berkeley Multimodal Human Action Database (MHAD) available on Kaggle. The dataset features 11 actions recorded with diverse demographics, including 7 males, 5 females, and one elderly participant. Each action is recorded with 5 repetitions, totaling approximately 82 minutes of recording time.

Preprocessing Steps:
Data Format: Joint coordinates of 43 markers on human body parts were recorded in "txt" files, organized into a dataframe.
Dataframe Structure: A dataframe was created with 129 XYZ joint coordinates for each action, and a 'class' column for the action label.
Label Conversion: Initial string-formatted labels, with the 12th class denoted as "t-", were converted into integer format, replacing "t-" with "12."

Project Description:
What the Application Does:
The project focuses on Human Action Recognition (HAR) using deep learning models, specifically CNN-1D, LSTM, and Sequential Functional API models. 

Technologies Used:
Python, Keras, PyTorch for model implementation.
Kaggle for dataset access.
Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) for feature extraction.

Models Implemented:
CNN-1D:

6-layer architecture with Conv1D layers for feature extraction.
Achieved 93.80% accuracy on unseen test samples.
10 epochs, 0.001 learning rate, Adam Optimization.
LSTM:

PyTorch-based model capturing sequence patterns.
Trained for 5 epochs, batch size 64, sequence length 5.
CrossEntropyLoss and Adam (0.001) for optimization.
Sequential API:

Four dense layers with dropout for overfitting.
Achieved accurate pattern capturing.
10 epochs, 0.0001 learning rate, Adam Optimization.

Challenges Faced:
Handling diverse demographic data in the MHAD dataset.
Optimizing model architectures for effective human action recognition.
Managing the trade-off between model complexity and overfitting.

Features for Future Implementation:
Real-time human action recognition.
Integration with video streams for dynamic action recognition.

How to Use:
Clone the repository.
Install dependencies.
Access MHAD dataset.
Run preprocessing script.
Choose and run the training script for the desired model.

Experimental Results:
The study utilized Python, Kaggle, and Google Colab with GPU-T4 support to evaluate a Human Action Recognition framework. TensorFlow and PyTorch were key libraries, and the dataset was sourced from Kaggle. The Convolutional 1D model achieved 93.80% test accuracy on 1,441,152 samples, outperforming LSTM (51.17%) and Sequential Functional API (38.50%). The learning curve showed CNN-1D's superior performance, and detailed metrics, classification reports, and a confusion matrix provided insights. The findings emphasize the importance of selecting a customized architecture for Human Action Recognition tasks. The accuracy, specificity, sensitivity, and F1 score formulas were applied, and a confusion matrix for CNN-1D was presented.
