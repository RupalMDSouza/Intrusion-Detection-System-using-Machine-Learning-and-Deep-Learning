# Intrusion-Detection-System-using-Machine-Learning-and-Deep-Learning

**Objective:**
The project aims to develop an AI-driven Intrusion Detection System (IDS) to detect Distributed Denial of Service (DDoS) attacks within a communication network. By leveraging machine learning and deep learning models, the system identifies abnormal activities and ensures network availability.

**Dataset:**
The project utilizes the CIC-DDoS 2019 dataset, developed by the Canadian Institute for Cybersecurity. This dataset, consisting of 2.8 million network packets, represents recent network traffic and contains seven attack types: brute force, Heartbleed, Botnet, DoS, DDoS, Web Attack, and Infiltration.

**Scope:**
**Implement machine learning models:** Logistic Regression (LR), Support Vector Machines (SVM), Random Forest (RF), and K-Nearest Neighbor (KNN).
**Implement deep learning models:** Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM).
Evaluate models with and without feature selection using metrics such as accuracy, precision, recall, and F1-score.
Compare performance across models to identify the best-suited approach for DDoS detection.

**Methodology:**
Data Preprocessing: Cleaning the dataset by handling missing and duplicate values, and encoding categorical data into numerical formats.

**Feature Selection:** Implementing methods like Principal Component Analysis (PCA) and SelectKBest to improve computational efficiency and reduce overfitting.

**Model Implementation:**
Machine Learning: LR, SVM, RF, and KNN are applied to classify attacks using binary and multiclass classification.
Deep Learning: CNN and LSTM models are implemented for comparative analysis.
Evaluation: The models are evaluated based on accuracy, precision, recall, F1-score, and confusion matrices.

**Results:**
Random Forest demonstrated the highest accuracy among machine learning models in both binary and multiclass classifications.
CNN achieved a perfect accuracy of 100%, while LSTM closely followed with 99% accuracy.
Feature selection methods improved model performance, with SelectKBest yielding efficient results, particularly for RF and KNN models.

**Key Findings:**
Random Forest and CNN are the most effective models for DDoS detection.
Deep learning models outperform traditional machine learning models in terms of accuracy and generalization.
Feature selection methods significantly enhance model efficiency and accuracy.

**Conclusion:**
The project successfully develops an effective IDS for detecting DDoS attacks. Random Forest and CNN emerge as the best-performing models, with CNN achieving the highest accuracy. This system contributes to robust network security by leveraging advanced machine learning and deep learning techniques.
