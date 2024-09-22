

# Enhancing Text Classification in Imbalanced Domains: A Multimodal Approach

This project tackles the challenge of identifying whether a given text is human-generated or machine-generated across two distinct domains. The project was conducted as part of the **COMP90051 Statistical Machine Learning** course at the University of Melbourne.  The model outperformed over 80 groups(more than 200 people), securing a first-place in both the kaggle competetion and final report result. 

![image](https://github.com/user-attachments/assets/28a1ab6a-91bd-4744-a149-e9d72f3b91e2
## Project Overview

The dataset consists of two domains with both human- and machine-generated text, with **Domain 2** having a significant imbalance between the two classes. The task is to predict the origin of the text (human or machine), requiring effective handling of this imbalance while maintaining accuracy across domains.

We developed a multimodal approach that includes separate models for domain classification, balanced domain text prediction, and imbalanced domain text prediction.

### Key Metrics:
- **F1 score**
- **Validation accuracy**
- **Test accuracy**

## Data Inspection

- **Text Length Differences:** We observed differences in the length of text between human- and machine-generated samples, which informed our feature engineering. 
- **Imbalance in Domain 2:** Domain 2 contains significantly more machine-generated data than human-generated data, and the distribution across the machine models is uneven.

## Hypothesis and Approach

Initially, we hypothesized that human- and machine-generated sentences would have domain-independent characteristics. We attempted to augment Domain 2's human-generated data with Domain 1 data but found that this integration led to poor test performance.

### Project Aims:
1. **Domain classification**: Classify data as originating from Domain 1 or Domain 2.
2. **Domain 1 classification**: Use a complex model for the balanced data in Domain 1.
3. **Domain 2 classification**: Handle imbalanced data in Domain 2 with robust models.

## Methodology

### 1. Domain Classification (Aim 1)
- **Model:** CNN-BiLSTM  
- **Results:** Achieved 98.9% accuracy.  
We employed a BiLSTM model to leverage contextual information and CNN layers for dimensionality reduction, significantly improving classification accuracy.

### 2. Domain 1 Classification (Aim 2)
- **Baseline Model:** Logistic Regression (92.36% accuracy)
- **Optimized Model:** CNN-BiLSTM (95.36% accuracy)  
The CNN-BiLSTM model improved performance for Domain 1, demonstrating the benefits of a more complex model on balanced data.

### 3. Domain 2 Classification (Aim 3)
- **Initial Model:** CNN-BiLSTM  
  - This model struggled with the imbalanced data, prompting us to try several techniques:
    - **Loss Function Adjustment**: Using weighted cross-entropy, but with limited success.
    - **Up-sampling & Down-sampling**: Both approaches failed due to overfitting and underfitting.
    - **Multitask Learning**: This also did not improve results.
  
- **Final Model:** LightGBM  
  - **F1-score:** 0.42  
  - **Accuracy:** 81%  
  LightGBM, a tree-based learning algorithm, outperformed neural networks on imbalanced data. Using Bayesian optimization to tune hyperparameters, we achieved significant improvements.

## Summary of Prediction Program

We implemented a three-part model:
1. **Domain Classification**: CNN-BiLSTM (98.9% accuracy)
2. **Domain 1 Text Classification**: CNN-BiLSTM (95.36% accuracy)
3. **Domain 2 Text Classification**: LightGBM (F1: 0.42, Accuracy: 81%)

The final composite prediction, combining all models, achieved an overall accuracy of **82.6%** on test data.

## How to Run the Project

1. **Dependencies**:
   - Python 3.x
   - Libraries: `scikit-learn`, `numpy`, `pandas`, `tensorflow`, `lightgbm`
   - See `requirements.txt` for the full list.

2. **Data**:
   - Training and test data are available via Kaggle. (Data access is restricted to the competition page.)

3. **Steps**:
   - Preprocess the datasets.
   - Train models using the respective scripts for each aim.
   - Generate predictions on the test set and submit to Kaggle for evaluation.
