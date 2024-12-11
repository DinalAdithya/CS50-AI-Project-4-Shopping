# Online Shopping Predictive Model

This project analyzes customer data from an online shopping platform to predict whether a customer will make a purchase. Using supervised learning, the model evaluates features such as page visits, visit durations, and demographics to classify user intent.

## Features

- **Supervised Learning**: Implements machine learning techniques to classify customer behavior.
- **Data Analysis**: Processes and cleans customer data for training and testing.
- **Evaluation Metrics**: Uses metrics like sensitivity and specificity to assess model performance.

## Project Files

- `shopping.py`: The main script for loading data, training the model, and evaluating its performance.
- `shopping.csv`: Example dataset used for training and testing the model.
- `README.md`: Documentation for the project.

## How It Works

1. **Input Data**:
   - Data includes features like page visits, durations, month, region, and weekend activity.
   - The target label indicates whether a purchase was completed.

2. **Model Training**:
   - The data is split into training and testing sets.
   - A k-Nearest Neighbors (k-NN) classifier is trained on the data.

3. **Evaluation**:
   - The model's predictions are compared against actual outcomes.
   - Sensitivity (true positive rate) and specificity (true negative rate) are calculated to measure performance.
