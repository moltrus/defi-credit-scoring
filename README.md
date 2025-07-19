# DeFi Credit Scoring

Credit scoring for DeFi using **supervised learning with retrospective labeling** to predict wallet liquidation risk based on transaction history.

## Model Architecture

The credit scoring system employs a **Random Forest classifier** with the following architecture:

### Data Pipeline
```
JSON Transaction Data -> Feature Engineering -> SMOTE Balancing -> Random Forest -> Credit Scores
```

### Core Components

#### 1. **Data Preprocessing**
- **Asset Normalization**: Converts the transaction amounts from smallest units to native token amounts using asset-specific decimal mappings
- **USD Conversion**: Calculates USD values using historical asset prices from the transaction data
- **Temporal Processing**: Converts Unix timestamps to datetime objects for time-based feature extraction

#### 2. **Feature Engineering**
The model extracts wallet-level features by aggregating individual transactions

**Transaction Volume Features:**
- `num_transactions`: Total number of transactions per wallet
- `num_deposits`, `num_borrows`, `num_repays`, `num_liquidations`: Count of each transaction type
- `total_deposit_usd`, `total_borrow_usd`, `total_repay_usd`: Log-transformed cumulative USD amounts

**Behavioral Features:**
- `borrow_proportion`: Ratio of borrow transactions to total transactions
- `avg_tx_size_usd`: Log-transformed average transaction size (excluding zero amounts)

**Temporal Features:**
- `time_span_days`: Number of days between the first and the last transaction
- `avg_time_between_tx_days`: Average time interval between consecutive transactions

#### 3. **Target Variable Creation**
- **Binary Classification**: Wallets are labeled as high-risk (1), if they experienced any liquidation events, low-risk (0) otherwise
- This creates a supervised learning problem where historical liquidations are used to predict the future liquidation risks

#### 4. **Class Imbalance Handling**
- **SMOTE (Synthetic Minority Oversampling Technique)**: Generates synthetic samples of the minority class (liquidated wallets)
- Uses k=5 nearest neighbors for creating realistic synthetic examples
- This balances the class imbalance in the dataset, to improve model performance

#### 5. **Model Training**
- **Algorithm**: Random Forest Classifier with balanced class weights
- **Hyperparameter Optimization**: RandomizedSearchCV with 30 iterations
  - `n_estimators`: 50-150 trees
  - `max_depth`: 2-4 levels (to prevent overfitting)
  - `min_samples_split`: 30-60 (for statistical significance)
  - `min_samples_leaf`: 30-60 (has smooth decision boundaries)
- **Validation**: 8-fold cross-validation using F1-weighted scoring
- **Feature Scaling**: StandardScaler normalization before training

#### 6. **Score Generation**
- **Risk Probability**: Direct output from Random Forest probability prediction
- **Smoothing Function**: Sigmoid transformation `1 / (1 + exp(-2.5 * (proba - 0.5)))` to improve score separation
- **Credit Score**: Inverted and scaled to 0-1000 range: `(1 - smoothed_score) * 1000`

## Features
- Loads and processes wallet transaction data from JSON dataset
- Feature engineering based on transaction types, amounts, and timing
- Labels wallets based on liquidation events
- Handles class imbalance using SMOTE oversampling
- Trains a Random Forest classifier with hyperparameter optimization
- Outputs credit scores and risk probabilities for each wallet address

## Requirements
Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning algorithms
- imbalanced-learn: SMOTE implementation
- scipy: Statistical functions

## Usage
The script is CLI-based to allow one-step training or prediction of wallet scores from a JSON file containing user transactions.

### Training the Model
Trains a new model using the given file:

```bash
python main.py train --input user-wallet-transactions.json
```

This will:
- Process transaction data and engineer features
- Display label distribution and model performance metrics
- Save the trained model, scaler, and feature names to `model.pkl`

### Predicting Credit Scores
Predicts credit scores for wallets using a trained model:

```bash
python main.py predict --input user-wallet-transactions.json --output wallet_scoring.csv --model model.pkl
```

The results will be saved to `wallet_scoring.csv` with columns:
- `userWallet`: Wallet address
- `credit_score`: Credit score (0-1000, higher = lower risk)
- `risk_probability`: Raw liquidation probability (0-1)

## File Overview
- `main.py`: Main script for training and prediction
- `model.pkl`: Serialized Random Forest model, scaler, and metadata
- `requirements.txt`: Python dependencies
- `user-wallet-transactions.json`: Input dataset containing wallet transactions
- `wallet_scoring.csv`: Output file with credit scores
- `understanding_dataset/`: Scripts for dataset exploration and analysis

## Model Performance
The model uses F1-weighted scoring for evaluation, which balances precision and recall while accounting for class imbalance. Cross-validation provides robust performance estimates across different data splits.
