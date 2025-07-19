import json
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")


def load_data(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    if df['actionData'].dtype == 'object' and isinstance(df['actionData'].iloc[0], str):
        df['actionData'] = df['actionData'].apply(json.loads)


    asset_decimals = {
        'USDC': 6, 'WMATIC': 18, 'DAI': 6, 'WBTC': 8, 'WETH': 18, 'USDT': 6, 'WPOL': 18, 'AAVE': 18, '': 18
    }

    df['amount_native'] = df.apply(
        lambda x: float(x['actionData'].get('amount', 0)) / (10 ** asset_decimals.get(x['actionData'].get('assetSymbol', ''), 18)),
        axis=1
    )
    df['amount_usd'] = df.apply(
        lambda x: 0 if x['actionData'].get('assetSymbol', '') == '' or  x['actionData'].get('assetPriceUSD', '0') == '0' or float(x['actionData'].get('amount', 0)) <= 0
        else x['amount_native'] * float(x['actionData'].get('assetPriceUSD', 0)),
        axis=1
    )
    df['reserve'] = df['actionData'].apply(lambda x: x.get('assetSymbol', 'unknown'))

    return df

def preprocess_features(df):

    df['deposit_usd'] = df['amount_usd'].where(df['action'] == 'deposit', 0)
    df['borrow_usd'] = df['amount_usd'].where(df['action'] == 'borrow', 0)
    df['repay_usd'] = df['amount_usd'].where(df['action'] == 'repay', 0)

    wallet_features = df.groupby('userWallet').agg({
        'action': [
            ('num_transactions', 'count'),
            ('num_deposits', lambda x: (x == 'deposit').sum()),
            ('num_borrows', lambda x: (x == 'borrow').sum()),
            ('num_repays', lambda x: (x == 'repay').sum()),
            ('num_liquidations', lambda x: (x == 'liquidationcall').sum()),
            ('borrow_proportion', lambda x: (x == 'borrow').sum() / x.count() if x.count() > 0 else 0)
        ],
        'deposit_usd': [('total_deposit_usd', lambda x: np.log1p(x.abs().sum()))],
        'borrow_usd': [('total_borrow_usd', lambda x: np.log1p(x.abs().sum()))],
        'repay_usd': [('total_repay_usd', lambda x: np.log1p(x.abs().sum()))],
        'amount_usd': [('avg_tx_size_usd', lambda x: np.log1p(x[x > 0].mean()) if x[x > 0].size > 0 else 0)],
        'timestamp': [
            ('time_span_days', lambda x: (x.max() - x.min()).days),
            ('avg_time_between_tx_days', lambda x: x.diff().mean().total_seconds() / 86400 if len(x) > 1 else 0)
        ]
    }).fillna(0)

    wallet_features.columns = ['_'.join(col).strip() for col in wallet_features.columns.values]
    df.drop(['deposit_usd', 'borrow_usd', 'repay_usd'], axis=1, inplace=True)

    return wallet_features

def create_labels(df):
    df['label'] = (df['action_num_liquidations'] > 0).astype(int)
    return df

def smoothout_scores(proba):
    return 1 / (1 + np.exp(-2.5 * (proba - 0.5)))

def train_random_forest_model(X, y):

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_dist = {
        'n_estimators': randint(50, 150),
        'max_depth': [2, 3, 4],
        'min_samples_split': randint(30, 60),
        'min_samples_leaf': randint(30, 60)
    }

    search = RandomizedSearchCV(
        rf_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring='f1_weighted',
        cv=8,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    cv_scores = cross_val_score(best_model, X_balanced, y_balanced, cv=8, scoring='f1_weighted')
    mean_cv_score = cv_scores.mean()

    print(f"cross validation F1 score: {mean_cv_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"best parameters: {search.best_params_}")

    return best_model

def train_model(data_file):

    df = load_data(data_file)

    features_df = preprocess_features(df)
    features_df = create_labels(features_df)

    print(f"label distribution: {features_df['label'].value_counts().to_dict()}")

    features = [col for col in features_df.columns if col not in ['label', 'action_num_liquidations']]
    X = features_df[features]
    y = features_df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)

    print("training a random forest..")
    best_model = train_random_forest_model(X_scaled, y)

    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': features,
        'model_name': 'RandomForest'
    }

    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("model saved as model.pkl")
    return best_model, scaler, features

def predict_scores(input_file, output_file, model_file='model.pkl'):

    print("loading model..")
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    model_name = model_data['model_name']

    print("loading input data..")
    df = load_data(input_file)
    features_df = preprocess_features(df)

    X = features_df[feature_names]

    X_scaled = scaler.transform(X)

    print("finding predictions..")
    proba = model.predict_proba(X_scaled)[:, 1]

    smoothed_scores = smoothout_scores(proba)
    credit_scores = (1 - smoothed_scores) * 1000

    results_df = pd.DataFrame({
        'userWallet': features_df.index,
        'credit_score': credit_scores,
        'risk_probability': proba
    })

    results_df.to_csv(output_file, index=False)
    print(f"predictions saved to '{output_file}'")
    print(f"processed {len(results_df)} wallets")
    print(f"score range: {credit_scores.min():.1f} - {credit_scores.max():.1f}")

def main():
    parser = argparse.ArgumentParser(description='AAVE V2 wallet credit scoring')
    parser.add_argument('mode', choices=['train', 'predict'], help='mode: train model or predict scores')
    parser.add_argument('--input', required=True, help='input JSON file path')
    parser.add_argument('--output', help='output CSV file path (required for predict mode)')
    parser.add_argument('--model', default='model.pkl', help='model file path (default: model.pkl)')

    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args.input)
    elif args.mode == 'predict':
        if not args.output:
            print("error: --output is required for predict mode")
            return
        predict_scores(args.input, args.output, args.model)

if __name__ == "__main__":
    main()
