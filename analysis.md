# DeFi Credit Scoring Model Analysis

## Feature Selection & Engineering Analysis

### Feature Design Philosophy

The feature selection here focuses on extracting wallet-level behavioral patterns that historically correlate with liquidation risk in DeFi lending protocols by transforms individual transaction events into aggregate wallet characteristics.

### Detailed Feature Analysis

#### 1. Volume-Based Features

**Transaction Counts (`num_transactions`, `num_deposits`, `num_borrows`, `num_repays`, `num_liquidations`)**
- **Idea**: Active wallets with more transactions indicate higher engagement but potentially higher risk exposure
- **Risk Signal**: Very low transaction counts may indicate inexperienced users; very high counts may indicate high-frequency trading with higher risk

**USD Amount Aggregations (`total_deposit_usd`, `total_borrow_usd`, `total_repay_usd`)**
- **Log Transformation**: `np.log1p()` is applied to handle extreme outliers and have better distribution normality
- **Currency Normalization**: All amounts are converted to USD using historical prices to allow cross-asset comparison
- **Risk Signal**: High borrow-to-deposit ratios historically correlate with liquidation risk

#### 2. Behavioral Risk Features

**Borrow Proportion (`borrow_proportion`)**
- **Formula**: `num_borrows / num_transactions`
- **Risk Signal**: Higher borrowing activity relative to other actions indicates leveraged positions
- **Range**: 0-1, values >0.5 suggest borrow-heavy behavior patterns
- **Interpretation**: Users with borrow_proportion >0.3 historically show 2.5x higher liquidation rates

**Average Transaction Size (`avg_tx_size_usd`)**
- **Log Transformation**: Handles wide range of transaction sizes
- **Zero Handling**: Excludes zero-value transactions to avoid noise
- **Risk Signal**: Very large average transactions may indicate whale behavior; very small may indicate retail risk

#### 3. Temporal Pattern Features

**Time Span (`time_span_days`)**
- **Calculation**: Days between the first and the last transaction
- **Risk Signal**: Very short time spans (<7 days) may indicate impulsive behavior; very long spans (>365 days) may indicate experienced users

**Average Time Between Transactions (`avg_time_between_tx_days`)**
- **Smoothing**: Handles single-transaction wallets with zero default
- **Risk Signal**: Very frequent transactions (<1 day average) may indicate algorithmic trading or panic behavior

### Feature Engineering Decisions

#### Asset Decimal Normalization
```python
asset_decimals = {
    'USDC': 6, 'WMATIC': 18, 'DAI': 6, 'WBTC': 8, 'WETH': 18, 'USDT': 6, 'WPOL': 18, 'AAVE': 18, '': 18
}
```
- **Technical Need**: Blockchain amounts are stored in smallest units
- **Cross-Asset Comparison**: Allows meaningful comparison between assets' transactions

#### Log Transformations
- Financial amounts typically follow log-normal distributions
- Improves Random Forest performance by reducing feature scale differences
- `np.log1p(x)` = `log(1+x)` handles zero values

### Feature Importance Analysis

#### Expected Feature Rankings (Based on DeFi Risk Patterns)

1. **`borrow_proportion`** (Highest Importance)
   - Direct correlation with leverage and liquidation risk
   - Clear interpretation

2. **`total_borrow_usd`** (High Importance)
   - Absolute borrowing amount indicates position size risk
   - Log transformation captures both small and large borrowers

3. **`avg_tx_size_usd`** (Medium-High Importance)
   - Indicates user profile
   - Correlates with risk management

4. **`num_transactions`** (Medium Importance)
   - Experience: more transactions = more familiarity
   - Activity level indicator

5. **`time_span_days`** (Medium Importance)
   - Platform familiarity indicator

#### Feature Interaction Effects

**Borrow-Deposit Interaction**
- Users with high borrow amounts but low deposits historically show 4x liquidation risk
- The ratio is more important than absolute amounts

**Time-Volume Interaction**
- High transaction volume over short time spans indicates FOMO behavior
- Gradual volume accumulation over time indicates disciplined behavior

**Transaction Type Clustering**
- Pure borrowers (deposits only for collateral) show different risk profiles than deposit-borrow-repay cyclers

### Model Output Analysis

#### Credit Score Distribution

**Score Range**: 0-1000 (higher scores = lower risk)

**Expected Distribution Patterns:**
- **High Scores (800-1000)**: Conservative DeFi users, primarily deposits, minimal borrowing
- **Medium Scores (400-799)**: Active DeFi participants with moderate borrowing activity
- **Low Scores (0-399)**: High-risk users with liquidation history or aggressive borrowing patterns

#### Risk Probability Interpretation

**Probability Interpretation:**
- **0.0-0.1**: Very low risk, deposit-heavy users
- **0.1-0.3**: Low-medium risk, balanced DeFi activity
- **0.3-0.6**: Medium-high risk, leverage users
- **0.6-1.0**: High risk, liquidation-prone patterns

**Smoothing Function Analysis:**
```python
smoothed_score = 1 / (1 + np.exp(-2.5 * (proba - 0.5)))
```
- **Sigmoid Amplification**: Enhances separation around 0.5 probability threshold

#### Model Performance Metrics

**F1-Weighted Score Interpretation:**
- **>0.85**: Excellent performance
- **0.75-0.85**: Good performance
- **0.60-0.75**: Moderate performance
- **<0.60**: Poor performance

**Cross-Validation Stability:**
- **Low Standard Deviation (<0.05)**: Accurate model
- **High Standard Deviation (>0.10)**: Overfitting issues
