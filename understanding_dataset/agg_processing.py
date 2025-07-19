import json
import pandas as pd
from collections import defaultdict

with open('user-wallet-transactions.json') as f:
    data = json.load(f)

wallets = defaultdict(lambda: defaultdict(float))
counts = defaultdict(lambda: defaultdict(int))

def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

for i in data:
    wallet = i.get("userWallet")
    action = i.get("action")
    d = i.get("actionData", {})

    if not wallet or not action or not d:
        continue

    amount = safe_float(d.get("amount", 0))
    price = safe_float(d.get("assetPriceUSD", 1))
    usd_value = amount * price / 1e6

    wallets[wallet][f"total_{action}_usd"] += usd_value
    counts[wallet][f"{action}_count"] += 1

    asset_symbol = d.get("assetSymbol")
    if asset_symbol:
        counts[wallet][f"{action}_{asset_symbol}_count"] += 1
        wallets[wallet][f"{action}_{asset_symbol}_usd"] += usd_value

    if action == "deposit":
        pool_id = d.get("poolId")
        if pool_id:
            counts[wallet][f"deposit_pool_{pool_id}_count"] += 1

    elif action == "borrow":
        rate = safe_float(d.get("borrowRate", 0))
        wallets[wallet]["borrow_rate_sum"] += rate
        counts[wallet]["borrow_rate_count"] += 1
        
        stable_debt = safe_float(d.get("stableTokenDebt", 0))
        variable_debt = safe_float(d.get("variableTokenDebt", 0))
        wallets[wallet]["total_stable_debt"] += stable_debt
        wallets[wallet]["total_variable_debt"] += variable_debt

    elif action == "repay":
        use_atokens = d.get("useATokens", False)
        if use_atokens:
            counts[wallet]["repay_with_atokens_count"] += 1

    elif action == "liquidationcall":
        collateral_amount = safe_float(d.get("collateralAmount", 0))
        collateral_price = safe_float(d.get("collateralAssetPriceUSD", 1))
        principal_amount = safe_float(d.get("principalAmount", 0))
        borrow_price = safe_float(d.get("borrowAssetPriceUSD", 1))
        
        collateral_usd = collateral_amount * collateral_price / 1e6
        principal_usd = principal_amount * borrow_price / 1e6
        
        wallets[wallet]["total_liquidation_collateral_usd"] += collateral_usd
        wallets[wallet]["total_liquidation_principal_usd"] += principal_usd
        
        collateral_symbol = d.get("collateralReserveSymbol")
        principal_symbol = d.get("principalReserveSymbol")
        if collateral_symbol:
            counts[wallet][f"liquidation_collateral_{collateral_symbol}_count"] += 1
        if principal_symbol:
            counts[wallet][f"liquidation_principal_{principal_symbol}_count"] += 1

rows = []
for wallet in wallets:
    row = {"userWallet": wallet}
    row.update(wallets[wallet])
    row.update(counts[wallet])

    if row.get("borrow_rate_count", 0) > 0:
        row["avg_borrow_rate"] = row["borrow_rate_sum"] / row["borrow_rate_count"]
    else:
        row["avg_borrow_rate"] = 0.0
    row.pop("borrow_rate_sum", None)
    row.pop("borrow_rate_count", None)

    row["total_transactions"] = (row.get("deposit_count", 0) + 
        row.get("redeemunderlying_count", 0) + 
        row.get("borrow_count", 0) + 
        row.get("repay_count", 0) + 
        row.get("liquidationcall_count", 0))

    row["total_volume_usd"] = (row.get("total_deposit_usd", 0) + 
        row.get("total_redeemunderlying_usd", 0) + 
        row.get("total_borrow_usd", 0) + 
        row.get("total_repay_usd", 0) + 
        row.get("total_liquidationcall_usd", 0))

    row["net_position_usd"] = (row.get("total_deposit_usd", 0) + 
        row.get("total_repay_usd", 0) - 
        row.get("total_borrow_usd", 0) - 
        row.get("total_redeemunderlying_usd", 0))

    rows.append(row)

df = pd.DataFrame(rows).fillna(0)
df.to_csv("wallet_aggregates_all.csv", index=False)
print("aggregated wallet data saved to wallet_aggregates_all.csv")
print(f"processed {len(df)} unique wallets")
print(f"generated {len(df.columns)} features per wallet")
