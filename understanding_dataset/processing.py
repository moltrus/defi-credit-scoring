import json
import csv
from datetime import datetime

input_file = 'user-wallet-transactions.json'
output_file = 'processed_transactions.csv'

def extract_value(obj):
    if isinstance(obj, dict):
        if '$oid' in obj:
            return obj['$oid']
        elif '$date' in obj:
            return obj['$date']
    return obj

with open(input_file, 'r') as f:
    data = json.load(f)

level_0_keys = ['_id', 'userWallet', 'network', 'protocol', 'txHash', 'logId', 'timestamp', 
                'blockNumber', 'action', '__v', 'createdAt', 'updatedAt']

action_keys = {
    'deposit': ['amount', 'assetPriceUSD', 'assetSymbol', 'poolId', 'type', 'userId'],
    'redeemunderlying': ['amount', 'assetPriceUSD', 'assetSymbol', 'poolId', 'toId', 'type', 'userId'],
    'borrow': ['amount', 'assetPriceUSD', 'assetSymbol', 'borrowRate', 'borrowRateMode', 'callerId', 
               'poolId', 'stableTokenDebt', 'type', 'userId', 'variableTokenDebt'],
    'repay': ['amount', 'assetPriceUSD', 'assetSymbol', 'poolId', 'repayerId', 'type', 'useATokens', 'userId'],
    'liquidationcall': ['amount', 'assetPriceUSD', 'assetSymbol', 'borrowAssetPriceUSD', 'collateralAmount', 
                        'collateralAssetPriceUSD', 'collateralReserveId', 'collateralReserveSymbol', 
                        'liquidatorId', 'poolId', 'principalAmount', 'principalReserveId', 
                        'principalReserveSymbol', 'type', 'userId']
}

csv_headers = level_0_keys[:]
for action, keys in action_keys.items():
    csv_headers += [f"{action}_{key}" for key in keys]

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()

    for entry in data:
        row = {}
        for key in level_0_keys:
            value = entry.get(key, '')
            row[key] = extract_value(value)
        for action, keys in action_keys.items():
            for key in keys:
                row[f"{action}_{key}"] = ''
        
        action = entry.get('action', '')
        action_data = entry.get('actionData', {})

        if action in action_keys:
            for key in action_keys[action]:
                value = action_data.get(key, '')
                row[f"{action}_{key}"] = extract_value(value)

        writer.writerow(row)

print(f"CSV file '{output_file}' has been created successfully.")