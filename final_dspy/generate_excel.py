import pandas as pd
from datetime import datetime
import os

os.makedirs('final_dspy', exist_ok=True)

# Orders and Accounts
orders = pd.DataFrame([
    {'order_id': 'O1001', 'account_id': 'A001', 'security': 'ETF-ABC', 'order_type': 'buy', 'quantity': 100, 'price': 50.25, 'status': 'pending', 'placed_time': datetime(2024,6,1,10,0), 'executed_time': None},
    {'order_id': 'O1002', 'account_id': 'A002', 'security': 'STOCK-XYZ', 'order_type': 'sell', 'quantity': 50, 'price': 120.5, 'status': 'filled', 'placed_time': datetime(2024,6,1,9,30), 'executed_time': datetime(2024,6,1,9,45)},
    {'order_id': 'O1003', 'account_id': 'A001', 'security': 'ETF-ABC', 'order_type': 'buy', 'quantity': 20, 'price': 51.0, 'status': 'cancelled', 'placed_time': datetime(2024,6,1,11,0), 'executed_time': None}
])
accounts = pd.DataFrame([
    {'account_id': 'A001', 'owner': 'Alice', 'account_type': 'individual', 'balance': 10000.0},
    {'account_id': 'A002', 'owner': 'Bob', 'account_type': 'retirement', 'balance': 25000.0}
])
with pd.ExcelWriter('final_dspy/accounts_orders.xlsx', engine='openpyxl') as writer:
    orders.to_excel(writer, sheet_name='orders', index=False)
    accounts.to_excel(writer, sheet_name='accounts', index=False)

# Securities, Holdings, Fees
securities = pd.DataFrame([
    {'security_id': 'ETF-ABC', 'name': 'Growth ETF', 'type': 'ETF', 'current_price': 50.5, 'risk_level': 'growth'},
    {'security_id': 'STOCK-XYZ', 'name': 'XYZ Corp', 'type': 'stock', 'current_price': 120.0, 'risk_level': 'stability'},
    {'security_id': 'ETF-DEF', 'name': 'Stable ETF', 'type': 'ETF', 'current_price': 40.0, 'risk_level': 'stability'}
])
holdings = pd.DataFrame([
    {'etf_id': 'ETF-ABC', 'holding_security': 'STOCK-XYZ', 'weight_percent': 60.0},
    {'etf_id': 'ETF-ABC', 'holding_security': 'STOCK-123', 'weight_percent': 40.0},
    {'etf_id': 'ETF-DEF', 'holding_security': 'STOCK-XYZ', 'weight_percent': 30.0},
    {'etf_id': 'ETF-DEF', 'holding_security': 'STOCK-456', 'weight_percent': 70.0}
])
fees = pd.DataFrame([
    {'security_id': 'ETF-ABC', 'fee_type': 'management', 'amount': 0.5},
    {'security_id': 'ETF-ABC', 'fee_type': 'commission', 'amount': 1.0},
    {'security_id': 'STOCK-XYZ', 'fee_type': 'commission', 'amount': 1.5}
])
with pd.ExcelWriter('final_dspy/securities_info.xlsx', engine='openpyxl') as writer:
    securities.to_excel(writer, sheet_name='securities', index=False)
    holdings.to_excel(writer, sheet_name='holdings', index=False)
    fees.to_excel(writer, sheet_name='fees', index=False)