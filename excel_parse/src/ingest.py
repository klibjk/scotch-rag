import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
import json

# LlamaParse API integration (updated for new endpoint)
LLAMAPARSE_API_URL = "https://api.cloud.llamaindex.ai/api/v1/parsing/upload"

# Helper to poll job status and get result
import time

def poll_llamaparse_job(job_id, api_key, poll_interval=2, timeout=300):
    status_url = f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}"
    headers = {'Authorization': f'Bearer {api_key}', 'accept': 'application/json'}
    start = time.time()
    last_status = None
    while True:
        resp = requests.get(status_url, headers=headers)
        resp.raise_for_status()
        status = resp.json()
        if status.get('status') != last_status:
            print(f"LlamaParse job status: {status.get('status')}")
            last_status = status.get('status')
        if status.get('status') in ('COMPLETED', 'SUCCESS'):
            return status
        if status.get('status') == 'FAILED':
            raise RuntimeError(f"LlamaParse job failed: {status}")
        if time.time() - start > timeout:
            raise TimeoutError("LlamaParse job polling timed out.")
        time.sleep(poll_interval)

def get_llamaparse_result(job_id, api_key):
    result_url = f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}/result/json"
    headers = {'Authorization': f'Bearer {api_key}', 'accept': 'application/json'}
    resp = requests.get(result_url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def generate_sample_excel(excel_path):
    data = {
        'txn_id': [f'TXN{i+1:03d}' for i in range(10)],
        'date': [(datetime.today() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)],
        'amount': [1000.0, -200.0, 500.0, 1200.0, -150.0, 300.0, 700.0, -50.0, 900.0, 400.0],
        'description': [
            'Payment received', 'Refund issued', 'Invoice paid', 'Salary credited', 'Fee charged',
            'Bonus received', 'Purchase', 'Adjustment', 'Transfer', 'Miscellaneous'
        ]
    }
    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)
    print(f"Generated valid sample Excel file at {excel_path}")

def generate_finance_excel(excel_path):
    expenses = pd.DataFrame({
        'date': pd.date_range(end=datetime.today(), periods=7).strftime('%Y-%m-%d'),
        'category': ['Travel', 'Supplies', 'Meals', 'Utilities', 'Travel', 'Supplies', 'Meals'],
        'amount': [-120.50, -45.00, -30.25, -200.00, -80.00, -60.00, -25.00],
        'description': [
            'Flight to NYC', 'Printer ink', 'Team lunch', 'Electric bill',
            'Taxi to airport', 'Office paper', 'Coffee meeting'
        ]
    })
    revenue = pd.DataFrame({
        'date': pd.date_range(end=datetime.today(), periods=7).strftime('%Y-%m-%d'),
        'source': ['Sales', 'Consulting', 'Sales', 'Investment', 'Sales', 'Consulting', 'Sales'],
        'amount': [5000.00, 1200.00, 4500.00, 800.00, 6000.00, 1500.00, 7000.00],
        'description': [
            'Product sale', 'Client project', 'Online order', 'Interest',
            'Bulk order', 'Strategy session', 'End-of-quarter sale'
        ]
    })
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        expenses.to_excel(writer, sheet_name='Expenses', index=False)
        revenue.to_excel(writer, sheet_name='Revenue', index=False)
    print(f"Generated valid finance Excel file at {excel_path}")

def llamaparse_excel(excel_path, api_key):
    # Not used for multi-sheet, fallback to pandas for now
    raise NotImplementedError("LlamaParse multi-sheet not implemented in this stub.")

def ingest_all_excels(data_dir='data'):
    load_dotenv()
    manifest = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.xlsx'):
            file_path = os.path.join(data_dir, fname)
            file_id = os.path.splitext(fname)[0]
            # Check if file is valid, if not, regenerate
            try:
                xls = pd.ExcelFile(file_path, engine='openpyxl')
            except Exception:
                if file_id == 'finance':
                    generate_finance_excel(file_path)
                    xls = pd.ExcelFile(file_path, engine='openpyxl')
                elif file_id == 'sample':
                    generate_sample_excel(file_path)
                    xls = pd.ExcelFile(file_path, engine='openpyxl')
                else:
                    print(f"Skipping invalid Excel file: {file_path}")
                    continue
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
                df['row_index_original'] = df.index
                df = df.convert_dtypes()
                sheet_id = sheet_name.lower().replace(' ', '_')
                table_id = f"{file_id}_{sheet_id}"
                df['file_id'] = file_id
                df['sheet_id'] = sheet_id
                df['table_id'] = table_id
                parquet_path = os.path.join(data_dir, f"{file_id}_{sheet_id}.parquet")
                table = pa.Table.from_pandas(df)
                pq.write_table(table, parquet_path)
                manifest.append({
                    'file_id': file_id,
                    'sheet_id': sheet_id,
                    'table_id': table_id,
                    'parquet_path': parquet_path,
                    'columns': list(df.columns)
                })
    print(f"Ingested {len(manifest)} tables from Excel files.")
    with open(os.path.join(data_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest

def main():
    ingest_all_excels('data')

if __name__ == '__main__':
    main()
