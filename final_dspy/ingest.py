import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import requests
import time

def llamaparse_ingest(file_path, api_key):
    url = "https://api.cloud.llamaindex.ai/api/v1/parsing/upload"
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        headers = {'Authorization': f'Bearer {api_key}'}
        resp = requests.post(url, files=files, headers=headers)
        resp.raise_for_status()
        job_id = resp.json()['id']
    # Poll for completion
    status_url = f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}"
    for _ in range(60):
        status_resp = requests.get(status_url, headers=headers)
        status_resp.raise_for_status()
        status = status_resp.json()
        if status.get('status') in ('COMPLETED', 'SUCCESS'):
            break
        if status.get('status') == 'FAILED':
            raise RuntimeError(f"LlamaParse job failed: {status}")
        time.sleep(3)
    else:
        raise TimeoutError("LlamaParse job polling timed out.")
    # Download result
    result_url = f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}/result/json"
    result_resp = requests.get(result_url, headers=headers)
    result_resp.raise_for_status()
    result = result_resp.json()
    # Convert LlamaParse JSON to dict of DataFrames (sheet_name: df)
    sheets = {}
    for sheet in result.get('sheets', []):
        name = sheet.get('name', 'Sheet1')
        rows = sheet.get('rows', [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        sheets[name] = df
    return sheets

def ingest_all_excels(data_dir='data'):
    manifest = []
    llama_api_key = os.getenv('LLAMAPARSE_API_KEY')
    for fname in os.listdir(data_dir):
        if fname.endswith('.xlsx'):
            file_path = os.path.join(data_dir, fname)
            file_id = os.path.splitext(fname)[0]
            used_llamaparse = False
            sheets = None
            if llama_api_key:
                try:
                    print(f"Trying LlamaParse for {fname} ...")
                    sheets = llamaparse_ingest(file_path, llama_api_key)
                    used_llamaparse = True
                except Exception as e:
                    print(f"LlamaParse failed for {fname}: {e}. Falling back to pandas.")
            if sheets is None:
                try:
                    xls = pd.ExcelFile(file_path, engine='openpyxl')
                    sheets = {sheet: pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl') for sheet in xls.sheet_names}
                    print(f"Used pandas for {fname}.")
                except Exception as e:
                    print(f"Skipping invalid Excel file: {file_path} ({e})")
                    continue
            for sheet_name, df in sheets.items():
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
                print(f"Processed: {file_id} | sheet: {sheet_id} | rows: {len(df)} | parquet: {parquet_path} | method: {'llamaparse' if used_llamaparse else 'pandas'}")
    print(f"Ingested {len(manifest)} tables from Excel files.")
    with open(os.path.join(data_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest

def main():
    ingest_all_excels('data')

if __name__ == '__main__':
    main()
