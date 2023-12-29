import pandas as pd

# Replace these file paths with your actual file paths
csv_file_paths = ['data/raw_data/train_essays.csv']

for csv_file_path in csv_file_paths:
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Create a corresponding Parquet file path
    parquet_file_path = csv_file_path.replace('.csv', '.parquet')

    # Convert DataFrame to Parquet format
    df.to_parquet(parquet_file_path, index=False)

    print(f"Conversion complete: {csv_file_path} -> {parquet_file_path}")