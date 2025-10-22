# The initial data injection from the three files found in the data folder. Here we get 10k of each

# === load_for_graph_building.py === 3rd step
# Load Amazon, International, AND Sale-Report (as product attribute lookup)
import pandas as pd
import os

# ———————————————————————
# CONFIG
# ———————————————————————
DATA_DIR = 'processed_data'
TRAINING_CUTOFF_TIME = pd.Timestamp('2022-12-31')
DVID = 1  # Initial data version

# Input files
amazon_file = os.path.join(DATA_DIR, 'sampled_amazon_sale_report_dvid1.csv')
international_file = os.path.join(DATA_DIR, 'sampled_international_sale_report_dvid1.csv')
sale_file_path = 'Sale-Report.csv'  # Product catalog — no dates

# Output files
final_output = os.path.join(DATA_DIR, 'combined_sales_dvid1.csv')  # Only transactions
product_attributes_output = os.path.join(DATA_DIR, 'product_attributes_dvid1.csv')  # Only product metadata

# Create processed_data folder if not exists
os.makedirs(DATA_DIR, exist_ok=True)

print("🚀 Starting data loading and integration...")

# ———————————————————————
# 1. LOAD AND CLEAN AMAZON DATA
# ———————————————————————
try:
    amazon_df = pd.read_csv(amazon_file, low_memory=False)
    print(f"📥 Loaded {amazon_file}. Shape: {amazon_df.shape}")

    # Standardize column names
    amazon_df.columns = [col.strip().replace("-", "_").replace(" ", "_").lower() for col in amazon_df.columns]

    # Drop rows with any missing values
    amazon_df_cleaned = amazon_df.dropna().copy()
    amazon_df_cleaned['source'] = 'amazon'
    amazon_df_cleaned['dvid'] = DVID
    print(f"✅ Cleaned Amazon data shape: {amazon_df_cleaned.shape}")

except FileNotFoundError:
    print(f"❌ Error: {amazon_file} not found.")
    amazon_df_cleaned = pd.DataFrame()

print("-" * 50)

# ———————————————————————
# 2. LOAD AND CLEAN INTERNATIONAL DATA
# ———————————————————————
try:
    international_df = pd.read_csv(international_file, low_memory=False)
    print(f"📥 Loaded {international_file}. Shape: {international_df.shape}")

    # Standardize column names
    international_df.columns = [col.strip().replace("-", "_").replace(" ", "_").lower() for col in international_df.columns]

    # Convert 'customer' to string and lowercase
    international_df['customer'] = international_df['customer'].astype(str).str.lower()

    # Filter out month names
    month_names = ['aug-21', 'dec-21', 'feb-22', 'jan-22', 'jul-21', 'mar-22', 'nov-21', 'oct-21', 'sep-21', 'jun-21']
    international_df_filtered = international_df[~international_df['customer'].isin(month_names)].copy()

    # Drop rows with any missing values
    international_df_cleaned = international_df_filtered.dropna().copy()
    international_df_cleaned['source'] = 'international'
    international_df_cleaned['dvid'] = DVID
    print(f"✅ Cleaned International data shape: {international_df_cleaned.shape}")

except FileNotFoundError:
    print(f"❌ Error: {international_file} not found.")
    international_df_cleaned = pd.DataFrame()

print("-" * 50)

# ———————————————————————
# 3. LOAD SALE REPORT AS PRODUCT CATALOG (NO DATE FILTERING)
# ———————————————————————
try:
    sale_df = pd.read_csv(sale_file_path, low_memory=False)
    print(f"📥 Loaded {sale_file_path}. Shape: {sale_df.shape}")

    # Standardize column names
    sale_df.columns = [col.strip().replace("-", "_").replace(" ", "_").lower() for col in sale_df.columns]

    # Drop rows with any missing values
    sale_df_cleaned = sale_df.dropna().copy()
    print(f"✅ Cleaned Sale Report data shape: {sale_df_cleaned.shape}")

    # Rename columns to match your graph schema
    sale_df_cleaned.rename(columns={
        'sku code': 'sku',
        'design no.': 'design_no',
        'stock': 'stock',
        'category': 'category',
        'size': 'size',
        'color': 'color'
    }, inplace=True)

    # Save as product attributes (no dvid needed — it's static metadata)
    sale_df_cleaned.to_csv(product_attributes_output, index=False)
    print(f"✅ Saved product attributes to: {product_attributes_output}")

except FileNotFoundError:
    print(f"❌ Error: {sale_file_path} not found.")
    print("⚠️ Skipping Sale Report data loading.")
    sale_df_cleaned = pd.DataFrame()

print("-" * 50)
print("✅ Data loading and cleaning complete.")

# ———————————————————————
# 4. COMBINE TRANSACTION DATA (Amazon + International)
# ———————————————————————
dfs_to_combine = []
for df in [amazon_df_cleaned, international_df_cleaned]:
    if not df.empty:
        # Ensure 'date' exists and is datetime
        if 'date' not in df.columns:
            raise ValueError(f"❌ Missing 'date' column in {df['source'].iloc[0]}")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df = df[df['date'] <= TRAINING_CUTOFF_TIME]  # Time filter
        dfs_to_combine.append(df)

if dfs_to_combine:
    combined_transactions = pd.concat(dfs_to_combine, ignore_index=True)
    combined_transactions.to_csv(final_output, index=False)
    print(f"✅ Final transaction dataset saved to: {final_output}")
    print(f"📁 Total records in final dataset: {len(combined_transactions):,}")
else:
    raise ValueError("❌ No transaction data available after cleaning and filtering.")

# ———————————————————————
# 5. FINAL SUMMARY
# ———————————————————————
print("\n" + "="*70)
print("✅ DATA INTEGRATION COMPLETE — READY FOR GRAPH BUILDING")
print(f"   Training cutoff: {TRAINING_CUTOFF_TIME.date()}")
print(f"   Data Version ID: {DVID}")
print(f"   Transaction dataset: {len(combined_transactions):,} records")
print(f"   Product attributes: {len(sale_df_cleaned):,} SKUs")
print("="*70)
