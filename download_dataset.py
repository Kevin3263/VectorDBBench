import s3fs
import pathlib
from vectordb_bench import config

fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})

# First, let's list what's actually available
print(f"AWS_S3_URL from config: {config.AWS_S3_URL}")
print(f"USE_SHUFFLED_DATA: {config.USE_SHUFFLED_DATA}")

# Fix the path - remove trailing slash from AWS_S3_URL if present
base_url = config.AWS_S3_URL.rstrip('/')
dataset_name = "cohere_large_10m"
dataset_path = f"{base_url}/{dataset_name}"

print(f"\nCorrected dataset path: {dataset_path}")

# Now download with correct paths
local_dir = pathlib.Path("/tmp/vectordb_bench/dataset/cohere/cohere_large_10m/")
local_dir.mkdir(parents=True, exist_ok=True)

# Determine file prefix based on shuffled config
# Files are named like: shuffle_train-00-of-10.parquet or train-00-of-10.parquet
file_prefix = "shuffle_train" if config.USE_SHUFFLED_DATA else "train"

print(f"\nStarting download (looking for files with prefix: {file_prefix}-)...")

# Download train files with correct naming format
for i in range(10):
    # Format: shuffle_train-00-of-10.parquet
    remote_file = f"{dataset_path}/{file_prefix}-{i:02d}-of-10.parquet"
    local_file = local_dir / f"{file_prefix}-{i:02d}-of-10.parquet"
    print(f"\nDownloading {remote_file}...")
    try:
        fs.download(remote_file, str(local_file))
        file_size = local_file.stat().st_size / (1024**3)  # Size in GB
        print(f"✓ Successfully downloaded {file_prefix}-{i:02d}-of-10.parquet ({file_size:.2f} GB)")
    except Exception as e:
        print(f"✗ Failed to download: {e}")

# Download test file
remote_file = f"{dataset_path}/test.parquet"
local_file = local_dir / "test.parquet"
print(f"\nDownloading {remote_file}...")
try:
    fs.download(remote_file, str(local_file))
    file_size = local_file.stat().st_size / (1024**2)  # Size in MB
    print(f"✓ Successfully downloaded test.parquet ({file_size:.2f} MB)")
except Exception as e:
    print(f"✗ Failed to download: {e}")

# Download scalar_labels.parquet (needed for label filtering)
remote_file = f"{dataset_path}/scalar_labels.parquet"
local_file = local_dir / "scalar_labels.parquet"
print(f"\nDownloading {remote_file}...")
try:
    fs.download(remote_file, str(local_file))
    file_size = local_file.stat().st_size / (1024**2)  # Size in MB
    print(f"✓ Successfully downloaded scalar_labels.parquet ({file_size:.2f} MB)")
except Exception as e:
    print(f"✗ Failed to download: {e}")

# Note: neighbors.parquet should be skipped (custom L2 ground truth)
print("\n" + "="*60)
print("Note: neighbors.parquet is NOT downloaded - it should be")
print("generated locally using custom L2 ground truth generation.")
print("="*60)

print("\n" + "="*60)
print("Download complete! Files are in:", local_dir)
print("="*60)

# List what we actually downloaded
print("\nDownloaded files:")
total_size = 0
for f in sorted(local_dir.glob("*.parquet")):
    size = f.stat().st_size / (1024**3)  # Size in GB
    total_size += size
    print(f"  - {f.name} ({size:.2f} GB)")
print(f"\nTotal size: {total_size:.2f} GB")