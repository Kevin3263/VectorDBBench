#!/usr/bin/env python3
"""
Download GIST 1M dataset for VectorDBBench.

GIST dataset:
- Dimension: 960
- Size: 1M vectors
- Metric: L2
- Files: train.parquet, test.parquet
"""

import s3fs
import pathlib
from vectordb_bench import config

print("="*70)
print("GIST 1M Dataset Download")
print("="*70)

# Initialize S3 filesystem
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})

# Dataset configuration
base_url = config.AWS_S3_URL.rstrip('/')
dataset_name = "gist_medium_1m"  # Standard naming pattern
dataset_path = f"{base_url}/{dataset_name}"

print(f"AWS S3 URL: {base_url}")
print(f"Dataset: {dataset_name}")
print(f"Full path: {dataset_path}")

# Local directory
local_dir = pathlib.Path("/tmp/vectordb_bench/dataset/gist/gist_medium_1m/")
local_dir.mkdir(parents=True, exist_ok=True)

print(f"Local directory: {local_dir}")
print("="*70)

# Download train.parquet
print("\n📥 Downloading train.parquet...")
remote_file = f"{dataset_path}/train.parquet"
local_file = local_dir / "train.parquet"

try:
    fs.download(remote_file, str(local_file))
    file_size = local_file.stat().st_size / (1024**3)  # GB
    print(f"✓ Successfully downloaded train.parquet ({file_size:.2f} GB)")
except Exception as e:
    print(f"✗ Failed to download train.parquet: {e}")
    print(f"   Trying alternate naming...")

    # Try alternate naming
    try:
        remote_file = f"{dataset_path}/gist_train.parquet"
        fs.download(remote_file, str(local_file))
        file_size = local_file.stat().st_size / (1024**3)
        print(f"✓ Successfully downloaded train.parquet ({file_size:.2f} GB)")
    except Exception as e2:
        print(f"✗ Also failed with alternate naming: {e2}")

# Download test.parquet
print("\n📥 Downloading test.parquet...")
remote_file = f"{dataset_path}/test.parquet"
local_file = local_dir / "test.parquet"

try:
    fs.download(remote_file, str(local_file))
    file_size = local_file.stat().st_size / (1024**2)  # MB
    print(f"✓ Successfully downloaded test.parquet ({file_size:.2f} MB)")
except Exception as e:
    print(f"✗ Failed to download test.parquet: {e}")
    print(f"   Trying alternate naming...")

    try:
        remote_file = f"{dataset_path}/gist_test.parquet"
        fs.download(remote_file, str(local_file))
        file_size = local_file.stat().st_size / (1024**2)
        print(f"✓ Successfully downloaded test.parquet ({file_size:.2f} MB)")
    except Exception as e2:
        print(f"✗ Also failed with alternate naming: {e2}")

print("\n" + "="*70)
print("Note: neighbors.parquet (ground truth) will be generated separately")
print("using generate_neighbors.py for L2 metric compatibility")
print("="*70)

# List downloaded files
print("\n📊 Downloaded files:")
total_size = 0
for f in sorted(local_dir.glob("*.parquet")):
    size = f.stat().st_size / (1024**3)  # GB
    total_size += size
    print(f"  - {f.name:20s} ({size:.3f} GB)")

print(f"\nTotal size: {total_size:.3f} GB")

if total_size > 0:
    print("\n✓ Download complete!")
    print(f"  Dataset location: {local_dir}")
else:
    print("\n⚠️  No files were downloaded. Checking what's available in S3...")

    # List available files
    try:
        print(f"\nListing files in {dataset_path}...")
        files = fs.ls(dataset_path)
        print(f"Found {len(files)} items:")
        for f in files[:20]:  # Show first 20
            print(f"  - {f}")
    except Exception as e:
        print(f"✗ Failed to list S3 directory: {e}")

print("="*70)
