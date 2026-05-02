import os
import requests
from tqdm import tqdm
import csv
import time

def download_file(url, directory, max_retries=3):
    """Download a font file with retry logic"""
    # Extract the filename from the URL
    filename = url.split("/")[-1]
    # Create the full path for the download
    file_path = os.path.join(directory, filename)
    
    # Skip if already downloaded
    if os.path.exists(file_path):
        return True, "Already exists"

    # Send a GET request to the URL
    full_url = "https://www.1001fonts.com" + url
    
    for attempt in range(max_retries):
        try:
            response = requests.get(full_url, timeout=30)
            # Check if the request was successful
            if response.status_code == 200:
                # Write the content to a file
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                return True, "Downloaded"
            else:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                return False, f"HTTP {response.status_code}"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return False, str(e)
    
    return False, "Max retries exceeded"

# Read URLs from CSV file
csv_path = r"E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic\font_links_license.csv"
urls = []

print(f"Reading font links from: {csv_path}")
with open(csv_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        urls.append(row['link'])

print(f"Total fonts to download: {len(urls)}")

# Directory to save the downloaded files
download_directory = r"E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic\fonts"

# Create the directory if it doesn't exist
os.makedirs(download_directory, exist_ok=True)

# Download statistics
stats = {"success": 0, "failed": 0, "skipped": 0}
failed_fonts = []

# Download each file
for url in tqdm(urls, desc="Downloading fonts"):
    success, message = download_file(url, download_directory)
    if success:
        if message == "Already exists":
            stats["skipped"] += 1
        else:
            stats["success"] += 1
    else:
        stats["failed"] += 1
        failed_fonts.append((url, message))

# Print summary
print("\n" + "="*50)
print("DOWNLOAD SUMMARY")
print("="*50)
print(f"Successfully downloaded: {stats['success']}")
print(f"Already existed (skipped): {stats['skipped']}")
print(f"Failed: {stats['failed']}")
print(f"Total: {len(urls)}")

if failed_fonts:
    print("\nFailed downloads:")
    for url, reason in failed_fonts[:10]:  # Show first 10
        print(f"  {url}: {reason}")
    if len(failed_fonts) > 10:
        print(f"  ... and {len(failed_fonts) - 10} more")
    
    # Save failed fonts to file
    failed_file = os.path.join(download_directory, "failed_downloads.txt")
    with open(failed_file, 'w') as f:
        for url, reason in failed_fonts:
            f.write(f"{url}\t{reason}\n")
    print(f"\nFailed downloads saved to: {failed_file}")