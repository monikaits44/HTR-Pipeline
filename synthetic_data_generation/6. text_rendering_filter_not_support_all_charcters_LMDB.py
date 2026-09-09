import os
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from fontTools.ttLib import TTFont
import io
import lmdb
import pickle

print("="*60)
print("STEP 6: FINAL IMAGE GENERATION WITH LMDB")
print("="*60)

# Paths
DATA_DIR = '/home/woody/iwi5/iwi5369h/projects/synth_htr'
font_dir = '/home/woody/iwi5/iwi5369h/projects/synth_htr/fonts_extracted'
text_file = os.path.join(DATA_DIR, 'cc100_random_subset_1M_modified_filtered.txt')
lmdb_output_path = os.path.join(DATA_DIR, 'synthesized_images_1M.lmdb')

print(f"Font directory: {font_dir}")
print(f"Text file: {text_file}")
print(f"Output LMDB: {lmdb_output_path}")

# Get list of font files with extensions .ttf, .otf, etc.
print("\nScanning for font files...")
font_files = []
for root, dirs, files in os.walk(font_dir):
    for file in files:
        if file.lower().endswith(('.otf', '.ttf')):
            font_files.append(os.path.join(root, file))

print(f"Found {len(font_files)} font files")

# Read the text lines from your text file
print(f"Reading text lines from {text_file}...")
with open(text_file, 'r', encoding='utf-8') as file:
    texts = file.readlines()

print(f"Loaded {len(texts)} text lines")

def check_font_support(font_path, characters):
    # Load the font
    try:
        font = TTFont(font_path)
        font_cmaps = []
        for table in font['cmap'].tables:
            font_cmaps.extend(table.cmap.keys())
        font_chars = set(chr(codepoint) for codepoint in font_cmaps)
        missing = [char for char in characters if char not in font_chars]
        return not missing  # True if no missing characters
    except Exception as e:
        print(f"Error reading font {font_path}: {e}")
        return False  # Return False if loading the font fails

def generate_image(args):
    i, text, font_files_global = args
    text = text.strip()
    # font_files = font_files_global.copy()
    font_files = list(font_files_global)

    font_loaded = False
    attempts = 0
    max_attempts = len(font_files)

    # Attempt to load a font that supports all characters
    while not font_loaded and attempts < max_attempts:
        font_path = random.choice(font_files)
        font_files.remove(font_path)
        if check_font_support(font_path, text):
            try:
                font_size = 100
                font = ImageFont.truetype(font_path, font_size)
                font_loaded = True
            except Exception as e:
                print(f"Failed to load font: {font_path}. Error: {e}")
                attempts += 1
        else:
            attempts += 1

    if not font_loaded:
        print(f"Could not find a font that supports all characters in text '{text}'. Skipping index {i}.")
        return None  # Return None to indicate failure

    if not text:
        print(f"Text is empty after stripping. Skipping index {i}.")
        return None

    # Create a dummy image to get a drawing context
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except Exception as e:
        print(f"Error getting textbbox for text '{text}': {e}")
        return None

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Check if the dimensions are positive and reasonable (max 10000 pixels in either dimension)
    MAX_DIMENSION = 10000
    if text_width <= 0 or text_height <= 0:
        print(f"Text '{text}' results in non-positive dimensions ({text_width}x{text_height}). Skipping.")
        return None
    
    if text_width > MAX_DIMENSION or text_height > MAX_DIMENSION:
        print(f"Text dimensions ({text_width}x{text_height}) exceed maximum allowed ({MAX_DIMENSION}). Skipping index {i}.")
        return None

    # Create a new image with the size of the text
    img = Image.new('RGB', (text_width, text_height), color='white')
    draw = ImageDraw.Draw(img)
    try:
        draw.text((-bbox[0], -bbox[1]), text, font=font, fill='black')
    except Exception as e:
        print(f"Error drawing text for index {i}: {e}")
        return None

    # Convert image to bytes in PNG format
    try:
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
    except Exception as e:
        print(f"Error saving image to buffer for index {i}: {e}")
        return None

    return (i, img_bytes, text)

def worker_initializer(font_files_global_):
    # Make font_files_global accessible in worker processes
    global font_files_global
    font_files_global = font_files_global_

if __name__ == '__main__':
    print("\nPreparing inputs for image generation...")
    
    # Prepare the inputs for multiprocessing - pass font_files directly as a regular list
    inputs = [(i, text, font_files) for i, text in enumerate(texts)]

    num_workers = min(cpu_count(), 8)  # Limit workers to control memory usage
    print(f"\nGenerating {len(inputs)} synthetic images using {num_workers} processes...")
    print("Writing directly to LMDB to avoid OOM...\n")
    
    # Create LMDB database - write incrementally to avoid OOM
    env = lmdb.open(lmdb_output_path, map_size=int(40e9))  # 40GB map size for 1M images
    
    successful_count = 0
    failed_count = 0
    batch_size = 1000  # Commit every N images to balance speed and memory
    batch = []
    
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(generate_image, inputs), 
                           total=len(inputs), 
                           desc="Generating images"):
            if result is not None:
                batch.append(result)
                if len(batch) >= batch_size:
                    # Write batch to LMDB
                    with env.begin(write=True) as txn:
                        for i, img_bytes, text in batch:
                            key = f'{i:010}'.encode('ascii')
                            data = {'image': img_bytes, 'text': text}
                            txn.put(key, pickle.dumps(data))
                    successful_count += len(batch)
                    batch = []
            else:
                failed_count += 1
    
    # Write remaining batch
    if batch:
        with env.begin(write=True) as txn:
            for i, img_bytes, text in batch:
                key = f'{i:010}'.encode('ascii')
                data = {'image': img_bytes, 'text': text}
                txn.put(key, pickle.dumps(data))
        successful_count += len(batch)
    
    env.close()
    
    print(f"\n{'='*60}")
    print("IMAGE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully generated: {successful_count} images")
    print(f"Failed: {failed_count} images")
    print(f"Total processed: {len(texts)}")
    print(f"LMDB database saved to: {lmdb_output_path}")
    lmdb_data_file = os.path.join(lmdb_output_path, 'data.mdb')
    if os.path.exists(lmdb_data_file):
        print(f"Database size: {os.path.getsize(lmdb_data_file) / (1024**3):.2f} GB")