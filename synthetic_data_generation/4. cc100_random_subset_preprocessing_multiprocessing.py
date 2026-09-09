import random
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm

# Paths
DATA_DIR = '/home/woody/iwi5/iwi5369h/projects/synth_htr'
LENGTH_COUNTS_FILE = os.path.join(DATA_DIR, "text_length_counts.txt")
INPUT_FILE = os.path.join(DATA_DIR, "cc100_random_subset_1M.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "cc100_random_subset_1M_modified.txt")

# Global variables
distribution = {}
total_count = 0
percentage_distribution = {}
cumulative_distribution = []

def build_distributions():
    global distribution, total_count, percentage_distribution, cumulative_distribution
    # Step 1: Parse the text_length_counts.txt to get the target distribution
    distribution = {}
    total_count = 0

    print(f"Loading IAM length distribution from: {LENGTH_COUNTS_FILE}")
    with open(LENGTH_COUNTS_FILE, "r", encoding='utf-8') as file:
        next(file)  # Skip header
        for line in file:
            length, count = line.strip().split("\t")
            distribution[int(length)] = int(count)
            total_count += int(count)  # Sum up the total number of lines
    
    print(f"  Loaded {len(distribution)} different lengths")
    print(f"  Total lines in IAM: {total_count}")

    # Step 2: Convert the distribution to percentages (probabilities)
    percentage_distribution = {length: (count / total_count) for length, count in distribution.items()}

    # Step 3: Create a cumulative distribution for selecting a length by probability
    cumulative_distribution = []
    current_cumulative = 0.0
    for length, probability in sorted(percentage_distribution.items()):
        current_cumulative += probability
        cumulative_distribution.append((length, current_cumulative))

# Function to randomly select a length based on cumulative distribution
def select_length_by_probability():
    random_value = random.random()  # Get a random number between 0 and 1
    for length, cumulative in cumulative_distribution:
        if random_value <= cumulative:
            return length
    return max(distribution.keys())  # Return the max length as a fallback

def process_chunk(lines):
    modified_lines = []
    for text in lines:
        text = text.strip()

        if len(text) == 0:
            continue

        selected_length = select_length_by_probability()

        if len(text) > selected_length:
            max_offset = len(text) - selected_length
            offset = random.randint(0, max_offset)
            new_text = text[offset:offset + selected_length]
        elif len(text) < selected_length:
            padding = text * ((selected_length // len(text)) + 1)
            new_text = padding[:selected_length]
        else:
            new_text = text

        modified_lines.append(new_text)
    return modified_lines

def chunked_file_reader(file_path, chunk_size):
    with open(file_path, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

if __name__ == '__main__':
    # Build distributions in the main process
    print("="*60)
    print("STEP 4: TEXT PREPROCESSING - Match IAM Length Distribution")
    print("="*60)
    build_distributions()

    chunk_size = 10000  # Adjust chunk size as needed
    
    print(f"\nProcessing text from: {INPUT_FILE}")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    
    # Count total lines for progress bar
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines to process: {total_lines:,}")

    processed_lines = 0
    with ProcessPoolExecutor(initializer=build_distributions) as executor, \
         open(OUTPUT_FILE, "w", encoding='utf-8') as out_file:
        
        chunks = list(chunked_file_reader(INPUT_FILE, chunk_size))
        print(f"\nProcessing {len(chunks)} chunks...")
        
        with tqdm(total=total_lines, desc="Processing") as pbar:
            for modified_lines in executor.map(process_chunk, chunks):
                for line in modified_lines:
                    out_file.write(line + "\n")
                    processed_lines += 1
                pbar.update(len(modified_lines))

    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {processed_lines:,} lines")
    print(f"Output saved to: {OUTPUT_FILE}")