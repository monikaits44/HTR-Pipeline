import os
import multiprocessing
import re
from tqdm import tqdm

# Define the set of allowed characters (IAM character set)
set_all_characters = ''' !"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'''

# Paths
DATA_DIR = '/home/woody/iwi5/iwi5369h/projects/synth_htr'
input_file = os.path.join(DATA_DIR, 'cc100_random_subset_1M_modified.txt')
output_file = os.path.join(DATA_DIR, 'cc100_random_subset_1M_modified_filtered.txt')

def process_chunk(input_file, output_file, start, end, pattern):
    with open(input_file, 'rb') as infile:
        infile.seek(start)
        if start != 0:
            infile.readline()  # Skip incomplete line
        with open(output_file, 'w', encoding='utf-8') as outfile:
            while True:
                pos = infile.tell()
                if pos >= end:
                    break
                line = infile.readline()
                if not line:
                    break
                line = line.decode('utf-8', errors='ignore').strip()
                # Remove disallowed characters
                new_line = re.sub(pattern, '', line)
                if new_line:
                    outfile.write(new_line + '\n')

def main():
    print("="*60)
    print("STEP 4-1: CHARACTER FILTERING - Remove Non-IAM Characters")
    print("="*60)
    print(f"Allowed characters: {set_all_characters}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    N = multiprocessing.cpu_count()
    print(f"\nUsing {N} processes...")
    
    file_size = os.path.getsize(input_file)
    chunk_size = file_size // N
    processes = []

    # Create a regex pattern for character filtering
    allowed_chars_re = re.escape(set_all_characters)
    pattern = f'[^{allowed_chars_re}]'

    for i in range(N):
        start = i * chunk_size
        end = file_size if i == N - 1 else (i + 1) * chunk_size
        p = multiprocessing.Process(
            target=process_chunk,
            args=(input_file, f'{output_file}_{i}', start, end, pattern)
        )
        processes.append(p)

    print("Filtering characters...")
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # Merge the output files
    print("Merging output files...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i in range(N):
            part_file = f'{output_file}_{i}'
            with open(part_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            os.remove(part_file)
    
    # Count final lines
    with open(output_file, 'r', encoding='utf-8') as f:
        final_count = sum(1 for _ in f)
    
    print(f"\n{'='*60}")
    print("CHARACTER FILTERING COMPLETE")
    print(f"{'='*60}")
    print(f"Final lines: {final_count:,}")
    print(f"Output saved to: {output_file}")

if __name__ == '__main__':
    main()