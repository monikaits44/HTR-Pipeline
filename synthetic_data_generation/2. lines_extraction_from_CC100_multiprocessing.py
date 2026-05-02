from datasets import load_dataset
import random
import argparse
from multiprocessing import Pool
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Extract random text lines from C4 dataset (replaces CC100).')
    parser.add_argument('--split', type=str, default="train", help='split type (train/validation)')
    parser.add_argument('--num_lines', type=int, default=10000000, help='number of lines to extract')
    parser.add_argument('--num_processes', type=int, default=None, help='number of processes (default: CPU count)')
    return parser.parse_args()

args = parse_args()

def init_worker(lang, split, cache_dir):
    global dataset
    # Use C4 dataset instead of CC100 (better maintained)
    dataset = load_dataset('allenai/c4', lang, split=split, cache_dir=cache_dir, streaming=False)
    # Disable caching to prevent disk space issues
    dataset.set_format(type=None)

def worker_process(args):
    indices, temp_output_file, temp_index_file = args
    with open(temp_output_file, 'w', encoding='utf-8') as outfile, \
         open(temp_index_file, 'w', encoding='utf-8') as indexfile:
        for idx in indices:
            item = dataset[idx]
            outfile.write(item['text'] + '\n')
            indexfile.write(str(idx) + '\n')

def extract_random_lines_from_hf_dataset(num_lines=10000000, lang='en',
                                         output_file='c4_random_subset_10M.txt',
                                         index_file='c4_random_indices_10M.txt',
                                         num_processes=None, cache_dir=None):
    
    # Set defaults
    if num_processes is None:
        num_processes = os.cpu_count()
    
    if cache_dir is None:
        # Use data/synthetic directory for cache
        cache_dir = r"E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic\cache"
        os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Number of lines: {num_lines:,}")
    print(f"  Language: {lang}")
    print(f"  Split: {args.split}")
    print(f"  Processes: {num_processes}")
    print(f"  Cache directory: {cache_dir}")
    
    # Load the dataset once to get the size
    print("\nLoading C4 dataset (English web crawl, replacing CC100)...")
    dataset = load_dataset('allenai/c4', lang, split=args.split, cache_dir=cache_dir, streaming=False)
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size:,} lines")

    # Check if we should exclude previously used indices
    exclude_prev = False
    prev_index_file = os.path.join(os.path.dirname(index_file), 'cc100_random_indices_10M.txt')
    
    if os.path.exists(prev_index_file):
        print(f"\nFound previous index file: {prev_index_file}")
        response = input("Exclude previously used indices? (y/n): ").strip().lower()
        exclude_prev = (response == 'y')
    
    if exclude_prev:
        print("Loading previously used indices...")
        used_indices = set()
        with open(prev_index_file, 'r') as f:
            for line in f:
                used_indices.add(int(line.strip()))
        print(f"Previously used indices: {len(used_indices):,}")

        # Generate new random indices, excluding used indices
        remaining_indices = set(range(dataset_size)) - used_indices
        if len(remaining_indices) < num_lines:
            raise ValueError(f"Not enough unused indices. Need {num_lines:,}, have {len(remaining_indices):,}")

        print(f"Sampling {num_lines:,} from {len(remaining_indices):,} remaining indices...")
        random_indices = random.sample(list(remaining_indices), num_lines)
    else:
        # Generate random indices
        print(f"Sampling {num_lines:,} random indices from dataset...")
        random_indices = random.sample(range(dataset_size), num_lines)

    # Split indices among processes
    chunk_size = num_lines // num_processes
    index_chunks = [random_indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]
    # Add remaining indices to the last chunk
    index_chunks[-1].extend(random_indices[num_processes*chunk_size:])

    # Prepare arguments for worker processes
    print(f"\nPreparing {num_processes} worker processes...")
    temp_files = []
    args_list = []
    for i, indices in enumerate(index_chunks):
        temp_output_file = f'{output_file}_part_{i}'
        temp_index_file = f'{index_file}_part_{i}'
        temp_files.append((temp_output_file, temp_index_file))
        args_list.append((indices, temp_output_file, temp_index_file))
        print(f"  Process {i+1}: {len(indices):,} lines")

    # Initialize worker processes
    print("\nExtracting lines (this may take a while)...")
    with Pool(processes=num_processes, initializer=init_worker, initargs=(lang, args.split, cache_dir)) as pool:
        pool.map(worker_process, args_list)

    # Merge temporary files
    print("\nMerging temporary files...")
    with open(output_file, 'w', encoding='utf-8') as outfile, \
         open(index_file, 'w', encoding='utf-8') as indexfile:
        for i, (temp_output_file, temp_index_file) in enumerate(temp_files):
            print(f"  Merging part {i+1}/{len(temp_files)}...")
            with open(temp_output_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            os.remove(temp_output_file)

            with open(temp_index_file, 'r', encoding='utf-8') as infile:
                indexfile.write(infile.read())
            os.remove(temp_index_file)

    print(f'\n{"="*60}')
    print('EXTRACTION COMPLETE')
    print(f'{"="*60}')
    print(f'{num_lines:,} random lines extracted to:')
    print(f'  {output_file}')
    print(f'Indices saved to:')
    print(f'  {index_file}')

# Example usage
if __name__ == "__main__":
    # Set output paths in data/synthetic directory
    output_dir = r"E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'cc100_random_subset_10M.txt')
    index_file = os.path.join(output_dir, 'cc100_random_indices_10M.txt')
    
    num_processes = args.num_processes if args.num_processes else os.cpu_count()
    
    extract_random_lines_from_hf_dataset(
        num_lines=args.num_lines,
        output_file=output_file,
        index_file=index_file,
        num_processes=num_processes,
        cache_dir=None  # Will use default
    )