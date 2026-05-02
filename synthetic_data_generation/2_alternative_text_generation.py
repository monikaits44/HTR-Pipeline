"""
Alternative to CC100: Generate synthetic English text using Wikipedia or random sentences
This replaces Step 2 since CC100 is deprecated
"""

import random
import os
from tqdm import tqdm

# Sample English words for text generation
COMMON_WORDS = [
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'is', 'was', 'are', 'been', 'has', 'had', 'were', 'said', 'did', 'having',
    'may', 'should', 'could', 'would', 'might', 'must', 'can', 'shall', 'will',
    'going', 'doing', 'making', 'working', 'trying', 'looking', 'thinking', 'feeling',
    'man', 'woman', 'child', 'person', 'family', 'friend', 'house', 'home', 'room',
    'hand', 'part', 'place', 'case', 'point', 'government', 'company', 'number',
    'group', 'problem', 'fact', 'world', 'school', 'state', 'country', 'area',
    'book', 'story', 'word', 'line', 'letter', 'name', 'right', 'left', 'side',
    'water', 'fire', 'air', 'earth', 'sun', 'moon', 'star', 'sky', 'sea', 'river',
    'mountain', 'tree', 'flower', 'grass', 'food', 'fish', 'bird', 'animal', 'dog', 'cat',
    'walk', 'run', 'sit', 'stand', 'play', 'read', 'write', 'speak', 'listen', 'watch',
    'eat', 'drink', 'sleep', 'wake', 'start', 'stop', 'open', 'close', 'move', 'turn',
    'big', 'small', 'long', 'short', 'high', 'low', 'old', 'new', 'young', 'good',
    'bad', 'hot', 'cold', 'fast', 'slow', 'early', 'late', 'right', 'wrong', 'true',
    'false', 'easy', 'hard', 'happy', 'sad', 'beautiful', 'ugly', 'strong', 'weak',
]

# Add some proper nouns and specific words
NAMES = ['John', 'Mary', 'David', 'Sarah', 'Michael', 'Emma', 'James', 'Lisa', 'Robert', 'Anna']
PLACES = ['London', 'Paris', 'New York', 'Tokyo', 'Berlin', 'Rome', 'Sydney', 'Moscow']

def generate_sentence(min_words=3, max_words=15):
    """Generate a random English sentence"""
    num_words = random.randint(min_words, max_words)
    words = []
    
    # Start with capital
    if random.random() < 0.2:
        words.append(random.choice(NAMES))
    else:
        words.append(random.choice(COMMON_WORDS).capitalize())
    
    # Add middle words
    for _ in range(num_words - 1):
        if random.random() < 0.05:  # 5% chance of name/place
            words.append(random.choice(NAMES + PLACES))
        else:
            words.append(random.choice(COMMON_WORDS))
    
    sentence = ' '.join(words)
    
    # Add punctuation
    if random.random() < 0.3:
        sentence += random.choice(['.', '!', '?'])
    
    return sentence

def generate_text_lines(num_lines, output_file):
    """Generate random English text lines"""
    print(f"Generating {num_lines:,} synthetic text lines...")
    
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(num_lines), desc="Generating text"):
            # Generate 1-3 sentences per line
            num_sentences = random.randint(1, 3)
            sentences = [generate_sentence() for _ in range(num_sentences)]
            line = ' '.join(sentences)
            f.write(line + '\n')
    
    print(f"\n{'='*60}")
    print('TEXT GENERATION COMPLETE')
    print(f"{'='*60}")
    print(f'{num_lines:,} lines generated and saved to:')
    print(f'  {output_file}')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic English text')
    parser.add_argument('--num_lines', type=int, default=50000, help='number of lines to generate')
    args = parser.parse_args()
    
    output_dir = r"E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic"
    output_file = os.path.join(output_dir, f'synthetic_text_{args.num_lines//1000}K.txt')
    
    generate_text_lines(args.num_lines, output_file)
