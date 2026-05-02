from collections import Counter
import matplotlib.pyplot as plt
import os

# Initialize an empty list to store the lengths of each text
text_lengths = []

# Define IAM data paths
iam_base = r"E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\IAM\processed_lines"
output_dir = r"E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic"

# Read from all splits (train, val, test) to get complete distribution
splits = ['train', 'val', 'test']

print("Reading IAM transcriptions from all splits...")
for split in splits:
    gt_file = os.path.join(iam_base, split, 'gt.txt')
    if not os.path.exists(gt_file):
        print(f"Warning: {gt_file} not found, skipping...")
        continue
    
    print(f"Processing {split}...")
    with open(gt_file, "r", encoding='utf-8') as file:
        for line in file:
            # Format: image_name transcription
            try:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    _, text = parts
                    # Calculate the length in characters
                    length = len(text)
                    text_lengths.append(length)
                else:
                    print(f"Skipping malformed line: {line[:50]}...")
            except Exception as e:
                print(f"Error processing line: {line[:50]}... - {e}")

print(f"\nTotal lines processed: {len(text_lengths)}")

# Calculate the distribution of text lengths
length_distribution = Counter(text_lengths)

# Sort the lengths and corresponding counts
lengths = sorted(length_distribution.keys())
counts = [length_distribution[length] for length in lengths]

# Print statistics
print(f"\nLength Distribution Statistics:")
print(f"  Min length: {min(text_lengths)}")
print(f"  Max length: {max(text_lengths)}")
print(f"  Average length: {sum(text_lengths)/len(text_lengths):.2f}")

# Plot the distribution
plt.figure(figsize=(12, 8))
plt.bar(lengths, counts, color='skyblue')

# Add labels and title
plt.xlabel('Text Length (Number of Characters)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Text Lengths in IAM Dataset (Max Length = {max(text_lengths)})')

# Save the plot as a PNG file
plot_path = os.path.join(output_dir, 'text_length_distribution.png')
plt.savefig(plot_path, format='png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")

# Save the counts to a text file
count_file_path = os.path.join(output_dir, 'text_length_counts.txt')
with open(count_file_path, "w") as count_file:
    count_file.write("Text Length\tFrequency\n")
    for length, count in sorted(length_distribution.items()):
        count_file.write(f"{length}\t{count}\n")

print(f"Length counts saved to: {count_file_path}")

# Show the plot (optional - commented out for automation)
# plt.show()

print("\n" + "="*60)
print("IAM LENGTH DISTRIBUTION ANALYSIS COMPLETE")
print("="*60)