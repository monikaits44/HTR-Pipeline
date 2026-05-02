import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Directory containing font files
DATA_DIR = r'E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic'
font_dir = os.path.join(DATA_DIR, 'fonts_extracted')

output_dir = os.path.join(DATA_DIR, 'synthesized_images')
os.makedirs(output_dir, exist_ok=True)

# Fixed text to use for each font
text = '''!"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'''

# Get list of font files with extensions .ttf, .otf, etc.
font_files = []
for root, dirs, files in os.walk(font_dir):
    for file in files:
        if file.lower().endswith(('.otf', '.ttf')):
            font_files.append(os.path.join(root, file))


def read_and_concatenate_with_font_dir(file_path, font_dir):
    # Initialize an empty list to store the concatenated paths
    concatenated_paths = []

    # Open the text file and read each line
    with open(file_path, 'r') as file:
        for line in file:
            # Remove any leading/trailing whitespace and join with font_dir
            concatenated_path = os.path.join(font_dir, line.strip())
            concatenated_paths.append(concatenated_path)

    return concatenated_paths

file_path = os.path.join(DATA_DIR, 'missing_characters_fonts.txt')
missing_c_fonts = read_and_concatenate_with_font_dir(file_path, font_dir)

def remove_missing_fonts(font_files, missing_c_fonts):
    # Use list comprehension to filter out elements that are in missing_c_fonts
    updated_font_files = [font for font in font_files if font not in missing_c_fonts]
    return updated_font_files

font_files = remove_missing_fonts(font_files, missing_c_fonts)


# List to store skipped fonts
skipped_fonts = []

def generate_image(i, text, font_files):
    text = text.strip()

    # Attempt to load the font corresponding to the current index
    font_path = font_files[i]

    # Set a base font size
    font_size = 100
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"Failed to load font: {font_path}. Skipping font.")
        skipped_fonts.append(font_path)  # Add skipped font to the list
        return ''

    # Create a dummy image to get a drawing context
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    # Get the bounding box of the text
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except Exception as e:
        print(f"Error getting textbbox for text '{text}': {e}")
        skipped_fonts.append(font_path)  # Add skipped font to the list
        return ''

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Check if the dimensions are positive and reasonable (max 10000 pixels in either dimension)
    MAX_DIMENSION = 10000
    if text_width <= 0 or text_height <= 0:
        print(f"Text '{text}' results in non-positive dimensions ({text_width}x{text_height}). Skipping.")
        skipped_fonts.append(font_path)  # Add skipped font to the list
        return ''
    
    if text_width > MAX_DIMENSION or text_height > MAX_DIMENSION:
        print(f"Text dimensions ({text_width}x{text_height}) exceed maximum allowed ({MAX_DIMENSION}). Skipping font: {font_path}")
        skipped_fonts.append(font_path)  # Add skipped font to the list
        return ''

    # Create a new image with the size of the text
    img = Image.new('RGB', (text_width, text_height), color='white')
    draw = ImageDraw.Draw(img)
    # Draw the text starting at the adjusted position
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill='black')

    # Save the image
    output_path = os.path.join(output_dir, f'{os.path.basename(font_path)}.png')
    try:
        img.save(output_path)
    except Exception as e:
        print(f"Error saving image '{output_path}': {e}")
        skipped_fonts.append(font_path)  # Add skipped font to the list
        return ''

    # Return the image name and text for mapping
    return f'{output_path}\t{text}\n'

# Main execution
if __name__ == '__main__':
    # Iterate over all font files
    for i in tqdm(range(len(font_files)), total=len(font_files)):
        generate_image(i, text, font_files)

    # Save the skipped fonts to a .txt file
    skipped_fonts_path = os.path.join(DATA_DIR, 'skipped_fonts.txt')
    with open(skipped_fonts_path, 'w') as f:
        for font in skipped_fonts:
            f.write(os.path.basename(font) + '\n')

    print(f"Skipped fonts have been saved to 'skipped_fonts.txt'.")