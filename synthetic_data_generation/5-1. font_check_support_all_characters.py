import os
from fontTools.ttLib import TTFont
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

print("="*60)
print("STEP 5-1: CHECK CHARACTER SUPPORT IN FONTS")
print("="*60)

def check_font_support(font_path, characters):
    # Load the font
    try:
        font = TTFont(font_path)
        # Get all character codes from all cmap tables
        font_chars = set()
        for table in font['cmap'].tables:
            font_chars.update([chr(x) for x in table.cmap.keys()])
        
        missing = [char for char in characters if char not in font_chars]
        return font_path, missing
    except Exception as e:
        return font_path, "load_failed"

def check_font_wrapper(args):
    """Wrapper for multiprocessing"""
    font_path, characters = args
    return check_font_support(font_path, characters)

def get_all_font_files(directory):
    """Recursively get all font files"""
    font_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.ttf', '.otf', '.TTF', '.OTF')):
                font_files.append(os.path.join(root, file))
    return font_files

def check_fonts_in_directory(directory, characters, output_file):
    print(f"Font directory: {directory}")
    print(f"Required characters: {characters}")
    print()
    
    # Get all font files
    print("Scanning for font files...")
    font_files = get_all_font_files(directory)
    print(f"Found {len(font_files)} font files\n")
    
    missing_fonts = []
    valid_fonts = 0
    failed_fonts = 0
    
    # Prepare arguments for multiprocessing
    args_list = [(font_path, characters) for font_path in font_files]
    
    # Process fonts in parallel
    print("Checking character support...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(check_font_wrapper, args_list), 
                           total=len(font_files), desc="Processing"))
    
    # Analyze results
    for font_path, missing in results:
        font_name = os.path.basename(font_path)
        if missing == "load_failed":
            missing_fonts.append(f"{font_name}\tLOAD_FAILED")
            failed_fonts += 1
        elif missing:
            missing_chars = ''.join(missing)
            missing_fonts.append(f"{font_name}\tMISSING: {missing_chars}")
        else:
            valid_fonts += 1

    # Write fonts with missing characters or load failures to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Font Name\tIssue\n")
        for font in missing_fonts:
            f.write(f"{font}\n")

    print(f"\n{'='*60}")
    print("CHARACTER SUPPORT CHECK COMPLETE")
    print(f"{'='*60}")
    print(f"Total fonts checked: {len(font_files)}")
    print(f"Valid fonts (support all characters): {valid_fonts}")
    print(f"Fonts with missing characters: {len(missing_fonts) - failed_fonts}")
    print(f"Fonts that failed to load: {failed_fonts}")
    print(f"Results saved to: {output_file}")

# Specify the directory with fonts and the character set to check
font_directory = r'E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic\fonts_extracted'
output_file = os.path.join(r'E:\Projects\HTR_PR_Lab\HTR-Pipeline\data\synthetic', 'missing_characters_fonts.txt')
characters = ''' !"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'''

# Run the function
if __name__ == "__main__":
    check_fonts_in_directory(font_directory, characters, output_file)