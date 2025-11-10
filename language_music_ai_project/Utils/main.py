import os
import argparse
import logging
# from language_detector import process_file # process_file is the core function
# from demcus import separate_music
# from deepfake_detector import detect_deepfake

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define media extensions as a module-level constant
MEDIA_EXTENSIONS = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')

# Mock process_file for demonstration purposes as its actual implementation is external
# In a real scenario, you would uncomment 'from language_detector import process_file'
def process_file(file_path):
    """
    Mock function for language detection and segregation.
    Assumes it performs a move operation, moving the processed file to a new location.
    In a real implementation, this function might return the new path of the moved file.
    """
    logging.info(f"Processing language for: {file_path}")
    # Simulate a move operation
    # For actual integration, the language_detector's process_file would handle this.
    # If subsequent steps need the new path, language_detector.process_file should return it.
    pass


def main(folder_path):
    """
    Iterates through all media files in the given folder and processes them 
    one by one using the language detection and segregation logic.
    """
    # Check if the path is a valid directory
    if not os.path.isdir(folder_path):
        logging.error(f"Error: Path is not a valid folder: {folder_path}")
        return

    logging.info(f"\n--- Starting batch processing in: {folder_path} ---")

    # Iterate through all files in the directory
    for file_name in os.listdir(folder_path):
        full_file_path = os.path.join(folder_path, file_name)
        
        # Check if the item is a file and has a supported media extension
        if os.path.isfile(full_file_path) and full_file_path.lower().endswith(MEDIA_EXTENSIONS):
            try:
                # Step 1: Detect & segregate language (Sequential Processing)
                # 'process_file' is expected to handle file movement or segregation internally.
                # If subsequent processing needs to operate on the file's new location
                # or a categorized copy, `process_file` should return the new path,
                # or subsequent steps must infer it based on 'process_file's' logic.
                process_file(full_file_path)
                logging.info(f"Successfully processed: {file_name}")

                # --- Integration Points (Optional/Future Steps) ---
                # These steps would typically happen *after* language segregation 
                # but *before* the file is moved, or you would process the file's 
                # copy from the output directory created by 'process_file'.

                # Example for future integration, assuming `process_file` moved `full_file_path`
                # to `new_processed_path`:
                # new_processed_path = process_file(full_file_path) # If process_file returns new path
                # if new_processed_path:
                #    output_dir = os.path.join(os.getcwd(), "separated_music_files")
                #    os.makedirs(output_dir, exist_ok=True)
                #    separate_music(new_processed_path, output_dir)
                #    detect_deepfake(new_processed_path)

            except Exception as e:
                logging.exception(f"An error occurred while processing {file_name}")
                
        else:
            logging.info(f"Skipping non-file or unsupported file: {file_name}")

    logging.info("\n--- Batch processing complete! All media files have been classified and moved. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process media files in a specified folder for language detection and other analyses."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The path to the folder containing media files to be processed."
    )
    args = parser.parse_args()
    
    main(args.folder_path)