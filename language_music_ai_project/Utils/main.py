import os
import argparse
import logging
from language_detector import process_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global constant for supported media extensions
MEDIA_EXTENSIONS = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')

def main(folder_path: str) -> None:
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
                # The loop waits here until process_file completes the move operation.
                logging.info(f"Processing file: {file_name}")
                process_file(full_file_path)
                logging.info(f"Successfully processed and moved: {file_name}")

                # --- Integration Points (Optional/Future Steps) ---
                # If 'separate_music' or 'detect_deepfake' were to be integrated,
                # they would likely operate on the original file path BEFORE
                # 'process_file' moves it, or they would operate on the file
                # once it's moved to its segregated output directory.
                # Example:
                # separated_output_dir = os.path.join(os.getcwd(), "separated_music_files")
                # os.makedirs(separated_output_dir, exist_ok=True)
                # separate_music(full_file_path, separated_output_dir)
                # detect_deepfake(full_file_path)

            except Exception as e:
                logging.error(f"An error occurred while processing {file_name}: {e}")
                
        else:
            logging.debug(f"Skipping non-file or unsupported file: {file_name}")

    logging.info("\n--- Batch processing complete! All media files have been classified and moved. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process media files in a given folder for language detection and segregation.")
    parser.add_argument("folder_path", type=str, 
                        help="The path to the folder containing media files to process.")
    
    args = parser.parse_args()
    
    main(args.folder_path)