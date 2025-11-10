import argparse
import logging
import os
from language_detector import process_file

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global constant for supported media extensions
SUPPORTED_MEDIA_EXTENSIONS = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')

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
        if os.path.isfile(full_file_path) and full_file_path.lower().endswith(SUPPORTED_MEDIA_EXTENSIONS):
            try:
                # Step 1: Detect & segregate language (Sequential Processing)
                # The loop waits here until process_file completes its operation.
                process_file(full_file_path)
                logging.info(f"✅ Successfully processed: {file_name}")

            except Exception as e:
                logging.error(f"❌ An error occurred while processing {file_name}: {e}")
                
        else:
            logging.info(f"⏭️ Skipping non-file or unsupported file: {file_name}")

    logging.info("\n--- Batch processing complete! All media files have been classified and moved. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process media files in a given folder for language detection and segregation.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing media files to process.")
    
    args = parser.parse_args()
    
    main(args.folder_path)