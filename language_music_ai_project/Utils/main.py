import os
import argparse
import logging
from language_detector import process_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration variable for supported media extensions
MEDIA_EXTENSIONS = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')

def main(folder_path):
    """
    Iterates through all media files in the given folder and processes them
    one by one using the language detection and segregation logic.
    """
    # Check if the path is a valid directory
    if not os.path.isdir(folder_path):
        logging.error(f"Error: Path is not a valid folder: {folder_path}")
        return

    logging.info(f"--- Starting batch processing in: {folder_path} ---")

    # Iterate through all files in the directory
    for file_name in os.listdir(folder_path):
        full_file_path = os.path.join(folder_path, file_name)
        
        # Check if the item is a file and has a supported media extension
        if os.path.isfile(full_file_path) and full_file_path.lower().endswith(MEDIA_EXTENSIONS):
            try:
                # Step 1: Detect & segregate language (Sequential Processing)
                # Assuming process_file now returns the new path after processing/moving
                logging.info(f"Processing file: {file_name}")
                processed_file_path = process_file(full_file_path)

                # If process_file indeed moves the file, subsequent steps would use processed_file_path
                # Example:
                # if processed_file_path:
                #    output_dir = os.path.join(os.getcwd(), "separated_files")
                #    os.makedirs(output_dir, exist_ok=True)
                #    separate_music(processed_file_path, output_dir)
                #    detect_deepfake(processed_file_path)

            except Exception as e:
                logging.error(f"An error occurred while processing {file_name}: {e}")
                
        else:
            logging.info(f"Skipping non-file or unsupported file: {file_name}")

    logging.info("--- Batch processing complete! All media files have been classified and moved. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process media files in a given folder for language detection and segregation.")
    parser.add_argument("folder_path", type=str,
                        help="The path to the folder containing media files to process.")

    args = parser.parse_args()

    main(args.folder_path)