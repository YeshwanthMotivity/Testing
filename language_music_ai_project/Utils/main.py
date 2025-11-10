import os
import argparse
import logging
from language_detector import process_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(folder_path):
    """
    Iterates through all media files in the given folder and processes them
    one by one using the language detection and segregation logic.
    """
    # Check if the path is a valid directory
    if not os.path.isdir(folder_path):
        logging.error(f"Path is not a valid folder: {folder_path}")
        return

    logging.info(f"Starting batch processing in: {folder_path}")

    # List of media extensions to process
    media_extensions = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')

    # Iterate through all files in the directory
    for file_name in os.listdir(folder_path):
        full_file_path = os.path.join(folder_path, file_name)

        # Check if the item is a file and has a supported media extension
        if os.path.isfile(full_file_path) and full_file_path.lower().endswith(media_extensions):
            try:
                # Step 1: Detect & segregate language (Sequential Processing)
                # Note: If process_file modifies the file's location, subsequent steps
                # would need to be aware of the new path, potentially by
                # receiving it as a return value from process_file.
                logging.info(f"Processing file: {file_name}")
                process_file(full_file_path)

            except Exception as e:
                logging.error(f"An error occurred while processing {file_name}: {e}")

        else:
            logging.info(f"Skipping non-file or unsupported file: {file_name}")

    logging.info("Batch processing complete! All media files have been classified and processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process media files in a given folder for language detection.")
    parser.add_argument("folder_path", type=str,
                        help="The path to the folder containing media files to process.")

    args = parser.parse_args()

    # The main function is now called with the folder path from command-line arguments
    main(args.folder_path)