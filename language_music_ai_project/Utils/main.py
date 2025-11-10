import os
import argparse
import logging

from language_detector import process_file

# Global constant for media extensions
MEDIA_EXTENSIONS = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')

def _process_single_file(full_file_path: str):
    """
    Processes a single media file using the language detection and segregation logic.
    Logs success or failure.
    """
    file_name = os.path.basename(full_file_path)
    try:
        logging.info(f"Processing file: {file_name}")
        process_file(full_file_path)
        logging.info(f"✅ Successfully processed {file_name}")
    except Exception as e:
        logging.error(f"❌ An error occurred while processing {file_name}: {e}")

def main(folder_path: str):
    """
    Iterates through all media files in the given folder and processes them
    one by one using the language detection and segregation logic.
    """
    if not os.path.isdir(folder_path):
        logging.error(f"Error: Path is not a valid folder: {folder_path}")
        return

    logging.info(f"--- Starting batch processing in: {folder_path} ---")

    for file_name in os.listdir(folder_path):
        full_file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_file_path) and full_file_path.lower().endswith(MEDIA_EXTENSIONS):
            _process_single_file(full_file_path)
        else:
            logging.info(f"⏭️ Skipping non-file or unsupported file: {file_name}")

    logging.info("--- Batch processing complete! All media files have been classified and moved. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process media files in a folder for language detection.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing media files.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (shows info messages like skips).")

    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    main(args.folder_path)