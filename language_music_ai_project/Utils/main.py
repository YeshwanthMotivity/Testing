import os
import argparse
import logging
from typing import Callable, List, Tuple

from language_detector import process_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Abstraction for processing steps
# This list holds functions that will be applied sequentially to each media file.
# Each function should accept a single string argument (the file path).
PROCESSING_STEPS: List[Callable[[str], None]] = [
    process_file,
    # Future processing steps (e.g., separate_music, detect_deepfake) can be added here
    # after their respective modules are imported and functions are defined to take
    # a single file path argument.
]

def main(folder_path: str, media_extensions: Tuple[str, ...]) -> None:
    """
    Iterates through all media files in the given folder and processes them
    one by one using the configured processing steps.
    """
    if not os.path.isdir(folder_path):
        logging.error(f"Error: Path is not a valid folder: {folder_path}")
        return

    logging.info(f"\n--- Starting batch processing in: {folder_path} ---")
    logging.info(f"Supported media extensions: {', '.join(media_extensions)}")

    for file_name in os.listdir(folder_path):
        full_file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_file_path) and full_file_path.lower().endswith(media_extensions):
            logging.info(f"Processing file: {file_name}")
            try:
                for step_func in PROCESSING_STEPS:
                    logging.info(f"  - Running step '{step_func.__name__}' for {file_name}")
                    step_func(full_file_path)
                logging.info(f"✅ Successfully processed {file_name}")
            except Exception as e:
                logging.error(f"❌ An error occurred while processing {file_name}: {e}")
        else:
            logging.info(f"⏭️ Skipping non-file or unsupported file: {file_name}")

    logging.info("\n--- Batch processing complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process media files for language detection and other tasks."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing media files to process."
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".mp3,.mp4,.mkv,.avi,.mov,.wav,.flac",
        help="Comma-separated list of media file extensions to process (e.g., .mp3,.mp4)."
    )

    args = parser.parse_args()

    # Convert extensions string to a tuple of lowercase strings
    processed_extensions = tuple(ext.strip().lower() for ext in args.extensions.split(','))

    main(args.folder_path, processed_extensions)