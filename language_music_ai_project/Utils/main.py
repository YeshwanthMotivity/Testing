import os
import argparse
import logging
from language_detector import process_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_language_detection(file_path):
    """
    Performs language detection and segregation on the given file.
    This function wraps the external `process_file` for pipeline integration.
    """
    logger.info(f"Detecting & segregating language for: {file_path}")
    process_file(file_path) # Assuming process_file handles moving the file
    logger.info(f"Language processing complete for: {file_path}")

# Define the processing pipeline.
# This list can be extended with more processor functions in the future.
# Each function in the pipeline should accept a file_path argument.
processing_pipeline = [
    process_language_detection,
    # Example for future integration:
    # separate_music_step,
    # detect_deepfake_step,
]

def main(folder_path):
    """
    Iterates through all media files in the given folder and processes them
    one by one using the configured processing pipeline.
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Error: Path is not a valid directory: {folder_path}")
        return

    logger.info(f"\n--- Starting batch processing in: {folder_path} ---")

    media_extensions = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')

    try:
        for file_name in os.listdir(folder_path):
            full_file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(full_file_path) and full_file_path.lower().endswith(media_extensions):
                logger.info(f"Processing file: {file_name}")
                try:
                    # Execute each step in the processing pipeline
                    for processor_func in processing_pipeline:
                        processor_func(full_file_path)
                except Exception as e:
                    # Catch general exceptions for individual file processing steps
                    logger.error(f"❌ An error occurred during processing pipeline for {file_name}: {e}")
            else:
                logger.info(f"⏭️ Skipping non-file or unsupported file: {file_name}")
    except FileNotFoundError:
        logger.error(f"Error: The directory '{folder_path}' does not exist.")
    except PermissionError:
        logger.error(f"Error: Permission denied when accessing '{folder_path}'.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during directory listing for '{folder_path}': {e}")

    logger.info("\n--- Batch processing complete! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process media files in a given folder for language detection and segregation."
    )
    parser.add_argument(
        "-p", "--path",
        type=str,
        help="The path to the folder containing media files to process.",
        required=True
    )

    args = parser.parse_args()
    main(args.path)