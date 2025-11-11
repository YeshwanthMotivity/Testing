import os
import argparse
import logging
from language_detector import process_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Processor Definitions (for Modularity/Extensibility) ---
def run_language_detection(file_path):
    """
    Processor for language detection and segregation.
    This function calls the external `process_file` which is assumed to handle
    the core logic including moving the processed file.
    """
    logging.info(f"Processing language for: {file_path}")
    process_file(file_path)
    logging.info(f"Completed language processing for: {file_path}")

def run_music_separation_placeholder(file_path):
    """
    Placeholder processor for music separation.
    This function represents a future integration point.
    """
    logging.info(f"Future step: Music separation for {file_path} (not implemented yet).")
    # Example implementation might look like:
    # output_dir = os.path.join(os.getcwd(), "separated_files")
    # os.makedirs(output_dir, exist_ok=True)
    # separate_music(file_path, output_dir)

def run_deepfake_detection_placeholder(file_path):
    """
    Placeholder processor for deepfake detection.
    This function represents a future integration point.
    """
    logging.info(f"Future step: Deepfake detection for {file_path} (not implemented yet).")
    # Example implementation might look like:
    # detect_deepfake(file_path)

# --- Main Logic ---
def main(folder_path):
    """
    Iterates through all media files in the given folder and processes them
    one by one using a defined pipeline of processors.
    """
    if not os.path.isdir(folder_path):
        logging.error(f"Path is not a valid folder: {folder_path}")
        return

    logging.info(f"--- Starting batch processing in: {folder_path} ---")

    media_extensions = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')

    # Define the processing pipeline here.
    # The order of processors matters. If a processor modifies or moves the file,
    # subsequent processors need to be aware of that or operate on the original path.
    # Currently, `run_language_detection` (via `process_file`) moves the file.
    # If other processors need to run on the *original* file, they should be placed
    # *before* `run_language_detection` in this list.
    # To activate future steps, uncomment them below.
    processing_pipeline = [
        # run_music_separation_placeholder,   # Example of a future step to be added
        # run_deepfake_detection_placeholder, # Example of a future step to be added
        run_language_detection              # The currently active core processing step
    ]

    for file_name in os.listdir(folder_path):
        full_file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_file_path) and full_file_path.lower().endswith(media_extensions):
            logging.info(f"Attempting to process file: {file_name}")
            try:
                for processor in processing_pipeline:
                    # Each processor runs on the current full_file_path.
                    # If a processor (like run_language_detection) moves the file,
                    # subsequent processors in the same loop iteration for *this* file_path
                    # might encounter a FileNotFoundError if they expect the file to still be there.
                    # This design assumes either:
                    # 1. Only one processor in the pipeline modifies/moves the file, and it's the last one
                    #    or the subsequent ones don't rely on the original path.
                    # 2. Processors are designed to handle files being moved (e.g., by returning new paths).
                    processor(full_file_path)
            except FileNotFoundError:
                logging.error(f"File not found during processing of {file_name}. It might have been moved or deleted unexpectedly.")
            except PermissionError:
                logging.error(f"Permission denied while processing {file_name}. Check file permissions for the folder and the file.")
            except IOError as io_e:
                logging.error(f"I/O error during processing {file_name}: {io_e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing {file_name}: {e}")
        else:
            logging.info(f"Skipping non-file or unsupported file: {file_name}")

    logging.info("--- Batch processing complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process media files in a specified folder using a modular pipeline.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing media files to process.")
    args = parser.parse_args()

    main(args.folder_path)