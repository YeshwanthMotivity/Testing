import os
import argparse
from language_detector import process_file

def find_media_files(folder_path, media_extensions):
    """
    Discovers and returns a list of paths of supported media files within the given folder.
    Raises ValueError if the folder_path is not a valid directory.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Error: Path is not a valid folder: {folder_path}")

    print(f"\n--- Discovering media files in: {folder_path} ---")
    
    found_files = []
    for file_name in os.listdir(folder_path):
        full_file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(full_file_path) and full_file_path.lower().endswith(media_extensions):
            found_files.append(full_file_path)
        else:
            print(f"⏭️ Skipping non-file or unsupported file: {file_name}")
    
    return found_files

def main(folder_path):
    """
    Iterates through all media files in the given folder and processes them 
    one by one using the language detection and segregation logic.
    """
    
    # List of media extensions to process
    media_extensions = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')

    try:
        media_files_to_process = find_media_files(folder_path, media_extensions)
    except ValueError as e:
        print(e)
        return

    if not media_files_to_process:
        print(f"No supported media files found in {folder_path}.")
        return

    print(f"\n--- Starting batch processing of {len(media_files_to_process)} files ---")

    for full_file_path in media_files_to_process:
        file_name = os.path.basename(full_file_path)
        try:
            # Step 1: Detect & segregate language (Sequential Processing)
            # The process_file function is expected to move the file after processing.
            # Any subsequent processing steps would need to account for the new file path.
            print(f"✨ Processing: {file_name}")
            process_file(full_file_path)
            print(f"✅ Successfully processed: {file_name}")

        except Exception as e:
            print(f"❌ An error occurred while processing {file_name}: {e}")
                
    print("\n--- Batch processing complete! All media files have been classified and moved. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process media files for language detection and segregation.")
    parser.add_argument("folder_path", type=str, 
                        help="The path to the folder containing media files to process.")
    
    args = parser.parse_args()
    
    main(args.folder_path)