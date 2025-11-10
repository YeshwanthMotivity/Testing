# from language_detector import process_file
# # from demcus import separate_music
# import os
# # from deepfake_detector import detect_deepfake

# def main(file_path):
#     # Step 1: Detect & segregate language
#     process_file(file_path)

#     # # Step 2: Separate BGM & Lyrics if video
    
#     # Create an output folder
#     output_dir = os.path.join(os.getcwd(), "separated_files")
#     os.makedirs(output_dir, exist_ok=True)

#     # # Call music separation with file path and output dir
#     # separate_music(file_path, output_dir)

#     # # Step 3: Detect AI vs Real
#     # detect_deepfake(file_path)

# if __name__ == "__main__":
#     file_path = r"C:\Users\Yeshwanth\Documents\songs\ORQUESTRA MALDITA (BRAZILIAN PHONK).mp3"
#     main(file_path)


# main.py

# main.py

from language_detector import process_file # process_file is the core function
import os
# from demcus import separate_music
# from deepfake_detector import detect_deepfake

def main(folder_path):
    """
    Iterates through all media files in the given folder and processes them 
    one by one using the language detection and segregation logic.
    """
    # Check if the path is a valid directory
    if not os.path.isdir(folder_path):
        print(f"Error: Path is not a valid folder: {folder_path}")
        return

    print(f"\n--- Starting batch processing in: {folder_path} ---")

    # List of media extensions to process
    media_extensions = ('.mp3', '.mp4', '.mkv', '.avi', '.mov', '.wav', '.flac')

    # Iterate through all files in the directory
    for file_name in os.listdir(folder_path):
        full_file_path = os.path.join(folder_path, file_name)
        
        # Check if the item is a file and has a supported media extension
        if os.path.isfile(full_file_path) and full_file_path.lower().endswith(media_extensions):
            try:
                # Step 1: Detect & segregate language (Sequential Processing)
                # The loop waits here until process_file completes the move operation.
                process_file(full_file_path)

                # --- Integration Points (Optional/Future Steps) ---
                # These steps would typically happen *after* language segregation 
                # but *before* the file is moved, or you would process the file's 
                # copy from the output directory.

                # output_dir = os.path.join(os.getcwd(), "separated_files")
                # os.makedirs(output_dir, exist_ok=True)
                
                # # Call music separation with file path and output dir
                # # separate_music(full_file_path, output_dir)

                # # Step 3: Detect AI vs Real
                # # detect_deepfake(full_file_path)

            except Exception as e:
                print(f"❌ An error occurred while processing {file_name}: {e}")
                
        else:
            print(f"⏭️ Skipping non-file or unsupported file: {file_name}")

    print("\n--- Batch processing complete! All media files have been classified and moved. ---")


if __name__ == "__main__":
    # ⚠️ UPDATE THIS LINE to the path of your folder containing multiple files
    folder_path_to_process = r"C:\Users\Yeshwanth\Documents\songs"
    
    # The main function is now called with the folder path
    main(folder_path_to_process)