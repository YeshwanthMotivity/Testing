import os
import shutil
import subprocess
import logging
import tempfile
import json
import hashlib

# --- Configuration (Moved to a separate config.py in a real-world scenario) ---
# For this exercise, we keep it integrated but clearly marked.

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Change to logging.DEBUG for verbose output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("language_detector.log"),
        logging.StreamHandler()
    ]
)
logging.info("--- Initializing Language Detector Script ---")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
TEMP_DEMUCS_DIR = os.path.join(BASE_DIR, "temp_demucs") # Still here if Demucs is re-integrated

LANGUAGES_DIR = os.path.join(OUTPUT_DIR, "Languages")
BGM_DIR = os.path.join(OUTPUT_DIR, "bgm")
REMIX_DIR = os.path.join(OUTPUT_DIR, "remix")

# Persistent hash storage file
HASH_STORAGE_FILE = os.path.join(BASE_DIR, "processed_hashes.json")

# Language map
LANG_MAP = {
    "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
    "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
    "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
    "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
}

# Whisper model parameters
WHISPER_MODEL_NAME = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# --- End Configuration ---


# ---------- Stability ----------
# DetectorFactory.seed = 0 is for langdetect.detect_langs, which is now secondary
# to faster-whisper's segment language detection for primary classification.
# However, if langdetect.detect_langs is used as a fallback or for finer
# multi-language detection on the full text (which is no longer the primary method
# for overall classification), keeping it can still be useful for consistency.
# For current code, it's not strictly necessary as faster-whisper handles primary language.
# Removed since langdetect.detect_langs is not actively used for primary classification.

# ---------- Global state for processed file hashes ----------
PROCESSED_FILE_HASHES = set()

def load_processed_hashes():
    """Loads processed file hashes from persistent storage."""
    global PROCESSED_FILE_HASHES
    if os.path.exists(HASH_STORAGE_FILE):
        try:
            with open(HASH_STORAGE_FILE, 'r', encoding='utf-8') as f:
                hashes = json.load(f)
                PROCESSED_FILE_HASHES = set(hashes)
            logging.info(f"Loaded {len(PROCESSED_FILE_HASHES)} processed hashes from {HASH_STORAGE_FILE}")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {HASH_STORAGE_FILE}: {e}")
            PROCESSED_FILE_HASHES = set() # Reset to empty if file is corrupt
        except Exception as e:
            logging.error(f"An unexpected error occurred loading hashes: {e}")
            PROCESSED_FILE_HASHES = set()
    else:
        logging.info(f"No hash storage file found at {HASH_STORAGE_FILE}.")

def save_processed_hashes():
    """Saves processed file hashes to persistent storage."""
    try:
        with open(HASH_STORAGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(PROCESSED_FILE_HASHES), f, indent=4)
        logging.info(f"Saved {len(PROCESSED_FILE_HASHES)} processed hashes to {HASH_STORAGE_FILE}")
    except Exception as e:
        logging.error(f"Error saving processed hashes to {HASH_STORAGE_FILE}: {e}")

# Call on script start
load_processed_hashes()

# Create output directories
os.makedirs(LANGUAGES_DIR, exist_ok=True)
os.makedirs(BGM_DIR, exist_ok=True)
os.makedirs(REMIX_DIR, exist_ok=True)
os.makedirs(TEMP_DEMUCS_DIR, exist_ok=True) # If Demucs is re-integrated in future.

logging.info(f"BASE_DIR: {BASE_DIR}")
logging.info(f"Output directories setup: Languages={LANGUAGES_DIR}, BGM={BGM_DIR}, REMIX={REMIX_DIR}")


# ---------- Whisper Model Loading ----------
from faster_whisper import WhisperModel
logging.info("📥 Loading Whisper model...")
try:
    model = WhisperModel(
        WHISPER_MODEL_NAME,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE
    )
    logging.info(f"✅ Whisper model loaded (Model: {WHISPER_MODEL_NAME}, Device: {WHISPER_DEVICE}, Compute: {WHISPER_COMPUTE_TYPE})")
except Exception as e:
    logging.error(f"❌ Failed to load Whisper model: {e}")
    exit(1) # Exit if model can't be loaded


# ---------- Helper: File Hashing ----------
def get_file_hash(file_path):
    """Generates an MD5 hash for a file, handling large files efficiently."""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536) # Read in 64KB chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except FileNotFoundError:
        logging.error(f"File not found for hashing: {file_path}")
        return None
    except Exception as e:
        logging.error(f"❌ ERROR: Could not get hash for {file_path}. Error: {e}")
        return None

# ---------- Helper: File Moving with Conflict Resolution ----------
def _move_file_with_rename_on_conflict(source_path, destination_dir):
    """
    Moves a file to a destination directory, resolving filename conflicts
    by appending '_copyX' if the file already exists.
    """
    os.makedirs(destination_dir, exist_ok=True)
    
    base_name = os.path.basename(source_path)
    destination_path = os.path.join(destination_dir, base_name)
    
    logging.debug(f"Initial destination path: {destination_path}")

    if os.path.exists(destination_path):
        base, ext = os.path.splitext(destination_path)
        i = 1
        while os.path.exists(f"{base}_copy{i}{ext}"):
            i += 1
        destination_path = f"{base}_copy{i}{ext}"
        logging.warning(f"File conflict detected. Renaming to: {destination_path}")

    try:
        shutil.move(source_path, destination_path)
        logging.info(f"✅ File moved successfully → {destination_path}")
        return destination_path
    except Exception as e:
        logging.error(f"❌ ERROR: File move failed for {source_path} to {destination_path}. Error: {e}")
        raise # Re-raise the exception to indicate failure


# ---------- Audio extractor ----------
def extract_audio(video_path):
    """
    Extracts mono 16kHz audio using ffmpeg to a unique temporary file.
    Returns the path to the extracted audio file.
    """
    logging.info(f"🎬 Extracting audio from video: {video_path}")
    
    temp_audio_file = None
    try:
        # Create a unique temporary file for audio extraction
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP_DEMUCS_DIR)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close() # Close the file handle so ffmpeg can write to it

        command = [
            "ffmpeg", "-i", video_path,
            "-ar", "16000", "-ac", "1",
            temp_audio_path, "-y"
        ]
        logging.debug(f"FFmpeg command: {' '.join(command)}")
        
        # Add check=True for robust error handling
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if os.path.exists(temp_audio_path):
            logging.info(f"✅ Extracted audio → {temp_audio_path}")
            return temp_audio_path
        else:
            logging.error(f"❌ ERROR: Audio extraction failed for {video_path}. Output file not found: {temp_audio_path}")
            return None
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ ERROR: FFmpeg failed for {video_path} with exit code {e.returncode}. Command: {e.cmd}. Stderr: {e.stderr.decode().strip()}")
        if temp_audio_file and os.path.exists(temp_audio_file.name):
            os.remove(temp_audio_file.name) # Clean up partial temp file
        return None
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during audio extraction for {video_path}: {e}")
        if temp_audio_file and os.path.exists(temp_audio_file.name):
            os.remove(temp_audio_file.name)
        return None


# ---------- Transcribe (Optimized Function for faster-whisper) ----------
def transcribe_audio(audio_path):
    """
    Transcribe entire audio using faster-whisper's optimized engine,
    reporting language detections at segment level.
    """
    logging.info(f"--- Starting transcription for: {audio_path}")
    
    texts = []
    langs_detected = {} # Stores language and weighted counts (by word count)
    
    # faster-whisper model.transcribe returns segments and info
    segments_generator, info = model.transcribe(audio_path, beam_size=5)

    logging.debug(f"Model detected dominant language for entire file: {info.language} with probability {info.language_probability:.2f}")
    logging.debug(f"Model transcription speed RTF (Real-Time Factor, lower is better): {info.initial_prompt_time}") # initial_prompt_time is not RTF, this was a previous mistake. info.time_precision is for time segments.

    segment_count = 0
    for segment in segments_generator:
        segment_count += 1
        seg_text = segment.text.strip()
        
        # Crucially, use segment.language for individual segment language detection
        seg_lang = segment.language
        
        if seg_text:
            word_count = len(seg_text.split())
            texts.append(seg_text)
            langs_detected[seg_lang] = langs_detected.get(seg_lang, 0) + word_count
            logging.debug(f"   🎧 Segment {segment_count}: Detected {seg_lang}, Words: {word_count}. Text: '{seg_text[:70]}...'")
        else:
            logging.debug(f"   🎧 Segment {segment_count}: Skipped (No text found).")

    logging.info(f"--- Total segments processed: {segment_count}")
    return " ".join(texts), langs_detected


# ---------- Core File Processing Logic ----------
def process_file(file_path):
    logging.info(f"\n=======================================================")
    logging.info(f"🚀 Processing file: {os.path.basename(file_path)}")
    logging.info(f"   FULL PATH: {file_path}")
    logging.info(f"=======================================================")
    
    # 1. Check for duplicates (highest precedence)
    file_hash = get_file_hash(file_path)
    if file_hash:
        logging.debug(f"File hash generated: {file_hash}")
        if file_hash in PROCESSED_FILE_HASHES:
            logging.warning("⚠️ DUPLICATE DETECTED! File with this hash has already been processed.")
            try:
                os.remove(file_path)
                logging.info(f"🗑️ Successfully deleted duplicate file: {file_path}")
                save_processed_hashes() # Save after removal
            except Exception as e:
                logging.error(f"❌ ERROR: Failed to delete duplicate file: {file_path}. Error: {e}")
            return # Exit the function immediately
        else:
            PROCESSED_FILE_HASHES.add(file_hash)
            logging.info("✅ File is unique. Proceeding with processing.")
            # We save the hash once the file is successfully classified and moved later.

    # 2. Filename-based Remix Check (Pre-transcription)
    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    if " X " in file_name_no_ext.upper():
        logging.info("✅ CLASSIFICATION (Filename): REMIX. Reason: Filename contains ' X ' pattern.")
        
        try:
            _move_file_with_rename_on_conflict(file_path, REMIX_DIR)
            save_processed_hashes() # Save after successful move
        except Exception as e:
            logging.error(f"❌ ERROR: Failed to move file '{file_path}' after filename classification. Error: {e}")
        return # Exit the function after moving

    # 3. Proceed with transcription for other files
    original_audio_path_is_temp = False
    audio_path = file_path
    
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
        logging.info(f"File recognized as VIDEO ({file_ext}). Extracting audio.")
        extracted_audio_path = extract_audio(file_path)
        if extracted_audio_path:
            audio_path = extracted_audio_path
            original_audio_path_is_temp = True
        else:
            logging.error(f"❌ Skipping transcription: Audio extraction failed for {file_path}.")
            if file_hash: PROCESSED_FILE_HASHES.remove(file_hash) # Remove hash if not processed
            save_processed_hashes()
            return
    else:
        logging.info(f"File recognized as AUDIO ({file_ext} or unknown). Using file directly.")

    text, lang_weights = transcribe_audio(audio_path)
    
    # Clean up temporary audio file if it was created
    if original_audio_path_is_temp and audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            logging.debug(f"Cleaned up temporary audio file: {audio_path}")
        except Exception as e:
            logging.error(f"❌ ERROR: Failed to remove temporary audio file {audio_path}: {e}")

    # Process detected languages and text
    langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
    detected_langs = [l for l, _ in langs_sorted if l != "unknown"] # Exclude 'unknown' from detected_langs for classification logic

    logging.debug(f"Sorted language weights (Word Count): {langs_sorted}")
    logging.info(f"🌍 Languages detected with weights: {lang_weights}")

    cleaned_text = text.replace("♪", "").replace("♫", "").strip()
    word_count = len(cleaned_text.split())
    
    logging.debug(f"Total cleaned transcription length: {len(cleaned_text)} characters")
    logging.info(f"🔍 Word count from transcription: {word_count}")

    # 4. Transcription-based Classification Logic (Refined Hierarchy)
    base_folder = None
    
    # Highest priority after filename remix/duplicates: BGM (low word count)
    if word_count <= 10:
        base_folder = BGM_DIR
        logging.info("➡️ CLASSIFICATION: PURE BGM. Reason: Word count <= 10.")
    # Next priority: Hindi/Urdu special handling (if intended to be classified as 'vocals' within Hindi)
    elif "hi" in detected_langs and "ur" in detected_langs:
        # If the intent is to treat hi+ur as a specific "Hindi vocals" category rather than a general remix
        primary_lang = "hi" # Arbitrarily pick Hindi as the primary folder name
        language_name = LANG_MAP.get(primary_lang, primary_lang)
        base_folder = os.path.join(LANGUAGES_DIR, language_name, "vocals")
        logging.info(f"➡️ CLASSIFICATION: VOCALS ({language_name}). Reason: Hindi/Urdu mix detected.")
    # Next priority: Generic multi-language remix
    elif len(detected_langs) >= 2:
        base_folder = REMIX_DIR
        logging.info(f"➡️ CLASSIFICATION: REMIX. Reason: Multiple distinct languages detected ({', '.join(detected_langs)}).")
    # Lowest priority: Single dominant language (vocals)
    elif detected_langs: # If there's at least one language detected
        primary_lang = detected_langs[0]
        language_name = LANG_MAP.get(primary_lang, primary_lang)
        base_folder = os.path.join(LANGUAGES_DIR, language_name, "vocals")
        logging.info(f"➡️ CLASSIFICATION: VOCALS ({language_name}). Reason: Single primary language detected.")
    else: # Fallback if no discernible language or text
        base_folder = os.path.join(LANGUAGES_DIR, "unknown") # Or another default folder
        logging.info(f"➡️ CLASSIFICATION: UNKNOWN. Reason: No significant language or text detected.")


    if not base_folder:
        logging.error("❌ ERROR: Classification logic failed to assign a base folder.")
        if file_hash: PROCESSED_FILE_HASHES.remove(file_hash) # Remove hash if not processed
        save_processed_hashes()
        return

    # 5. Move file to its final destination
    try:
        _move_file_with_rename_on_conflict(file_path, base_folder)
        save_processed_hashes() # Save after successful move
    except Exception as e:
        logging.error(f"❌ ERROR: Failed to move file '{file_path}' to '{base_folder}'. Error: {e}")
        if file_hash: PROCESSED_FILE_HASHES.remove(file_hash) # Remove hash if not processed
        save_processed_hashes()
# This ensures that `save_processed_hashes()` is called at the end of `process_file`
# after a file has been either deleted (as a duplicate) or successfully moved.

# If you need to run this script as a standalone (e.g., to test process_file),
# you would add a main block like this:
# if __name__ == "__main__":
#     # Example usage
#     test_file_path = "path/to/your/test_video.mp4"
#     if os.path.exists(test_file_path):
#         process_file(test_file_path)
#     else:
#         logging.warning(f"Test file not found: {test_file_path}")
#     
#     # Ensure hashes are saved when the script exits, even if not explicitly
#     # called after every single file. A final save can be good.
#     save_processed_hashes()