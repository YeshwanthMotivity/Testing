import os
import shutil
import subprocess
import hashlib
import json
import tempfile
import configparser
import logging
import argparse
from typing import Dict, List, Tuple, Set, Optional

# --- 1. Configuration Loading ---
config = configparser.ConfigParser()
config_file_path = 'config.ini' # Assumes config.ini is in the same directory as the script
if not os.path.exists(config_file_path):
    # Create a default config.ini if it doesn't exist
    config['PATHS'] = {
        'output_base_dir': 'data/output',
        'processed_hashes_file': 'processed_hashes.json',
        'temp_audio_prefix': 'temp_audio_'
    }
    config['WHISPER'] = {
        'model_size': 'small',
        'device': 'cpu',
        'compute_type': 'int8',
        'min_word_count_bgm': '10'
    }
    config['CLASSIFICATION'] = {
        'remix_filename_pattern': ' X '
    }
    config['LANG_MAP'] = {
        "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
        "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
        "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
        "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
    }
    with open(config_file_path, 'w', encoding='utf-8') as f:
        config.write(f)
    print(f"Created default '{config_file_path}'. Please review and modify if needed.")

config.read(config_file_path)

# --- 2. Logging Setup ---
# Logging level will be set by argparse later
logging.basicConfig(
    level=logging.INFO, # Default to INFO, will be overridden by --log-level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("language_detector.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 3. Stability ---
DetectorFactory.seed = 0
logger.debug("Stability seed for langdetect set to 0.")

# --- 4. Global Paths and Configurations ---
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE_DIR: str = os.path.join(BASE_DIR, config.get('PATHS', 'output_base_dir'))
PROCESSED_HASHES_FILE: str = os.path.join(BASE_DIR, config.get('PATHS', 'processed_hashes_file'))
TEMP_AUDIO_PREFIX: str = config.get('PATHS', 'temp_audio_prefix')

LANGUAGES_DIR: str = os.path.join(OUTPUT_BASE_DIR, "Languages")
BGM_DIR: str = os.path.join(OUTPUT_BASE_DIR, "bgm")
REMIX_DIR: str = os.path.join(OUTPUT_BASE_DIR, "remix")

WHISPER_MODEL_SIZE: str = config.get('WHISPER', 'model_size')
WHISPER_DEVICE: str = config.get('WHISPER', 'device')
WHISPER_COMPUTE_TYPE: str = config.get('WHISPER', 'compute_type')
MIN_WORD_COUNT_BGM: int = config.getint('WHISPER', 'min_word_count_bgm')

REMIX_FILENAME_PATTERN: str = config.get('CLASSIFICATION', 'remix_filename_pattern')

LANG_MAP: Dict[str, str] = dict(config.items('LANG_MAP'))

# Ensure output directories exist
os.makedirs(LANGUAGES_DIR, exist_ok=True)
os.makedirs(BGM_DIR, exist_ok=True)
os.makedirs(REMIX_DIR, exist_ok=True)
logger.info(f"Output directories ensured: Languages={LANGUAGES_DIR}, BGM={BGM_DIR}, REMIX={REMIX_DIR}")

# --- 5. Processed File Hashes Persistence ---
PROCESSED_FILE_HASHES: Set[str] = set()

def load_processed_hashes(file_path: str) -> Set[str]:
    """Loads processed file hashes from a JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                hashes = json.load(f)
                logger.info(f"Loaded {len(hashes)} processed hashes from {file_path}")
                return set(hashes)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}. Starting with empty hashes.")
            return set()
        except Exception as e:
            logger.error(f"Error loading processed hashes from {file_path}: {e}. Starting with empty hashes.")
            return set()
    logger.info(f"No processed hashes file found at {file_path}. Starting with empty hashes.")
    return set()

def save_processed_hashes(file_path: str, hashes: Set[str]) -> None:
    """Saves processed file hashes to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(list(hashes), f, indent=4)
        logger.info(f"Saved {len(hashes)} processed hashes to {file_path}")
    except Exception as e:
        logger.error(f"Error saving processed hashes to {file_path}: {e}")

# Load hashes at startup
PROCESSED_FILE_HASHES = load_processed_hashes(PROCESSED_HASHES_FILE)

# --- 6. Whisper Model Loading ---
logger.info("📥 Loading Whisper model...")
try:
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE
    )
    logger.info(f"✅ Whisper model loaded (Model: {WHISPER_MODEL_SIZE}, Device: {WHISPER_DEVICE}, Compute: {WHISPER_COMPUTE_TYPE})")
except Exception as e:
    logger.critical(f"❌ Failed to load Whisper model: {e}")
    exit(1) # Exit if model cannot be loaded

# --- 7. Helper Functions ---

def get_file_hash(file_path: str) -> Optional[str]:
    """Generates an MD5 hash for a file, handling large files efficiently."""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"❌ ERROR: Could not get hash for {file_path}. Error: {e}")
        return None

def extract_audio(video_path: str) -> Optional[str]:
    """Extracts mono 16kHz audio using ffmpeg to a temporary file."""
    temp_audio_file: Optional[str] = None
    try:
        # Create a named temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=TEMP_AUDIO_PREFIX) as tmp:
            temp_audio_file = tmp.name
        
        logger.info(f"🎬 Extracting audio from video: {video_path} to temporary file: {temp_audio_file}")
        command = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", temp_audio_file, "-y"]
        logger.debug(f"FFmpeg command: {' '.join(command)}")
        
        process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        
        if process.returncode == 0 and os.path.exists(temp_audio_file):
            logger.info(f"✅ Extracted audio → {temp_audio_file}")
            return temp_audio_file
        else:
            logger.error(f"❌ ERROR: Audio extraction failed for {video_path}. FFmpeg exited with code {process.returncode}.")
            if temp_audio_file and os.path.exists(temp_audio_file):
                os.remove(temp_audio_file) # Clean up failed temp file
            return None
    except Exception as e:
        logger.error(f"❌ ERROR: Exception during audio extraction for {video_path}: {e}")
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        return None

def transcribe_audio(audio_path: str) -> Tuple[str, Dict[str, int]]:
    """Transcribe entire audio using faster-whisper's optimized engine."""
    logger.debug(f"Starting transcription for: {audio_path}")
    
    segments, info = model.transcribe(audio_path, beam_size=5)

    logger.debug(f"Model detected language for entire file: {info.language}")
    # language_probability is usually less than RTF for speed, let's just log language_probability
    logger.debug(f"Model transcription language probability: {info.language_probability:.2f}")

    langs_detected: Dict[str, int] = {}
    texts: List[str] = []
    segment_count: int = 0
    
    for segment in segments:
        segment_count += 1
        seg_text: str = segment.text.strip()
        seg_lang: str = info.language # Use the language detected for the whole file
        
        if seg_text:
            word_count: int = len(seg_text.split())
            texts.append(seg_text)
            langs_detected[seg_lang] = langs_detected.get(seg_lang, 0) + word_count
            logger.debug(f"   🎧 Segment {segment_count}: Detected {seg_lang}, Words: {word_count}. Text: '{seg_text[:50]}...'")
        else:
            logger.debug(f"   🎧 Segment {segment_count}: Skipped (No text found).")

    logger.debug(f"Total segments processed: {segment_count}")
    return " ".join(texts), langs_detected

def _handle_duplicate_filename(destination: str) -> str:
    """Generates a unique filename by appending '_copyX' if the destination exists."""
    if not os.path.exists(destination):
        return destination
    
    base, ext = os.path.splitext(destination)
    i = 1
    while os.path.exists(f"{base}_copy{i}{ext}"):
        i += 1
    new_destination: str = f"{base}_copy{i}{ext}"
    logger.debug(f"Renaming required. New destination path: {new_destination}")
    return new_destination

def _move_file(source_path: str, destination_dir: str) -> bool:
    """Moves a file to a destination directory, handling existing filenames."""
    os.makedirs(destination_dir, exist_ok=True)
    destination: str = os.path.join(destination_dir, os.path.basename(source_path))
    destination = _handle_duplicate_filename(destination) # Get unique path
    
    logger.info(f"Attempting to move '{os.path.basename(source_path)}' to '{destination}'")
    try:
        shutil.move(source_path, destination)
        logger.info(f"✅ Final move successful → {destination}")
        return True
    except Exception as e:
        logger.error(f"❌ ERROR: File move failed for {source_path}. Error: {e}")
        return False

def _classify_file(text: str, lang_weights: Dict[str, int], file_name_no_ext: str) -> Tuple[Optional[str], str]:
    """
    Classifies a file based on transcription results and filename.
    Returns (base_folder, classification_reason).
    """
    
    # Filename-based Remix Check is done *before* transcription in process_file.
    # This helper focuses on transcription-based classification.

    cleaned_text: str = text.replace("♪", "").replace("♫", "").strip()
    word_count: int = len(cleaned_text.split())
    logger.debug(f"Total cleaned transcription length: {len(cleaned_text)} characters, Word count: {word_count}")

    langs_sorted: List[Tuple[str, int]] = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
    detected_langs: List[str] = [l for l, _ in langs_sorted]
    logger.debug(f"Sorted language weights (Word Count): {langs_sorted}")

    base_folder: Optional[str] = None
    classification_reason: str = "Unknown"

    if "hi" in detected_langs and "ur" in detected_langs and word_count > MIN_WORD_COUNT_BGM:
        primary_lang: str = "hi"
        language: str = LANG_MAP.get(primary_lang, primary_lang)
        base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
        classification_reason = f"VOCALS ({language}). Reason: Hindi/Urdu Mix and word count > {MIN_WORD_COUNT_BGM}."
    elif word_count <= MIN_WORD_COUNT_BGM:
        base_folder = BGM_DIR
        classification_reason = f"PURE BGM. Reason: Word count <= {MIN_WORD_COUNT_BGM}."
    elif len(detected_langs) >= 2:
        base_folder = REMIX_DIR
        classification_reason = f"REMIX. Reason: Multiple Languages detected ({', '.join(detected_langs)})."
    else:
        primary_lang = detected_langs[0] if detected_langs else "unknown"
        language = LANG_MAP.get(primary_lang, primary_lang)
        base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
        classification_reason = f"VOCALS ({language}). Reason: Primary language detected."

    logger.info(f"➡️ CLASSIFICATION: {classification_reason}")
    if not base_folder:
        logger.error("❌ Classification logic failed to assign a base folder.")
    return base_folder, classification_reason

def process_file(file_path: str) -> None:
    logger.info(f"\n=======================================================")
    logger.info(f"🚀 Processing file: {os.path.basename(file_path)}")
    logger.info(f"   FULL PATH: {file_path}")
    logger.info(f"=======================================================")
    
    # CHECK FOR DUPLICATES BEFORE ANYTHING ELSE
    file_hash: Optional[str] = get_file_hash(file_path)
    if file_hash:
        logger.debug(f"File hash generated: {file_hash}")
        if file_hash in PROCESSED_FILE_HASHES:
            logger.warning("⚠️ DUPLICATE DETECTED! File with this hash has already been processed.")
            try:
                os.remove(file_path)
                logger.info(f"🗑️ Successfully deleted duplicate file: {file_path}")
            except Exception as e:
                logger.error(f"❌ ERROR: Failed to delete duplicate file: {file_path}. Error: {e}")
            return # Exit the function immediately
        else:
            PROCESSED_FILE_HASHES.add(file_hash)
            logger.info("✅ File is unique. Proceeding with processing.")
    else:
        logger.error(f"❌ Could not generate hash for {file_path}. Cannot reliably check for duplicates.")

    file_name_no_ext: str = os.path.splitext(os.path.basename(file_path))[0]

    # Filename-based Remix Check (Short-circuit before transcription)
    if REMIX_FILENAME_PATTERN in file_name_no_ext.upper():
        logger.info(f"✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains configured pattern '{REMIX_FILENAME_PATTERN}'.")
        target_dir: str = REMIX_DIR
        if _move_file(file_path, target_dir):
            logger.info("✅ File pre-classified as remix and moved successfully (skipped transcription).")
        return # Exit after moving

    # CONTINUE WITH REGULAR PROCESSING (If not a filename-based remix)
    
    file_ext: str = os.path.splitext(file_path)[1].lower()
    temporary_audio_path: Optional[str] = None
    audio_source_for_transcription: Optional[str] = None

    if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
        logger.debug(f"File recognized as VIDEO ({file_ext}). Extracting audio.")
        temporary_audio_path = extract_audio(file_path)
        if not temporary_audio_path:
            logger.error(f"❌ Skipping {file_path} due to failed audio extraction.")
            return
        audio_source_for_transcription = temporary_audio_path
    elif file_ext in [".wav", ".mp3", ".flac", ".aac"]: # Explicitly supported audio files
        logger.debug(f"File recognized as AUDIO ({file_ext}). Using file directly.")
        audio_source_for_transcription = file_path
    else:
        logger.warning(f"⚠️ Unsupported file format: {file_ext} for {file_path}. Skipping.")
        return

    # Transcription
    text, lang_weights = transcribe_audio(audio_source_for_transcription)
    
    # If audio_source_for_transcription was a temporary file, clean it up
    if temporary_audio_path and os.path.exists(temporary_audio_path):
        try:
            os.remove(temporary_audio_path)
            logger.debug(f"Cleaned up temporary audio file: {temporary_audio_path}")
        except OSError as e:
            logger.error(f"Error cleaning up temporary audio file {temporary_audio_path}: {e}")

    # Classification
    target_dir, classification_reason = _classify_file(text, lang_weights, file_name_no_ext)

    if not target_dir:
        logger.error(f"❌ Skipping {file_path} because classification failed.")
        return

    # Move file
    _move_file(file_path, target_dir)

# --- 8. Main Function and Argparse ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Classify media files by language and move them.")
    parser.add_argument("input_path", type=str, help="Path to a file or directory to process.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO).")
    args = parser.parse_args()

    # Update logging level based on argument
    logger.setLevel(getattr(logging, args.log_level.upper()))
    logger.info(f"Log level set to {args.log_level.upper()}")

    # Process files
    if os.path.isfile(args.input_path):
        process_file(args.input_path)
    elif os.path.isdir(args.input_path):
        for root, _, files in os.walk(args.input_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                process_file(file_path)
    else:
        logger.error(f"Invalid input path: {args.input_path}. Must be a file or directory.")

    # Save hashes before exit
    save_processed_hashes(PROCESSED_HASHES_FILE, PROCESSED_FILE_HASHES)

if __name__ == "__main__":
    main()