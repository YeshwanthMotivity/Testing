import os
import shutil
import subprocess
import tempfile
import hashlib
import logging
import configparser
from typing import Dict, List, Set, Tuple

from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect_langs

# --- 1. Centralized Configuration ---
CONFIG_FILE = "config.ini"
config = configparser.ConfigParser()
# Add default values before reading to ensure keys exist even if file is empty/missing
config.read_string("""
[PATHS]
base_dir = .
output_root_dir_name = data/output
languages_dir_name = Languages
bgm_dir_name = bgm
remix_dir_name = remix
processed_hashes_file = processed_hashes.txt

[WHISPER]
model_size = small
device = cpu
compute_type = int8
beam_size = 5

[CLASSIFICATION]
word_count_threshold_bgm = 10
filename_remix_pattern =  X 

[LANG_MAP]
en = english
hi = hindi
te = telugu
ta = tamil
ml = malayalam
kn = kannada
gu = gujarati
bn = bengali
pa = punjabi
ur = urdu
fr = french
es = spanish
de = german
it = italian
zh = chinese
ja = japanese
ko = korean
mr = marathi
unknown = unknown

[LOGGING]
level = INFO
""")
config.read(CONFIG_FILE) # Read user-defined config

# --- 4. Logging Configuration ---
log_level_str = config['LOGGING']['level'].upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Stability ---
DetectorFactory.seed = 0
logger.debug("Stability seed set to 0.")

# --- Paths ---
BASE_DIR = os.path.abspath(config['PATHS'].get('base_dir', '.'))
OUTPUT_ROOT_DIR = os.path.join(BASE_DIR, config['PATHS']['output_root_dir_name'])

LANGUAGES_DIR = os.path.join(OUTPUT_ROOT_DIR, config['PATHS']['languages_dir_name'])
BGM_DIR = os.path.join(OUTPUT_ROOT_DIR, config['PATHS']['bgm_dir_name'])
REMIX_DIR = os.path.join(OUTPUT_ROOT_DIR, config['PATHS']['remix_dir_name'])
PROCESSED_HASHES_FILE = os.path.join(BASE_DIR, config['PATHS']['processed_hashes_file'])

os.makedirs(LANGUAGES_DIR, exist_ok=True)
os.makedirs(BGM_DIR, exist_ok=True)
os.makedirs(REMIX_DIR, exist_ok=True)
logger.info(f"Output directories ensured: Languages={LANGUAGES_DIR}, BGM={BGM_DIR}, REMIX={REMIX_DIR}")

# --- 7. Supported Extensions Constants ---
VIDEO_EXTENSIONS: Set[str] = {".mp4", ".mkv", ".avi", ".mov"}
AUDIO_EXTENSIONS: Set[str] = {".wav", ".mp3", ".flac", ".aac"}

# --- Lang map ---
LANG_MAP: Dict[str, str] = {k: v for k, v in config['LANG_MAP'].items()}

# --- Whisper (Updated for faster-whisper) ---
logger.info("📥 Loading Whisper model...")
model = WhisperModel(
    config['WHISPER']['model_size'],
    device=config['WHISPER']['device'],
    compute_type=config['WHISPER']['compute_type']
)
logger.info(f"✅ Whisper model loaded (Model: {config['WHISPER']['model_size']}, Device: {config['WHISPER']['device']}, Compute: {config['WHISPER']['compute_type']})")

# --- 6. Persistent Duplicate Tracking ---
PROCESSED_FILE_HASHES: Set[str] = set()

def _load_processed_hashes() -> None:
    """Loads previously processed file hashes from a persistent file."""
    if os.path.exists(PROCESSED_HASHES_FILE):
        try:
            with open(PROCESSED_HASHES_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    PROCESSED_FILE_HASHES.add(line.strip())
            logger.info(f"Loaded {len(PROCESSED_FILE_HASHES)} hashes from {PROCESSED_HASHES_FILE}.")
        except Exception as e:
            logger.error(f"Failed to load processed hashes from {PROCESSED_HASHES_FILE}: {e}")
    else:
        logger.info("No processed hashes file found. Starting fresh.")

def _save_processed_hashes() -> None:
    """Saves current set of processed file hashes to a persistent file."""
    try:
        with open(PROCESSED_HASHES_FILE, 'w', encoding='utf-8') as f:
            for file_hash in PROCESSED_FILE_HASHES:
                f.write(f"{file_hash}\n")
        logger.info(f"Saved {len(PROCESSED_FILE_HASHES)} hashes to {PROCESSED_HASHES_FILE}.")
    except Exception as e:
        logger.error(f"Failed to save processed hashes to {PROCESSED_HASHES_FILE}: {e}")

# Call load at startup
_load_processed_hashes()

def get_file_hash(file_path: str) -> str | None:
    """
    Generates an MD5 hash for a file, handling large files efficiently.

    Args:
        file_path: The path to the file.

    Returns:
        The MD5 hash as a hexadecimal string, or None if an error occurs.
    """
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536) # Read in 64KB chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"❌ ERROR: Could not get hash for {file_path}. Error: {e}")
        return None

# --- Helper Functions for File Operations ---

def _move_file_with_rename_on_conflict(source_path: str, destination_dir: str) -> str:
    """
    Moves a file to a destination directory, renaming it if a file with the same
    name already exists to prevent overwriting.

    Args:
        source_path: The current path of the file to move.
        destination_dir: The directory where the file should be moved.

    Returns:
        The final destination path of the moved file.
    """
    os.makedirs(destination_dir, exist_ok=True)
    file_name = os.path.basename(source_path)
    destination = os.path.join(destination_dir, file_name)

    if os.path.exists(destination):
        base, ext = os.path.splitext(file_name)
        i = 1
        while os.path.exists(os.path.join(destination_dir, f"{base}_copy{i}{ext}")):
            i += 1
        destination = os.path.join(destination_dir, f"{base}_copy{i}{ext}")
        logger.info(f"File '{file_name}' already exists. Renaming to '{os.path.basename(destination)}'.")

    try:
        shutil.move(source_path, destination)
        logger.info(f"✅ Moved '{os.path.basename(source_path)}' → '{destination}'")
        return destination
    except Exception as e:
        logger.error(f"❌ ERROR: File move failed for {source_path}. Error: {e}")
        raise # Re-raise the exception after logging

# --- Core Functions (Refactored from process_file) ---

def _extract_audio_if_video(file_path: str, file_ext: str) -> str:
    """
    Extracts mono 16kHz audio from a video file using ffmpeg, or returns the
    original path if it's already an audio file. Uses a temporary file for extraction.

    Args:
        file_path: The path to the input file (video or audio).
        file_ext: The lowercase extension of the input file.

    Returns:
        The path to the extracted audio file (temporary) or the original audio file path.
        The temporary file will be automatically cleaned up after the function scope,
        so the caller must process it within this context or copy it.
    
    Raises:
        RuntimeError: If audio extraction fails.
    """
    if file_ext in VIDEO_EXTENSIONS:
        logger.info(f"🎬 Extracting audio from video: {file_path}")
        
        # 2. Implement robust temporary file management
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=True)
        output_audio_path = temp_audio_file.name
        
        command = [
            "ffmpeg", "-i", file_path,
            "-ar", "16000", "-ac", "1",
            output_audio_path, "-y"
        ]
        logger.debug(f"FFmpeg command: {' '.join(command)}")
        
        process = subprocess.run(command, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"❌ ERROR: Audio extraction failed for {file_path}.")
            logger.error(f"FFmpeg STDOUT: {process.stdout}")
            logger.error(f"FFmpeg STDERR: {process.stderr}")
            temp_audio_file.close() # Ensure temp file is closed and deleted
            raise RuntimeError(f"FFmpeg audio extraction failed: {process.stderr}")
        
        logger.info(f"✅ Extracted audio → {output_audio_path}")
        return output_audio_path
    elif file_ext in AUDIO_EXTENSIONS:
        logger.debug(f"File recognized as AUDIO ({file_ext}). Using file directly.")
        return file_path
    else:
        logger.warning(f"Unsupported file type for audio extraction: {file_ext}")
        return file_path # Or raise an error, depending on desired strictness

def _transcribe_audio_content(audio_path: str) -> Tuple[str, Dict[str, int]]:
    """
    Transcribes the entire audio using faster-whisper's optimized engine
    and aggregates detected languages and word counts.

    Args:
        audio_path: The path to the audio file.

    Returns:
        A tuple containing:
            - The full transcribed text as a single string.
            - A dictionary mapping detected language codes to their total word counts.
    """
    logger.info(f"--- Starting transcription for: {os.path.basename(audio_path)}")
    
    segments_generator, info = model.transcribe(
        audio_path,
        beam_size=config.getint('WHISPER', 'beam_size')
    )

    logger.debug(f"Model detected primary language for entire file: {info.language}")
    logger.debug(f"Model transcription speed RTF (Lower is better): {info.language_probability}") # This is RTF or probability, depends on version

    langs_detected: Dict[str, int] = {}
    texts: List[str] = []
    segment_count = 0
    
    for segment in segments_generator:
        segment_count += 1
        seg_text = segment.text.strip()
        seg_lang = info.language # Use the language detected for the whole file
        
        if seg_text:
            word_count = len(seg_text.split())
            texts.append(seg_text)
            langs_detected[seg_lang] = langs_detected.get(seg_lang, 0) + word_count
            logger.debug(f"   🎧 Segment {segment_count}: Detected {seg_lang}, Words: {word_count}. Text: '{seg_text[:50]}...'")
        else:
             logger.debug(f"   🎧 Segment {segment_count}: Skipped (No text found).")

    logger.info(f"--- Total segments processed: {segment_count}")
    return " ".join(texts), langs_detected

def _classify_content(text: str, lang_weights: Dict[str, int], file_name_no_ext: str) -> str:
    """
    Classifies the content into BGM, Remix, or Language-specific Vocals based on
    transcription, language weights, and filename patterns.

    Args:
        text: The full transcribed text.
        lang_weights: Dictionary of language codes to word counts.
        file_name_no_ext: The filename without extension.

    Returns:
        The target base directory for the file (e.g., BGM_DIR, REMIX_DIR, or specific language folder).

    Raises:
        ValueError: If classification logic fails to determine a base folder.
    """
    cleaned_text = text.replace("♪", "").replace("♫", "").strip()
    word_count = len(cleaned_text.split())
    
    logger.info(f"📝 Text snippet: {text[:100]}...")
    logger.debug(f"Total cleaned transcription length: {len(cleaned_text)} characters")
    logger.info(f"🔍 Word count: {word_count}")

    langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
    detected_langs = [l for l, _ in langs_sorted]
    
    logger.debug(f"Sorted language weights (Word Count): {langs_sorted}")
    logger.info(f"🌍 Languages detected with weights: {lang_weights}")

    base_folder: str | None = None

    # --- NEW FEATURE: Filename-based Remix Check (Configurable) ---
    filename_remix_pattern = config['CLASSIFICATION']['filename_remix_pattern'].strip().upper()
    if filename_remix_pattern and filename_remix_pattern in file_name_no_ext.upper():
        base_folder = REMIX_DIR
        logger.info(f"➡️ CLASSIFICATION: REMIX. Reason: Filename contains '{filename_remix_pattern}' pattern.")
    
    # --- Transcription-based Classification (if not already classified by filename) ---
    if base_folder is None:
        word_count_threshold_bgm = config.getint('CLASSIFICATION', 'word_count_threshold_bgm')
        
        if "hi" in detected_langs and "ur" in detected_langs and word_count > word_count_threshold_bgm:
            primary_lang = "hi" # Prioritize Hindi for mixed Hindi/Urdu content
            language = LANG_MAP.get(primary_lang, primary_lang)
            base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
            logger.info(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Hindi/Urdu Mix and word count > {word_count_threshold_bgm}.")
        elif word_count <= word_count_threshold_bgm:
            base_folder = BGM_DIR
            logger.info(f"➡️ CLASSIFICATION: PURE BGM. Reason: Word count <= {word_count_threshold_bgm}.")
        elif len(detected_langs) >= 2:
            base_folder = REMIX_DIR
            logger.info(f"➡️ CLASSIFICATION: REMIX. Reason: Multiple Languages detected ({detected_langs}).")
        else:
            primary_lang = detected_langs[0] if detected_langs else "unknown"
            language = LANG_MAP.get(primary_lang, primary_lang)
            base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
            logger.info(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Primary language detected.")

    if base_folder is None:
        logger.error("❌ ERROR: Classification logic failed to assign a base folder.")
        raise ValueError("Failed to classify content and determine a destination folder.")
    
    logger.debug(f"Determined target directory: {base_folder}")
    return base_folder

def process_file(file_path: str) -> None:
    """
    Processes a single audio or video file to detect its language, classify its content
    (BGM, Remix, Vocals), and move it to the appropriate destination folder.
    Includes duplicate detection and temporary file management.

    Args:
        file_path: The full path to the file to be processed.
    """
    logger.info(f"\n=======================================================")
    logger.info(f"🚀 Processing file: {os.path.basename(file_path)}")
    logger.info(f"   FULL PATH: {file_path}")
    logger.info(f"=======================================================")
    
    # Check for duplicates before expensive operations
    file_hash = get_file_hash(file_path)
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
        logger.error(f"Could not generate hash for {file_path}. Cannot check for duplicates.")

    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    file_ext = os.path.splitext(file_path)[1].lower()

    # Determine audio source
    audio_source: str
    temp_file_obj = None

    try:
        # Use a context manager for temporary audio file
        if file_ext in VIDEO_EXTENSIONS:
            # We need to handle the NamedTemporaryFile lifecycle carefully
            # The _extract_audio_if_video will return the name of a temporary file,
            # but that file object needs to persist until transcription is done.
            # A simpler way is to let _extract_audio_if_video create and manage
            # its own temp file (which it does with delete=True), and copy the content
            # if we truly need it outside its scope, or just rely on its immediate
            # return and usage. For faster_whisper, it directly reads path.
            # So, the tempfile handling is already internal to extract_audio.
            
            # The tempfile approach from prior versions would return the *object*,
            # but with NamedTemporaryFile(delete=True), the file is deleted
            # when the *object* is closed or garbage collected.
            # `_extract_audio_if_video` returns *only the path*, making it problematic for automatic cleanup.
            # Re-thinking tempfile:
            # We need the temp file to exist for transcription. It must be cleaned *after* transcription.
            
            # A more robust pattern:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
                logger.info(f"🎬 Extracting audio from video: {file_path} into temporary file: {tf.name}")
                command = [
                    "ffmpeg", "-i", file_path,
                    "-ar", "16000", "-ac", "1",
                    tf.name, "-y"
                ]
                process = subprocess.run(command, capture_output=True, text=True)
                if process.returncode != 0:
                    logger.error(f"❌ ERROR: Audio extraction failed for {file_path}. STDOUT: {process.stdout}, STDERR: {process.stderr}")
                    raise RuntimeError(f"FFmpeg audio extraction failed: {process.stderr}")
                audio_source = tf.name
                logger.info(f"✅ Extracted audio to temporary file: {audio_source}")
                
                # Now transcribe from this temporary file
                text, lang_weights = _transcribe_audio_content(audio_source)
            # `tf` is now out of scope, temp file is deleted.
            
        else:
            audio_source = file_path
            logger.debug(f"File recognized as AUDIO ({file_ext}). Using file directly.")
            text, lang_weights = _transcribe_audio_content(audio_source)

        # Classify and move
        final_destination_dir = _classify_content(text, lang_weights, file_name_no_ext)
        _move_file_with_rename_on_conflict(file_path, final_destination_dir)

    except Exception as e:
        logger.error(f"❌ Processing failed for {file_path}: {e}")
    finally:
        # Ensure that any temporary audio files created *outside* a `with` statement are cleaned up.
        # With the `with tempfile.NamedTemporaryFile` block, this is mostly handled.
        pass

# Call save at script shutdown
import atexit
atexit.register(_save_processed_hashes)

# Example usage (will not be part of the final returned code, but for completeness)
# if __name__ == "__main__":
#     # Create dummy files for testing
#     if not os.path.exists("test_files"):
#         os.makedirs("test_files")
#     with open("test_files/english_song.mp3", "w") as f:
#         f.write("dummy audio content")
#     with open("test_files/hindi X french remix.mp4", "w") as f:
#         f.write("dummy video content")
#     with open("test_files/pure_bgm.wav", "w") as f:
#         f.write("dummy audio content")
    
#     # Mock ffmpeg and whisper for testing without actual binaries/models
#     # This part would require more advanced mocking setup (e.g., unittest.mock)
#     # For this exercise, assume ffmpeg and WhisperModel are functional.
    
#     # process_file("test_files/english_song.mp3")
#     # process_file("test_files/hindi X french remix.mp4")
#     # process_file("test_files/pure_bgm.wav")
#     # process_file("test_files/another_english_song.mp3") # Duplicate test
#     pass