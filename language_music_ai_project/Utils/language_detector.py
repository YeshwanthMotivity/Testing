import os
import shutil
import subprocess
import tempfile
import hashlib
import json
import logging
from typing import Dict, Tuple, Optional

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Management ---
CONFIG_FILE = "config.json"
CONFIG = {}
try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)
    logger.info(f"Configuration loaded from {CONFIG_FILE}")
except FileNotFoundError:
    logger.error(f"Configuration file '{CONFIG_FILE}' not found. Please create it.")
    exit(1)
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON from '{CONFIG_FILE}': {e}")
    exit(1)

# --- Define Constants from config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, CONFIG["output_base_dir_name"])

LANGUAGES_DIR = os.path.join(OUTPUT_DIR, "Languages")
BGM_DIR = os.path.join(OUTPUT_DIR, "bgm")
REMIX_DIR = os.path.join(OUTPUT_DIR, "remix")

os.makedirs(LANGUAGES_DIR, exist_ok=True)
os.makedirs(BGM_DIR, exist_ok=True)
os.makedirs(REMIX_DIR, exist_ok=True)

LANG_MAP = CONFIG["lang_map"]

MIN_WORDS_FOR_VOCALS = CONFIG["classification_thresholds"]["min_words_for_vocals"]
MIN_SECOND_LANG_SHARE_FOR_REMIX = CONFIG["classification_thresholds"]["min_second_lang_share_for_remix"]

PROCESSED_HASHES_FILE = CONFIG["processed_hashes_file"]
PROCESSED_FILE_HASHES = set()

# --- Persist Duplicate Hashes: Load at startup ---
def _load_processed_hashes():
    """
    Loads processed file hashes from a dedicated text file at startup.
    Each line in the file is expected to be a unique hash.
    """
    if os.path.exists(PROCESSED_HASHES_FILE):
        try:
            with open(PROCESSED_HASHES_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    hash_value = line.strip()
                    if hash_value:
                        PROCESSED_FILE_HASHES.add(hash_value)
            logger.info(f"Loaded {len(PROCESSED_FILE_HASHES)} hashes from {PROCESSED_HASHES_FILE}")
        except IOError as e:
            logger.error(f"Error loading processed hashes from {PROCESSED_HASHES_FILE}: {e}")

def _save_processed_hashes():
    """
    Saves the current set of processed file hashes to the dedicated text file.
    Each hash is written on a new line.
    """
    try:
        with open(PROCESSED_HASHES_FILE, 'w', encoding='utf-8') as f:
            for hash_value in PROCESSED_FILE_HASHES:
                f.write(hash_value + '\n')
        logger.debug(f"Saved {len(PROCESSED_FILE_HASHES)} hashes to {PROCESSED_HASHES_FILE}")
    except IOError as e:
        logger.error(f"Error saving processed hashes to {PROCESSED_HASHES_FILE}: {e}")

_load_processed_hashes()

# --- Whisper Model Load ---
from faster_whisper import WhisperModel
logger.info("📥 Loading Whisper model...")
model = WhisperModel(
    CONFIG["whisper_model"]["model_size"],
    device=CONFIG["whisper_model"]["device"],
    compute_type=CONFIG["whisper_model"]["compute_type"]
)
logger.info(f"✅ Whisper model loaded (Model: {CONFIG['whisper_model']['model_size']}, Device: {CONFIG['whisper_model']['device']}, Compute: {CONFIG['whisper_model']['compute_type']})")

# --- Audio extractor ---
def _extract_audio_ffmpeg(video_path: str, output_audio_path: str):
    """
    Extracts mono 16kHz audio from a video file using ffmpeg.
    This function is a helper for robust temporary file handling in `process_file`.

    Args:
        video_path: The path to the input video file.
        output_audio_path: The path where the extracted audio WAV file will be saved.
    """
    logger.info(f"🎬 Extracting audio from video: {video_path}")
    command = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", output_audio_path, "-y"]
    logger.debug(f"FFmpeg command: {' '.join(command)}")

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(output_audio_path):
            logger.info(f"✅ Extracted audio → {output_audio_path}")
        else:
            logger.error(f"❌ ERROR: Audio extraction failed, output file not found: {output_audio_path}.")
            raise RuntimeError(f"Audio extraction failed for {video_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ ERROR: FFmpeg command failed for {video_path}. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ ERROR: Unexpected error during audio extraction for {video_path}. Error: {e}")
        raise

# --- Transcribe (Corrected Multi-language Detection Logic) ---
def transcribe_audio(audio_path: str) -> Tuple[str, Dict[str, int]]:
    """
    Transcribes audio using faster-whisper's optimized engine and detects segment-level languages.

    Args:
        audio_path: The path to the audio file to transcribe.

    Returns:
        A tuple containing:
        - The concatenated transcribed text (str).
        - A dictionary with language codes as keys and total word counts for that language as values.
    """
    logger.info(f"--- Starting transcription for: {audio_path}")

    segments, info = model.transcribe(audio_path, beam_size=5)

    # Note: info.language provides an overall language for the file.
    # The suggestion mandates using segment.language for multi-language detection within a single file.
    logger.debug(f"Model detected overall language for file (info.language): {info.language} (will use segment-level detection)")
    logger.debug(f"Model transcription speed RTF (Lower is better): {info.language_probability}")

    langs_detected = {}
    texts = []
    segment_count = 0

    for segment in segments:
        segment_count += 1
        seg_text = segment.text.strip()
        # Correct Multi-language Detection Logic: Use segment.language for finer granularity
        seg_lang = segment.language

        if seg_text:
            word_count = len(seg_text.split())
            texts.append(seg_text)
            langs_detected[seg_lang] = langs_detected.get(seg_lang, 0) + word_count
            logger.debug(f"   🎧 Segment {segment_count}: Detected {seg_lang}, Words: {word_count}. Text: '{seg_text[:50]}...'")
        else:
            logger.debug(f"   🎧 Segment {segment_count}: Skipped (No text found).")

    logger.info(f"--- Total segments processed: {segment_count}")
    return " ".join(texts), langs_detected

# --- Helper: Get file hash ---
def get_file_hash(file_path: str) -> Optional[str]:
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
            # Read in 64k chunks
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"❌ ERROR: Could not get hash for {file_path}. Error: {e}")
        return None

# --- Helper: Classify Content ---
def _classify_content(
    text: str,
    lang_weights: Dict[str, int]
) -> Tuple[str, str]:
    """
    Classifies the content (BGM, Remix, or Vocal-Language) based on transcription results.

    Args:
        text: The full transcribed text.
        lang_weights: Dictionary of language codes and their accumulated word counts.

    Returns:
        A tuple: (classification type string, absolute path to the destination folder).
    """
    cleaned_text = text.replace("♪", "").replace("♫", "").strip()
    word_count = len(cleaned_text.split())

    logger.debug(f"Total cleaned transcription length: {len(cleaned_text)} characters")
    logger.info(f"🔍 Word count: {word_count}")

    langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
    detected_langs = [l for l, _ in langs_sorted]
    total_weight = sum(lang_weights.values()) or 1 # Avoid division by zero

    logger.debug(f"Sorted language weights (Word Count): {langs_sorted}")
    logger.info(f"🌍 Languages detected with weights: {lang_weights}")

    # Classification Logic
    if not detected_langs or word_count <= MIN_WORDS_FOR_VOCALS:
        logger.info("➡️ CLASSIFICATION: PURE BGM. Reason: No significant vocals or word count <= threshold.")
        return "BGM", BGM_DIR
    
    # Check for Remix based on multiple languages and share
    is_remix_by_language_share = False
    if len(langs_sorted) >= 2:
        # Calculate share of the second most dominant language
        second_lang_share = langs_sorted[1][1] / total_weight
        if second_lang_share >= MIN_SECOND_LANG_SHARE_FOR_REMIX:
            is_remix_by_language_share = True

    # Hindi/Urdu special case (if both are highly present, consider primary Hindi for folder)
    # This logic aims to prevent classifying as Remix if it's primarily one language with a strong presence of the other
    # within the same linguistic group (e.g., Hindi/Urdu overlap).
    if "hi" in detected_langs and "ur" in detected_langs and word_count > MIN_WORDS_FOR_VOCALS:
        # If both are present and there are enough words, classify based on "hi" as primary for consistency
        # This prevents accidental remix classification for common Hindi/Urdu content.
        primary_lang = "hi"
        language_folder_name = LANG_MAP.get(primary_lang, primary_lang)
        logger.info(f"➡️ CLASSIFICATION: VOCALS ({language_folder_name}). Reason: Hindi/Urdu Mix and significant word count.")
        return f"VOCALS ({language_folder_name})", os.path.join(LANGUAGES_DIR, language_folder_name, "vocals")
    elif is_remix_by_language_share:
        # If multiple languages detected with significant shares, classify as Remix
        lang_names = " + ".join(LANG_MAP.get(l, l) for l, _ in langs_sorted[:2]) # Top 2 languages
        logger.info(f"➡️ CLASSIFICATION: REMIX ({lang_names}). Reason: Multiple languages with significant shares.")
        return f"REMIX ({lang_names})", REMIX_DIR
    else:
        # Primary language classification
        primary_lang = detected_langs[0]
        language_folder_name = LANG_MAP.get(primary_lang, primary_lang)
        logger.info(f"➡️ CLASSIFICATION: VOCALS ({language_folder_name}). Reason: Primary language detected.")
        return f"VOCALS ({language_folder_name})", os.path.join(LANGUAGES_DIR, language_folder_name, "vocals")

# --- Helper: Move File ---
def _move_file(source_path: str, destination_dir: str, file_name: Optional[str] = None):
    """
    Moves a file to the specified destination directory, handling existing filenames
    by appending a copy number if necessary.

    Args:
        source_path: The current path of the file to move.
        destination_dir: The target directory to move the file into.
        file_name: Optional new filename. If None, uses the original filename.
    """
    os.makedirs(destination_dir, exist_ok=True)
    
    if file_name is None:
        file_name = os.path.basename(source_path)

    destination_path = os.path.join(destination_dir, file_name)
    logger.debug(f"Initial destination path: {destination_path}")

    # Handle duplicate filenames in destination
    if os.path.exists(destination_path):
        base, ext = os.path.splitext(destination_path)
        i = 1
        while os.path.exists(f"{base}_copy{i}{ext}"):
            i += 1
        destination_path = f"{base}_copy{i}{ext}"
        logger.warning(f"File with same name exists in {destination_dir}. Renaming to: {os.path.basename(destination_path)}")

    try:
        shutil.move(source_path, destination_path)
        logger.info(f"✅ File moved successfully → {destination_path}")
    except Exception as e:
        logger.error(f"❌ ERROR: File move failed for {source_path} to {destination_path}. Error: {e}")
        raise # Re-raise to signal failure to the caller

# --- Core: Process File ---
def process_file(file_path: str):
    """
    Processes a single audio/video file to classify its content (BGM, Remix, or Vocal-Language)
    and moves it to the appropriate organized directory.

    This function orchestrates:
    1. Duplicate file detection using hashes.
    2. Optional filename-based remix pre-classification for early exit.
    3. Audio extraction for video files using robust temporary file handling.
    4. Transcription using faster-whisper.
    5. Content classification based on transcription results.
    6. Moving the original file to its final categorized destination.

    Args:
        file_path: The absolute path to the file to be processed.
    """
    logger.info(f"\n=======================================================")
    logger.info(f"🚀 Processing file: {os.path.basename(file_path)}")
    logger.info(f"   FULL PATH: {file_path}")
    logger.info(f"=======================================================")

    # 1. Check for duplicates
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
            _save_processed_hashes() # Save after adding a new hash
            logger.info("✅ File is unique. Proceeding with processing.")
    else:
        logger.error(f"❌ Cannot generate hash for {file_path}. Skipping duplicate check.")

    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    file_ext = os.path.splitext(file_path)[1].lower()

    # 2. Filename-based Remix Check (Pre-Transcription)
    # This provides an early exit for files clearly marked as remixes in their name.
    if " X " in file_name_no_ext.upper():
        logger.info("✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains ' X ' pattern.")
        try:
            _move_file(file_path, REMIX_DIR)
            return # Exit after moving
        except Exception as e:
            logger.error(f"❌ Error moving file during filename-based remix classification: {e}")
            return

    # 3. Audio extraction or direct use
    audio_source_path: str = file_path
    cleanup_temp_audio: bool = False

    if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
        logger.debug(f"File recognized as VIDEO ({file_ext}). Extracting audio.")
        cleanup_temp_audio = True
        # Robust Temporary File Handling: Create temp file without immediate deletion
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_source_path = temp_audio_file.name
        temp_audio_file.close() # Close the handle so ffmpeg can write to it without interference
        try:
            _extract_audio_ffmpeg(file_path, audio_source_path)
        except Exception:
            # If extraction fails, ensure temp file is cleaned up and exit
            if os.path.exists(audio_source_path):
                os.remove(audio_source_path)
            return
    else:
        logger.debug(f"File recognized as AUDIO ({file_ext} or unknown). Using file directly.")

    try:
        # 4. Transcribe audio
        text, lang_weights = transcribe_audio(audio_source_path)

        # 5. Content Classification (Transcription-based)
        classification_type, final_destination_dir = _classify_content(text, lang_weights)
        logger.debug(f"Target move directory: {final_destination_dir}")

        # 6. Move original file
        _move_file(file_path, final_destination_dir)

    except Exception as e:
        logger.error(f"❌ An error occurred during transcription or classification for {file_path}: {e}")
    finally:
        # 7. Cleanup temporary audio file if created
        if cleanup_temp_audio and os.path.exists(audio_source_path):
            logger.debug(f"🗑️ Cleaning up temporary audio file: {audio_source_path}")
            os.remove(audio_source_path)