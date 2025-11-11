import os
import shutil
import subprocess
import logging
import tempfile
from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect_langs, LangDetectException
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logging.info("--- Stability seed set to 0 ---")
DetectorFactory.seed = 0

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

LANGUAGES_DIR = os.path.join(OUTPUT_DIR, "Languages")
BGM_DIR = os.path.join(OUTPUT_DIR, "bgm")
REMIX_DIR = os.path.join(OUTPUT_DIR, "remix")

logging.info(f"BASE_DIR: {BASE_DIR}")
logging.info(f"Output directories setup: Languages={LANGUAGES_DIR}, BGM={BGM_DIR}, REMIX={REMIX_DIR}")

os.makedirs(LANGUAGES_DIR, exist_ok=True)
os.makedirs(BGM_DIR, exist_ok=True)
os.makedirs(REMIX_DIR, exist_ok=True)

# Global variable to store hashes of processed files
PROCESSED_FILE_HASHES = set()

# ---------- Custom Exception for Audio Extraction ----------
class AudioExtractionError(Exception):
    """Custom exception for ffmpeg audio extraction failures."""
    pass

# ---------- Lang map ----------
LANG_MAP = {
    "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
    "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
    "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
    "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
}

# ---------- Whisper (Updated for faster-whisper) ----------
logging.info("📥 Loading Whisper model...")
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)
logging.info("✅ Whisper model loaded (Model: small, Device: CPU, Compute: INT8)")

# ---------- Helper: File Moving and Renaming Logic Refactor ----------
def _move_file_to_destination(source_path, target_dir):
    """
    Moves a file to a target directory, handling existing file names by adding _copy{i}.
    Logs operations using the logging module.
    """
    logging.debug(f"Attempting to move '{source_path}' to '{target_dir}'")
    os.makedirs(target_dir, exist_ok=True)

    base_name = os.path.basename(source_path)
    destination = os.path.join(target_dir, base_name)

    logging.debug(f"Initial destination path: {destination}")

    if os.path.exists(destination):
        base, ext = os.path.splitext(destination)
        i = 1
        while os.path.exists(f"{base}_copy{i}{ext}"):
            i += 1
        destination = f"{base}_copy{i}{ext}"
        logging.info(f"Renaming required. New destination path: {destination}")

    try:
        shutil.move(source_path, destination)
        logging.info(f"✅ Final move successful → {destination}")
        return destination
    except Exception as e:
        logging.error(f"❌ ERROR: File move failed for '{source_path}' to '{destination}'. Error: {e}")
        raise # Re-raise the exception to be handled upstream

# ---------- Audio extractor (Enhanced error handling and temp file management) ----------
def extract_audio(video_path):
    """
    Extracts mono 16kHz audio using ffmpeg into a temporary file.
    Raises AudioExtractionError if ffmpeg fails.
    """
    logging.info(f"\n🎬 Extracting audio from video: {video_path}")
    
    # Use tempfile for automatic cleanup
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_audio_path = temp_audio_file.name
    temp_audio_file.close() # Close the file handle created by NamedTemporaryFile, ffmpeg will write to it

    command = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", output_audio_path, "-y"]
    logging.debug(f"FFmpeg command: {' '.join(command)}")
    
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logging.info(f"✅ Extracted audio → {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ ERROR: FFmpeg audio extraction failed for '{video_path}'. Error: {e}")
        # Ensure the temporary file is cleaned up if ffmpeg fails
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        raise AudioExtractionError(f"Audio extraction failed for {video_path}") from e

# ---------- Transcribe (New Optimized Function) ----------
def transcribe_audio(audio_path):
    """Transcribe entire audio using faster-whisper's optimized engine."""
    logging.info(f"--- Starting transcription for: {audio_path}")
    
    segments, info = model.transcribe(audio_path, beam_size=5)

    logging.info(f"--- Faster-Whisper detected primary language: {info.language}")
    # Corrected debug message for language_probability
    logging.debug(f"--- Faster-Whisper detected language probability: {info.language_probability:.4f}")

    # Note: lang_weights will likely only contain one language as faster-whisper's info.language
    # typically gives a single dominant language for the entire file.
    # We will use langdetect for multi-language detection.
    langs_detected = {} # This will store the primary language detected by faster-whisper and its word count
    texts = []
    segment_count = 0
    
    for segment in segments:
        segment_count += 1
        seg_text = segment.text.strip()
        seg_lang = info.language # Use the primary language detected for the whole file by faster-whisper
        
        if seg_text:
            word_count = len(seg_text.split())
            texts.append(seg_text)
            langs_detected[seg_lang] = langs_detected.get(seg_lang, 0) + word_count
            logging.debug(f"   🎧 Segment {segment_count}: Detected {seg_lang}, Words: {word_count}. Text: '{seg_text[:50]}...'")
        else:
            logging.debug(f"   🎧 Segment {segment_count}: Skipped (No text found).")

    logging.debug(f"Total segments processed: {segment_count}")
    return " ".join(texts), langs_detected # langs_detected will essentially contain only one entry

# New function to get file hash
def get_file_hash(file_path):
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
        logging.error(f"❌ ERROR: Could not get hash for {file_path}. Error: {e}")
        return None

# ---------- Core (Updated to include duplicate check, refactored move logic, improved error handling) ----------
def process_file(file_path):
    logging.info(f"\n=======================================================")
    logging.info(f"🚀 Processing file: {os.path.basename(file_path)}")
    logging.info(f"   FULL PATH: {file_path}")
    logging.info(f"=======================================================")
    
    # CHECK FOR DUPLICATES BEFORE ANYTHING ELSE
    file_hash = get_file_hash(file_path)
    if file_hash:
        logging.debug(f"File hash generated: {file_hash}")
        if file_hash in PROCESSED_FILE_HASHES:
            logging.warning("⚠️ DUPLICATE DETECTED! File with this hash has already been processed.")
            try:
                os.remove(file_path)
                logging.info(f"🗑️ Successfully deleted duplicate file: {file_path}")
            except Exception as e:
                logging.error(f"❌ ERROR: Failed to delete duplicate file: {file_path}. Error: {e}")
            return # Exit the function immediately
        else:
            PROCESSED_FILE_HASHES.add(file_hash)
            logging.info("✅ File is unique. Proceeding with processing.")

    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    
    # --- Filename-based Remix Check (Pre-Transcription) ---
    if " X " in file_name_no_ext.upper():
        logging.info("✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains ' X ' pattern.")
        base_folder = REMIX_DIR
        try:
            _move_file_to_destination(file_path, base_folder)
            return # Exit after moving
        except Exception as e:
            logging.error(f"❌ ERROR: File move failed during pre-classification for {file_path}. Error: {e}")
            return
    
    # --- CONTINUE WITH REGULAR PROCESSING (If not a filename-based remix) ---
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    audio_path = None
    is_temp_audio = False

    # Extract audio if it's a video
    if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
        logging.debug(f"File recognized as VIDEO ({file_ext}). Extracting audio.")
        try:
            audio_path = extract_audio(file_path)
            is_temp_audio = True
        except AudioExtractionError:
            logging.error(f"Skipping file {file_path} due to audio extraction failure.")
            return # Skip this file
    else:
        logging.debug(f"File recognized as AUDIO ({file_ext} or unknown). Using file directly.")
        audio_path = file_path

    if not audio_path or not os.path.exists(audio_path):
        logging.error(f"❌ ERROR: Audio path is invalid or does not exist after extraction for {file_path}. Skipping.")
        return

    # Use try-finally to ensure temporary audio file is deleted
    try:
        # Transcribe audio 
        full_text, faster_whisper_lang_weights = transcribe_audio(audio_path)
        
        # faster-whisper provides a single dominant language via info.language,
        # so faster_whisper_lang_weights will only have one entry.
        primary_whisper_lang = list(faster_whisper_lang_weights.keys())[0] if faster_whisper_lang_weights else "unknown"

        logging.debug(f"Faster-Whisper's primary language: {primary_whisper_lang}")
        logging.debug(f"Faster-Whisper's detected language weights (word count): {faster_whisper_lang_weights}")

        # Cleanup + heuristic
        cleaned_text = full_text.replace("♪", "").replace("♫", "").strip()
        word_count = len(cleaned_text.split())
        
        logging.debug(f"Total cleaned transcription length: {len(cleaned_text)} characters")
        logging.info(f"🔍 Word count from transcription: {word_count}")

        # Clarify and align the multi-language detection strategy with langdetect
        langdetect_result_codes = []
        if cleaned_text and word_count > 50: # Only try langdetect if enough text
            try:
                # detect_langs returns a list of Language objects, e.g., [en:0.99, fr:0.01]
                detected_lang_objects = detect_langs(cleaned_text)
                langdetect_result_codes = [lang.lang.split("-")[0] for lang in detected_lang_objects if lang.prob > 0.05] # Consider languages with significant probability
                logging.info(f"🔎 Langdetect detected languages (codes, prob > 0.05): {langdetect_result_codes}")
            except LangDetectException as e:
                logging.warning(f"⚠️ Langdetect failed for text: '{cleaned_text[:100]}...'. Error: {e}")
            except Exception as e: # Catch other potential errors from langdetect
                logging.warning(f"⚠️ Unexpected error with Langdetect: {e}")
        else:
            logging.debug(f"Skipping Langdetect due to insufficient word count ({word_count}) or no text.")

        # --- Classification Logic (Transcription-based, enhanced for multi-language) ---
        base_folder = None
        
        # Remix condition: Multi-language detected by langdetect with significant text
        if len(langdetect_result_codes) > 1 and word_count > 100: # Threshold for considering it a true multi-language remix
            logging.info(f"➡️ CLASSIFICATION: REMIX. Reason: Langdetect found multiple significant languages ({langdetect_result_codes}) and sufficient word count.")
            
            # Construct a more descriptive remix folder name
            primary_langs_for_folder = [LANG_MAP.get(l, l) for l in langdetect_result_codes[:3]] # Take top 3 for folder name
            remix_folder_name = " + ".join(primary_langs_for_folder)
            base_folder = os.path.join(REMIX_DIR, f"Remix ({remix_folder_name})")

        # Special Hindi/Urdu handling (if not already classified as a general remix by langdetect)
        elif "hi" in langdetect_result_codes and "ur" in langdetect_result_codes and word_count > 10:
            primary_lang = "hi" # Prioritize Hindi for folder naming if both are present
            language = LANG_MAP.get(primary_lang, primary_lang)
            base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
            logging.info(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Hindi/Urdu Mix detected by langdetect and word count > 10.")
        
        # Pure BGM if very little speech
        elif word_count <= 10:
            base_folder = BGM_DIR
            logging.info("➡️ CLASSIFICATION: PURE BGM. Reason: Word count <= 10.")
        
        # Single dominant language (based on faster-whisper)
        else:
            # Use faster-whisper's primary language as the single language classification
            language = LANG_MAP.get(primary_whisper_lang, primary_whisper_lang)
            base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
            logging.info(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Primary language ({language}) detected by Faster-Whisper.")


        if not base_folder:
            logging.error("❌ ERROR: Classification logic failed to assign a base folder.")
            return

        # Move file using the refactored helper
        try:
            _move_file_to_destination(file_path, base_folder)
        except Exception:
            # _move_file_to_destination already logs the error, just return here
            return

    finally:
        # Ensure temporary audio file is cleaned up
        if is_temp_audio and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logging.debug(f"🗑️ Cleaned up temporary audio file: {audio_path}")
            except Exception as e:
                logging.error(f"❌ ERROR: Failed to clean up temporary audio file {audio_path}. Error: {e}")