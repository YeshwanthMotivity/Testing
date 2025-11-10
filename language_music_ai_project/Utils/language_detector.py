import os
import shutil
import subprocess
from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect_langs
import hashlib
import logging
import tempfile
from typing import Set, Dict, Tuple, List, Optional

# Set langdetect stability seed for consistent results
DetectorFactory.seed = 0

# ---------- Configuration Section ----------
class Config:
    """Centralized configuration for the language detection script."""

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_ROOT_DIR: str = os.path.join(BASE_DIR, "data", "output")
    LANGUAGES_SUBDIR: str = "Languages"
    BGM_SUBDIR: str = "bgm"
    REMIX_SUBDIR: str = "remix"

    LANGUAGES_DIR: str = os.path.join(OUTPUT_ROOT_DIR, LANGUAGES_SUBDIR)
    BGM_DIR: str = os.path.join(OUTPUT_ROOT_DIR, BGM_SUBDIR)
    REMIX_DIR: str = os.path.join(OUTPUT_ROOT_DIR, REMIX_SUBDIR)
    
    # Whisper Model Parameters
    WHISPER_MODEL_SIZE: str = "small" # Options: tiny, base, small, medium, large, large-v2, large-v3
    WHISPER_DEVICE: str = "cpu"       # Options: "cpu", "cuda", "auto"
    WHISPER_COMPUTE_TYPE: str = "int8" # Options: "int8", "int8_float16", "int16", "float16", "float32"
    WHISPER_BEAM_SIZE: int = 5         # Beam search width for transcription

    # Language Map
    LANG_MAP: Dict[str, str] = {
        "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
        "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
        "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
        "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
    }

    # Language Detection & Classification Thresholds
    MIN_SECOND_LANG_SHARE_FOR_REMIX: float = 0.20 # Min share for a second lang to classify as remix
    MIN_WORD_COUNT_FOR_VOCALS: int = 10         # Min word count to consider as having vocals (not BGM)
    MIN_WORD_COUNT_FOR_LANGUAGE: int = 5        # Min word count for a language to be considered "detected" (for hi/ur mix, etc.)
    REMIX_FILENAME_PATTERN: str = " X "         # Pattern in filename to pre-classify as remix (e.g., "Song A X Song B")
    REMIX_LANG_COUNT_FOR_NAME: int = 4          # Number of top languages to include in remix folder name
    LANGDETECT_PROB_THRESHOLD: float = 0.5      # Minimum probability for langdetect to consider a language valid

    # Logging
    LOG_LEVEL: str = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Duplicate File Handling
    PERSISTENT_HASH_FILE: str = os.path.join(BASE_DIR, "processed_hashes.txt") # File to store processed hashes

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"--- Initializing: Log level set to {Config.LOG_LEVEL} ---")
logging.debug(f"BASE_DIR: {Config.BASE_DIR}")
logging.debug(f"Output directories setup: Languages={Config.LANGUAGES_DIR}, BGM={Config.BGM_DIR}, REMIX={Config.REMIX_DIR}")

# Create output directories if they don't exist
os.makedirs(Config.LANGUAGES_DIR, exist_ok=True)
os.makedirs(Config.BGM_DIR, exist_ok=True)
os.makedirs(Config.REMIX_DIR, exist_ok=True)

# Global variable to store hashes of processed files
PROCESSED_FILE_HASHES: Set[str] = set()

def load_processed_hashes() -> Set[str]:
    """
    Loads processed file hashes from a persistent file.

    Returns:
        A set of MD5 hashes of previously processed files.
    """
    hashes = set()
    if os.path.exists(Config.PERSISTENT_HASH_FILE):
        try:
            with open(Config.PERSISTENT_HASH_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    h = line.strip()
                    if h:
                        hashes.add(h)
            logging.info(f"Loaded {len(hashes)} processed file hashes from {Config.PERSISTENT_HASH_FILE}")
        except Exception as e:
            logging.error(f"Error loading hashes from {Config.PERSISTENT_HASH_FILE}: {e}")
    return hashes

def save_processed_hashes(hashes_to_save: Set[str]):
    """
    Saves processed file hashes to a persistent file.

    Args:
        hashes_to_save: The set of hashes to save.
    """
    try:
        with open(Config.PERSISTENT_HASH_FILE, 'w', encoding='utf-8') as f:
            for h in sorted(list(hashes_to_save)): # Sort for consistent file content
                f.write(f"{h}\n")
        logging.info(f"Saved {len(hashes_to_save)} processed file hashes to {Config.PERSISTENT_HASH_FILE}")
    except Exception as e:
        logging.error(f"Error saving hashes to {Config.PERSISTENT_HASH_FILE}: {e}")

# Initialize PROCESSED_FILE_HASHES at script startup
PROCESSED_FILE_HASHES = load_processed_hashes()

# ---------- Whisper Model Loading ----------
logging.info(f"📥 Loading Whisper model (size: {Config.WHISPER_MODEL_SIZE}, device: {Config.WHISPER_DEVICE}, compute_type: {Config.WHISPER_COMPUTE_TYPE})...")
model = WhisperModel(
    Config.WHISPER_MODEL_SIZE,
    device=Config.WHISPER_DEVICE,
    compute_type=Config.WHISPER_COMPUTE_TYPE
)
logging.info("✅ Whisper model loaded.")

# ---------- Helper Functions ----------

def get_file_hash(file_path: str) -> Optional[str]:
    """
    Generates an MD5 hash for a file, handling large files efficiently.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        The MD5 hash string or None if an error occurs.
    """
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536) # Read in chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        logging.error(f"❌ ERROR: Could not get hash for {file_path}. Error: {e}")
        return None

def extract_audio(video_path: str) -> Optional[str]:
    """
    Extracts mono 16kHz audio from a video file using ffmpeg to a temporary file.
    
    Args:
        video_path: The path to the input video file.
        
    Returns:
        The path to the temporary audio file, or None if extraction fails.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_audio_path = temp_file.name
    temp_file.close() # Close the handle immediately so ffmpeg can write to it

    logging.info(f"🎬 Extracting audio from video: {video_path} to {output_audio_path}")
    command = [
        "ffmpeg", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        output_audio_path, "-y"
    ]
    logging.debug(f"FFmpeg command: {' '.join(command)}")

    try:
        # Use check=True to raise an exception on non-zero exit codes
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(output_audio_path):
            logging.info(f"✅ Extracted audio → {output_audio_path}")
            return output_audio_path
        else:
            logging.error(f"❌ ERROR: Audio extraction failed for {video_path}. Output file {output_audio_path} not found.")
            # Ensure the temp file is removed even if extraction didn't create it but path exists
            if os.path.exists(output_audio_path):
                os.remove(output_audio_path)
            return None
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ ERROR: FFmpeg failed for {video_path} with exit code {e.returncode}. Error: {e}")
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        return None
    except Exception as e:
        logging.error(f"❌ ERROR: An unexpected error occurred during audio extraction for {video_path}. Error: {e}")
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        return None

def transcribe_audio(audio_path: str) -> Tuple[str, Dict[str, int]]:
    """
    Transcribes audio using faster-whisper, collecting segment-level language weights
    based on word count.

    Args:
        audio_path: The path to the audio file to transcribe.

    Returns:
        A tuple containing:
        - The full transcribed text as a single string.
        - A dictionary mapping detected language codes to their total word count (weight)
          from faster_whisper's segment-level detection.
    """
    logging.info(f"--- Starting transcription for: {audio_path}")
    
    segments, info = model.transcribe(audio_path, beam_size=Config.WHISPER_BEAM_SIZE)

    logging.debug(f"Model detected overall language for entire file: {info.language}")
    logging.debug(f"Model detected overall language probability for entire file: {info.language_probability:.2f}")

    langs_detected_by_fw: Dict[str, int] = {}
    full_texts: List[str] = []
    segment_count = 0
    
    # Iterate through segments returned by the model to get segment-level language
    for segment in segments:
        segment_count += 1
        seg_text = segment.text.strip()
        seg_lang = segment.language # Use segment's specific language detection
        seg_prob = segment.probability # Probability of the detected language for this segment

        if seg_text:
            word_count = len(seg_text.split())
            full_texts.append(seg_text)
            langs_detected_by_fw[seg_lang] = langs_detected_by_fw.get(seg_lang, 0) + word_count
            logging.debug(f"   🎧 Segment {segment_count}: Detected {seg_lang} (P={seg_prob:.2f}), Words: {word_count}. Text: '{seg_text[:50]}...'")
        else:
            logging.debug(f"   🎧 Segment {segment_count}: Skipped (No text found).")

    logging.debug(f"Total segments processed: {segment_count}")
    return " ".join(full_texts), langs_detected_by_fw

def resolve_destination_path(target_dir: str, original_filename: str) -> str:
    """
    Resolves a unique destination path in target_dir by appending _copy{i}
    if a file with the original_filename already exists.
    
    Args:
        target_dir: The directory where the file is intended to be moved.
        original_filename: The base filename (e.g., "my_song.mp3").
        
    Returns:
        A unique, non-existent path in target_dir for the file.
    """
    base_name, ext = os.path.splitext(original_filename)
    destination_path = os.path.join(target_dir, original_filename)

    if not os.path.exists(destination_path):
        logging.debug(f"Destination path is unique: {destination_path}")
        return destination_path
    
    logging.debug(f"Destination path '{destination_path}' already exists. Resolving new path.")
    i = 1
    while True:
        new_filename = f"{base_name}_copy{i}{ext}"
        new_destination_path = os.path.join(target_dir, new_filename)
        if not os.path.exists(new_destination_path):
            logging.debug(f"Resolved new unique destination path: {new_destination_path}")
            return new_destination_path
        i += 1

# ---------- Core Processing Function ----------
def process_file(file_path: str):
    """
    Processes a single audio/video file to detect its language(s), classify it
    as BGM, Remix, or Vocals (with primary language), and move it to the
    appropriate organized output directory.

    Handles duplicate file detection, temporary file management, and filename
    based pre-classification for remixes.

    Args:
        file_path: The full path to the input file.
    """
    logging.info(f"\n=======================================================")
    logging.info(f"🚀 Processing file: {os.path.basename(file_path)}")
    logging.info(f"   FULL PATH: {file_path}")
    logging.info(f"=======================================================")
    
    original_filename = os.path.basename(file_path)

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
        logging.error(f"❌ ERROR: Could not generate hash for {file_path}. Skipping duplicate check for this run.")

    # --- Filename-based Remix Check (Pre-Transcription) ---
    file_name_no_ext = os.path.splitext(original_filename)[0]
    if Config.REMIX_FILENAME_PATTERN in file_name_no_ext.upper():
        logging.info(f"✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains '{Config.REMIX_FILENAME_PATTERN}' pattern.")
        
        target_dir = Config.REMIX_DIR
        logging.debug(f"Target move directory (Pre-classified): {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        
        destination = resolve_destination_path(target_dir, original_filename)
        
        try:
            shutil.move(file_path, destination)
            if file_hash and file_hash not in PROCESSED_FILE_HASHES:
                PROCESSED_FILE_HASHES.add(file_hash)
                save_processed_hashes(PROCESSED_FILE_HASHES) # Save immediately for persistence
            logging.info(f"✅ Final move successful (Skipped transcription) → {destination}")
            return # IMPORTANT: Exit the function after moving

        except Exception as e:
            logging.error(f"❌ ERROR: File move failed for {file_path}. Error: {e}")
            return
    
    # --- CONTINUE WITH REGULAR PROCESSING (If not a filename-based remix or duplicate) ---
    
    file_ext = os.path.splitext(file_path)[1].lower()
    audio_path_to_process: str = file_path # Default to original file
    temp_audio_created: bool = False
    
    try:
        # Extract audio if it's a video
        if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
            logging.debug(f"File recognized as VIDEO ({file_ext}). Extracting audio.")
            extracted_audio_path = extract_audio(file_path)
            if extracted_audio_path:
                audio_path_to_process = extracted_audio_path
                temp_audio_created = True
            else:
                logging.error(f"❌ Audio extraction failed for {file_path}. Cannot proceed with transcription.")
                return # Cannot proceed if audio extraction failed
        else:
            logging.debug(f"File recognized as AUDIO ({file_ext} or unknown). Using file directly.")

        # Transcribe audio using faster-whisper's segment-level detection
        full_transcribed_text, fw_lang_weights = transcribe_audio(audio_path_to_process)
        
        logging.debug(f"Faster-Whisper segment-wise language weights (Word Count): {fw_lang_weights}")

        # Secondary language detection on full text using langdetect
        langdetect_langs: Set[str] = set()
        cleaned_text = full_transcribed_text.replace("♪", "").replace("♫", "").strip()
        if cleaned_text:
            try:
                for d in detect_langs(cleaned_text):
                    if d.prob >= Config.LANGDETECT_PROB_THRESHOLD:
                        langdetect_langs.add(d.lang.split("-")[0])
            except Exception as e:
                logging.warning(f"Langdetect failed for text (likely short/unclear text). Error: {e}")
        
        logging.info(f"🌍 Detected languages (Faster-Whisper segments by word count): {sorted(fw_lang_weights.items(), key=lambda x: x[1], reverse=True)}")
        logging.info(f"🔎 Detected languages (Langdetect on full text, >{Config.LANGDETECT_PROB_THRESHOLD:.0%} prob): {sorted(list(langdetect_langs))}")

        # Combine language information for robust classification
        all_detected_langs_codes = set(fw_lang_weights.keys()).union(langdetect_langs)
        
        # Determine primary language from FW weights (most reliable for primary speech)
        langs_sorted_by_word_count = sorted(fw_lang_weights.items(), key=lambda x: x[1], reverse=True)
        primary_fw_lang = langs_sorted_by_word_count[0][0] if langs_sorted_by_word_count else "unknown"

        # Heuristic for vocals presence
        word_count = len(cleaned_text.split())
        has_vocals = word_count >= Config.MIN_WORD_COUNT_FOR_VOCALS
        logging.debug(f"Total cleaned transcription word count: {word_count}")
        logging.info(f"🎙 Vocals detected: {'yes' if has_vocals else 'no'} (Word count: {word_count} vs threshold: {Config.MIN_WORD_COUNT_FOR_VOCALS})")

        # Remix detection logic
        is_remix = False
        total_fw_word_count = sum(fw_lang_weights.values())
        if len(langs_sorted_by_word_count) >= 2 and total_fw_word_count > 0:
            second_lang_share = langs_sorted_by_word_count[1][1] / total_fw_word_count
            is_remix = second_lang_share >= Config.MIN_SECOND_LANG_SHARE_FOR_REMIX
            logging.debug(f"Second language share: {second_lang_share:.2f} (Remix threshold: {Config.MIN_SECOND_LANG_SHARE_FOR_REMIX:.2f}). Is Remix: {is_remix}")
        
        # Hindi/Urdu specific handling
        is_hindi_urdu_mix = False
        if "hi" in all_detected_langs_codes and "ur" in all_detected_langs_codes:
            # Check if both have significant presence based on FW word counts
            if fw_lang_weights.get("hi", 0) >= Config.MIN_WORD_COUNT_FOR_LANGUAGE and \
               fw_lang_weights.get("ur", 0) >= Config.MIN_WORD_COUNT_FOR_LANGUAGE:
                is_hindi_urdu_mix = True
        logging.debug(f"Hindi/Urdu mix detected: {is_hindi_urdu_mix}")

        # --- Final Classification Logic (Transcription-based) ---
        base_folder: Optional[str] = None
        routing_reason: str = ""
        classification_type: str = "UNKNOWN"
        
        if not has_vocals:
            base_folder = Config.BGM_DIR
            routing_reason = "No significant speech detected."
            classification_type = "PURE BGM"
        elif is_remix:
            # Use top languages for remix folder naming
            pretty_langs = " + ".join(Config.LANG_MAP.get(l, l) for l, _ in langs_sorted_by_word_count[:Config.REMIX_LANG_COUNT_FOR_NAME])
            base_folder = os.path.join(Config.REMIX_DIR, f"Remix ({pretty_langs})")
            routing_reason = f"Multiple languages with significant shares detected ({pretty_langs})."
            classification_type = "REMIX"
        elif is_hindi_urdu_mix: # Prioritize Hindi for folder if it's a specific Hi/Ur mix and not a broader remix
            language = Config.LANG_MAP.get("hi", "hindi")
            base_folder = os.path.join(Config.LANGUAGES_DIR, language, "vocals")
            routing_reason = "Hindi/Urdu mix detected (primarily)."
            classification_type = f"VOCALS ({language})"
        else: # Single dominant language (or "unknown" if no detection)
            language = Config.LANG_MAP.get(primary_fw_lang, primary_fw_lang)
            base_folder = os.path.join(Config.LANGUAGES_DIR, language, "vocals")
            routing_reason = f"Primary language detected ({language})."
            classification_type = f"VOCALS ({language})"

        if not base_folder:
            logging.error("❌ ERROR: Classification logic failed to assign a base folder.")
            return

        logging.info(f"➡️ CLASSIFICATION: {classification_type}. Reason: {routing_reason}")
        
        os.makedirs(base_folder, exist_ok=True)
        logging.debug(f"Target move directory: {base_folder}")

        # Move file (Final move for non-preclassified files)
        destination = resolve_destination_path(base_folder, original_filename)
        
        try:
            shutil.move(file_path, destination)
            if file_hash and file_hash not in PROCESSED_FILE_HASHES:
                PROCESSED_FILE_HASHES.add(file_hash)
                save_processed_hashes(PROCESSED_FILE_HASHES) # Save immediately for persistence
            logging.info(f"✅ Final move successful → {destination}")
        except Exception as e:
            logging.error(f"❌ ERROR: File move failed for {file_path}. Error: {e}")

    finally:
        # Robust Temporary File Management: Cleanup
        if temp_audio_created and os.path.exists(audio_path_to_process):
            try:
                os.remove(audio_path_to_process)
                logging.info(f"🗑️ Cleaned up temporary audio file: {audio_path_to_process}")
            except Exception as e:
                logging.error(f"❌ ERROR: Failed to remove temporary audio file {audio_path_to_process}. Error: {e}")