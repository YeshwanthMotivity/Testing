import logging
import subprocess
import shutil
import json
import hashlib
import tempfile
import argparse
from pathlib import Path

from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect_langs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Stability ----------
DetectorFactory.seed = 0
logger.debug("Stability seed set to 0")

# Global variable for processed hashes persistence
PROCESSED_FILE_HASHES_FILE = Path("processed_hashes.json")
PROCESSED_FILE_HASHES = set()

# ---------- Language map ----------
LANG_MAP = {
    "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
    "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
    "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
    "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
}

# Whisper Model (will be loaded in main based on configuration)
model = None

# ---------- Persistence for PROCESSED_FILE_HASHES ----------
def load_processed_hashes():
    """Loads processed file hashes from a persistent JSON file."""
    global PROCESSED_FILE_HASHES
    if PROCESSED_FILE_HASHES_FILE.exists():
        try:
            with PROCESSED_FILE_HASHES_FILE.open('r', encoding='utf-8') as f:
                PROCESSED_FILE_HASHES = set(json.load(f))
            logger.info(f"Loaded {len(PROCESSED_FILE_HASHES)} processed file hashes.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding processed hashes file: {e}. Starting with empty set.")
            PROCESSED_FILE_HASHES = set()
    else:
        logger.info("No processed hashes file found. Starting with empty set.")

def save_processed_hashes():
    """Saves processed file hashes to a persistent JSON file."""
    try:
        with PROCESSED_FILE_HASHES_FILE.open('w', encoding='utf-8') as f:
            json.dump(list(PROCESSED_FILE_HASHES), f, indent=4)
        logger.info(f"Saved {len(PROCESSED_FILE_HASHES)} processed file hashes.")
    except Exception as e:
        logger.error(f"Error saving processed hashes file: {e}")

# ---------- Centralized File Movement Logic ----------
def move_file_with_conflict_resolution(source_path: Path, destination_dir: Path) -> Path:
    """
    Moves a file to a destination directory, handling filename conflicts by adding a '_copyX' suffix.

    Args:
        source_path: The original path of the file to move.
        destination_dir: The directory where the file should be moved.

    Returns:
        The final destination path of the moved file.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source_path.name

    logger.debug(f"Initial destination path: {destination_path}")

    if destination_path.exists():
        base_name = destination_path.stem
        extension = destination_path.suffix
        i = 1
        while (destination_dir / f"{base_name}_copy{i}{extension}").exists():
            i += 1
        destination_path = destination_dir / f"{base_name}_copy{i}{extension}"
        logger.info(f"Renaming required due to conflict. New destination: {destination_path}")

    try:
        shutil.move(str(source_path), str(destination_path))
        logger.info(f"✅ Moved '{source_path.name}' to '{destination_path}'")
        return destination_path
    except Exception as e:
        logger.error(f"❌ ERROR: File move failed for '{source_path}'. Error: {e}")
        raise # Re-raise to indicate failure

# ---------- New function to get file hash ----------
def get_file_hash(file_path: Path) -> str | None:
    """Generates an MD5 hash for a file, handling large files efficiently."""
    try:
        hasher = hashlib.md5()
        with file_path.open('rb') as f:
            buf = f.read(65536) # Read in 64KB chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"❌ ERROR: Could not get hash for '{file_path}'. Error: {e}")
        return None

# ---------- Audio extractor ----------
def extract_audio(video_path: Path) -> Path:
    """
    Extracts mono 16kHz audio using ffmpeg to a temporary file.

    Args:
        video_path: Path to the input video file.

    Returns:
        Path to the temporary audio file.
    """
    logger.info(f"🎬 Extracting audio from video: {video_path.name}")
    
    temp_audio_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_audio_file = Path(tf.name)

        command = ["ffmpeg", "-i", str(video_path), "-ar", "16000", "-ac", "1", str(temp_audio_file), "-y"]
        logger.debug(f"FFmpeg command: {' '.join(command)}")

        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info(f"✅ Extracted audio to '{temp_audio_file.name}'")
        return temp_audio_file
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ ERROR: Audio extraction failed for '{video_path}'. FFmpeg error: {e}")
        if temp_audio_file and temp_audio_file.exists():
            temp_audio_file.unlink() # Clean up failed temp file
        raise
    except Exception as e:
        logger.error(f"❌ ERROR: An unexpected error occurred during audio extraction for '{video_path.name}': {e}")
        if temp_audio_file and temp_audio_file.exists():
            temp_audio_file.unlink()
        raise

# ---------- Transcribe with segment-level language detection ----------
def transcribe_audio(audio_path: Path):
    """
    Transcribe audio using faster-whisper and perform segment-level language detection with langdetect.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A tuple containing:
            - The full transcribed text (str).
            - A dictionary of language codes and their aggregated word counts (dict).
    """
    logger.info(f"Starting transcription for: {audio_path.name}")
    
    global model

    if model is None:
        logger.error("Whisper model not loaded. Cannot transcribe.")
        raise RuntimeError("Whisper model not loaded.")

    segments_generator, info = model.transcribe(str(audio_path), beam_size=5, return_segments=True)

    logger.debug(f"Faster-Whisper model detected dominant language for entire file: {info.language} with probability {info.language_probability:.2f}")

    all_texts = []
    lang_weights = {} # Will store language codes -> word count
    segment_count = 0
    
    # Iterate through segments returned by faster-whisper
    for segment in segments_generator:
        segment_count += 1
        seg_text = segment.text.strip()
        
        if not seg_text:
            logger.debug(f"Segment {segment_count}: Skipped (No text found).")
            continue

        all_texts.append(seg_text)
        word_count = len(seg_text.split())

        # Use langdetect for segment-level language detection
        try:
            detected_langs = detect_langs(seg_text)
            if detected_langs:
                # Take the top detected language for the segment, weighted by its confidence
                primary_segment_lang = detected_langs[0].lang.split("-")[0]
                lang_weights[primary_segment_lang] = lang_weights.get(primary_segment_lang, 0) + word_count
                logger.debug(f"Segment {segment_count}: LangDetect primary: {primary_segment_lang}, Words: {word_count}. Text: '{seg_text[:50]}...'")
            else:
                # Fallback if langdetect fails for a segment, use faster-whisper's overall language
                lang_weights[info.language] = lang_weights.get(info.language, 0) + word_count
                logger.debug(f"Segment {segment_count}: LangDetect failed, using Whisper overall '{info.language}', Words: {word_count}. Text: '{seg_text[:50]}...'")

        except Exception as e: # Catch any langdetect errors (e.g., text too short/no-lang)
            lang_weights[info.language] = lang_weights.get(info.language, 0) + word_count
            logger.debug(f"Segment {segment_count}: LangDetect error '{e}', using Whisper overall '{info.language}', Words: {word_count}. Text: '{seg_text[:50]}...'")

    full_transcribed_text = " ".join(all_texts)
    logger.debug(f"Total segments processed: {segment_count}")
    logger.debug(f"Aggregated language weights: {lang_weights}")
    return full_transcribed_text, lang_weights

# ---------- Core processing logic ----------
def process_file(file_path: Path, config: argparse.Namespace):
    """
    Processes a single audio/video file for language detection and moves it to the appropriate output folder.

    Args:
        file_path: Path to the file to process.
        config: Configuration namespace from argparse.
    """
    logger.info(f"\n=======================================================")
    logger.info(f"🚀 Processing file: {file_path.name}")
    logger.info(f"   FULL PATH: {file_path}")
    logger.info(f"=======================================================")
    
    # Check for duplicates first
    file_hash = get_file_hash(file_path)
    if file_hash:
        logger.debug(f"File hash generated: {file_hash}")
        if file_hash in PROCESSED_FILE_HASHES:
            logger.warning("⚠️ DUPLICATE DETECTED! File with this hash has already been processed.")
            try:
                file_path.unlink() # Use pathlib for deletion
                logger.info(f"🗑️ Successfully deleted duplicate file: {file_path.name}")
            except Exception as e:
                logger.error(f"❌ ERROR: Failed to delete duplicate file: '{file_path.name}'. Error: {e}")
            return # Exit the function immediately
        else:
            PROCESSED_FILE_HASHES.add(file_hash)
            logger.info("File is unique. Proceeding with processing.")
    else:
        logger.warning("Could not generate file hash. Skipping duplicate check for this file.")

    # --- NEW FEATURE: Filename-based Remix Check (Pre-Transcription) ---
    file_name_no_ext = file_path.stem
    if " X " in file_name_no_ext.upper():
        logger.info("✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains ' X ' pattern.")
        
        try:
            move_file_with_conflict_resolution(file_path, config.remix_dir)
            return # IMPORTANT: Exit the function after moving
        except Exception as e:
            logger.error(f"❌ ERROR: Failed to move pre-classified remix file: {e}")
            return
    
    # --- CONTINUE WITH REGULAR PROCESSING (If not a filename-based remix) ---
    
    audio_path = None
    temp_audio_file = None # To keep track of temp file for cleanup
    
    try:
        if file_path.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov"]:
            logger.debug(f"File recognized as VIDEO ({file_path.suffix}). Extracting audio.")
            temp_audio_file = extract_audio(file_path)
            audio_path = temp_audio_file
        else:
            logger.debug(f"File recognized as AUDIO ({file_path.suffix} or unknown). Using file directly.")
            audio_path = file_path

        # Transcribe audio 
        text, lang_weights = transcribe_audio(audio_path)
        
        langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
        detected_langs = [l for l, _ in langs_sorted]
        
        logger.debug(f"Sorted language weights (Word Count): {langs_sorted}")
        logger.info(f"🌍 Languages detected with weights: {lang_weights}")

        # Cleanup + heuristic
        cleaned_text = text.replace("♪", "").replace("♫", "").strip()
        word_count = len(cleaned_text.split())
        
        logger.debug(f"Total cleaned transcription length: {len(cleaned_text)} characters")
        logger.info(f"🔍 Word count: {word_count}")

        # --- Classification Logic (Transcription-based) ---
        base_folder = None
        
        if "hi" in detected_langs and "ur" in detected_langs and word_count > config.min_vocals_word_count:
            primary_lang = "hi"
            language = LANG_MAP.get(primary_lang, primary_lang)
            base_folder = config.languages_dir / language / "vocals"
            logger.info(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Hindi/Urdu Mix and word count > {config.min_vocals_word_count}.")
        elif word_count <= config.min_vocals_word_count:
            base_folder = config.bgm_dir
            logger.info(f"➡️ CLASSIFICATION: PURE BGM. Reason: Word count <= {config.min_vocals_word_count}.")
        elif len(detected_langs) >= 2: # Multi-language detection now based on langdetect for segments
            base_folder = config.remix_dir
            logger.info(f"➡️ CLASSIFICATION: REMIX. Reason: Multiple Languages detected ({', '.join(detected_langs)}).")
        else:
            primary_lang = detected_langs[0] if detected_langs else "unknown"
            language = LANG_MAP.get(primary_lang, primary_lang)
            base_folder = config.languages_dir / language / "vocals"
            logger.info(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Primary language detected.")

        if not base_folder:
            logger.error("❌ ERROR: Classification logic failed to assign a base folder.")
            return

        # Move file
        move_file_with_conflict_resolution(file_path, base_folder)

    except Exception as e:
        logger.error(f"❌ An error occurred during processing of '{file_path.name}': {e}", exc_info=True)
    finally:
        if temp_audio_file and temp_audio_file.exists():
            try:
                temp_audio_file.unlink()
                logger.debug(f"Cleaned up temporary audio file: '{temp_audio_file.name}'")
            except Exception as e:
                logger.error(f"❌ ERROR: Failed to clean up temporary audio file '{temp_audio_file.name}': {e}")

# ---------- Main function and configurability ----------
def main():
    global model

    parser = argparse.ArgumentParser(description="Process audio/video files for language detection and classification.")
    parser.add_argument("--input_dir", type=Path, default="input",
                        help="Directory containing files to process.")
    parser.add_argument("--output_dir", type=Path, default=Path("data") / "output",
                        help="Base output directory for classified files.")
    parser.add_argument("--whisper_model", type=str, default="small",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Size of the Faster-Whisper model to load.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to run Whisper model on (cpu or cuda).")
    parser.add_argument("--compute_type", type=str, default="int8",
                        choices=["int8", "float16", "float32"],
                        help="Compute type for Whisper model (int8, float16, float32). int8 is fastest for CPU.")
    parser.add_argument("--min_vocals_word_count", type=int, default=10,
                        help="Minimum word count to classify as vocals. Below this, it's considered BGM.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level.")

    args = parser.parse_args()

    # Configure logging based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    logger.info(f"Logging level set to {args.log_level.upper()}")

    # Define output directories based on configurable base output directory
    args.languages_dir = args.output_dir / "Languages"
    args.bgm_dir = args.output_dir / "bgm"
    args.remix_dir = args.output_dir / "remix"

    # Create output directories
    args.languages_dir.mkdir(parents=True, exist_ok=True)
    args.bgm_dir.mkdir(parents=True, exist_ok=True)
    args.remix_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories setup: Languages={args.languages_dir}, BGM={args.bgm_dir}, REMIX={args.remix_dir}")

    # Load Whisper model
    logger.info(f"📥 Loading Whisper model: {args.whisper_model} on {args.device} with {args.compute_type} compute type...")
    try:
        model = WhisperModel(
            args.whisper_model,
            device=args.device,
            compute_type=args.compute_type
        )
        logger.info("✅ Whisper model loaded successfully.")
    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR: Failed to load Whisper model: {e}")
        return

    # Load processed hashes at startup
    load_processed_hashes()

    # Process files in the input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    files_to_process = []
    for f in input_path.iterdir():
        if f.is_file() and f.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov", ".wav", ".mp3", ".flac", ".aac"]:
            files_to_process.append(f)
    
    if not files_to_process:
        logger.info(f"No supported media files found in input directory: {input_path}")
        return

    logger.info(f"Found {len(files_to_process)} files to process in {input_path}")

    for file_path in files_to_process:
        try:
            process_file(file_path, args)
        except Exception as e:
            logger.error(f"❌ Failed to process '{file_path.name}': {e}", exc_info=True)

    # Save processed hashes before exiting
    save_processed_hashes()

if __name__ == "__main__":
    main()