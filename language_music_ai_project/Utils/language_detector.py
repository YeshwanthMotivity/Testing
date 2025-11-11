python
import os
import shutil
import subprocess
from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect_langs
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Stability ----------
DetectorFactory.seed = 0
logger.debug("Stability seed set to 0")

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

LANGUAGES_DIR = os.path.join(OUTPUT_DIR, "Languages")
BGM_DIR = os.path.join(OUTPUT_DIR, "bgm")
REMIX_DIR = os.path.join(OUTPUT_DIR, "remix")

logger.debug(f"BASE_DIR: {BASE_DIR}")
logger.debug(f"Output directories setup: Languages={LANGUAGES_DIR}, BGM={BGM_DIR}, REMIX={REMIX_DIR}")

os.makedirs(LANGUAGES_DIR, exist_ok=True)
os.makedirs(BGM_DIR, exist_ok=True)
os.makedirs(REMIX_DIR, exist_ok=True)

# ---------- Lang map (remains the same) ----------
LANG_MAP = {
    "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
    "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
    "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
    "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
}

# ---------- Whisper (Updated for faster-whisper) ----------
logger.info("📥 Loading Whisper model...")
model = WhisperModel(
    "small",             # Keeps your desired 'small' model size
    device="cpu",        # Explicitly targets the CPU
    compute_type="int8"  # Enables INT8 quantization for max speed
)
logger.info("✅ Whisper model loaded (Model: small, Device: CPU, Compute: INT8)")

# ---------- Helper: File moving and renaming logic ----------
def _move_file_to_destination(source_path, target_dir):
    """
    Moves a file to the target directory, handling potential filename conflicts
    by appending '_copy{i}' to the filename.
    """
    os.makedirs(target_dir, exist_ok=True)
    destination = os.path.join(target_dir, os.path.basename(source_path))

    original_destination = destination
    if os.path.exists(destination):
        base, ext = os.path.splitext(destination)
        i = 1
        while os.path.exists(f"{base}_copy{i}{ext}"):
            i += 1
        destination = f"{base}_copy{i}{ext}"
        logger.warning(f"File '{os.path.basename(original_destination)}' already exists in '{target_dir}'. Renaming to '{os.path.basename(destination)}'.")

    try:
        shutil.move(source_path, destination)
        logger.info(f"✅ Moved '{os.path.basename(source_path)}' → '{destination}'")
        return destination
    except Exception as e:
        logger.error(f"❌ ERROR: File move failed for '{source_path}' to '{destination}'. Error: {e}")
        raise

# ---------- Audio extractor ----------
def extract_audio(video_path):
    """
    Extracts mono 16kHz audio using ffmpeg to a temporary file.
    Returns the path to the temporary audio file. Raises an exception on failure.
    """
    logger.info(f"🎬 Extracting audio from video: {video_path}")
    
    # Create a temporary file for the audio
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_path = temp_audio_file.name
    temp_audio_file.close() # Close the file handle so ffmpeg can write to it

    command = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", temp_audio_path, "-y"]
    logger.debug(f"FFmpeg command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
            logger.info(f"✅ Extracted audio to temporary file: {temp_audio_path}")
            return temp_audio_path
        else:
            raise RuntimeError(f"FFmpeg produced an empty or non-existent audio file: {temp_audio_path}. Stderr: {result.stderr.decode()}")
    except subprocess.CalledProcessError as e:
        logger.critical(f"❌ ERROR: FFmpeg failed to extract audio from '{video_path}'. Command: {' '.join(command)}. Stderr: {e.stderr.decode()}")
        # Clean up the temporary file if it was created but extraction failed
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise RuntimeError(f"FFmpeg extraction failed for {video_path}") from e
    except Exception as e:
        logger.critical(f"❌ ERROR: An unexpected error occurred during audio extraction from '{video_path}': {e}")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise

# ---------- Transcribe (New Optimized Function) ----------
def transcribe_audio(audio_path):
    """
    Transcribe entire audio using faster-whisper's optimized engine
    and perform an independent multi-language detection on the full text.
    """
    logger.info(f"--- DEBUG: Starting transcription for: {audio_path}")
    
    segments_generator, info = model.transcribe(audio_path, beam_size=5)

    logger.debug(f"Faster-Whisper's dominant language for entire file: {info.language}")
    # Corrected debug message for language_probability
    logger.debug(f"Faster-Whisper's dominant language probability: {info.language_probability:.2f}")

    whisper_langs_detected = {}
    full_text = []
    segment_count = 0
    
    for segment in segments_generator:
        segment_count += 1
        seg_text = segment.text.strip()
        
        # faster-whisper provides overall language in 'info', so we use that for primary language tracking
        seg_lang = info.language
        
        if seg_text:
            word_count = len(seg_text.split())
            full_text.append(seg_text)
            whisper_langs_detected[seg_lang] = whisper_langs_detected.get(seg_lang, 0) + word_count
            logger.debug(f"   🎧 Segment {segment_count}: Detected {seg_lang}, Words: {word_count}. Text: '{seg_text[:50]}...'")
        else:
             logger.debug(f"   🎧 Segment {segment_count}: Skipped (No text found).")

    cleaned_text = " ".join(full_text).replace("♪", "").replace("♫", "").strip()
    logger.debug(f"Total segments processed by Faster-Whisper: {segment_count}")

    # Independent multi-language detection using langdetect on the full cleaned text
    langdetect_langs = set()
    if cleaned_text:
        try:
            detected_with_probs = detect_langs(cleaned_text)
            for d in detected_with_probs:
                langdetect_langs.add(d.lang.split("-")[0])
            logger.debug(f"Langdetect: Found languages {detected_with_probs}")
        except Exception as e:
            logger.warning(f"Langdetect failed for text snippet: '{cleaned_text[:100]}...'. Error: {e}")
            pass # Continue without langdetect results if it fails

    # Combine results: Whisper's dominant language with langdetect's broader view
    # Prioritize whisper's dominant language, but add any strong secondary languages from langdetect
    final_detected_langs = list(whisper_langs_detected.keys())
    for ld_lang in langdetect_langs:
        if ld_lang not in final_detected_langs:
            # We don't have word count for langdetect's contribution,
            # so we'll just ensure it's considered in the list of detected languages
            final_detected_langs.append(ld_lang)

    # For the `lang_weights` dict, we'll primarily stick to what Whisper provided,
    # as its segmentation gives us better granular control for remix logic based on actual speech.
    # The `final_detected_langs` list (derived from langdetect) will be used to signal remix.
    
    return cleaned_text, whisper_langs_detected, list(langdetect_langs)

# ---------- Core ----------
def process_file(file_path):
    logger.info(f"\n=======================================================")
    logger.info(f"🚀 Processing file: {os.path.basename(file_path)}")
    logger.info(f"   FULL PATH: {file_path}")
    logger.info(f"=======================================================")
    
    original_file_path = file_path # Keep original path for final move attempt
    temp_audio_path = None # Initialize to ensure cleanup

    try:
        file_name_no_ext = os.path.splitext(os.path.basename(original_file_path))[0]
        
        # --- FILENAME-BASED REMIX CHECK (Pre-Transcription) ---
        if " X " in file_name_no_ext.upper():
            logger.info("✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains ' X ' pattern.")
            _move_file_to_destination(original_file_path, REMIX_DIR)
            return # Exit after moving
        
        file_ext = os.path.splitext(original_file_path)[1].lower()

        # Extract audio if it's a video
        if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
            logger.debug(f"File recognized as VIDEO ({file_ext}). Extracting audio.")
            temp_audio_path = extract_audio(original_file_path)
            audio_source_for_transcription = temp_audio_path
        elif file_ext in [".wav", ".mp3", ".flac", ".aac"]:
            logger.debug(f"File recognized as AUDIO ({file_ext}). Using file directly.")
            audio_source_for_transcription = original_file_path
        else:
            logger.warning(f"❌ Unsupported file format: {file_ext}. Skipping file '{original_file_path}'.")
            return

        # Transcribe audio and get language info
        text, whisper_lang_weights, langdetect_langs = transcribe_audio(audio_source_for_transcription)
        
        # Sort Whisper languages by word count contribution
        whisper_langs_sorted = sorted(whisper_lang_weights.items(), key=lambda x: x[1], reverse=True)
        dominant_whisper_lang = whisper_langs_sorted[0][0] if whisper_langs_sorted else "unknown"

        logger.debug(f"Sorted Whisper language weights (Word Count): {whisper_langs_sorted}")
        logger.info(f"🌍 Whisper-detected languages with weights: {whisper_lang_weights}")
        logger.info(f"🌍 Langdetect-detected languages (independent check): {langdetect_langs}")

        # Cleanup + heuristic
        word_count = len(text.split())
        
        logger.debug(f"Total cleaned transcription length: {len(text)} characters")
        logger.info(f"🔍 Total transcribed word count: {word_count}")

        # --- Classification Logic (Transcription-based) ---
        base_folder = None
        
        # Check for multi-language specifically for remix, using both whisper's info and langdetect
        # A remix is indicated if Faster-Whisper reports multiple languages (less likely) or
        # if langdetect finds multiple distinct languages.
        is_remix = False
        if len(langdetect_langs) > 1:
            is_remix = True
            logger.info(f"Remix detected by Langdetect due to multiple languages: {langdetect_langs}")
        elif len(whisper_langs_sorted) > 1:
            # Fallback if whisper somehow gives multiple different language codes (less common for info.language)
            total_weight = sum(whisper_lang_weights.values())
            if total_weight > 0 and len(whisper_langs_sorted) >= 2 and (whisper_langs_sorted[1][1] / total_weight) >= 0.20:
                is_remix = True
                logger.info(f"Remix detected by Faster-Whisper's language weights (secondary language share > 20%): {whisper_langs_sorted}")

        if "hi" in langdetect_langs and "ur" in langdetect_langs and word_count > 10:
            primary_lang = "hi" # Prioritize Hindi for Hindi/Urdu mixes as per previous logic
            language = LANG_MAP.get(primary_lang, primary_lang)
            base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
            logger.info(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Hindi/Urdu Mix detected by Langdetect and word count > 10.")
        elif word_count <= 10:
            base_folder = BGM_DIR
            logger.info("➡️ CLASSIFICATION: PURE BGM. Reason: Word count <= 10.")
        elif is_remix:
            # Use detected languages from langdetect for remix folder naming if available, otherwise whisper's.
            remix_langs_for_name = langdetect_langs if langdetect_langs else [l for l, _ in whisper_langs_sorted]
            pretty_langs = " + ".join(LANG_MAP.get(l, l) for l in remix_langs_for_name[:4])
            base_folder = os.path.join(REMIX_DIR, f"Remix ({pretty_langs})")
            logger.info(f"➡️ CLASSIFICATION: REMIX. Reason: Multiple languages detected ({', '.join(remix_langs_for_name)}).")
        else:
            primary_lang_code = dominant_whisper_lang # Fallback to whisper's dominant if not remix or BGM
            language = LANG_MAP.get(primary_lang_code, primary_lang_code)
            base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
            logger.info(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Primary language detected by Whisper.")

        if not base_folder:
            logger.critical("❌ ERROR: Classification logic failed to assign a base folder.")
            return

        _move_file_to_destination(original_file_path, base_folder)

    except Exception as e:
        logger.exception(f"❌ CRITICAL ERROR: Processing failed for file '{original_file_path}'. Error: {e}")
    finally:
        # Clean up temporary audio file if it was created
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.debug(f"🗑️ Cleaned up temporary audio file: {temp_audio_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove temporary audio file '{temp_audio_path}'. Error: {e}")