import os
import shutil
import subprocess
import logging
import hashlib
import tempfile
import json

from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect_langs

# ---------- Logging Configuration ----------
# Configure logging before any other module prints, so all output goes through logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Stability ----------
DetectorFactory.seed = 0
logger.debug("Stability seed set to 0.")

# ---------- Configuration (Can be externalized further to a separate config file) ----------
CONFIG = {
    "whisper_model_size": "small",
    "whisper_device": "cpu",
    "whisper_compute_type": "int8",
    "whisper_beam_size": 5,
    "bgm_word_count_threshold": 10,
    "remix_filename_pattern": " X ", # Used for early exit remix detection (e.g., "song X remix")
    "min_second_lang_share_for_remix": 0.20, # Minimum share for a second language to classify as remix
    "processed_hashes_file": "processed_hashes.json", # File to store processed hashes
    "base_output_dir": "data/output",
}

# ---------- Language Map (Static lookup data) ----------
LANG_MAP = {
    "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
    "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
    "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
    "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
}

# ----------------------------------------------------------------------------------------------------
# Encapsulated Logic in a Class: MediaClassifier
# ----------------------------------------------------------------------------------------------------

class MediaClassifier:
    def __init__(self, config=CONFIG):
        self.config = config
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Output Directories ---
        self.output_dir = os.path.join(self.base_dir, self.config["base_output_dir"])
        self.languages_dir = os.path.join(self.output_dir, "Languages")
        self.bgm_dir = os.path.join(self.output_dir, "bgm")
        self.remix_dir = os.path.join(self.output_dir, "remix")
        
        os.makedirs(self.languages_dir, exist_ok=True)
        os.makedirs(self.bgm_dir, exist_ok=True)
        os.makedirs(self.remix_dir, exist_ok=True)
        logger.info(f"Output directories setup: Languages={self.languages_dir}, BGM={self.bgm_dir}, REMIX={self.remix_dir}")

        # --- Whisper Model ---
        logger.info("📥 Loading Whisper model...")
        self.model = WhisperModel(
            self.config["whisper_model_size"],
            device=self.config["whisper_device"],
            compute_type=self.config["whisper_compute_type"]
        )
        logger.info(f"✅ Whisper model loaded (Model: {self.config['whisper_model_size']}, Device: {self.config['whisper_device']}, Compute: {self.config['whisper_compute_type']})")

        # --- Processed Hashes for Duplicate Detection ---
        self.processed_file_hashes = set()
        self._load_hashes()
        logger.info(f"Loaded {len(self.processed_file_hashes)} previously processed file hashes.")

    def _load_hashes(self):
        """Loads previously processed file hashes from a persistent storage."""
        hashes_file_path = os.path.join(self.base_dir, self.config["processed_hashes_file"])
        if os.path.exists(hashes_file_path):
            try:
                with open(hashes_file_path, 'r', encoding='utf-8') as f:
                    self.processed_file_hashes = set(json.load(f))
                logger.debug(f"Successfully loaded hashes from {hashes_file_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading hashes from {hashes_file_path}: {e}")
        else:
            logger.info(f"Hashes file not found at {hashes_file_path}. Starting with empty hash set.")

    def _save_hashes(self):
        """Saves current processed file hashes to persistent storage."""
        hashes_file_path = os.path.join(self.base_dir, self.config["processed_hashes_file"])
        try:
            # Convert set to list for JSON serialization
            with open(hashes_file_path, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed_file_hashes), f, indent=2)
            logger.info(f"Successfully saved {len(self.processed_file_hashes)} hashes to {hashes_file_path}")
        except IOError as e:
            logger.error(f"Error saving hashes to {hashes_file_path}: {e}")

    def __del__(self):
        """Ensures hashes are saved when the object is garbage collected or script exits."""
        self._save_hashes()
        logger.debug("MediaClassifier object destroyed, hashes saved.")
    
    def get_file_hash(self, file_path):
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
            logger.error(f"File not found when trying to hash: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Could not get hash for {file_path}. Error: {e}")
            return None

    def extract_audio(self, video_path):
        """
        Extracts mono 16kHz audio using ffmpeg into a temporary file.
        Returns the path to the temporary audio file.
        """
        logger.info(f"🎬 Extracting audio from video: {video_path}")
        
        # Use NamedTemporaryFile for automatic cleanup and unique names
        # delete=False so that ffmpeg can open it, and we manually delete it later in `finally` block
        temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_audio_path = temp_audio_file.name
        temp_audio_file.close() # Close the file handle, ffmpeg will open it

        command = [
            "ffmpeg", "-i", video_path,
            "-ar", "16000", "-ac", "1",
            output_audio_path, "-y"
        ]
        logger.debug(f"FFmpeg command: {' '.join(command)}")
        
        # Capture stderr for improved error reporting
        process = subprocess.run(command, capture_output=True, text=True)
        
        if process.returncode == 0:
            logger.info(f"✅ Extracted audio → {output_audio_path}")
        else:
            logger.error(f"❌ ERROR: Audio extraction failed for {video_path}.")
            logger.error(f"FFmpeg stdout: {process.stdout.strip()}")
            logger.error(f"FFmpeg stderr: {process.stderr.strip()}")
            if os.path.exists(output_audio_path):
                os.remove(output_audio_path) # Clean up partial temp file if created
            raise RuntimeError(f"FFmpeg failed with exit code {process.returncode}: {process.stderr.strip()}")
        
        return output_audio_path

    def transcribe_audio(self, audio_path):
        """
        Transcribes audio using faster-whisper.
        Populates language weights based on segment-level language detection
        from faster-whisper.
        """
        logger.info(f"--- Starting transcription for: {os.path.basename(audio_path)}")
        
        # Use vad_filter=True to enable segment-level language detection,
        # and allow Whisper to detect language per segment without pre-specifying.
        segments_generator, info = self.model.transcribe(
            audio_path,
            beam_size=self.config["whisper_beam_size"],
            vad_filter=True, # Enable VAD filter for better segmenting and language per segment
            language=None # Allows Whisper to detect language per segment initially
        )

        overall_lang = info.language
        logger.info(f"--- Whisper's overall detected language for file: {overall_lang} (prob: {info.language_probability:.2f})")

        texts = []
        whisper_segment_langs_weighted = {} # {lang_code: total_word_count}

        for i, segment in enumerate(segments_generator):
            seg_text = segment.text.strip()
            # Use segment.language for more granular language detection provided by faster-whisper
            seg_lang = segment.language 
            
            if seg_text:
                word_count = len(seg_text.split())
                texts.append(seg_text)
                whisper_segment_langs_weighted[seg_lang] = whisper_segment_langs_weighted.get(seg_lang, 0) + word_count
                logger.debug(f"   🎧 Segment {i+1}: Lang='{seg_lang}', Words={word_count}, Text='{seg_text[:70]}...'")
            else:
                logger.debug(f"   🎧 Segment {i+1}: Skipped (No text found).")

        full_transcription_text = " ".join(texts).strip()

        # Refine Multilingual Detection: Use `langdetect.detect_langs` on the full text
        # for a secondary confirmation or to catch languages Whisper might have missed entirely
        # in segments due to very short or very mixed utterances.
        if full_transcription_text:
            try:
                langdetect_results = detect_langs(full_transcription_text)
                # langdetect_results would be like [en:0.999992, fr:6.89e-06]
                logger.debug(f"--- LangDetect on full text: {langdetect_results}")
                
                # Integrate langdetect results for remix classification if they are significant
                for ld_result in langdetect_results:
                    if ld_result.prob > 0.15: # Consider languages with at least 15% probability from langdetect
                        lang_code = ld_result.lang.split("-")[0]
                        # If langdetect finds a language that Whisper didn't assign any words to,
                        # give it a small artificial weight to be considered for remix.
                        # This ensures multi-language detection is more robust.
                        if lang_code not in whisper_segment_langs_weighted:
                            whisper_segment_langs_weighted[lang_code] = whisper_segment_langs_weighted.get(lang_code, 0) + 5 # Add a small weight (e.g., 5 words)
            except Exception as e:
                logger.warning(f"LangDetect failed on full text: {e}")
        else:
            logger.info("No transcription text to run LangDetect on.")


        logger.info(f"--- Transcription complete. Languages (from segments, weighted): {whisper_segment_langs_weighted}")
        return full_transcription_text, whisper_segment_langs_weighted

    def process_file(self, file_path):
        """
        Processes a single media file: checks for duplicates, extracts audio,
        transcribes, detects languages, and moves the file to the appropriate folder.
        """
        logger.info(f"\n=======================================================")
        logger.info(f"🚀 Processing file: {os.path.basename(file_path)}")
        logger.info(f"   FULL PATH: {file_path}")
        logger.info(f"=======================================================")
        
        # 1. CHECK FOR DUPLICATES BEFORE ANYTHING ELSE
        file_hash = self.get_file_hash(file_path)
        if file_hash:
            logger.debug(f"File hash generated: {file_hash}")
            if file_hash in self.processed_file_hashes:
                logger.warning("⚠️ DUPLICATE DETECTED! File with this hash has already been processed.")
                try:
                    os.remove(file_path)
                    logger.info(f"🗑️ Successfully deleted duplicate file: {file_path}")
                except Exception as e:
                    logger.error(f"❌ ERROR: Failed to delete duplicate file: {file_path}. Error: {e}")
                return # Exit the function immediately
            else:
                # Add the new file's hash to our set
                self.processed_file_hashes.add(file_hash)
                logger.info("✅ File is unique. Proceeding with processing.")
        else:
            logger.error(f"Could not generate hash for {file_path}. Skipping duplicate check. File will be processed and its hash will not be saved.")
            # If hash generation fails, we still process the file but cannot mark it as processed persistently.

        file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        
        # 2. FILENAME-BASED REMIX CHECK (Early exit)
        if self.config["remix_filename_pattern"] in file_name_no_ext.upper():
            logger.info(f"✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains '{self.config['remix_filename_pattern']}' pattern.")
            base_folder = self.remix_dir
            self._move_file(file_path, base_folder)
            return # IMPORTANT: Exit the function after moving
        
        # 3. EXTRACT AUDIO IF VIDEO
        file_ext = os.path.splitext(file_path)[1].lower()
        temp_audio_path = None # Initialize to None for cleanup
        
        try:
            if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
                logger.debug(f"File recognized as VIDEO ({file_ext}). Extracting audio.")
                temp_audio_path = self.extract_audio(file_path)
                audio_source_path = temp_audio_path
            else:
                logger.debug(f"File recognized as AUDIO ({file_ext} or unknown). Using file directly.")
                audio_source_path = file_path

            # 4. TRANSCRIBE AUDIO
            text, lang_weights = self.transcribe_audio(audio_source_path)
            
            # Sort detected languages by their accumulated word count
            langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
            detected_langs_codes = [l for l, _ in langs_sorted] # e.g., ['en', 'hi']
            
            logger.debug(f"Sorted language weights (Word Count): {langs_sorted}")
            logger.info(f"🌍 Languages detected (Whisper segments & LangDetect hints): {lang_weights}")

            # 5. CLEANUP + HEURISTIC
            cleaned_text = text.replace("♪", "").replace("♫", "").strip()
            word_count = len(cleaned_text.split())
            
            logger.debug(f"Total cleaned transcription length: {len(cleaned_text)} characters")
            logger.info(f"🔍 Word count: {word_count}")

            # 6. CLASSIFICATION LOGIC (Transcription-based)
            base_folder = None
            
            # Special case: Hindi/Urdu. If both are significantly present and text exists.
            if "hi" in detected_langs_codes and "ur" in detected_langs_codes and word_count > self.config["bgm_word_count_threshold"]:
                primary_lang = "hi" # Arbitrarily pick hi as primary if both are present
                language = LANG_MAP.get(primary_lang, primary_lang)
                base_folder = os.path.join(self.languages_dir, language, "vocals")
                logger.info(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Hindi/Urdu Mix and word count > {self.config['bgm_word_count_threshold']}.")
            # Case: Pure BGM (very few words)
            elif word_count <= self.config["bgm_word_count_threshold"]:
                base_folder = self.bgm_dir
                logger.info(f"➡️ CLASSIFICATION: PURE BGM. Reason: Word count <= {self.config['bgm_word_count_threshold']}.")
            # Case: Remix (multiple languages detected based on segment weights and share threshold)
            else: # If not BGM and not Hindi/Urdu special, check for general remix or single language
                is_remix_by_share = False
                total_weight = sum(lang_weights.values())
                if len(langs_sorted) >= 2 and total_weight > 0:
                    second_lang_share = langs_sorted[1][1] / total_weight
                    is_remix_by_share = second_lang_share >= self.config["min_second_lang_share_for_remix"]
                    logger.debug(f"Second language share: {second_lang_share:.2f} (Threshold: {self.config['min_second_lang_share_for_remix']})")

                if is_remix_by_share:
                    # Construct a more descriptive remix folder name
                    # Take top 2 languages for the folder name
                    pretty_langs = [LANG_MAP.get(l, l) for l, _ in langs_sorted[:2]]
                    remix_folder_name = f"Remix ({' + '.join(pretty_langs)})"
                    base_folder = os.path.join(self.remix_dir, remix_folder_name)
                    logger.info(f"➡️ CLASSIFICATION: REMIX. Reason: Multiple prominent languages detected (Share-based).")
                # Case: Single language vocals
                else:
                    primary_lang_code = detected_langs_codes[0] if detected_langs_codes else "unknown"
                    language_name = LANG_MAP.get(primary_lang_code, primary_lang_code)
                    base_folder = os.path.join(self.languages_dir, language_name, "vocals")
                    logger.info(f"➡️ CLASSIFICATION: VOCALS ({language_name}). Reason: Primary language detected.")

            if not base_folder:
                logger.error("❌ ERROR: Classification logic failed to assign a base folder.")
                return

            # 7. MOVE FILE
            self._move_file(file_path, base_folder)

        except RuntimeError as e: # Catch FFmpeg errors, etc.
            logger.error(f"❌ Processing failed for {file_path}: {e}")
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while processing {file_path}: {e}", exc_info=True)
        finally:
            # 8. CLEANUP TEMPORARY AUDIO FILE
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    logger.debug(f"Cleaned up temporary audio file: {temp_audio_path}")
                except Exception as e:
                    logger.error(f"❌ ERROR: Failed to remove temporary audio file {temp_audio_path}: {e}")

    def _move_file(self, source_path, destination_dir):
        """Helper to move a file, handling name conflicts."""
        logger.debug(f"Target move directory: {destination_dir}")
        os.makedirs(destination_dir, exist_ok=True)

        original_file_name = os.path.basename(source_path)
        destination_path = os.path.join(destination_dir, original_file_name)
        
        logger.debug(f"Initial destination path: {destination_path}")

        # Handle duplicate filenames in the target directory by appending a copy number
        if os.path.exists(destination_path):
            base, ext = os.path.splitext(destination_path)
            i = 1
            while os.path.exists(f"{base}_copy{i}{ext}"):
                i += 1
            destination_path = f"{base}_copy{i}{ext}"
            logger.warning(f"Filename collision, renaming to: {destination_path}")

        try:
            shutil.move(source_path, destination_path)
            logger.info(f"✅ Final move successful → {destination_path}")
        except Exception as e:
            logger.error(f"❌ ERROR: File move failed for {source_path}. Error: {e}")
            raise # Re-raise to allow main processing to catch if needed