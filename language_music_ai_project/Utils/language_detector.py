python
import os
import shutil
import subprocess
import hashlib
import json
import logging
import tempfile
from typing import Dict, List, Tuple

# Import the accepted configuration
import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Remove unused imports from langdetect
# from langdetect import DetectorFactory, detect_langs

class LanguageClassifier:
    """
    A class to classify audio/video files based on detected languages from speech
    using the Faster Whisper model. It handles audio extraction, transcription,
    language detection, duplicate file tracking, and file organization.
    """

    def __init__(self):
        """
        Initializes the LanguageClassifier, loads the Whisper model, and sets up
        output directories and persistent duplicate file tracking.
        """
        logger.info("Initializing LanguageClassifier...")

        # Initialize output directories
        self.output_dir = config.OUTPUT_DIR
        self.languages_dir = os.path.join(self.output_dir, config.LANGUAGES_SUBDIR)
        self.bgm_dir = os.path.join(self.output_dir, config.BGM_SUBDIR)
        self.remix_dir = os.path.join(self.output_dir, config.REMIX_SUBDIR)

        os.makedirs(self.languages_dir, exist_ok=True)
        os.makedirs(self.bgm_dir, exist_ok=True)
        os.makedirs(self.remix_dir, exist_ok=True)
        logger.info(f"Output directories ensured: {self.languages_dir}, {self.bgm_dir}, {self.remix_dir}")

        # Load Whisper model
        logger.info(f"Loading Whisper model: {config.WHISPER_MODEL_SIZE} on {config.WHISPER_DEVICE} with {config.WHISPER_COMPUTE_TYPE} compute type...")
        self.model = config.WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE
        )
        logger.info("Whisper model loaded successfully.")

        # Initialize persistent duplicate file tracking
        self.processed_file_hashes_path = config.PROCESSED_HASHES_FILE
        self.processed_file_hashes = self._load_processed_hashes()
        logger.info(f"Loaded {len(self.processed_file_hashes)} processed file hashes from {self.processed_file_hashes_path}")

    def _load_processed_hashes(self) -> set:
        """Loads processed file hashes from a JSON file."""
        if os.path.exists(self.processed_file_hashes_path):
            try:
                with open(self.processed_file_hashes_path, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except json.JSONDecodeError as e:
                logger.warning(f"Could not decode JSON from {self.processed_file_hashes_path}: {e}. Starting with empty hashes.")
                return set()
        return set()

    def _save_processed_hashes(self) -> None:
        """Saves current processed file hashes to a JSON file."""
        os.makedirs(os.path.dirname(self.processed_file_hashes_path), exist_ok=True)
        try:
            with open(self.processed_file_hashes_path, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed_file_hashes), f, indent=2)
            logger.debug(f"Saved {len(self.processed_file_hashes)} processed file hashes to {self.processed_file_hashes_path}")
        except IOError as e:
            logger.error(f"Failed to save processed hashes to {self.processed_file_hashes_path}: {e}")

    def _get_file_hash(self, file_path: str) -> str:
        """
        Generates an MD5 hash for a file, handling large files efficiently.

        Args:
            file_path: The path to the file.

        Returns:
            The MD5 hash of the file as a hexadecimal string, or an empty string if an error occurs.
        """
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                buf = f.read(config.HASH_CHUNK_SIZE)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(config.HASH_CHUNK_SIZE)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Could not get hash for {file_path}. Error: {e}")
            return ""

    def _get_unique_destination_path(self, destination_dir: str, original_filename: str) -> str:
        """
        Generates a unique destination path for a file by appending '_copyX' if needed.

        Args:
            destination_dir: The target directory.
            original_filename: The base filename (e.g., "audio.mp3").

        Returns:
            A unique file path in the destination directory.
        """
        base_name, ext = os.path.splitext(original_filename)
        destination = os.path.join(destination_dir, original_filename)
        
        if not os.path.exists(destination):
            return destination
        
        i = 1
        while os.path.exists(os.path.join(destination_dir, f"{base_name}_copy{i}{ext}")):
            i += 1
        return os.path.join(destination_dir, f"{base_name}_copy{i}{ext}")

    def extract_audio(self, video_path: str, output_audio_path: str) -> str:
        """
        Extracts mono 16kHz audio from a video file using ffmpeg.

        Args:
            video_path: The path to the input video file.
            output_audio_path: The desired path for the extracted audio file.

        Returns:
            The path to the extracted audio file.

        Raises:
            subprocess.CalledProcessError: If the ffmpeg command fails.
        """
        logger.info(f"🎬 Extracting audio from video: {video_path} to {output_audio_path}")
        command = [
            config.FFMPEG_PATH, "-i", video_path,
            "-ar", str(config.AUDIO_SAMPLE_RATE), "-ac", str(config.AUDIO_CHANNELS),
            output_audio_path, "-y"
        ]
        logger.debug(f"FFmpeg command: {' '.join(command)}")
        
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            logger.info(f"✅ Extracted audio → {output_audio_path}")
            return output_audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ ERROR: FFmpeg audio extraction failed for {video_path}. Error: {e}")
            raise

    def transcribe_audio(self, audio_path: str) -> Tuple[str, Dict[str, int]]:
        """
        Transcribes audio using faster-whisper's optimized engine and aggregates
        language detection from individual segments.

        Args:
            audio_path: The path to the audio file.

        Returns:
            A tuple containing:
            - The full transcribed text as a string.
            - A dictionary where keys are language codes (str) and values are
              the total word count detected for that language across all segments.
        """
        logger.info(f"Starting transcription for: {audio_path}")
        
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=config.WHISPER_BEAM_SIZE,
            condition_on_previous_text=False # Each segment is independent for better multi-language detection
        )

        logger.debug(f"Model overall detected language for file: {info.language}")
        logger.debug(f"Model transcription speed RTF: {info.language_probability}") # This is actually language probability, not RTF.

        langs_detected: Dict[str, int] = {}
        texts: List[str] = []
        segment_count = 0
        
        for segment in segments:
            segment_count += 1
            seg_text = segment.text.strip()
            # CORRECTED: Use segment.language instead of info.language for accurate multi-language detection
            seg_lang = segment.language 
            
            if seg_text:
                word_count = len(seg_text.split())
                texts.append(seg_text)
                langs_detected[seg_lang] = langs_detected.get(seg_lang, 0) + word_count
                logger.debug(f"   🎧 Segment {segment_count}: Detected {seg_lang}, Words: {word_count}. Text: '{seg_text[:50]}...'")
            else:
                logger.debug(f"   🎧 Segment {segment_count}: Skipped (No text found).")

        logger.info(f"Total segments processed: {segment_count}")
        return " ".join(texts), langs_detected

    def process_file(self, file_path: str) -> None:
        """
        Processes a single audio/video file to classify its language content
        (BGM, Vocals-Language, or Remix) and moves it to the appropriate directory.

        Args:
            file_path: The path to the file to process.
        """
        logger.info(f"\n=======================================================")
        logger.info(f"🚀 Processing file: {os.path.basename(file_path)}")
        logger.info(f"   FULL PATH: {file_path}")
        logger.info(f"=======================================================")
        
        # 1. Check for duplicates
        file_hash = self._get_file_hash(file_path)
        if file_hash:
            logger.debug(f"File hash generated: {file_hash}")
            if file_hash in self.processed_file_hashes:
                logger.warning(f"⚠️ DUPLICATE DETECTED! File with this hash has already been processed: {file_path}")
                try:
                    os.remove(file_path)
                    logger.info(f"🗑️ Successfully deleted duplicate file: {file_path}")
                except OSError as e:
                    logger.error(f"❌ ERROR: Failed to delete duplicate file: {file_path}. Error: {e}")
                return # Exit the function immediately
            else:
                self.processed_file_hashes.add(file_hash)
                self._save_processed_hashes()
                logger.info("✅ File is unique. Proceeding with processing.")
        else:
            logger.warning(f"Could not generate hash for {file_path}. Skipping duplicate check.")

        file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        
        # 2. Filename-based Remix Check (Pre-Transcription)
        if config.FILENAME_REMIX_IDENTIFIER in file_name_no_ext.upper():
            logger.info(f"✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains '{config.FILENAME_REMIX_IDENTIFIER}' pattern.")
            target_dir = self.remix_dir
            
            os.makedirs(target_dir, exist_ok=True)
            destination_path = self._get_unique_destination_path(target_dir, os.path.basename(file_path))
            
            try:
                shutil.move(file_path, destination_path)
                logger.info(f"✅ Final move successful (Skipped transcription) → {destination_path}")
                return
            except Exception as e:
                logger.error(f"❌ ERROR: File move failed for {file_path}. Error: {e}")
                return
        
        # 3. Process audio (if not pre-classified as remix)
        file_ext = os.path.splitext(file_path)[1].lower()
        temp_audio_file = None
        audio_to_transcribe = file_path

        try:
            if file_ext in config.VIDEO_EXTENSIONS:
                logger.debug(f"File recognized as VIDEO ({file_ext}). Extracting audio.")
                with tempfile.NamedTemporaryFile(suffix=config.TEMP_AUDIO_EXTENSION, delete=False) as tf:
                    temp_audio_file = tf.name
                audio_to_transcribe = self.extract_audio(file_path, temp_audio_file)
            elif file_ext not in config.AUDIO_EXTENSIONS:
                logger.warning(f"Unsupported file type for transcription: {file_ext}. Attempting to process as audio.")

            # 4. Transcribe audio
            text, lang_weights = self.transcribe_audio(audio_to_transcribe)
            
            langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
            detected_langs = [l for l, _ in langs_sorted]
            
            logger.debug(f"Sorted language weights (Word Count): {langs_sorted}")
            logger.info(f"🌍 Languages detected with weights: {lang_weights}")

            # 5. Cleanup + heuristic for word count
            cleaned_text = text.replace("♪", "").replace("♫", "").strip()
            word_count = len(cleaned_text.split())
            
            logger.debug(f"Total cleaned transcription length: {len(cleaned_text)} characters")
            logger.info(f"🔍 Word count from transcription: {word_count}")

            # 6. Classification Logic (Transcription-based)
            target_dir = None
            
            if config.CLASSIFY_HINDI_URDU_AS_HINDI and "hi" in detected_langs and "ur" in detected_langs and word_count > config.MIN_VOCAL_WORD_COUNT:
                primary_lang_code = "hi"
                language_name = config.LANG_MAP.get(primary_lang_code, primary_lang_code)
                target_dir = os.path.join(self.languages_dir, language_name, "vocals")
                logger.info(f"➡️ CLASSIFICATION: VOCALS ({language_name}). Reason: Hindi/Urdu Mix detected and word count > {config.MIN_VOCAL_WORD_COUNT}.")
            elif word_count <= config.MIN_VOCAL_WORD_COUNT:
                target_dir = self.bgm_dir
                logger.info(f"➡️ CLASSIFICATION: PURE BGM. Reason: Word count ({word_count}) <= {config.MIN_VOCAL_WORD_COUNT}.")
            elif len(detected_langs) >= 2 and config.TRANSCRIPTION_REMIX_ENABLED:
                target_dir = self.remix_dir
                logger.info(f"➡️ CLASSIFICATION: REMIX. Reason: Multiple Languages detected ({', '.join(detected_langs)}).")
            elif detected_langs:
                primary_lang_code = detected_langs[0]
                language_name = config.LANG_MAP.get(primary_lang_code, primary_lang_code)
                target_dir = os.path.join(self.languages_dir, language_name, "vocals")
                logger.info(f"➡️ CLASSIFICATION: VOCALS ({language_name}). Reason: Primary language detected.")
            else:
                target_dir = os.path.join(self.languages_dir, config.LANG_MAP["unknown"], "vocals")
                logger.warning(f"Could not detect clear language. Classifying as UNKNOWN VOCALS.")

            if not target_dir:
                logger.error("❌ ERROR: Classification logic failed to assign a target directory.")
                return

            os.makedirs(target_dir, exist_ok=True)

            # 7. Move file
            destination_path = self._get_unique_destination_path(target_dir, os.path.basename(file_path))
            
            try:
                shutil.move(file_path, destination_path)
                logger.info(f"✅ Final move successful → {destination_path}")
            except Exception as e:
                logger.error(f"❌ ERROR: File move failed for {file_path}. Error: {e}")

        finally:
            # Clean up temporary audio file if it was created
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.remove(temp_audio_file)
                    logger.debug(f"Removed temporary audio file: {temp_audio_file}")
                except OSError as e:
                    logger.warning(f"Could not remove temporary audio file {temp_audio_file}: {e}")

# Example usage (if this were run as a script)
if __name__ == "__main__":
    # This part is just for demonstration if you want to run the code
    # directly for testing. In a real scenario, you'd import and use the class.
    
    # Create some dummy files for testing
    if not os.path.exists("test_input"):
        os.makedirs("test_input")

    with open("test_input/english_song.mp3", "w") as f: f.write("dummy audio content")
    with open("test_input/hindi_X_telugu_remix.mp4", "w") as f: f.write("dummy video content")
    with open("test_input/bgm_track.wav", "w") as f: f.write("dummy audio content")
    with open("test_input/another_english_song.mp3", "w") as f: f.write("dummy audio content") # Duplicate hash for testing
    with open("test_input/another_english_song_copy1.mp3", "w") as f: f.write("dummy audio content") # Another duplicate with same hash
    with open("test_input/french_vocals.flac", "w") as f: f.write("dummy audio content")

    classifier = LanguageClassifier()

    test_files = [
        "test_input/english_song.mp3",
        "test_input/hindi_X_telugu_remix.mp4",
        "test_input/bgm_track.wav",
        "test_input/another_english_song.mp3",
        "test_input/another_english_song_copy1.mp3", # Should be detected as duplicate if hash is same
        "test_input/french_vocals.flac"
    ]

    for file in test_files:
        if os.path.exists(file): # Only process if file still exists (not moved by previous run)
            classifier.process_file(file)
        else:
            logger.info(f"Skipping {file} as it no longer exists (likely processed in a previous run).")

    logger.info("Demonstration finished. Check 'data/output' directory.")