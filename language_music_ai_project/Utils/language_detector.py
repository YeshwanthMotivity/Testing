# # import os
# # import shutil
# # import subprocess
# # import whisper
# # from langdetect import detect

# # OUTPUT_DIR = "data/output/Language/"

# # # Load Whisper model (you can use "tiny", "base", "small", "medium", "large")
# # model = whisper.load_model("small")

# # def extract_audio(video_path, output_audio="temp_audio.wav"):
# #     """Extracts audio from video using ffmpeg (16kHz mono)."""
# #     command = [
# #         "ffmpeg", "-i", video_path,
# #         "-ar", "16000", "-ac", "1",
# #         output_audio, "-y"
# #     ]
# #     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# #     return output_audio

# # def transcribe_audio(audio_file):
# #     """Transcribes speech using Whisper."""
# #     result = model.transcribe(audio_file)
# #     text = result["text"]
# #     detected_lang = result.get("language", "unknown")
# #     return text.strip(), detected_lang

# # def process_language(file_path):
# #     """Detects language from speech/text and moves file to respective folder."""
# #     file_ext = os.path.splitext(file_path)[1].lower()

# #     # Step 1: Extract audio if video
# #     if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
# #         audio_path = extract_audio(file_path)
# #         text, whisper_lang = transcribe_audio(audio_path)
# #         os.remove(audio_path)

# #     # Step 2: Handle audio files (added .aac support)
# #     elif file_ext in [".wav", ".mp3", ".flac", ".aac"]:
# #         text, whisper_lang = transcribe_audio(file_path)

# #     # Step 3: Handle text files
# #     elif file_ext == ".txt":
# #         with open(file_path, "r", encoding="utf-8") as f:
# #             text = f.read()
# #         whisper_lang = detect(text)

# #     else:
# #         print(f"❌ Unsupported file format: {file_ext}")
# #         return

# #     if not text.strip():
# #         print("⚠️ No speech detected.")
# #         return

# #     # Step 4: Confirm language
# #     try:
# #         lang_detect = detect(text)
# #     except:
# #         lang_detect = "unknown"

# #     print(f"🗣 Whisper detected: {whisper_lang}")
# #     print(f"🔎 LangDetect detected: {lang_detect}")
# #     print(f"📝 Extracted Text: {text[:100]}...")

# #     final_lang = whisper_lang or lang_detect

# #     # Step 5: Move file into respective folder
# #     lang_folder = os.path.join(OUTPUT_DIR, final_lang)
# #     os.makedirs(lang_folder, exist_ok=True)

# #     shutil.move(file_path, os.path.join(lang_folder, os.path.basename(file_path)))
# #     print(f"✅ Moved {file_path} → {lang_folder}/")


# # import os
# # import shutil
# # import subprocess
# # import whisper
# # from langdetect import detect, DetectorFactory, detect_langs

# # # ---------- Stability ----------
# # DetectorFactory.seed = 0

# # # ---------- Paths ----------
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

# # LANGUAGES_DIR = os.path.join(OUTPUT_DIR, "Languages")
# # BGM_DIR = os.path.join(OUTPUT_DIR, "bgm")
# # REMIX_DIR = os.path.join(OUTPUT_DIR, "remix")
# # TEMP_DIR = os.path.join(BASE_DIR, "temp_demucs")

# # os.makedirs(LANGUAGES_DIR, exist_ok=True)
# # os.makedirs(BGM_DIR, exist_ok=True)
# # os.makedirs(REMIX_DIR, exist_ok=True)
# # os.makedirs(TEMP_DIR, exist_ok=True)

# # # ---------- Lang map ----------
# # LANG_MAP = {
# #     "en": "english",
# #     "hi": "hindi",
# #     "te": "telugu",
# #     "ta": "tamil",
# #     "ml": "malayalam",
# #     "kn": "kannada",
# #     "gu": "gujarati",
# #     "bn": "bengali",
# #     "pa": "punjabi",
# #     "ur": "urdu",
# #     "fr": "french",
# #     "es": "spanish",
# #     "de": "german",
# #     "it": "italian",
# #     "zh": "chinese",
# #     "ja": "japanese",
# #     "ko": "korean",
# #     "mr": "marathi",
# #     "unknown": "unknown",
# # }

# # # ---------- Whisper ----------
# # model = whisper.load_model("small")  # upgrade to medium/large for better detection

# # # ---------- Helper: ffmpeg audio extraction ----------
# # def extract_audio(video_path, output_audio="temp_audio.wav"):
# #     """Extracts mono 16kHz audio using ffmpeg."""
# #     command = [
# #         "ffmpeg", "-i", video_path,
# #         "-ar", "16000", "-ac", "1",
# #         output_audio, "-y"
# #     ]
# #     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# #     return output_audio

# # # ---------- NEW: Demucs separation ----------
# # def run_demucs(input_path):
# #     """Runs Demucs and returns path to vocals stem."""
# #     command = [
# #         "demucs", input_path, "-o", TEMP_DIR
# #     ]
# #     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# #     # Demucs saves inside TEMP_DIR/htdemucs/songname/
# #     song_name = os.path.splitext(os.path.basename(input_path))[0]
# #     stem_dir = os.path.join(TEMP_DIR, "htdemucs", song_name)

# #     if not os.path.exists(stem_dir):
# #         raise RuntimeError("Demucs output not found!")

# #     vocals_path = os.path.join(stem_dir, "vocals.wav")

# #     # Move other stems to bgm/
# #     for stem in ["drums.wav", "bass.wav", "other.wav"]:
# #         stem_path = os.path.join(stem_dir, stem)
# #         if os.path.exists(stem_path):
# #             dest = os.path.join(BGM_DIR, f"{song_name}_{stem}")
# #             shutil.move(stem_path, dest)

# #     return vocals_path  # return vocals for transcription

# # # ---------- Chunked transcription ----------
# # def transcribe_chunked(audio_source, chunk_sec=25, min_chunk_sec=5):
# #     import numpy as np
# #     audio = whisper.load_audio(audio_source)
# #     sr = 16000
# #     chunk_samples = int(chunk_sec * sr)
# #     min_chunk_samples = int(min_chunk_sec * sr)

# #     lang_weights = {}
# #     texts = []

# #     for start in range(0, len(audio), chunk_samples):
# #         seg = audio[start:start + chunk_samples]
# #         if len(seg) < min_chunk_samples:
# #             continue

# #         result = model.transcribe(
# #             seg,
# #             fp16=False,
# #             condition_on_previous_text=False,
# #             temperature=0.0,
# #             beam_size=5,
# #             task="transcribe",
# #         )

# #         seg_text = result.get("text", "").strip()
# #         seg_lang = result.get("language", "unknown")
# #         if not seg_text:
# #             continue

# #         weight = len(seg_text)
# #         code = seg_lang.split("-")[0]
# #         lang_weights[code] = lang_weights.get(code, 0) + weight
# #         texts.append(seg_text)

# #     return " ".join(texts).strip(), lang_weights

# # # ---------- Fallback multi-language detection ----------
# # def detect_multiple_languages_from_text(text):
# #     langs = set()
# #     if not text.strip():
# #         return []
# #     try:
# #         for d in detect_langs(text):
# #             langs.add(d.lang.split("-")[0])
# #     except:
# #         pass
# #     return list(langs)

# # # ---------- Core ----------
# # def process_language(file_path):
# #     file_ext = os.path.splitext(file_path)[1].lower()

# #     # 1) Extract audio if video
# #     if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
# #         audio_path = extract_audio(file_path)
# #     else:
# #         audio_path = file_path

# #     # 2) Run Demucs → get vocals
# #     vocals_path = run_demucs(audio_path)

# #     # 3) Transcribe vocals only
# #     text, lang_weights = transcribe_chunked(vocals_path)

# #     # 4) Heuristic vocals detection
# #     cleaned_text = text.replace("♪", "").replace("♫", "").strip()
# #     word_count = len(cleaned_text.split())
# #     has_vocals = word_count > 10

# #     # 5) Language & remix logic
# #     total_weight = sum(lang_weights.values()) or 1
# #     langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
# #     detected_langs = [l for l, _ in langs_sorted] or ["unknown"]

# #     MIN_SECOND_LANG_SHARE = 0.20
# #     is_remix = False
# #     if len(langs_sorted) >= 2:
# #         second_share = langs_sorted[1][1] / total_weight
# #         is_remix = second_share >= MIN_SECOND_LANG_SHARE

# #     shares_str = ", ".join([f"{LANG_MAP.get(l,l)}={w/total_weight:.0%}" for l, w in langs_sorted]) or "none"
# #     print(f"📝 Text snippet: {text[:100]}...")
# #     print(f"🌍 Chunk-wise language shares: {shares_str}")
# #     print(f"🎙 Vocals detected: {'yes' if has_vocals else 'no'} | Remix: {'yes' if is_remix else 'no'}")

# #     # 6) Routing
# #     if not has_vocals:
# #         base_folder = BGM_DIR
# #     elif is_remix:
# #         pretty = " + ".join(LANG_MAP.get(l, l) for l, _ in langs_sorted[:4])
# #         base_folder = os.path.join(REMIX_DIR, f"Remix ({pretty})")
# #     else:
# #         lang_code = detected_langs[0]
# #         language = LANG_MAP.get(lang_code, lang_code)
# #         base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")

# #     os.makedirs(base_folder, exist_ok=True)

# #     # 7) Move original file
# #     destination = os.path.join(base_folder, os.path.basename(file_path))
# #     if os.path.exists(destination):
# #         base, ext = os.path.splitext(destination)
# #         i = 1
# #         while os.path.exists(f"{base}_copy{i}{ext}"):
# #             i += 1
# #         destination = f"{base}_copy{i}{ext}"

# #     shutil.move(file_path, destination)
# #     print(f"✅ Moved {file_path} → {destination}")


# # # language_detector.py

# # import os
# # import shutil
# # import subprocess
# # # REMOVE: import whisper
# # # ADD: 
# # from faster_whisper import WhisperModel
# # from langdetect import DetectorFactory, detect_langs

# # # ---------- Stability ----------
# # DetectorFactory.seed = 0

# # # ---------- Paths ----------
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

# # LANGUAGES_DIR = os.path.join(OUTPUT_DIR, "Languages")
# # BGM_DIR = os.path.join(OUTPUT_DIR, "bgm")
# # REMIX_DIR = os.path.join(OUTPUT_DIR, "remix")

# # os.makedirs(LANGUAGES_DIR, exist_ok=True)
# # os.makedirs(BGM_DIR, exist_ok=True)
# # os.makedirs(REMIX_DIR, exist_ok=True)

# # # ---------- Lang map ----------
# # LANG_MAP = {
# #     "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
# #     "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
# #     "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
# #     "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
# # }

# # # ---------- Whisper (Updated for faster-whisper) ----------
# # print("📥 Loading Whisper model...")
# # # Load the 'small' model, specify 'cpu', and use 'int8' for fastest CPU speed
# # model = WhisperModel(
# #     "small",             # Keeps your desired 'small' model size
# #     device="cpu",        # Explicitly targets the CPU
# #     compute_type="int8"  # Enables INT8 quantization for max speed
# # )
# # print("✅ Whisper model loaded")

# # # ---------- Audio extractor ----------
# # def extract_audio(video_path, output_audio="temp_audio.wav"):
# #     """Extract mono 16kHz audio using ffmpeg."""
# #     print(f"🎬 Extracting audio from video: {video_path}")
# #     command = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", output_audio, "-y"]
# #     # We still use subprocess for ffmpeg
# #     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# #     print(f"✅ Extracted audio → {output_audio}")
# #     return output_audio

# # # ---------- Transcribe (New Optimized Function) ----------
# # # Replaces the old transcribe_chunked function
# # def transcribe_audio(audio_path):
# #     """Transcribe entire audio using faster-whisper's optimized engine."""
    
# #     # model.transcribe handles audio loading and chunking internally
# #     segments, info = model.transcribe(audio_path, beam_size=5)

# #     langs_detected = {}
# #     texts = []
    
# #     # Iterate through segments returned by the model
# #     for segment in segments:
# #         seg_text = segment.text.strip()
        
# #         # faster-whisper provides overall language in 'info'
# #         seg_lang = info.language
        
# #         if seg_text:
# #             texts.append(seg_text)
# #             langs_detected[seg_lang] = langs_detected.get(seg_lang, 0) + len(seg_text.split())
# #             print(f"   🎧 Segment: Detected {seg_lang}")

# #     return " ".join(texts), langs_detected

# # # ---------- Core ----------
# # def process_file(file_path):
# #     print(f"\n🚀 Processing file: {file_path}")
# #     file_ext = os.path.splitext(file_path)[1].lower()

# #     # Extract audio if it's a video
# #     if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
# #         audio_path = extract_audio(file_path)
# #     else:
# #         audio_path = file_path

# #     # Transcribe audio (UPDATED CALL)
# #     text, lang_weights = transcribe_audio(audio_path) # Calls the new optimized function
    
# #     langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
# #     detected_langs = [l for l, _ in langs_sorted]

# #     print(f"🌍 Languages detected with weights: {lang_weights}")

# #     # Cleanup + heuristic
# #     cleaned_text = text.replace("♪", "").replace("♫", "").strip()
# #     word_count = len(cleaned_text.split())
# #     print(f"🔍 Word count: {word_count}")

# #     # --- New Hindi/Urdu Logic ---
# #     if "hi" in detected_langs and "ur" in detected_langs:
# #         primary_lang = "hi"
# #         base_folder = os.path.join(LANGUAGES_DIR, LANG_MAP.get(primary_lang, primary_lang), "vocals")
# #         print(f"➡️ Classified as: VOCALS ({LANG_MAP.get(primary_lang, primary_lang)}) due to Hindi/Urdu mix")
# #     # --- Original Routing Logic ---
# #     elif word_count <= 10:
# #         base_folder = BGM_DIR
# #         print("➡️ Classified as: PURE BGM")
# #     elif len(detected_langs) >= 2:
# #         base_folder = REMIX_DIR
# #         print(f"➡️ Classified as: REMIX ({detected_langs})")
# #     else:
# #         primary_lang = detected_langs[0] if detected_langs else "unknown"
# #         language = LANG_MAP.get(primary_lang, primary_lang)
# #         base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
# #         print(f"➡️ Classified as: VOCALS ({language})")

# #     os.makedirs(base_folder, exist_ok=True)

# #     # Move file
# #     destination = os.path.join(base_folder, os.path.basename(file_path))
# #     if os.path.exists(destination):
# #         base, ext = os.path.splitext(destination)
# #         i = 1
# #         while os.path.exists(f"{base}_copy{i}{ext}"):
# #             i += 1
# #         destination = f"{base}_copy{i}{ext}"

# #     shutil.move(file_path, destination)
# #     print(f"✅ Final move → {destination}")


# # language_detector.py

# import os
# import shutil
# import subprocess
# from faster_whisper import WhisperModel
# from langdetect import DetectorFactory, detect_langs

# # ---------- Stability ----------
# DetectorFactory.seed = 0
# print("--- DEBUG: Stability seed set to 0 ---")

# # ---------- Paths ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

# LANGUAGES_DIR = os.path.join(OUTPUT_DIR, "Languages")
# BGM_DIR = os.path.join(OUTPUT_DIR, "bgm")
# REMIX_DIR = os.path.join(OUTPUT_DIR, "remix")

# print(f"--- DEBUG: BASE_DIR: {BASE_DIR}")
# print(f"--- DEBUG: Output directories setup: Languages={LANGUAGES_DIR}, BGM={BGM_DIR}, REMIX={REMIX_DIR}")

# os.makedirs(LANGUAGES_DIR, exist_ok=True)
# os.makedirs(BGM_DIR, exist_ok=True)
# os.makedirs(REMIX_DIR, exist_ok=True)

# # ---------- Lang map (remains the same) ----------
# LANG_MAP = {
#     "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
#     "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
#     "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
#     "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
# }

# # ---------- Whisper (Updated for faster-whisper) ----------
# print("📥 Loading Whisper model...")
# model = WhisperModel(
#     "small",             # Keeps your desired 'small' model size
#     device="cpu",        # Explicitly targets the CPU
#     compute_type="int8"  # Enables INT8 quantization for max speed
# )
# print("✅ Whisper model loaded (Model: small, Device: CPU, Compute: INT8)")

# # ---------- Audio extractor ----------
# def extract_audio(video_path, output_audio="temp_audio.wav"):
#     """Extract mono 16kHz audio using ffmpeg."""
#     print(f"\n🎬 Extracting audio from video: {video_path}")
#     command = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", output_audio, "-y"]
#     print(f"--- DEBUG: FFmpeg command: {' '.join(command)}")
    
#     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
#     if os.path.exists(output_audio):
#         print(f"✅ Extracted audio → {output_audio}")
#     else:
#         print(f"❌ ERROR: Audio extraction failed for {video_path}.")
#     return output_audio

# # ---------- Transcribe (New Optimized Function) ----------
# def transcribe_audio(audio_path):
#     """Transcribe entire audio using faster-whisper's optimized engine."""
#     print(f"--- DEBUG: Starting transcription for: {audio_path}")
    
#     # model.transcribe handles audio loading and chunking internally
#     segments, info = model.transcribe(audio_path, beam_size=5)

#     print(f"--- DEBUG: Model detected language for entire file: {info.language}")
#     print(f"--- DEBUG: Model transcription speed RTF (Lower is better): {info.language_probability}")

#     langs_detected = {}
#     texts = []
#     segment_count = 0
    
#     # Iterate through segments returned by the model
#     for segment in segments:
#         segment_count += 1
#         seg_text = segment.text.strip()
#         seg_lang = info.language # Use the language detected for the whole file
        
#         if seg_text:
#             word_count = len(seg_text.split())
#             texts.append(seg_text)
#             langs_detected[seg_lang] = langs_detected.get(seg_lang, 0) + word_count
#             print(f"   🎧 Segment {segment_count}: Detected {seg_lang}, Words: {word_count}. Text: '{seg_text[:50]}...'")
        
#         else:
#              print(f"   🎧 Segment {segment_count}: Skipped (No text found).")


#     print(f"--- DEBUG: Total segments processed: {segment_count}")
#     return " ".join(texts), langs_detected

# # ---------- Core ----------
# def process_file(file_path):
#     print(f"\n=======================================================")
#     print(f"🚀 Processing file: {os.path.basename(file_path)}")
#     print(f"   FULL PATH: {file_path}")
#     print(f"=======================================================")
    
#     file_ext = os.path.splitext(file_path)[1].lower()
    
#     # Extract audio if it's a video
#     if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
#         print(f"--- DEBUG: File recognized as VIDEO ({file_ext}). Extracting audio.")
#         audio_path = extract_audio(file_path)
#     else:
#         print(f"--- DEBUG: File recognized as AUDIO ({file_ext} or unknown). Using file directly.")
#         audio_path = file_path

#     # Transcribe audio 
#     text, lang_weights = transcribe_audio(audio_path)
    
#     langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
#     detected_langs = [l for l, _ in langs_sorted]
    
#     print(f"--- DEBUG: Sorted language weights (Word Count): {langs_sorted}")
#     print(f"🌍 Languages detected with weights: {lang_weights}")

#     # Cleanup + heuristic
#     cleaned_text = text.replace("♪", "").replace("♫", "").strip()
#     word_count = len(cleaned_text.split())
    
#     print(f"--- DEBUG: Total cleaned transcription length: {len(cleaned_text)} characters")
#     print(f"🔍 Word count: {word_count}")

#     # --- Classification Logic ---
#     base_folder = None
    
#     if "hi" in detected_langs and "ur" in detected_langs and word_count > 10:
#         primary_lang = "hi"
#         language = LANG_MAP.get(primary_lang, primary_lang)
#         base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
#         print(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Hindi/Urdu Mix and word count > 10.")
#     elif word_count <= 10:
#         base_folder = BGM_DIR
#         print("➡️ CLASSIFICATION: PURE BGM. Reason: Word count <= 10.")
#     elif len(detected_langs) >= 2:
#         base_folder = REMIX_DIR
#         print(f"➡️ CLASSIFICATION: REMIX. Reason: Multiple Languages detected ({detected_langs}).")
#     else:
#         primary_lang = detected_langs[0] if detected_langs else "unknown"
#         language = LANG_MAP.get(primary_lang, primary_lang)
#         base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
#         print(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Primary language detected.")

#     if not base_folder:
#         print("❌ ERROR: Classification logic failed to assign a base folder.")
#         return

#     final_destination_dir = base_folder
#     print(f"--- DEBUG: Target move directory: {final_destination_dir}")
#     os.makedirs(final_destination_dir, exist_ok=True)

#     # Move file
#     destination = os.path.join(final_destination_dir, os.path.basename(file_path))
    
#     print(f"--- DEBUG: Initial destination path: {destination}")

#     if os.path.exists(destination):
#         base, ext = os.path.splitext(destination)
#         i = 1
#         while os.path.exists(f"{base}_copy{i}{ext}"):
#             i += 1
#         destination = f"{base}_copy{i}{ext}"
#         print(f"--- DEBUG: Renaming required. New destination path: {destination}")

#     try:
#         shutil.move(file_path, destination)
#         print(f"✅ Final move successful → {destination}")
#     except Exception as e:
#         print(f"❌ ERROR: File move failed for {file_path}. Error: {e}")


# # language_detector.py

# import os
# import shutil
# import subprocess
# from faster_whisper import WhisperModel
# from langdetect import DetectorFactory, detect_langs

# # ---------- Stability ----------
# DetectorFactory.seed = 0
# print("--- DEBUG: Stability seed set to 0 ---")

# # ---------- Paths ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

# LANGUAGES_DIR = os.path.join(OUTPUT_DIR, "Languages")
# BGM_DIR = os.path.join(OUTPUT_DIR, "bgm")
# REMIX_DIR = os.path.join(OUTPUT_DIR, "remix")

# print(f"--- DEBUG: BASE_DIR: {BASE_DIR}")
# print(f"--- DEBUG: Output directories setup: Languages={LANGUAGES_DIR}, BGM={BGM_DIR}, REMIX={REMIX_DIR}")

# os.makedirs(LANGUAGES_DIR, exist_ok=True)
# os.makedirs(BGM_DIR, exist_ok=True)
# os.makedirs(REMIX_DIR, exist_ok=True)

# # ---------- Lang map (remains the same) ----------
# LANG_MAP = {
#     "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
#     "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
#     "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
#     "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
# }

# # ---------- Whisper (Updated for faster-whisper) ----------
# print("📥 Loading Whisper model...")
# model = WhisperModel(
#     "small",             # Keeps your desired 'small' model size
#     device="cpu",        # Explicitly targets the CPU
#     compute_type="int8"  # Enables INT8 quantization for max speed
# )
# print("✅ Whisper model loaded (Model: small, Device: CPU, Compute: INT8)")

# # ---------- Audio extractor ----------
# def extract_audio(video_path, output_audio="temp_audio.wav"):
#     """Extract mono 16kHz audio using ffmpeg."""
#     print(f"\n🎬 Extracting audio from video: {video_path}")
#     command = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", output_audio, "-y"]
#     print(f"--- DEBUG: FFmpeg command: {' '.join(command)}")
    
#     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
#     if os.path.exists(output_audio):
#         print(f"✅ Extracted audio → {output_audio}")
#     else:
#         print(f"❌ ERROR: Audio extraction failed for {video_path}.")
#     return output_audio

# # ---------- Transcribe (New Optimized Function) ----------
# def transcribe_audio(audio_path):
#     """Transcribe entire audio using faster-whisper's optimized engine."""
#     print(f"--- DEBUG: Starting transcription for: {audio_path}")
    
#     # model.transcribe handles audio loading and chunking internally
#     segments, info = model.transcribe(audio_path, beam_size=5)

#     print(f"--- DEBUG: Model detected language for entire file: {info.language}")
#     print(f"--- DEBUG: Model transcription speed RTF (Lower is better): {info.language_probability}")

#     langs_detected = {}
#     texts = []
#     segment_count = 0
    
#     # Iterate through segments returned by the model
#     for segment in segments:
#         segment_count += 1
#         seg_text = segment.text.strip()
#         seg_lang = info.language # Use the language detected for the whole file
        
#         if seg_text:
#             word_count = len(seg_text.split())
#             texts.append(seg_text)
#             langs_detected[seg_lang] = langs_detected.get(seg_lang, 0) + word_count
#             print(f"   🎧 Segment {segment_count}: Detected {seg_lang}, Words: {word_count}. Text: '{seg_text[:50]}...'")
        
#         else:
#              print(f"   🎧 Segment {segment_count}: Skipped (No text found).")


#     print(f"--- DEBUG: Total segments processed: {segment_count}")
#     return " ".join(texts), langs_detected

# # ---------- Core ----------
# def process_file(file_path):
#     print(f"\n=======================================================")
#     print(f"🚀 Processing file: {os.path.basename(file_path)}")
#     print(f"   FULL PATH: {file_path}")
#     print(f"=======================================================")
    
#     file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]

#     # --- NEW FEATURE: Filename-based Remix Check (Pre-Transcription) ---
#     # Check for the pattern " X " (case-insensitive) in the filename
#     if " X " in file_name_no_ext.upper():
#         print("✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains ' X ' pattern.")
#         base_folder = REMIX_DIR
        
#         # Skip transcription and immediately move the file
#         final_destination_dir = base_folder
#         print(f"--- DEBUG: Target move directory (Pre-classified): {final_destination_dir}")
#         os.makedirs(final_destination_dir, exist_ok=True)
        
#         destination = os.path.join(final_destination_dir, os.path.basename(file_path))
#         print(f"--- DEBUG: Initial destination path: {destination}")

#         if os.path.exists(destination):
#             base, ext = os.path.splitext(destination)
#             i = 1
#             while os.path.exists(f"{base}_copy{i}{ext}"):
#                 i += 1
#             destination = f"{base}_copy{i}{ext}"
#             print(f"--- DEBUG: Renaming required. New destination path: {destination}")
            
#         try:
#             shutil.move(file_path, destination)
#             print(f"✅ Final move successful (Skipped transcription) → {destination}")
#             return # IMPORTANT: Exit the function after moving

#         except Exception as e:
#             print(f"❌ ERROR: File move failed for {file_path}. Error: {e}")
#             return
    
#     # --- CONTINUE WITH REGULAR PROCESSING (If not a filename-based remix) ---
    
#     file_ext = os.path.splitext(file_path)[1].lower()

#     # Extract audio if it's a video
#     if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
#         print(f"--- DEBUG: File recognized as VIDEO ({file_ext}). Extracting audio.")
#         audio_path = extract_audio(file_path)
#     else:
#         print(f"--- DEBUG: File recognized as AUDIO ({file_ext} or unknown). Using file directly.")
#         audio_path = file_path

#     # Transcribe audio 
#     text, lang_weights = transcribe_audio(audio_path)
    
#     langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
#     detected_langs = [l for l, _ in langs_sorted]
    
#     print(f"--- DEBUG: Sorted language weights (Word Count): {langs_sorted}")
#     print(f"🌍 Languages detected with weights: {lang_weights}")

#     # Cleanup + heuristic
#     cleaned_text = text.replace("♪", "").replace("♫", "").strip()
#     word_count = len(cleaned_text.split())
    
#     print(f"--- DEBUG: Total cleaned transcription length: {len(cleaned_text)} characters")
#     print(f"🔍 Word count: {word_count}")

#     # --- Classification Logic (Transcription-based) ---
#     base_folder = None
    
#     if "hi" in detected_langs and "ur" in detected_langs and word_count > 10:
#         primary_lang = "hi"
#         language = LANG_MAP.get(primary_lang, primary_lang)
#         base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
#         print(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Hindi/Urdu Mix and word count > 10.")
#     elif word_count <= 10:
#         base_folder = BGM_DIR
#         print("➡️ CLASSIFICATION: PURE BGM. Reason: Word count <= 10.")
#     elif len(detected_langs) >= 2:
#         base_folder = REMIX_DIR
#         print(f"➡️ CLASSIFICATION: REMIX. Reason: Multiple Languages detected ({detected_langs}).")
#     else:
#         primary_lang = detected_langs[0] if detected_langs else "unknown"
#         language = LANG_MAP.get(primary_lang, primary_lang)
#         base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
#         print(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Primary language detected.")

#     if not base_folder:
#         print("❌ ERROR: Classification logic failed to assign a base folder.")
#         return

#     final_destination_dir = base_folder
#     print(f"--- DEBUG: Target move directory: {final_destination_dir}")
#     os.makedirs(final_destination_dir, exist_ok=True)

#     # Move file (Final move for non-preclassified files)
#     destination = os.path.join(final_destination_dir, os.path.basename(file_path))
    
#     print(f"--- DEBUG: Initial destination path: {destination}")

#     if os.path.exists(destination):
#         base, ext = os.path.splitext(destination)
#         i = 1
#         while os.path.exists(f"{base}_copy{i}{ext}"):
#             i += 1
#         destination = f"{base}_copy{i}{ext}"
#         print(f"--- DEBUG: Renaming required. New destination path: {destination}")

#     try:
#         shutil.move(file_path, destination)
#         print(f"✅ Final move successful → {destination}")
#     except Exception as e:
#         print(f"❌ ERROR: File move failed for {file_path}. Error: {e}")


# language_detector.py

import os
import shutil
import subprocess
from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect_langs
import hashlib

# ---------- Stability ----------
DetectorFactory.seed = 0
print("--- DEBUG: Stability seed set to 0 ---")

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

LANGUAGES_DIR = os.path.join(OUTPUT_DIR, "Languages")
BGM_DIR = os.path.join(OUTPUT_DIR, "bgm")
REMIX_DIR = os.path.join(OUTPUT_DIR, "remix")

print(f"--- DEBUG: BASE_DIR: {BASE_DIR}")
print(f"--- DEBUG: Output directories setup: Languages={LANGUAGES_DIR}, BGM={BGM_DIR}, REMIX={REMIX_DIR}")

os.makedirs(LANGUAGES_DIR, exist_ok=True)
os.makedirs(BGM_DIR, exist_ok=True)
os.makedirs(REMIX_DIR, exist_ok=True)

# 🆕 Global variable to store hashes of processed files
PROCESSED_FILE_HASHES = set()

# ---------- Lang map (remains the same) ----------
LANG_MAP = {
    "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "ml": "malayalam",
    "kn": "kannada", "gu": "gujarati", "bn": "bengali", "pa": "punjabi", "ur": "urdu",
    "fr": "french", "es": "spanish", "de": "german", "it": "italian", "zh": "chinese",
    "ja": "japanese", "ko": "korean", "mr": "marathi", "unknown": "unknown",
}

# ---------- Whisper (Updated for faster-whisper) ----------
print("📥 Loading Whisper model...")
model = WhisperModel(
    "small",             # Keeps your desired 'small' model size
    device="cpu",        # Explicitly targets the CPU
    compute_type="int8"  # Enables INT8 quantization for max speed
)
print("✅ Whisper model loaded (Model: small, Device: CPU, Compute: INT8)")

# ---------- Audio extractor ----------
def extract_audio(video_path, output_audio="temp_audio.wav"):
    """Extract mono 16kHz audio using ffmpeg."""
    print(f"\n🎬 Extracting audio from video: {video_path}")
    command = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", output_audio, "-y"]
    print(f"--- DEBUG: FFmpeg command: {' '.join(command)}")
    
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(output_audio):
        print(f"✅ Extracted audio → {output_audio}")
    else:
        print(f"❌ ERROR: Audio extraction failed for {video_path}.")
    return output_audio

# ---------- Transcribe (New Optimized Function) ----------
def transcribe_audio(audio_path):
    """Transcribe entire audio using faster-whisper's optimized engine."""
    print(f"--- DEBUG: Starting transcription for: {audio_path}")
    
    # model.transcribe handles audio loading and chunking internally
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"--- DEBUG: Model detected language for entire file: {info.language}")
    print(f"--- DEBUG: Model transcription speed RTF (Lower is better): {info.language_probability}")

    langs_detected = {}
    texts = []
    segment_count = 0
    
    # Iterate through segments returned by the model
    for segment in segments:
        segment_count += 1
        seg_text = segment.text.strip()
        seg_lang = info.language # Use the language detected for the whole file
        
        if seg_text:
            word_count = len(seg_text.split())
            texts.append(seg_text)
            langs_detected[seg_lang] = langs_detected.get(seg_lang, 0) + word_count
            print(f"   🎧 Segment {segment_count}: Detected {seg_lang}, Words: {word_count}. Text: '{seg_text[:50]}...'")
        
        else:
            print(f"   🎧 Segment {segment_count}: Skipped (No text found).")

    print(f"--- DEBUG: Total segments processed: {segment_count}")
    return " ".join(texts), langs_detected

# 🆕 New function to get file hash
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
        print(f"❌ ERROR: Could not get hash for {file_path}. Error: {e}")
        return None

# ---------- Core (Updated to include duplicate check) ----------
def process_file(file_path):
    print(f"\n=======================================================")
    print(f"🚀 Processing file: {os.path.basename(file_path)}")
    print(f"   FULL PATH: {file_path}")
    print(f"=======================================================")
    
    # 🆕 CHECK FOR DUPLICATES BEFORE ANYTHING ELSE
    file_hash = get_file_hash(file_path)
    if file_hash:
        print(f"--- DEBUG: File hash generated: {file_hash}")
        if file_hash in PROCESSED_FILE_HASHES:
            print("⚠️ DUPLICATE DETECTED! File with this hash has already been processed.")
            try:
                os.remove(file_path)
                print(f"🗑️ Successfully deleted duplicate file: {file_path}")
            except Exception as e:
                print(f"❌ ERROR: Failed to delete duplicate file: {file_path}. Error: {e}")
            return # Exit the function immediately
        else:
            # 🆕 Add the new file's hash to our set
            PROCESSED_FILE_HASHES.add(file_hash)
            print("✅ File is unique. Proceeding with processing.")

    # --- CONTINUE WITH REGULAR PROCESSING (If not a duplicate) ---
    
    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    
    # --- NEW FEATURE: Filename-based Remix Check (Pre-Transcription) ---
    if " X " in file_name_no_ext.upper():
        print("✅ FILENAME CLASSIFICATION: REMIX. Reason: Filename contains ' X ' pattern.")
        base_folder = REMIX_DIR
        
        final_destination_dir = base_folder
        print(f"--- DEBUG: Target move directory (Pre-classified): {final_destination_dir}")
        os.makedirs(final_destination_dir, exist_ok=True)
        
        destination = os.path.join(final_destination_dir, os.path.basename(file_path))
        print(f"--- DEBUG: Initial destination path: {destination}")

        if os.path.exists(destination):
            base, ext = os.path.splitext(destination)
            i = 1
            while os.path.exists(f"{base}_copy{i}{ext}"):
                i += 1
            destination = f"{base}_copy{i}{ext}"
            print(f"--- DEBUG: Renaming required. New destination path: {destination}")
            
        try:
            shutil.move(file_path, destination)
            print(f"✅ Final move successful (Skipped transcription) → {destination}")
            return # IMPORTANT: Exit the function after moving

        except Exception as e:
            print(f"❌ ERROR: File move failed for {file_path}. Error: {e}")
            return
    
    # --- CONTINUE WITH REGULAR PROCESSING (If not a filename-based remix) ---
    
    file_ext = os.path.splitext(file_path)[1].lower()

    # Extract audio if it's a video
    if file_ext in [".mp4", ".mkv", ".avi", ".mov"]:
        print(f"--- DEBUG: File recognized as VIDEO ({file_ext}). Extracting audio.")
        audio_path = extract_audio(file_path)
    else:
        print(f"--- DEBUG: File recognized as AUDIO ({file_ext} or unknown). Using file directly.")
        audio_path = file_path

    # Transcribe audio 
    text, lang_weights = transcribe_audio(audio_path)
    
    langs_sorted = sorted(lang_weights.items(), key=lambda x: x[1], reverse=True)
    detected_langs = [l for l, _ in langs_sorted]
    
    print(f"--- DEBUG: Sorted language weights (Word Count): {langs_sorted}")
    print(f"🌍 Languages detected with weights: {lang_weights}")

    # Cleanup + heuristic
    cleaned_text = text.replace("♪", "").replace("♫", "").strip()
    word_count = len(cleaned_text.split())
    
    print(f"--- DEBUG: Total cleaned transcription length: {len(cleaned_text)} characters")
    print(f"🔍 Word count: {word_count}")

    # --- Classification Logic (Transcription-based) ---
    base_folder = None
    
    if "hi" in detected_langs and "ur" in detected_langs and word_count > 10:
        primary_lang = "hi"
        language = LANG_MAP.get(primary_lang, primary_lang)
        base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
        print(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Hindi/Urdu Mix and word count > 10.")
    elif word_count <= 10:
        base_folder = BGM_DIR
        print("➡️ CLASSIFICATION: PURE BGM. Reason: Word count <= 10.")
    elif len(detected_langs) >= 2:
        base_folder = REMIX_DIR
        print(f"➡️ CLASSIFICATION: REMIX. Reason: Multiple Languages detected ({detected_langs}).")
    else:
        primary_lang = detected_langs[0] if detected_langs else "unknown"
        language = LANG_MAP.get(primary_lang, primary_lang)
        base_folder = os.path.join(LANGUAGES_DIR, language, "vocals")
        print(f"➡️ CLASSIFICATION: VOCALS ({language}). Reason: Primary language detected.")

    if not base_folder:
        print("❌ ERROR: Classification logic failed to assign a base folder.")
        return

    final_destination_dir = base_folder
    print(f"--- DEBUG: Target move directory: {final_destination_dir}")
    os.makedirs(final_destination_dir, exist_ok=True)

    # Move file (Final move for non-preclassified files)
    destination = os.path.join(final_destination_dir, os.path.basename(file_path))
    
    print(f"--- DEBUG: Initial destination path: {destination}")

    # The original check for duplicate filenames is still useful for unique naming
    if os.path.exists(destination):
        base, ext = os.path.splitext(destination)
        i = 1
        while os.path.exists(f"{base}_copy{i}{ext}"):
            i += 1
        destination = f"{base}_copy{i}{ext}"
        print(f"--- DEBUG: Renaming required. New destination path: {destination}")

    try:
        shutil.move(file_path, destination)
        print(f"✅ Final move successful → {destination}")
    except Exception as e:
        print(f"❌ ERROR: File move failed for {file_path}. Error: {e}")