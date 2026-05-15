from pathlib import Path
import wave

import librosa
import soundfile as sf


TARGET_SAMPLE_RATE = 16000


def preprocess_file_with_librosa(input_path, output_path, target_sr=TARGET_SAMPLE_RATE):
    """
    Preprocess one audio file:
    1. Load audio file
    2. Convert to mono
    3. Resample to 16 kHz
    4. Trim silence
    5. Peak-normalize
    6. Save as WAV
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load audio and resample directly to 16 kHz
    y, sr = librosa.load(
        input_path,
        sr=target_sr,
        mono=True
    )

    # Trim silence from beginning and end
    y_trimmed, _ = librosa.effects.trim(
        y,
        top_db=30
    )

    # If trimming removes everything, keep original audio
    if len(y_trimmed) == 0:
        y_trimmed = y

    # Peak normalize
    max_abs = abs(y_trimmed).max()
    if max_abs > 0:
        y_trimmed = 0.99 * y_trimmed / max_abs

    # Save as wav
    sf.write(
        file=str(output_path),
        data=y_trimmed,
        samplerate=target_sr
    )

    return str(output_path)


def check_audio_file(path):
    """
    Check that the saved WAV file exists, is not empty,
    and has 16 kHz sample rate.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_RATE} Hz, got {sample_rate} Hz for {path}"
        )

    if num_frames == 0:
        raise ValueError(f"Empty audio file: {path}")

    return True