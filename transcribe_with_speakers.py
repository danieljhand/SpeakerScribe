"""
Speaker-Aware Transcription Script

This script processes an audio file to:
1. Transcribe speech using Whisper
2. Perform speaker diarization using pyannote
3. Combine the results to create a transcript with speaker labels

Requirements:
- whisper
- pyannote.audio
- torch
- A valid Hugging Face token for pyannote (set in config.py or as environment variable)
"""

import os
import sys
import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from config import HUGGINGFACE_TOKEN, DEFAULT_WHISPER_MODEL, DEFAULT_NUM_SPEAKERS

# Get Hugging Face token from config
huggingface_token = HUGGINGFACE_TOKEN

# Get user input for audio file
input_audio_file = input("Enter the path to the audio file: ").strip()
if not input_audio_file:
    print("Error: No input file provided.")
    sys.exit(1)

# Set output files based on input filename
output_rttm_file = os.path.splitext(input_audio_file)[0] + ".rttm"
output_transcript_file = input("Enter the path for the transcript file [default: transcript.txt]: ").strip()
if not output_transcript_file:
    output_transcript_file = "transcript.txt"

# Get user input for model type
valid_models = ['tiny', 'base', 'small', 'medium', 'large']
while True:
    whisper_model = input(f"Enter Whisper model type (tiny, base, small, medium, large) [default: {DEFAULT_WHISPER_MODEL}]: ").strip().lower()
    if whisper_model == "":
        whisper_model = DEFAULT_WHISPER_MODEL
        break
    if whisper_model in valid_models:
        break
    print(f"Invalid model type. Please choose from: {', '.join(valid_models)}")

# Get user input for number of speakers
while True:
    try:
        num_speakers_input = input(f"Enter the number of speakers [default: {DEFAULT_NUM_SPEAKERS}]: ").strip()
        if num_speakers_input == "":
            num_speakers = DEFAULT_NUM_SPEAKERS
            break
        num_speakers = int(num_speakers_input)
        if num_speakers > 0:
            break
        print("Number of speakers must be greater than 0")
    except ValueError:
        print("Please enter a valid number")

# Check if input file exists
if not os.path.exists(input_audio_file):
    print(f"Error: Input file '{input_audio_file}' not found.")
    sys.exit(1)

print(f"Step 1: Transcribing audio using Whisper ({whisper_model} model)...")
try:
    model = whisper.load_model(whisper_model)
    result = model.transcribe(input_audio_file)
    print("Transcription completed successfully.")
    print("Transcription segments:")
    #pprint.pp(result["segments"])
except Exception as e:
    print(f"Error during transcription: {e}")
    sys.exit(1)


print("\nStep 2: Performing speaker diarization...")
try:
    # Initialize the speaker diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=huggingface_token)
    
    # Use appropriate device based on platform
    import platform
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # Use MPS (Metal Performance Shaders) if available
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        # Use CUDA if available on other platforms
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    pipeline.to(device)
    
    # Run the diarization pipeline with progress indicator
    with ProgressHook() as hook:
        diarization = pipeline(input_audio_file, num_speakers=num_speakers, hook=hook)
    
    diarization_segments = []
    
    # Extract and print speaker timestamps
    print("\nSpeaker segments:")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        #print(f"Speaker {speaker}: {turn.start:.2f} - {turn.end:.2f}")
        start_time = round(turn.start, 1)
        end_time = round(turn.end, 1)
        diarization_segments.append({'start': start_time, 'end': end_time, 'speaker': speaker})
    
    # Save diarization output to disk using RTTM format
    print(f"\nSaving diarization results to {output_rttm_file}")
    with open(output_rttm_file, "w") as rttm:
        diarization.write_rttm(rttm)
    
except Exception as e:
    print(f"Error during diarization: {e}")
    sys.exit(1)

def assign_speakers(whisper_segments, diarization_segments):
    """
    Align transcription segments with speaker segments to create a speaker-labeled transcript.
    
    Args:
        whisper_segments (list): List of transcription segments from Whisper
        diarization_segments (list): List of speaker segments from diarization
        
    Returns:
        list: List of strings with format "speaker: text"
    """
    speaker_transcript = []
    unassigned_segments = 0

    for whisper_segment in whisper_segments:
        whisper_start = whisper_segment["start"]
        whisper_end = whisper_segment["end"]
        assigned = False

        # Find matching speaker segment
        for diarization_segment in diarization_segments:
            diarization_start = diarization_segment["start"]
            diarization_end = diarization_segment["end"]
            speaker = diarization_segment["speaker"]

            # If Whisper segment overlaps with diarization segment
            if whisper_start < diarization_end and whisper_end > diarization_start:
                speaker_transcript.append(f"{speaker}: {whisper_segment['text']}")
                assigned = True
                break  # Assign to the first matching speaker
        
        if not assigned:
            speaker_transcript.append(f"UNKNOWN: {whisper_segment['text']}")
            unassigned_segments += 1
    
    if unassigned_segments > 0:
        print(f"Warning: {unassigned_segments} segments couldn't be assigned to a speaker")
        
    return speaker_transcript

def identify_speakers(formatted_output):
    """
    Ask the user to identify speakers based on sample text and return a mapping
    of speaker IDs to real names.
    
    Args:
        formatted_output (list): List of strings with format "speaker: text"
        
    Returns:
        dict: Mapping of speaker IDs to real names
    """
    # Extract unique speakers
    speakers = set()
    for line in formatted_output:
        speaker = line.split(':', 1)[0].strip()
        if speaker != "UNKNOWN":
            speakers.add(speaker)
    
    speaker_names = {}
    
    print("\nLet's identify the speakers in the transcript.")
    print("For each speaker, I'll show you some sample text they spoke.")
    
    # For each speaker, show 5 random examples of their speech
    import random
    for speaker in speakers:
        speaker_lines = [line for line in formatted_output if line.startswith(f"{speaker}:")]
        
        # Get up to 5 random samples
        samples = random.sample(speaker_lines, min(5, len(speaker_lines)))
        
        print(f"\nSpeaker: {speaker}")
        print("Sample text:")
        for i, sample in enumerate(samples, 1):
            _, text = sample.split(':', 1)
            print(f"  {i}. {text.strip()}")
        
        # Ask for the real name
        name = input(f"Who is {speaker}? Enter their real name: ").strip()
        if name:
            speaker_names[speaker] = name
        else:
            speaker_names[speaker] = speaker  # Keep original if no name provided
    
    return speaker_names

def update_transcript_with_names(formatted_output, speaker_names, output_file):
    """
    Update the transcript with real speaker names and save to file.
    
    Args:
        formatted_output (list): List of strings with format "speaker: text"
        speaker_names (dict): Mapping of speaker IDs to real names
        output_file (str): Path to save the updated transcript
    """
    updated_output = []
    
    for line in formatted_output:
        parts = line.split(':', 1)
        if len(parts) == 2:
            speaker, text = parts
            speaker = speaker.strip()
            if speaker in speaker_names:
                updated_output.append(f"{speaker_names[speaker]}: {text.strip()}")
            else:
                updated_output.append(line)
        else:
            updated_output.append(line)
    
    with open(output_file, "w") as transcript_file:
        for line in updated_output:
            transcript_file.write(f"{line}\n")

print("\nStep 3: Aligning transcription with speaker information...")
try:
    # Run alignment
    formatted_output = assign_speakers(result["segments"], diarization_segments)
    
    # Save initial transcript with speaker labels to file
    print(f"\nSaving initial transcript to {output_transcript_file}")
    with open(output_transcript_file, "w") as transcript_file:
        for line in formatted_output:
            transcript_file.write(f"{line}\n")
    
    # Ask user if they want to identify speakers
    identify = input("\nWould you like to identify the speakers? (y/n): ").strip().lower()
    
    if identify == 'y':
        # Get speaker names
        speaker_names = identify_speakers(formatted_output)
        
        # Update transcript with real names
        updated_transcript_file = os.path.splitext(output_transcript_file)[0] + "_named.txt"
        update_transcript_with_names(formatted_output, speaker_names, updated_transcript_file)
        print(f"\nUpdated transcript saved to {updated_transcript_file}")
    
    print("\nProcessing completed successfully!")
except Exception as e:
    print(f"Error during speaker assignment: {e}")
    sys.exit(1)
