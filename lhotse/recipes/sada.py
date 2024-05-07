"""
About the SADA corpus: https://www.kaggle.com/datasets/sdaiancai/sada2022
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import pandas as pd
from tqdm.auto import tqdm
import torchaudio

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, validate_recordings_and_supervisions, AudioSource
from lhotse.qa import fix_manifests
from lhotse.utils import Pathlike


# transform str to numeric column
def transform(df):
    numeric_columns = ["FullFileLength", "SegmentLength", "SegmentStart", "SegmentEnd"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])
    return df
    

# filter by ProcessedText and SegmentLength
def filter(df, part):
    df = df[df["ProcessedText"].notnull()]
    if part == "train":
        df = df[(df["SegmentLength"] > 1) & (df["SegmentLength"] < 54)]
    return df


# make utterance
def make_utterance(audio_file, utter_file, utter_offset, num_samples):
    waveform, sample_rate = torchaudio.load(audio_file, frame_offset=utter_offset, num_frames=num_samples)
    assert sample_rate == 16000, print(f"sample rate is: {sample_rate}")
    torchaudio.save(utter_file, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)


# make recording
def make_recording(utter_id, utter_file, duration, num_samples):
    recording = Recording(
    id=utter_id,
    sources=[AudioSource(type='file', channels=[0], source=utter_file)],
    sampling_rate=16000,
    duration=(duration),
    num_samples=num_samples,  
    )
    return recording


# make supervision
def make_supervision(row, utter_id, utter_start, duration):
    supervision = SupervisionSegment(
        id=utter_id,
        recording_id=utter_id,
        start=utter_start,
        duration=duration,
        channel=0,
        text=row["ProcessedText"],
        language=row["SpeakerDialect"],
        speaker=row["Speaker"],
        gender=row["SpeakerGender"],
    )
    return supervision


def make_manifest(row, utterance_dir, corpus_dir):
    _, row = row
    sample_rate = 16000
    utter_id = row["SegmentID"]
    utter_start = round(row["SegmentStart"], 3)
    utter_offset = int(utter_start * sample_rate)
    duration = round(row["SegmentEnd"] - row["SegmentStart"], 3)
    num_samples = int(duration * sample_rate)
    audio_file = corpus_dir / row["FileName"]
    utter_file = utterance_dir / f"{row['SegmentID']}.wav"
    make_utterance(audio_file, utter_file, utter_offset, num_samples)
    recording = make_recording(utter_id, utter_file, duration, num_samples)
    supervision = make_supervision(row, utter_id, utter_start, duration)
    return recording, supervision


def prepare_sada(
    corpus_dir: Pathlike,
    utterance_dir: Pathlike, 
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 20,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param utterance_dir: Pathlike, the path of the utterance dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    utterance_dir.mkdir(parents=True, exist_ok=True)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    manifests = defaultdict(dict)
    dataset_parts = ["train", "valid", "test"]
    for part in tqdm(dataset_parts, desc=f"Processing audios"):
        recordings = []
        supervisions = []
        orig_info = pd.read_csv(corpus_dir / f"{part}.csv")
        orig_info = transform(orig_info)
        clean_info = filter(orig_info, part)
        with ProcessPoolExecutor(max_workers=num_jobs) as ex:
            for recording, sueprvision in tqdm(
                ex.map(make_manifest, clean_info.iterrows(), repeat(utterance_dir), repeat(corpus_dir)),
                desc=f"making sada {part} manifests"
                ):
                recordings.append(recording)
                supervisions.append(sueprvision)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"sada_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"sada_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    corpus_dir = Path("/mgData4/yangb/data/sada2022")
    utterance_dir = Path("/mgData4/yangb/data/sada2022/utterances")
    output_dir = Path("/mgData2/yangb/icefall/egs/arabic/SADA_ASR/data/manifests")
    prepare_sada(corpus_dir, utterance_dir, output_dir)