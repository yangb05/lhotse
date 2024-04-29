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

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, validate_recordings_and_supervisions, AudioSource
from lhotse.qa import fix_manifests
from lhotse.utils import Pathlike


def make_manifest(row, corpus_dir):
    recording = Recording(
        id=row["SegmentID"].split('-', 1)[0],
        sources=[AudioSource(type='file', channels=[0], source=f'{str(corpus_dir)}/{row["FileName"]}')],
        sampling_rate=16000,
        duration=float(row["FullFileLength"]),
        num_samples=int(16000 * float(row["FullFileLength"])),  
    )
    segment = SupervisionSegment(
        id=row["SegmentID"],
        recording_id=recording.id,
        start=float(row["SegmentStart"]),
        duration=round(float(row["SegmentLength"]), 3),
        channel=0,
        text=row["ProcessedText"],
        language=row["SpeakerDialect"],
        speaker=row["Speaker"],
        gender=row["SpeakerGender"],
    )
    return recording, segment


def prepare_sada(
    corpus_dir: Pathlike, 
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    manifests = defaultdict(dict)
    dataset_parts = ["train", "valid", "test"]
    for part in tqdm(
        dataset_parts,
        desc="Process sada audios.",
    ):
        logging.info(f"Processing sada subset: {part}")
        recordings = []
        supervisions = []
        orig_info = pd.read_csv(corpus_dir / f"{part}.csv")
        with ProcessPoolExecutor(max_workers=num_jobs) as ex:
            for recording, segment in tqdm(
                ex.map(make_manifest, orig_info.iterrows(), repeat(corpus_dir)),
                desc="making sada manifests"
                ):
                recordings.append(recording)
                supervisions.append(segment)

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
    prepare_sada("/mgData4/yangb/data/sada2022", "/mgData4/yangb/data/sada2022/manifests")