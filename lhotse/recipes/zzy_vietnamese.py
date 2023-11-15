"""
Prepare for all the Vietnamese corpus provided by zhaozhiyuan.
"""

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Sequence

from tqdm.auto import tqdm

from lhotse import (
    fix_manifests,
    validate_recording,
    validate_recordings_and_supervisions,
    validate_supervision,
)
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from underthesea import text_normalize


VIET_PARTS = ('train', 'dev', 'test')


def preprocess_text(text):
    # % and @ are not considered as punctuations
    english_punctuations = '!"#$&\'()*+,-./:;<=>?[\\]^_`{|}~'
    # remove punctuations, basically ; and '
    text = text.translate(str.maketrans('', '', english_punctuations))
    # remove whitespace
    text = text.strip()
    # normalize text
    text = text_normalize(text)
    return text


def _read_raw_manifest(corpus_dir, subset):
    """Build a list of wav dir and corresponding txt dir pairs from a wenet data.list file.
    :param corpus_dir: Path, dir of the vietnamese data.
    :return: a list of the whole dataset.
    """
    raw_manifest = set()
    with open(corpus_dir / subset / 'data.list', 'r') as f:
        for line in tqdm(f.readlines(), desc=f'Reading {subset} datalist'):
            data = json.loads(line.strip())
            raw_manifest.add((data['key'], data['wav'], preprocess_text(data['txt'])))
    return raw_manifest


def prepare_vienamese(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "all",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: Number of workers to extract manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with
             the keys 'recordings' and 'supervisions'.
    """
    print(f"num of threads: {num_jobs}")
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    subsets = VIET_PARTS if "all" in dataset_parts else dataset_parts
    manifests = defaultdict(dict)
    for sub in subsets:
        if sub not in VIET_PARTS:
            raise ValueError(f"No such part of dataset in Vietnamese speech : {sub}")
        manifests[sub] = {"recordings": [], "supervisions": []}
        raw_manifest = _read_raw_manifest(corpus_dir, sub)
        with ProcessPoolExecutor(num_jobs) as ex:
            for recording, segment in tqdm(
                ex.map(parse_utterance, raw_manifest),
                desc=f"Processing Vietnamese {sub} manifest",
            ):
                if recording is not None:
                    manifests[sub]["recordings"].append(recording)
                    manifests[sub]["supervisions"].append(segment)
        recordings, supervisions = fix_manifests(
            recordings=RecordingSet.from_recordings(manifests[sub]["recordings"]),
            supervisions=SupervisionSet.from_segments(manifests[sub]["supervisions"]),
        )
        validate_recordings_and_supervisions(
            recordings=recordings, supervisions=supervisions
        )
        if output_dir is not None:
            supervisions.to_file(
                output_dir / f"vietnamese_supervisions_{sub}.jsonl.gz"
            )
            recordings.to_file(output_dir / f"vietnamese_recordings_{sub}.jsonl.gz")
        manifests[sub] = {
            "recordings": recordings,
            "supervisions": supervisions,
        }
    return manifests


def parse_utterance(
    data: Tuple,
) -> Tuple[Recording, Dict[str, List[SupervisionSegment]]]:
    try:
        recording = Recording.from_file(data[1], recording_id=data[0])
        validate_recording(recording)
        segment = SupervisionSegment(
            id=data[0],
            recording_id=data[0],
            start=0.0,
            duration=recording.duration,
            channel=0,
            language="Vietnamese",
            text=data[2],
        )
        validate_supervision(segment)
        return recording, segment
    except:
        return None, None


if __name__ == "__main__":
    corpus_dir = Path("/mgData1/yangb/data/zzy_vietnamese")
    output_dir = Path("/mgData2/yangb/icefall/egs/vietnamese/ASR/data/manifests")
    prepare_vienamese(corpus_dir=corpus_dir, dataset_parts=['dev', 'test'], output_dir=output_dir, num_jobs=20)
