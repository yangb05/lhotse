"""
Prepare for all the Arabic corpus collected for now(2023/10/23).
"""

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


def _read_raw_manifest(corpus_dir):
    """Build a list of wav dir and corresponding txt dir pairs from a wenet data.list file.

    :param corpus_dir: Path, dir of the ru_open_stt data.
    :return: a list of all the wav and corresponding txt absolute dir pairs of the whole dataset.
    """
    print(f"->Start reading raw manifests...")
    raw_train_manifest, raw_dev_manifest, raw_test_manifest = set(), set(), set()
    for subset, raw_manifest in zip(
        ('train', 'masc_clean_dev', 'mgb2_test'),
        [raw_train_manifest, raw_dev_manifest, raw_test_manifest],
    ):
        with open(corpus_dir / subset / 'data.list', 'r') as f:
            for line in f.readlines():
                data = json.loads(line.strip())
                raw_manifest.add((data['key'], data['wav'], data['txt']))
    print(
        f"<-Finish reading raw manifests, total samples: {len(raw_train_manifest) + len(raw_dev_manifest) + len(raw_test_manifest)}!"
    )
    return raw_train_manifest, raw_dev_manifest, raw_test_manifest


def prepare_arabic(
    corpus_dir: Pathlike,
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

    manifests = defaultdict(dict)
    raw_train_manifest, raw_dev_manifest, raw_test_manifest = _read_raw_manifest(corpus_dir)
    for split, raw_manifest in zip(
        ["train", "dev", "test"],
        [raw_train_manifest, raw_dev_manifest, raw_test_manifest],
    ):
        manifests[split] = {"recordings": [], "supervisions": []}
        with ProcessPoolExecutor(num_jobs) as ex:
            for recording, segment in tqdm(
                ex.map(parse_utterance, raw_manifest),
                desc=f"Processing Arabic {split} manifests",
            ):
                if recording is not None:
                    manifests[split]["recordings"].append(recording)
                    manifests[split]["supervisions"].append(segment)
        recordings, supervisions = fix_manifests(
            recordings=RecordingSet.from_recordings(manifests[split]["recordings"]),
            supervisions=SupervisionSet.from_segments(manifests[split]["supervisions"]),
        )
        validate_recordings_and_supervisions(
            recordings=recordings, supervisions=supervisions
        )
        if output_dir is not None:
            supervisions.to_file(
                output_dir / f"arabic_supervisions_{split}.jsonl.gz"
            )
            recordings.to_file(output_dir / f"arabic_recordings_{split}.jsonl.gz")
        manifests[split] = {
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
            language="Arabic",
            text=data[2],
        )
        validate_supervision(segment)
        return recording, segment
    except:
        return None, None


if __name__ == "__main__":
    corpus_dir = Path("/mgData4/yangb/data/Arabic/wavedata")
    output_dir = Path("/mgData4/yangb/data/Arabic/manifests")
    prepare_arabic(corpus_dir=corpus_dir, output_dir=output_dir, num_jobs=20)
