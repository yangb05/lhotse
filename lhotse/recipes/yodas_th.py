"""
Description taken from the official website of Russian Open Speech To Text
(https://learn.microsoft.com/en-us/azure/open-datasets/dataset-open-speech-text?tabs=azure-storage)

This Russian speech to text (STT) dataset includes:

~16 million utterances
~20,000 hours
2.3 TB (uncompressed in .wav format in int16), 356G in opus
All files were transformed to opus, except for validation datasets
The main purpose of the dataset is to train speech-to-text models.

See https://learn.microsoft.com/en-us/azure/open-datasets/dataset-open-speech-text?tabs=azure-storage for more details about Russian Open Speech To Text
"""
import re
import emoji
from itertools import repeat
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


TRAIN_PARTS = (
    "yodas_th000",
)


def preprocess(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace='') # 删掉 emoji
    text = text.replace(':D', '') # 删掉颜文字
    text = re.sub(r'\[âm nhạc\]|Top★Hmoob★Music|█|★|♪|♫|▢|ừ|à|…|<200b>', '', text) # 删掉无意义的字符
    text = re.sub(r'&.*;', '', text) # 删掉网页符号
    text = re.sub(r'^font color.*', '', text) # 删掉网页符号
    punctuations = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·""" # 删掉所有标点符号
    text = text.translate(str.maketrans('', '', punctuations))
    text = ' '.join(text.strip().split()) # 删掉多余的空格
    if len(text.split()) <= 3 and ("music" in text or "musc" in text or "nhạc" in text or "melody" in text): # 删掉音乐
        return None
    return text


def _read_raw_manifest(corpus_dir):
    """Build a list of all the wav files.

    :param corpus_dir: Path, dir of the ru_open_stt data.
    :return: a list of all the wav and corresponding text pairs of the whole dataset.
    """
    yodas_th000_manifest = set()
    for subset in sorted(corpus_dir.iterdir()):
        if subset.stem == "th000":
            target = yodas_th000_manifest
        # elif subset.stem == "yodas_vi100":
        #     target = yodas_vi100_manifest
        # elif subset.stem == "yodas_vi101":
        #     target = yodas_vi101_manifest
        else:
            continue
        print(f"----Start reading {subset.stem}...")
        wav_info = {}
        text_info = {}
        duration_info = {}
        # get wav info
        for audio in (subset / "untar").iterdir():
            for wav in (subset / "untar" / audio).iterdir():
                wav_info[wav.stem] = str(wav)
        # get text info 
        with open(subset / "text" / "all.txt") as f:
            for line in f.readlines():
                key, text = line.split(maxsplit=1)
                text  = preprocess(text)
                if text:
                    text_info[key] = text
        # get duration info
        with open(subset / "duration" / "all.txt") as f:
            for line in f.readlines():
                key, duration = line.split(maxsplit=1)
                if 1.0 <= float(duration) <= 20.0:
                    duration_info[key] = float(duration)
        # write to raw manifest
        for key, text in text_info.items():
            if key in wav_info and key in duration_info:
                target.add((key, wav_info[key], text))
        print(f"----Finish reading {subset.stem}!")
    return yodas_th000_manifest


def too_short_or_too_long(segment):
    if segment.duration < 1.0 or segment.duration > 20.0:
        print(
            f"Exclude segment with ID {segment.id} from training. Duration: {segment.duration}"
        )
        return True
    return False


def prepare_yodas_th(
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
    yodas_th000_manifest = _read_raw_manifest(corpus_dir)
    prefix = "yodas"
    for split, raw_manifest in zip(
        ["th000",],
        [yodas_th000_manifest,],
    ):
        manifests[split] = {"recordings": [], "supervisions": []}
        with ProcessPoolExecutor(num_jobs) as ex:
            for recording, segment in tqdm(
                ex.map(parse_utterance, raw_manifest),
                desc=f"Processing {split} manifests",
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
                output_dir / f"{prefix}_supervisions_{split}.jsonl.gz"
            )
            recordings.to_file(output_dir / f"{prefix}_recordings_{split}.jsonl.gz")
        manifests[split] = {
            "recordings": recordings,
            "supervisions": supervisions,
        }
    return manifests


def parse_utterance(
    utt: Tuple
) -> Tuple[Recording, Dict[str, List[SupervisionSegment]]]:
    try:
        recording = Recording.from_file(utt[1], recording_id=utt[0], force_opus_sampling_rate=16000)
        if too_short_or_too_long(recording):
            return None, None
        validate_recording(recording)
        segment = SupervisionSegment(
            id=utt[0],
            recording_id=utt[0],
            start=0.0,
            duration=recording.duration,
            channel=0,
            language="Thai",
            text=utt[2],
        )
        validate_supervision(segment)
        return recording, segment
    except:
        return None, None


# from lhotse import SupervisionSet


# def text_view(supervision_dir):
#     supervisions = SupervisionSet.from_file(supervision_dir)
#     with open(output_dir / "yodas_vi000.text", 'w') as f:
#         for supervision in supervisions:
#             key, text = supervision.id, supervision.text
#             f.write(f"{key} {text}\n")
    

if __name__ == "__main__":
    corpus_dir = Path("/mgData2/yangb/icefall/egs/th/ASR/download")
    output_dir = Path("/mgData2/yangb/icefall/egs/th/ASR/data/manifests")
    prepare_yodas_th(corpus_dir=corpus_dir, output_dir=output_dir, num_jobs=1)
    # text_view("/data_a100/userhome/yangb/data/yodas/manifests/yodas_vi000_supervisions.jsonl.gz")
