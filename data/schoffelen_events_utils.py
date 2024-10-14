# Code is based on code from the brainmagick repository. Original LICENSE is:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Union, Optional, Any

import mne
import numpy as np
import pandas as pd
from Levenshtein import editops
from mne_bids import BIDSPath
from scipy.stats import spearmanr

import textgrid
from consts import PH_DICT
from data.path_utils import PathArgs

# code graciously taken from brainmagick

logger = logging.getLogger(__name__)


def _process_log_block(block: str) -> list[dict[str, Any]]:
    """Parse a block of annotation log"""
    lines = block.split("\n")
    # find header line
    iterlines = enumerate(lines)
    ind, line = next(iterlines)
    while "Uncertainty" not in line:  # only present in the header
        ind, line = next(iterlines)
    # build header (Uncertainty is present twice and must be updated)
    headers = [x.replace(" ", "_") for x in line.split("\t")]
    replacements = iter(["time_uncertainty", "duration_uncertainty"])
    for k, name in enumerate(headers):
        if name == "Uncertainty":
            headers[k] = next(replacements)
    # build data
    data: list[dict[str, Any]] = []
    for line in lines[ind + 1:]:
        if not line:
            continue
        line_dict = dict(zip(headers, line.split("\t")))
        # convert to seconds if it's a time/duration field
        line_dict = {x: _seconds_if_time(x, y) for x, y in line_dict.items()}
        data.append(line_dict)
    return data


def _seconds_if_time(key: str, val: str) -> Any:
    """Converts time/duration field values to seconds (initially in 1e-4)"""
    if val.isnumeric() and any(z in key.lower() for z in ["time", "dur"]):
        return float(val) / 1e4
    return val


def _parse_log(log_fname: str):
    text = Path(log_fname).read_text()

    # Fixes broken inputs
    text = text.replace(".\n", ".")

    # used to avoid duplicates in some subjects
    # FIXME it's unclear to me why these subjects have duplicated logs
    text = text.split("Scenario -")[1]

    # file is made of two blocks
    data1, data2 = [_process_log_block(block) for block in text.split("\n\n\n")]
    df1 = pd.DataFrame(data1)

    # # the two dataframe are only synced on certains rows
    common_samples = ("Picture", "Sound", "Nothing")
    sel = df1["Event_Type"].apply(lambda x: x in common_samples)
    index = df1.loc[sel].index
    df2 = pd.DataFrame(data2, index=index)

    # remove duplicate
    duplicates = np.intersect1d(df1.keys(), df2.keys())
    for key in duplicates:
        assert (df1.loc[index, key] == df2[key].fillna("")).all()
        df2.pop(key)

    log = pd.concat((df1, df2), axis=1)
    return log


def _clean_log(log):
    # Relabel condition: only applies to sample where condition changes
    translate = dict(
        ZINNEN="sentence",
        WOORDEN="word_list",
        FIX="fix",
        QUESTION="question",
        Response="response",
        ISI="isi",
        blank="blank",
    )
    for key, value in translate.items():
        sel = log.Code.astype(str).str.contains(key)
        log.loc[sel, "condition"] = value
    log.loc[log.Code == "", "condition"] = "blank"

    # Annotate sequence idx and extend context to all trials
    start = 0
    block = 0
    context = "init"
    log["new_context"] = False
    query = 'condition in ("word_list", "sentence")'
    for row in log.query(query).itertuples():
        idx = row.Index
        log.loc[start:idx, "context"] = context
        log.loc[start:idx, "block"] = block
        log.loc[idx, "new_context"] = True
        context = row.condition
        block += 1
        start = idx
    log.loc[start:, "context"] = context
    log.loc[start:, "block"] = block

    # Format time
    log.loc[:, "Time"] = [0.0 if not isinstance(x, (int, float)) else x for x in log.Time]

    # Extract individual word
    log.loc[log.condition.isna(), "condition"] = "word"
    idx = log.condition == "word"
    words = log.Code.str.strip("0123456789 ")
    log.loc[idx, "word"] = words.loc[idx]
    sel = log.query('word=="" and condition=="word"').index
    log.loc[sel, "word"] = np.nan
    log.loc[log.word.isna() & (log.condition == "word"), "condition"] = "blank"
    log.loc[log.Code == "pause", "condition"] = "pause"
    log.columns = log.columns.str.lower()  # remove capitalization!
    log.loc[log.word == 'PULSE MODE', 'condition'] = 'pulse'
    return log


def add_word_sequence_and_position(log: pd.DataFrame) -> pd.DataFrame:
    """Add word_sequence (the sequence of words in the sentence/word list) and
    word_index (its position) for each event of the log
    """
    indices = log.loc[log.condition == "fix"].index.tolist()
    for ind1, ind2 in zip(indices, indices[1:] + [log.index[-1]]):
        sub = log.loc[ind1: ind2, :]
        is_word = sub.condition == "word"
        sequence = " ".join(sub.loc[is_word, :].word)
        if sequence:
            log.loc[ind1:ind2, "word_sequence"] = sequence
            log.loc[ind1:ind2, "word_index"] = np.maximum(0, np.cumsum(is_word) - 1)
    return log


def wave_file(path_args: PathArgs, name: str) -> Path:
    # eg name: 186.wav
    if name.startswith('/'):
        return Path(name)  # already a full absolute path  (useful for tests)
    return path_args.get_stimuli_path(name)


def _add_sound_events(path_args: PathArgs, log):
    # Extract wave fname from structure
    sel = log["event_type"] == "Sound"

    def get_fname(s): return str(wave_file(path_args, s.split("Start File ")[1]))  # noqa

    filepaths = log.loc[sel, "code"].apply(get_fname)
    log.loc[sel, "filepath"] = filepaths

    # add wave fname to audio onset
    sel = log.query("event_type == 'Sound'").index
    log.loc[sel + 1, "filepath"] = log.loc[sel, "filepath"].values

    log.loc[sel, "condition"] = "sound_legacy"
    log.loc[sel + 1, "condition"] = "sound"
    # features without "task" tag set are ignored during training,
    # so we set this tag properly
    # TODO move this one level up and mark "audio" for all "word" and "events" conditions
    return log


def add_sequence_uid(path_args: PathArgs, log):
    """Add sequence uid to the metadata.
    """
    # some trials missed the last word
    max_char = 45
    sequence_uids = dict()
    with open(path_args.get_stimuli_txt_path(), 'r') as f:
        lines = f.readlines()
        for line in lines:
            idx = line.find(' ')
            uid = int(line[:idx])
            sequence = line[idx + 1:].replace('\n', '')
            sequence = sequence[:max_char].lower()
            assert sequence not in sequence_uids.keys()
            sequence_uids[sequence] = uid
            assert uid != 0, "uid should not be 0"

    def _map(sequence):
        if not isinstance(sequence, str):
            return None
        key = sequence[:max_char].lower()
        assert key in sequence_uids, key
        return sequence_uids[key]

    sequence_uid = log.word_sequence.map(_map)
    first_idx = (sequence_uid.isna()).argmin()  # return first non NaN
    assert not (sequence_uid.iloc[first_idx:].isna()).any(), 'NaNs should be only at start'
    sequence_uid.iloc[:first_idx] = sequence_uid.iloc[first_idx]
    log['sequence_uid'] = sequence_uid
    return log


def _map_phonemes_to_ids_internal(phonemes_list, phonemes_ids_dict):
    phonemes_ids = []

    for phoneme in phonemes_list:
        key = phoneme.name
        assert key in phonemes_ids_dict, f"{key} not in dict {phonemes_ids_dict}"
        phonemes_ids.append(phonemes_ids_dict[key])
    return phonemes_ids


def _map_phonemes_to_ids(phonemes_list):
    return _map_phonemes_to_ids_internal(phonemes_list, PH_DICT)


def tgrid_to_dict(fname: str) -> list[dict[str, Any]]:
    """Parse TextGrid Praat file and generates a dataframe containing both
    words and phonemes"""
    tgrid = textgrid.read_textgrid(fname)  # type: ignore
    parts: dict[str, Any] = {}
    for p in tgrid:
        if p.name != "" and p.name != "<p:>":  # Remove empty entries
            parts.setdefault(p.tier, []).append(p)

    # Separate orthographics, phonetics, and phonemes
    words = parts["ORT-MAU"]
    phonemes = parts["MAU"]
    phonemes_ids = _map_phonemes_to_ids(phonemes)
    assert len(phonemes) == len(phonemes_ids)

    # Def concatenate orthographics and phonetics
    rows: list[dict[str, Any]] = []
    for word_index, word in enumerate(words):
        rows.append(
            dict(
                event_type="word",
                start=word.start,
                stop=word.stop,
                word_index=word_index,
                word=word.name,
                modality="audio",
            )
        )

    # Add timing of individual phonemes
    starts = np.array([i["start"] for i in rows])
    # phonemes and starts are both ordered so this could be further optimized if need be
    for phoneme, ph_id in zip(phonemes, phonemes_ids):
        idx = np.where(phoneme.start < starts)[0]
        idx = idx[0] - 1 if idx.size else len(rows) - 1
        row = rows[idx]
        rows.append(
            dict(
                event_type="phoneme",
                start=phoneme.start + 1e-6,
                stop=phoneme.stop,
                word_index=row["word_index"],
                word=row["word"],
                phoneme=phoneme.name,
                phoneme_id=ph_id,
                modality="audio",
            )
        )
    # not sure why sorting is needed, but otherwise a sample is dropped
    rows.sort(key=lambda x: float(x["start"]))
    return rows


def _add_phonemes(path_args: PathArgs, log: pd.DataFrame, phonemes_path: Optional[Path] = None) -> pd.DataFrame:
    """Add phonemes and word timing to the log of the auditory experiment"""
    if phonemes_path is None:
        phonemes_path = path_args.get_phoneme_path(0).parent

    # Add audio file name across dataframe
    file_ = np.nan
    prev_start = 0
    prev_stop = 0

    log["sequence_id"] = np.nan
    starts = np.where(log.word.apply(lambda x: "Start File" in str(x)))[0]
    stops = np.where(log.word.apply(lambda x: "End of file" in str(x)))[0]

    assert len(starts) == len(stops)

    for start, stop in zip(starts, stops):
        # set file to previous rows
        log.loc[slice(prev_start, prev_stop), "sequence_id"] = file_
        # update file name
        file_ = int(log.loc[start, "word"].split()[-1][:-4])
        prev_start, prev_stop = start, stop
    log.loc[slice(prev_start, prev_stop), "sequence_id"] = file_

    # For each audio file, add timing of words and phonemes
    starts = np.where(log.word == "Audio onset")[0]
    rows: list[dict[str, Any]] = []  # faster than appending on the fly
    for start in starts:
        row = log.loc[start, :]
        if not row.condition == "sound":  # should be used for SentenceWavFeature
            raise RuntimeError(f"Unexpected condition {row.condition}")
        fname = (
                str(phonemes_path) + "/EQ_Ramp_Int2_Int1LPF%.3i.TextGrid" % row.sequence_id
        )
        content = tgrid_to_dict(fname)
        for d in content:
            d.update(subject=row.subject, trial=row.trial, stim_type="sound",
                     context=row.context, block=row.block, sequence_id=row.sequence_id,
                     duration=d["stop"] - d["start"],
                     filepath=row.filepath,
                     time=row.time + d["start"]
                     )  # audio onset
        log.loc[start, "start"] = 0
        duration = content[-1]["stop"]
        log.loc[start, "stop"] = duration
        log.loc[start, "duration"] = duration
        rows.extend(content)
    log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True, sort=False)

    # homogeneize names
    for condition in ("word", "phoneme"):
        idx = log.query("event_type == @condition").index
        log.loc[idx, "condition"] = condition

    # fix
    idx = log.query('word=="End of file"').index
    log.loc[idx, "condition"] = "end"
    idx = log.query('event_type=="Nothing" and condition=="word"').index
    log.loc[idx, "condition"] = "nothing"
    return log.sort_values("time")


def read_log(path_args: PathArgs, verbose: bool = False) -> pd.DataFrame:
    log_fname = path_args.get_audio_log_path()
    log = _parse_log(log_fname)
    log = _clean_log(log)
    if "MEG-MOUS-Aud" in log_fname:
        log = _add_sound_events(path_args, log)
        log = _add_phonemes(path_args, log)
    elif "MEG-MOUS-Vis" in log_fname:
        words = log.query('condition == "word"')
        # TODO check duration?
        log.loc[words.index, "modality"] = "visual"
    else:
        raise ValueError(f"Unknown log type: {log_fname}")
    log = add_word_sequence_and_position(log)
    try:
        log = add_sequence_uid(path_args, log)
    except Exception:
        print("failure", log_fname)
        raise
    assert len(log)
    return log


def match_list(A, B, on_replace="delete"):
    """Match two lists of different sizes and return corresponding indice
    Parameters
    ----------
    A: list | array, shape (n,)
        The values of the first list
    B: list | array: shape (m, )
        The values of the second list
    Returns
    -------
    A_idx : array
        The indices of the A list that match those of the B
    B_idx : array
        The indices of the B list that match those of the A
    """

    if not isinstance(A, str):
        unique = np.unique(np.r_[A, B])
        label_encoder = dict((k, v) for v, k in enumerate(unique))

        def int_to_unicode(array: np.ndarray) -> str:
            return "".join([str(chr(label_encoder[ii])) for ii in array])

        A = int_to_unicode(A)
        B = int_to_unicode(B)

    changes = editops(A, B)
    B_sel = np.arange(len(B)).astype(float)
    A_sel = np.arange(len(A)).astype(float)
    for type_, val_a, val_b in changes:
        if type_ == "insert":
            B_sel[val_b] = np.nan
        elif type_ == "delete":
            A_sel[val_a] = np.nan
        elif on_replace == "delete":
            # print('delete replace')
            A_sel[val_a] = np.nan
            B_sel[val_b] = np.nan
        elif on_replace == "keep":
            # print('keep replace')
            pass
        else:
            raise NotImplementedError
    B_sel = B_sel[np.where(~np.isnan(B_sel))]
    A_sel = A_sel[np.where(~np.isnan(A_sel))]
    assert len(B_sel) == len(A_sel)
    return A_sel.astype(int), B_sel.astype(int)


def get_log_times(log: pd.DataFrame, events: np.ndarray, sfreq: float) -> pd.DataFrame:
    last_sample = events[-1, 0]
    sel: Union[np.ndarray, slice] = np.sort(
        np.r_[
            np.where(events[:, 2] == 20)[0],  # fixation
            np.where(events[:, 2] == 10)[0],  # context
        ]
    )
    common_megs = events[sel]
    common_logs = log.query('(new_context == True) or condition=="fix"')

    last_log = common_logs.time.values[0]
    last_meg = common_megs[0, 0]
    last_idx = 0

    # TODO FIXME match_list may be based on too few elements, and
    # generate random timings, hence the assert > 40 (chosen arbitrarily)
    # fix missing triggers with leventhstein distance
    fix_logs = common_logs.code.str.contains("FIX")
    fix_megs = common_megs[:, 2] == 20
    if len(fix_megs) < 40 or len(fix_logs) < 40:
        logger.warning("CAUTION: match_list may be based on too few elements, and "
                       "generate random timings")
    assert len(fix_megs) > 1 and len(fix_logs) > 1
    idx_logs, idx_megs = match_list(fix_logs, fix_megs)

    time_logs = common_logs.iloc[idx_logs].time
    time_meg = events[idx_megs, 0] * sfreq
    r, _ = spearmanr(time_logs, time_meg)
    # check that there is a perfect correlation between the log and meg timings
    assert r > 0.9999
    common_megs = common_megs[idx_megs]
    common_logs = common_logs.iloc[idx_logs]

    assert len(common_megs) == len(common_logs)
    for common_meg, common_log in zip(
            common_megs, common_logs.itertuples()
    ):
        idx = common_log.Index
        if common_meg[2] == 20:
            assert common_log.condition == "fix"
        else:
            assert common_log.condition in ("sentence", "word_list")

        log.loc[idx, "meg_time"] = common_meg[0] / sfreq

        sel = slice(last_idx + 1, idx)
        times = log.loc[sel, "time"] - last_log + last_meg / sfreq
        assert np.all(np.isfinite(times.astype(float)))
        log.loc[sel, "meg_time"] = times

        last_log = common_log.time
        last_meg = common_meg[0]
        last_idx = idx

        assert np.isfinite(last_log) * np.isfinite(last_meg)

    # last block
    sel = slice(last_idx + 1, None)
    times = log.loc[sel, "time"] - last_log + last_meg / sfreq
    log.loc[sel, "meg_time"] = times
    log.meg_time = log.meg_time.fillna(-1)
    log["meg_sample"] = np.array(log.meg_time.values * sfreq, int)

    # Filter out events that are after the last MEG trigger
    n_out = np.sum(log.meg_sample > last_sample) + np.sum(log.meg_sample < 0)
    if n_out:
        logger.warning(
            f"CAUTION: {n_out} events occur after the last MEG trigger and will thus be removed"
        )

    log = log.query(f"meg_sample<={last_sample} and meg_sample>=0")

    return log


def read_events_file_schoffelen(path_args, sfreq: float) -> pd.DataFrame:
    """Loads events from the dataset
    """
    # having a separate function is helpful for mocking
    bids_path = BIDSPath(subject=path_args.subj, session=path_args.sess, task=path_args.task, root=path_args.root)
    raw = mne.io.read_raw_ctf(bids_path.fpath, preload=False, verbose=False)
    raw.pick('stim')
    raw.resample(sfreq)
    events = mne.find_events(raw, shortest_event=1)

    # Read Event data
    metadata = read_log(path_args)
    # Align MEG and events
    metadata = get_log_times(metadata, events, sfreq)
    # rename
    metadata = metadata.rename(columns=dict(
        start="offset", meg_time="start", stop="legacy_stop", condition="kind"
    ))

    # Clean up dataframe
    events_df = metadata.drop(
        columns=[x for x in metadata.columns if x.startswith("legacy_")])
    cols_to_keep = [
        'start', 'duration', 'kind', 'context', 'word', 'filepath', 'sequence_id',
        'word_index', 'phoneme', 'phoneme_id', 'word_sequence', 'sequence_uid']
    events_df = events_df.loc[events_df.kind.isin(['word', 'phoneme', 'sound']), cols_to_keep]

    return events_df
