# Some code is taken from the brainmagick repository. Original LICENSE is:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from dataclasses import dataclass, astuple
from pathlib import Path


@dataclass
class PathArgs:
    subj: str
    sess: str
    task: str
    data_path: str
    dataset: str

    def __iter__(self):
        return iter(astuple(self))

    def get_voxel_cache_path(self, morph: bool = False):
        base_path = os.path.join(self.root, f"sub-{self.subj}")
        if self.sess is not None:
            base_path = os.path.join(base_path, f"ses-{self.sess}")
        return os.path.join(base_path, f"task-{self.task}", f"morph-{morph}", "voxels")

    @property
    def root(self):
        return os.path.join(self.data_path, self.dataset)

    def get_raw_cache_path(self, highpass: bool = False):
        base_path = os.path.join(self.root, f"sub-{self.subj}")
        if self.sess is not None:
            base_path = os.path.join(base_path, f"ses-{self.sess}")
        return os.path.join(base_path, f"task-{self.task}", f"highpass-{highpass}", "raw.fif")

    def get_audio_log_path(self):
        assert self.dataset == 'schoffelen', "Only Schoffelen dataset needs to use log files"
        # Extract subject number
        subj = self.subj.strip('sub-')

        log_dir = os.path.join(self.root, 'sourcedata', 'meg_task')

        if not os.path.isdir(log_dir):
            raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

        # Compile the regex to match filenames like 'subj-<number>-MEG-MOUS-Aud.log'
        pattern = re.compile(rf'{subj}-(\d+)-MEG-MOUS-Aud\.log')

        for filename in os.listdir(log_dir):
            if pattern.match(filename):
                return os.path.join(log_dir, filename)

        raise FileNotFoundError(f"No log file found for subject {subj} in {log_dir}")

    def get_stimuli_path(self, name: str):
        assert self.dataset == 'schoffelen', "Only Schoffelen dataset needs to use stimuli files"
        return os.path.join(self.root, 'stimuli', 'audio_files', f"EQ_Ramp_Int2_Int1LPF{name}")

    def get_stimuli_txt_path(self):
        assert self.dataset == 'schoffelen', "Only Schoffelen dataset needs to use stimuli"
        return os.path.join(self.root, 'stimuli', "stimuli.txt")

    def get_phoneme_path(self, sequ_id: int) -> Path:
        assert self.dataset == 'schoffelen', "Only Schoffelen dataset needs to use phoneme files"
        return Path(os.path.join(self.root, 'derivatives', 'phonemes', "EQ_Ramp_Int2_Int1LPF%.3i.TextGrid" % sequ_id))
