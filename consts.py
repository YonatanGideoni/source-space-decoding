import json
import os
from pathlib import Path

import torch

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
SOURCE_VOL_SPACING = 15.  # mm
is_windows = os.name == 'nt'
SCH_TEST_SUBJS = ('A2004', 'A2005')
SCH_VAL_SUBJS = ('A2002', 'A2003')
AR_VAL_SESS = '009'
AR_TEST_SESS = '010'

dir_path = Path(__file__).parent

with open(dir_path / "phonemes.json", 'r') as f:
    PH_DICT = json.load(f)

DEF_DATA_PATH = None
