import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import mne
from cachetools import TTLCache

from consts import SOURCE_VOL_SPACING
from data.path_utils import PathArgs

morph_cache = TTLCache(maxsize=10, ttl=9999)  # long calculations but can also be a bit memory intensive, hence the ttl


def get_stc_morph(src: mne.SourceSpaces, subj: str, subj_dir: str, verbose: bool = False) -> mne.SourceMorph:
    if (subj_dir, subj) in morph_cache:
        return morph_cache[(subj_dir, subj)]

    fsaverage_strucs = StructuralsData.get_def_structural(subjects_dir=Path(subj_dir).parent)
    subj_to = '../fsaverage'
    fsaverage_strucs.src[0]['subject_his_id'] = subj_to  # hack but works

    morph = mne.compute_source_morph(src=src, subject_from=f'sub-{subj}', subject_to=subj_to,
                                     subjects_dir=subj_dir, src_to=fsaverage_strucs.src,
                                     spacing=SOURCE_VOL_SPACING, verbose=verbose)

    morph_cache[(subj_dir, subj)] = morph

    return morph


@dataclass
class StructuralsData:
    # has properties+constructor stuff to avoid unnecessary computations if not using the structurals,
    # eg loading cached data
    _fiducials: Optional[list] = field(default=None, init=False)
    _src: Optional[mne.SourceSpaces] = field(default=None, init=False)
    _bem: Optional[mne.bem.ConductorModel] = field(default=None, init=False)
    constructor: Callable[[], 'StructuralsData'] = field(default=None)

    @classmethod
    def create(cls, fiducials: list, src: mne.SourceSpaces, bem: mne.bem.ConductorModel):
        structs = cls()
        structs._fiducials = fiducials
        structs._src = src
        structs._bem = bem
        return structs

    @property
    def fiducials(self) -> list:
        self._ensure_constructed()
        return self._fiducials

    @property
    def src(self) -> mne.SourceSpaces:
        self._ensure_constructed()
        return self._src

    @property
    def bem(self) -> mne.bem.ConductorModel:
        self._ensure_constructed()
        return self._bem

    def _ensure_constructed(self):
        if self.constructor is not None:
            constructed = self.constructor()
            self._fiducials = constructed._fiducials
            self._src = constructed._src
            self._bem = constructed._bem
            self.constructor = None  # Clear the constructor to avoid unnecessary future calls

    @classmethod
    def get_def_structural(cls, subjects_dir: str = 'bids_dataset', verbose: bool = False) -> 'StructuralsData':
        def constructor():
            # Load the BEM model (Boundary Element Model)
            try:
                mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=verbose)
            except FileNotFoundError:
                mne.datasets.download_fsaverage_data(subjects_dir)

            model = mne.make_bem_model(subject='fsaverage', subjects_dir=subjects_dir, verbose=verbose,
                                       conductivity=(0.3,))  # single layer model, sufficient for meg
            bem = mne.make_bem_solution(model, verbose=verbose)

            # Create the source volume space
            src = mne.setup_volume_source_space(subject='fsaverage', subjects_dir=subjects_dir, pos=SOURCE_VOL_SPACING,
                                                verbose=verbose)

            # Perform automatic coregistration using the digitized points
            fiducials = mne.coreg.get_mni_fiducials(subject='fsaverage', subjects_dir=subjects_dir, verbose=verbose)

            return cls.create(fiducials, src, bem)

        return cls(constructor=constructor)

    def set_subject_fiducials(self, path_args: 'PathArgs', verbose: bool = False):
        try:
            fiducials = mne.coreg.get_mni_fiducials(subject=path_args.subj, subjects_dir=path_args.root,
                                                    verbose=verbose)
            self._fiducials = fiducials
        except FileNotFoundError:
            return


def morph_stc_to_fsaverage(stc: mne.VolVectorSourceEstimate, structs: StructuralsData, path_args: PathArgs,
                           verbose: bool = False) -> mne.VolVectorSourceEstimate:
    if structs is None:
        return stc  # already fsaverage

    fsaverage_strucs = StructuralsData.get_def_structural(path_args.data_path, verbose=verbose)
    subj_to = os.path.join('..', 'fsaverage')
    fsaverage_strucs.src[0]['subject_his_id'] = subj_to  # hack but works

    morph = get_stc_morph(structs.src, path_args.subj, path_args.root, verbose=verbose)

    return morph.apply(stc)


def morph_stc_to_subj(stc: mne.VolVectorSourceEstimate, structs: StructuralsData, path_args: PathArgs,
                      morph_subj: str, verbose: bool = False) -> mne.VolVectorSourceEstimate:
    from data.dataset_utils import SchoffelenSubloader, ArmeniSubloader
    if morph_subj == 'fsaverage':
        return morph_stc_to_fsaverage(stc, structs, path_args, verbose=verbose)

    morph_dataset = 'schoffelen' if morph_subj.startswith('A') else 'armeni'  # todo implement for other datasets
    if morph_dataset == 'schoffelen':
        morph_path_args = PathArgs(subj=morph_subj, sess=None, task='auditory', data_path=path_args.data_path,
                                   dataset=morph_dataset)
        morph_structs = SchoffelenSubloader.get_structurals(morph_path_args, verbose=verbose)
    else:
        morph_path_args = PathArgs(subj=morph_subj, sess='001', task='compr', data_path=path_args.data_path,
                                   dataset=morph_dataset)
        morph_structs = ArmeniSubloader.get_structurals(morph_path_args, verbose=verbose)

    subj_to = os.path.join('..', morph_dataset, f'sub-{morph_subj}')
    morph_structs.src[0]['subject_his_id'] = subj_to
    morph = mne.compute_source_morph(src=structs.src, subject_from=f'sub-{path_args.subj}',
                                     subject_to=subj_to,
                                     subjects_dir=path_args.root, src_to=morph_structs.src,
                                     spacing=SOURCE_VOL_SPACING, verbose=verbose)

    return morph.apply(stc)
