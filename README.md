## Source space neural decoding

This is the repository for `Non-invasive Neural Decoding in Source Reconstructed Brain Space`.

### Environment

Create a virtual environment from the `requirements.txt` with Python 3.10. Make sure to use `numpy<2`.

### Downloading data

$\rightarrow$ `data/download_utils.py`

For the Armeni and Schoffelen datasets you need to register at the [Radboud data repository](https://data.ru.nl/).
To download each dataset run:  
```python -m data.download_utils --dataset <armeni/schoffelen/gwilliams> --dataset_path /path/to/datasets/directory```  
Note Schoffelen takes a long time to download and prepare its structurals.
GWilliams wasn't used in the paper due to not having public structurals but is included for convenience.

Make sure you have enough space, these are big datasets! After preprocessing you can delete non `.fif/.h5/.xfm/.mgz` files. `.fif/.h5` is used to cache sensor/source data respectively while `.xfm/.mgz` are needed for structurals. For
Schoffelen you can delete the `V****` subjects as only auditory (`A****`) subjects were used.
Having many CPU cores will help speed up the initial preprocessing for Armeni/Schoffelen. Note this isn't the source
reconstruction detailed in the paper but just preparing their
structurals. [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) should be installed in advance.

### Pipeline ablations

$\rightarrow$ `comparisons/ablate_pipeline.py`  
Note that the voxel size ablation is hackily carried out by changing `SOURCE_VOL_SPACING` in `consts.py`.

### Hyperparameter search

$\rightarrow$ `train/train_opt_model.py`

For example, to replicate the Schoffelen source space MLP hyperparameter search run:  
```python -m train.train_opt_model --dataset schoffelen --dataset_path /path/to/datasets/directory --n_repeats 1 --model MLP --res_path hypopt_res.txt --subjects A2006 A2007 A2008 A2009 A2010 --cache_dir /path/to/cache/best/models --multi_subj True --n_params 250000 --data_space shared_source```

`--res_path` is where the results will be saved, `cache_dir` is where the best (validation accuracy wise) models are
cached, `n_params` is how many parameters the models have, `data_space` is the space the data is in (
sensor/source/shared source), `n_repeats` is over how many random seeds to optimise each hyperparameter configuration,
`multi_subj` is whether to train on multiple subjects at once (`True` for Schoffelen/the combined dataset, `False` for
Armeni), and `subjects` is the list of subjects to train on.

For all configuration settings/hyperparameters see `train/parser_utils.py`. For the hyperparameter distributions see `train/train_opt_model.py`'s `main`. The distributions can be given as inputs, eg. `--batch_sizes 16 32 64`.

### Evaluating best models

$\rightarrow$ `train/train_opt_model.py` but with `--opt_hyps False`. For example, to evaluate the Schoffelen sensor
space MLP over 3 seeds:  
``python -m train.train_opt_model --data_path /path/to/datasets/directory --dataset schoffelen --model MLP --n_repeats 3 --subjects A2006 A2007 A2008 A2009 A2010 --seed 42 --cache_dir /path/to/cache/models --multi_subj True --n_params 250000 --data_space sensor --opt_hyps False --dropout_prob 0.1 --lr 5.4e-4 --batch_size 16 --weight_decay 1.7e-1``

It's convenient to run the script several times with different seeds if you want to train several models at once.

### Region Masking
$\rightarrow$ `comparisons/mask_regions_eval.py`

For example, to evaluate the Schoffelen CNN when masking different regions:  
```python -m comparisons.mask_regions_eval --data_path /path/to/datasets/directory --dataset schoffelen --model SECNN --cache_dir /path/where/best/models/are/cached --n_params 250000 --data_space shared_source```

A `.pkl` file is saved in the `cache_dir`, with plots made by running `plot_regionmask_res.ipynb`. `pkl` files used for the paper's plots are in subdirectories in `comparisons`.

### Cross-dataset evaluation

$\rightarrow$ `comparisons/cross_dataset_eval.py`

For example, to evaluate the Armeni single subject MLP on the Schoffelen dataset:  
```python -m comparisons.cross_dataset_eval --dataset armeni --data_path /path/to/datasets/directory --cache_dir /armeni/models/cache/dir --model MLP```

### Convenience

- `DEF_DATA_PATH` in `consts.py` can be set to what you'd put in `--data_path`.

### Sanity

See scripts in the `sanity` folder for tests.

### Potential improvements
- Here data was saved as `.fif/.h5` files for convenience as the preprocessing pipeline was extensively ablated. Once it's fixed you can save their data in better formats and save a lot of memory.
