import argparse
import os
import shutil
import subprocess
import typing as tp
import zipfile
from distutils.util import strtobool
from pathlib import Path
from urllib.request import urlretrieve

import gdown
import mne
from osfclient import OSF
from tqdm import tqdm


# much code was graciously taken from the brainmagick repo
# ensure you have freesurfer installed before running this!
# note the downloads can take a *very* long while, especially schoffelen. note you can remove the visual subjects as
# they aren't used here by default


def download_osf(
        study: str, dset_dir: tp.Union[str, Path], success="osf_download.txt"
):
    dset_dir = Path(dset_dir).resolve()
    assert dset_dir.parent.exists()

    success_file = dset_dir / success
    if success_file.exists():
        return
    print(f"Downloading {study} to {dset_dir.name}...")

    project = OSF().project(study)

    store = list(project.storages)
    assert len(store) == 1
    assert store[0].name == "osfstorage"

    pbar = tqdm()
    for source in store[0].files:
        path = source.path
        if path.startswith("/"):
            path = path[1:]

        file_ = dset_dir / path

        if file_.exists():
            continue

        pbar.set_description(file_.name)
        file_.parent.mkdir(parents=True, exist_ok=True)
        with file_.open("wb") as fb:
            source.write_to(fb)

    with success_file.open("w") as f:
        f.write("success")
    print("Done!")


def download_gwilliams(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)

    print("Downloading Gwilliams dataset...")
    download_osf('ag3kj', dataset_path, 'ag3kj')
    download_osf('h2tzn', dataset_path, 'h2tzn')
    download_osf('u5327', dataset_path, 'u5327')


def download_donders(study, dset_dir, parent="dccn", overwrite=False):
    dset_dir = Path(dset_dir).resolve()
    dset_dir.mkdir(exist_ok=True, parents=True)
    success = dset_dir / "download" / "success.txt"
    if not success.exists() or overwrite:
        print(f"Downloading {study} to {dset_dir}...")
        print("Please enter your Donders credentials. You can find them by going onto the website, "
              "eg. https://data.ru.nl/collections/di/dccn/DSC_3011020.09_236 for the schoeffelen dataset, and "
              "clicking on user->'Data access credentials'.")
        user = input("user:").strip()
        password = input("password:").strip()

        command = "wget -r -nH -np --cut-dirs=1"
        command += " --no-check-certificate -U Mozilla"
        command += f" --user={user} --password={password}"
        command += f" https://webdav.data.donders.ru.nl/{parent}/{study}/"
        command += f" -P {dset_dir}"
        command += ' -R "index.html*" -e robots=off'
        command += ' --show-progress'
        subprocess.run(command.split(), text=True, check=True)
        shutil.move(dset_dir / study, dset_dir / "download")
        print("Done!")
        with open(success, "w") as f:
            f.write("download success")


def download_schoffelen(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)

    # download the dataset from donders
    url = "https://data.donders.ru.nl/collections/di/dccn/DSC_3011020.09_236_v1"
    parent, study = url.split('/')[-2:]

    print("Downloading Schoffelen_2019 (Broderick2019) private files...")
    derivatives = Path(dataset_path) / 'derivatives'
    if not derivatives.exists():
        zip_derivatives = derivatives.parent / "derivatives.zip"
        if not zip_derivatives.exists():
            print("Downloading Broderick_2019 private files...")
            url = "https://ai.honu.io/papers/brainmagick/derivatives.zip"
            os.makedirs(zip_derivatives.parent, exist_ok=True)
            urlretrieve(url, zip_derivatives)
        print("Extracting Broderick_2019 private files...")
        with zipfile.ZipFile(str(zip_derivatives), "r") as zip:
            zip.extractall(derivatives.parent)

    download_donders(study, dataset_path, parent=parent)

    print("Note: you can delete the data for the visual subjects eg. sub-V* as they aren't used")

    # move data from DSC_.../schoffelen to /schoffelen
    shutil.move(dataset_path / 'DSC_3011020.09_236_v1' / 'schoffelen', dataset_path / 'schoffelen')


def download_armeni(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)

    # download the dataset from donders
    url = "https://data.ru.nl/collections/di/dccn/DSC_3011085.05_995"
    parent, study = url.split('/')[-2:]
    download_donders(study, dataset_path, parent=parent)


def download_fsaverage_data(dataset_path):
    # as only the freesurfer structure files are needed it's easiest to download some cached ones from gdrive
    # instead of installing fs from scratch each time
    path = Path(dataset_path).resolve()
    os.makedirs(path, exist_ok=True)

    # download the fsaverage files from gdrive
    drive_url = 'https://drive.google.com/uc?id=1uwj2YeOycEoAHC7e0bBUdMLx666Y4J5N'
    fsaverage_zip_path = path / 'fsaverage.zip'
    gdown.download(drive_url, str(fsaverage_zip_path), quiet=False)

    # Extract the ZIP file
    with zipfile.ZipFile(fsaverage_zip_path, 'r') as zf:
        zf.extractall(path)

    # Optionally, remove the downloaded ZIP file
    os.remove(fsaverage_zip_path)


def source_freesurfer_setup():
    # This will source the SetUpFreeSurfer.sh script and return the environment variables
    command = "bash -c 'source $FREESURFER_HOME/SetUpFreeSurfer.sh && env'"

    # Run the command in a shell and capture the output (environment variables)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    output, _ = proc.communicate()

    # Parse the output and set the environment variables in the Python process
    for line in output.decode('utf-8').splitlines():
        if '=' not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def freesurfer_source_recon(subjects_dir, n_workers: int = 8):
    subjects_dir = os.path.abspath(subjects_dir)
    os.environ['SUBJECTS_DIR'] = subjects_dir
    command = (f"cd {subjects_dir} && "
               r"ls */anat/*_T1w.nii | grep -v 'space-CTF' | "
               fr"parallel --jobs {n_workers} "
               r"'subj=$(echo {} | cut -d/ -f1); recon-all -s ${subj}_new -i {} -all'")

    try:
        # Run the command using subprocess
        process = subprocess.Popen(command, shell=True, executable='/bin/bash',
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Capture the output and error streams
        stdout, stderr = process.communicate()

        # Decode and print the output
        output = stdout.decode('utf-8')
        error = stderr.decode('utf-8')

        # Check the exit status of the process
        if process.returncode != 0:
            # Command failed, raise an error with the stderr message
            raise RuntimeError(f"Command failed with exit code {process.returncode}:\n{error}.\nOutput: {output}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # Re-raise the exception to propagate the error


def merge_mri_with_sensor_subj_folders(base_dir):
    # Change to the base directory
    os.chdir(base_dir)

    # Get all directories in the base folder
    directories = [d for d in os.listdir('.') if os.path.isdir(d)]

    # Filter for directories starting with 'sub-' and ending with '_new'
    new_dirs = [d for d in directories if d.startswith('sub-') and d.endswith('_new')]

    for new_dir in new_dirs:
        # Get the corresponding original directory name
        original_dir = new_dir[:-4]  # Remove '_new' from the end

        if os.path.exists(original_dir):
            # Copy all contents from new_dir to original_dir
            for item in os.listdir(new_dir):
                s = os.path.join(new_dir, item)
                d = os.path.join(original_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

            # Remove the new_dir
            shutil.rmtree(new_dir)
            print(f"Merged {new_dir} into {original_dir} and deleted {new_dir}")
        else:
            print(f"Warning: {original_dir} does not exist. Skipping {new_dir}")


def run_watershed_bem(subjects_dir):
    # Get all directories in the subjects_dir
    subject_folders = [d for d in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, d))]

    # Filter for directories starting with 'sub-'
    subject_folders = [d for d in subject_folders if d.startswith('sub-')]

    for subject in subject_folders:
        print(f"Processing subject: {subject}")
        try:
            mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True)
            print(f"Successfully processed {subject}")
        except Exception as e:
            print(f"Error processing {subject}: {str(e)}")


def preprocess_source_recon(subjects_dir, n_workers: int = 8):
    source_freesurfer_setup()

    freesurfer_source_recon(subjects_dir=subjects_dir, n_workers=n_workers)

    merge_mri_with_sensor_subj_folders(subjects_dir)

    run_watershed_bem(subjects_dir=subjects_dir)


if __name__ == '__main__':
    # get dataset and data_path as args from the user
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gwilliams')
    parser.add_argument('--data_path', type=str, default='datasets')
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--download', type=strtobool, nargs='?', const=True, default=True)
    args = parser.parse_args()

    print(f'Downloading fsaverage data to {args.data_path}')
    download_fsaverage_data(args.data_path)

    path = os.path.join(args.data_path, args.dataset)
    print(f'Downloading {args.dataset} dataset to {args.data_path}')
    if args.dataset == 'gwilliams':
        download_gwilliams(path) if args.download else None
    elif args.dataset == 'schoffelen':
        download_schoffelen(path) if args.download else None
        preprocess_source_recon(path, n_workers=args.n_workers)
    elif args.dataset == 'armeni':
        download_armeni(path) if args.download else None
        preprocess_source_recon(path, n_workers=args.n_workers)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
