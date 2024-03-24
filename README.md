# A GUI for batching WD14 v3 image to caption inference
![image](https://github.com/MNeMoNiCuZ/wdv3-jax-gui/assets/60541708/77f39500-107d-4d77-ac42-f255258f97fd)

## Original documentation
Base code ~~shamelessly stolen~~ borrowed from https://github.com/neggles/wdv3-timm

The Models directory has been copied from https://github.com/SmilingWolf/JAX-CV  

## How to install

1. clone the repository and enter the directory:
```sh
git clone https://github.com/SmilingWolf/wdv3-jax.git
cd wd3-jax
```

2. Create a virtual environment and install the Python requirements.

If you're using Linux, you can use the provided script:
```sh
bash setup.sh
```

Or if you're on Windows (or just want to do it manually), you can do the following:
```sh
# Create virtual environment
python3.11 -m venv .venv
# Activate it
source .venv/bin/activate
# Upgrade pip/setuptools/wheel
python -m pip install -U pip setuptools wheel
# At this point, optionally you can install JAX manually (e.g. if you are using an nVidia GPU)
python -m pip install -U "jax[cpu]"
# Install requirements
python -m pip install -r requirements.txt
```
Consult https://github.com/google/jax?tab=readme-ov-file#installation for more infos on how to install JAX with GPU/TPU/ROCm/Metal support

# Running the GUI
Run `py wdv3_jax_gui.py`

It may also work to just double-click the script, depending on your setup.

# GUI Settings
## Input Folder
Enter the path to the folder with your dataset in it.

## Caption Settings
### Prefix
Enter any prefix you wish to add to the captions. If you're training a LoRA, you may want to add an activation trigger word like this:

`MyModel, `

### Suffix
Enter any suffix you want to add to the end of each caption. Similar to the prefix.

### Blocked Tags
Enter a comma-separated list of blocked tags here and they will be removed from the output text files.

Example: `1girl, blonde_hair, black_hair, brunette`

Any instances of those tags should be removed from the output now.

## Configuration
### Multi-Threading (instead of multi-processing)
The script will already run via multi-processing, so the time saved shouldn't be significant. It's here as an option if you want it.

### Recursive Search
Enable this if you want sub-folders of your input folder to also be captioned.

This is useful if your input folder has each concept sorted in its own folder.

### Overwrite existing files
If enabled, it will overwrite existing text-files. If disabled, it will skip any text-file that already exists.

### Save captions with image
If true, the captions will be saved in the same folder as the images. This can be used at the same time as the option below.

### Save captions in subfolder
If true, the captions will be saved in a subfolder based on the chosen model name. This can be used at the same time as the option above.

## Select Model
Simply choose which model or models you wish to use.

If you choose only one model and you use the `Save captions with image`-option, the output text-files will have the same name as the input image file.

If you choose multiple models and you use the `Save captions with images`-option, the output text-files will have the model-name as part of the caption name. Like this: `ImageName.ModelName.txt`.

# Credits
Thanks to [borisignjatovic](https://github.com/borisignjatovic) for helping me out with the code!
