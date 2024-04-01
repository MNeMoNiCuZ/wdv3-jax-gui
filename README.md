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
- Specify the path to the folder containing your dataset.

## Caption Settings
### Prefix
- Add any prefix to your captions. For instance, when training a LoRA model, you might use an activation trigger like:

```
MyModel,
```


### Suffix
- Append any suffix to the end of each caption, similar to how you might use a prefix.

### Blocked Tags
- Provide a comma-separated list of tags to exclude from the output text files. 
- **Example**: 
  ```
  1girl, blonde_hair, black_hair, brunette
  ```
Tags listed will be omitted from the output.

## Configuration Options
### Multi-Threading
- The script defaults to multi-processing. Opting for multi-threading may offer marginal time savings. This setting is provided for those who prefer it.

### Recursive Search
- Activating this option enables the inclusion of sub-folders in the captioning process, useful for structured datasets.

### Overwrite Existing Files
- When enabled, the script overwrites existing text files; if disabled, it bypasses any existing text file.

### Save Captions with Image
- With this enabled, captions are saved in the same directory as their corresponding images. Can be used concurrently with the option to save captions in a subfolder.

### Save Captions in Subfolder
- Activating this saves captions in a subfolder named after the chosen model. It can work in tandem with saving captions alongside images.

## Select Model
- Choose one or multiple models for caption generation. 
- For a single model and using `Save captions with image`, output text files match the input image filenames.
- With multiple models and `Save captions with images`, output files include the model name, formatted as `ImageName.ModelName.txt`.

# Credits
Thanks to [borisignjatovic](https://github.com/borisignjatovic) for helping me out with the code!
