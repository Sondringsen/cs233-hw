## Getting the Code
All necessary starter code is already made available in this codebase.

## Environment and Code Setup
All requirements are listed in `setup.py`. We will walk through how to install these requirements and set up the code below. The code has been tested with Python 3.6 (but any 3.x version should work), and in these instructions we assume you are using an Ubuntu or MacOS environment (though similar steps can be taken on Windows). Note that some instructions are only relevant if setting up in a GPU environment (as on Google Cloud).

Make sure you are in the top level directory of the codebase (with `setup.py` and this README). First, set up a [virtual environment](https://docs.python.org/3/library/venv.html) that will be used to install all dependencies by running the following commands:
```
sudo apt-get install python3-venv       # install python virtual env on your system
python3 -m venv ./hw4_env               # create an environment called hw4_env
source hw4_env/bin/activate             # activate that environment
pip install --upgrade pip               # upgrade pip
```

**[GPU Only]** If you are on a GPU system, we will install PyTorch with a suitable GPU version (using CUDA 11.3):
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

**[CPU Only]** If you are not using GPU, we will install PyTorch without CUDA:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

This codebase includes two Chamfer distance implementations, one efficient one that can only run on the GPU and one that is significantly slower but also runs on CPU. When developing your code locally (likely on CPU), you will need to use the slower one while it's recommended to use the faster one when using a GPU.

## Getting Started
Start your work at `notebooks/main.ipynb` (or `notebooks_as_python_scripts/main.py` if you are not a fan of notebooks).

If you want to use the notebooks, make sure jupyter is installed with `pip install jupyter` then you can open the notebooks using `jupyter notebook` from the root directory.

Best of luck!

-----