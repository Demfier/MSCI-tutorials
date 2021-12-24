# MSCI-tutorials
Code for tutorials conducted in MSCI 641 (Text Analytics) @ Uwaterloo

**Note**: Video recordings of the tutorial sessions are posted on Learn.

# Instructions to run the code
**General Instructions**: The command to run any tutorial's driver script follows a consistent format. From the root of this repository, run `python3 tut_[tut_number]/main.py [appropriate argument(s)]` where `tut_number` is an integer representing the tutorial number and appropriate arguments could be infered from the description below:

- [Tutorial 1](tut_1):
  - *Driver script*: `python3 tut_1/main.py [dataset_path]`. Here `dataset_path` is the directory where you have the `pos.txt` and `neg.txt` files (`data/raw` in my case). It will generate the training split as well generate labels.
  - *Inference script* : `python3 tut_1/inference.py [sample_text_file_path]`. Here `sample_text_file_path` is the path to a sample text file which contains one word per line. In this case, its value would be `data/raw/sample_gensim.txt`. This inference script will print top-20 words from words given in a sample text file.
- [Tutorial 2](tut_2):
  - *Driver script*: `python3 tut_2/main.py [dataset_path]`. Here `dataset_path` is the directory where you have the data split and labels files (`data/processed` in my case). This script trains a simple MNB classifier and saves the requrired files inside `tut_2/data/` folder.
  - *Inference script*: `pythond3 tut_2/inference.py [sample_text_file_path] [model_code]`. Here, `sample_text_file_path` is the path to a sample text file with one sentence per line and `model_code` is the type of MNB classifier we need to use (we only have one in our case). The values for the two command-line arguments in this case are `data/raw/sample.txt` and `mnb_uni`, respectively. This script predicts a label for all the sentences in the sample text file.
- [Tutorial 3](tut_3):
  - *Driver script*: `python3 tut_3/main.py [dataset_path]`. Here `dataset_path` is the directory where you have the files for autoencoder training
  - *Inference script*: `pythond3 tut_3/inference.py [test_text_dir] [model_code]`. Here, `test_text_dir` is the path to the directory containig the test file with one sentence per line and `model_code` is the type of model to load (we only have one in our case, i.e, 'ae'). The values for the two command-line arguments in this case are `data/processed/` and `ae`, respectively. This script tries to reconstruct all the sentences in the sample text file. Note that the first argument is a directory, not the absolute path of the text file in this case, as all the output files are being stored in the same directory. You can easily change it to comply with your needs.

# Additional instructions:
Kindly follow a directory structure similar to what has been used here for your course's Github repository as well. It will smoothen the evaluation process and it may help you reuse modules developed across different assignments more seamlessly in the future.

# Credits and feedback
These tutorials have been compiled by [Gaurav Sahu (PhD Candidate, UWaterloo)](https://github.com/demfier). If you have any suggestions, feel free to open an issue and consider :star:ing the repository if you find the code useful :smile:
