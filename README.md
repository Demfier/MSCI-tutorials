# MSCI-tutorials
Code for tutorials conducted in MSCI 641 (Text Analytics) @ Uwaterloo

**Note**: Video recordings of the tutorial sessions are posted on Learn.

# Instructions to run the code
**General Instructions**: The command to run any tutorial's driver script follows a consistent format. From the root of this repository, run `python3 tut_[tut_number]/main.py [appropriate argument(s)]` where `tut_number` is an integer representing the tutorial number and appropriate arguments could be infered from the description below:

- [Tutorial 1](tut_1): `python3 tut_1/main.py [dataset_path]`. Here `dataset_path` is the directory where you have the `pos.txt` and `neg.txt` files (`data/raw` in my case). It will generate the training split as well generate labels.
- [Tutorial 2](tut_2): `python3 tut_2/main.py [dataset_path]`. Here `dataset_path` is the directory where you have the data split and labels files (`data/processed` in my case)

# Additional instructions:
Kindly follow a directory structure similar to what has been used here for your course's Github repository as well. It will smoothen the evaluation process and it may help you reuse modules developed across different assignments more seamlessly in the future.

# Credits and feedback
These tutorials have been compiled by [Gaurav Sahu (MMath, UWaterloo)](github.com/demfier). If you have any suggestions, feel free to open an issue and consider :star:ing the repository if you find the code useful :smile:
