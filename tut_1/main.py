"""
This is the driver script for tut1 which executes driver functions of the three
covered topics, namely, Python, NLTK, and Gensim.
"""

import sys
from py_tut import main as py_tut_driver
from nltk_tut import main as nltk_tut_driver
from gensim_tut import main as gensim_tut_driver


def main(raw_data_path):
    py_tut_driver(raw_data_path)
    nltk_tut_driver()
    gensim_tut_driver(raw_data_path)


if __name__ == '__main__':
    main(sys.argv[1])
