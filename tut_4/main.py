import scipy
import skimage
import librosa
from librosa.display import specshow
import matplotlib.pyplot as plt


def generate_spectrograms(read_dest, write_dest):
    try:
        y, _ = librosa.load(read_dest)
        plt.figure(0)
        plt.axis('off')
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        s = librosa.feature.melspectrogram(y)
        specshow(librosa.power_to_db(s, ref=np.max), fmax=8000)
        plt.tight_layout()
        plt.savefig(write_dest, bbox_inches='tight')
        plt.close(0)
    except Exception as e:
        print(e)


def read_spectrogram_batch(spectrogram_paths):
    # remove alpha dimension and resize to 224x224
    return [skimage.transform.resize(s[:, :, :3], (224, 224, 3))
            for s in skimage.io.imread_collection(spectrogram_paths)]


def read_spectrogram(spectrogram_path):
    # remove alpha dimension and resize to 224x224
    return scipy.misc.imresize(
        skimage.io.imread(spectrogram_path)[:, :, :3], (224, 224, 3)
        )
