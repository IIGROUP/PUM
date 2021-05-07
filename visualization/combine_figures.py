import cv2
import pickle
import os
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

pgf_with_latex = {
    "text.usetex": True,            # use LaTeX to write all text
    "pgf.rcfonts": False,           # Ignore Matplotlibrc
    "pgf.preamble": [
        r'\usepackage{color}'     # xcolor for colours
    ]
}
matplotlib.rcParams.update(pgf_with_latex)

matplotlib.use('pgf')

if __name__ == '__main__':
    file_dir = 'visualization/multiple_inferences_viz'
    file_names = ['11', '23', '78']
    nrows = len(file_names)
    fig = plt.figure(figsize=[10, 18], constrained_layout=True)  # the height should be 3-times of the single one
    fig_width, fig_height = fig.get_size_inches() * fig.dpi  # size in pixels
    print('figure size: %.2f x %.2f' % (fig_width, fig_height))
    for i, name in enumerate(file_names):
        ax = fig.add_subplot(nrows, 1, i+1)
        file_path = os.path.join(file_dir, 'hit-%s.pkl' % name)
        data = pickle.load(open(file_path, 'rb'))
        image = data['image']
        ax.imshow(image)
        ax.text(data['text_x'] * 0.8, data['text_y'], data['text'], fontsize=17)
        ax.axis('off')
    save_path = os.path.join(file_dir, 'MultipleInference.pdf')
    plt.savefig(save_path)
    print('Result figure saved at %s' % save_path)
