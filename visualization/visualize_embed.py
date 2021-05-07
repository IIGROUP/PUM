import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib
from ellipse import LsqEllipse
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import joblib
import os
import json
from config import cfg, OLD_DATA_PATH
from lib.pytorch_misc import sample_z

matplotlib.use('agg')


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Refer to https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def visualize_embed(prd2mus, prd2samples, prd_labels, save_path):
    fig, ax = plt.subplots()
    colors = cm.rainbow(np.linspace(0, 1, len(prd2mus)))
    for i, (mus, samples, c) in enumerate(zip(prd2mus, prd2samples, colors)):
        for j, (m, s) in enumerate(zip(mus, samples)):
            label = prd_labels[i] if j == 0 else None
            confidence_ellipse(s[:, 0], s[:, 1], ax, edgecolor=c)
            # ax.scatter(s[:, 0], s[:, 1], marker='.', color='yellow')
            ax.plot(m[0], m[1], marker='x', color=c, label=label)
    ax.legend()
    plt.xticks(())
    plt.yticks(())
    plt.savefig(save_path)
    plt.close()


def main():
    gaussians_list = joblib.load(cfg.gaussians_path)
    if cfg.prd_to_view is None:
        prd_to_view = ['on', 'has', 'wearing']
    else:
        prd_to_view = [x.replace('_', ' ') for x in cfg.prd_to_view]
    with open(os.path.join(OLD_DATA_PATH, 'vg/predicates.json')) as f:
        prd_label_list = ['__no_relation__'] + json.load(f)  # a list of labels
    prd_idx_to_view = [prd_label_list.index(x) for x in prd_to_view]
    num_example_per_prd = cfg.num_example_per_prd
    prd2mus = [[] for _ in range(len(prd_idx_to_view))]
    prd2samples = [[] for _ in range(len(prd_idx_to_view))]

    for entry in gaussians_list:
        mus, log_vars, preds = entry
        flag = 0
        for i, prd_idx in enumerate(prd_idx_to_view):
            if len(prd2mus[i]) >= num_example_per_prd:
                flag += 1
                continue
            target_pos = np.where(preds == prd_idx)[0]
            if len(target_pos):
                target_mus = mus[target_pos]
                target_log_vars = log_vars[target_pos]
                samples = sample_z(target_mus, target_log_vars, cfg.num_gaussian_samples)
                prd2mus[i].append(target_mus.numpy())
                prd2samples[i].append(samples.numpy())
        if flag == len(prd_idx_to_view):
            # Already collect enough data
            break

    data_to_reduce = []
    for i in range(len(prd_idx_to_view)):
        if len(prd2mus[i]):
            prd2mus[i] = np.concatenate(prd2mus[i])[:num_example_per_prd]
            prd2samples[i] = np.concatenate(prd2samples[i])[:num_example_per_prd*cfg.num_gaussian_samples]
            data_to_reduce.append(np.concatenate([prd2mus[i], prd2samples[i]]))
    data_to_reduce = np.concatenate(data_to_reduce)

    if cfg.reduce_method == 'pca':
        reducer = PCA(n_components=2)
    elif cfg.reduce_method == 'tsne':
        reducer = TSNE(n_components=2)
    else:
        raise ValueError('Unexpected reduce method')
    data_2d = reducer.fit_transform(data_to_reduce)
    prd2mus_2d = []
    prd2samples_2d = []
    start = 0
    for i in range(len(prd_idx_to_view)):
        end = start + len(prd2mus[i])
        prd2mus_2d.append(data_2d[start:end])
        start = end
        end = start + len(prd2samples[i])
        prd2samples_2d.append(data_2d[start:end])
        start = end
    assert end == len(data_2d)
    for i in range(len(prd2samples_2d)):
        num_dist = prd2mus_2d[i].shape[0]
        if num_dist:
            prd2samples_2d[i] = prd2samples_2d[i].reshape([num_dist, cfg.num_gaussian_samples, -1])

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embed_visu.jpg')
    visualize_embed(prd2mus_2d, prd2samples_2d, prd_to_view, save_path)
    print('Result saved at %s' % save_path)


if __name__ == '__main__':
    main()
