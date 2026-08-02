import torch
from matplotlib.figure import Figure

from pathml.training.calibration import reliability_bins
from pathml.training.metrics import metrics_at_threshold, precision_recall_curve
from pathml.training.plots import plot_confusion_matrix, plot_pr_curve, plot_reliability_diagram


def test_plot_confusion_matrix_runs_and_returns_figure():
    scores = torch.tensor([0.1, 0.4, 0.6, 0.9])
    labels = torch.tensor([0, 1, 0, 1])
    m = metrics_at_threshold(scores, labels, threshold=0.5)

    fig = plot_confusion_matrix(m)

    assert isinstance(fig, Figure)


def test_plot_pr_curve_runs_and_returns_figure():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    _, precision, recall = precision_recall_curve(scores, labels)

    fig = plot_pr_curve(precision, recall)

    assert isinstance(fig, Figure)


def test_plot_reliability_diagram_runs_and_returns_figure():
    torch.manual_seed(0)
    scores = torch.rand(50)
    labels = torch.randint(0, 2, (50,))
    bins = reliability_bins(scores, labels, n_bins=5)

    fig = plot_reliability_diagram(bins, title="test")

    assert isinstance(fig, Figure)
