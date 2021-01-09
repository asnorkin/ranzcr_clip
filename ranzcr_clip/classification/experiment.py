import os.path as osp

import numpy as np


class Experiment:
    PREFIX = '_experiment'

    def __init__(self, args, num_items, num_folds=5, num_classes=11):
        self.args = args
        self.num_items = num_items
        self.num_folds = num_folds
        self.num_classes = num_classes

        # State
        self.folds = -1 * np.ones(num_items)
        self.labels = -1 * np.ones((num_items, num_classes))
        self.probabilities = -1 * np.ones((num_items, num_classes))
        self.folds_auc = -1 * np.ones(num_folds)

        self.load_state()

    def load_state(self):
        folds = self._load_file('folds.npy')
        if folds is not None:
            self.folds = folds

        labels = self._load_file('labels.npy')
        if labels is not None:
            self.labels = labels

        probabilities = self._load_file('probabilities.npy')
        if probabilities is not None:
            self.probabilities = probabilities

        folds_auc = self._load_file('folds_auc.npy')
        if folds_auc is not None:
            self.folds_auc = folds_auc

    def save_state(self):
        self._save_file('folds.npy', self.folds)
        self._save_file('labels.npy', self.labels)
        self._save_file('probabilities.npy', self.probabilities)
        self._save_file('folds_auc.npy', self.folds_auc)

    def update_state(self, trainer, fold):
        model = trainer.model
        if hasattr(model, 'module'):  # DP, DDP
            model = model.module

        fold_indices = model.test_indices.cpu().numpy()

        self.folds[fold_indices] = fold
        self.labels[fold_indices] = model.test_labels.cpu().numpy()
        self.probabilities[fold_indices] = model.test_probabilities.cpu().numpy()
        self.folds_auc[fold] = model.test_roc_auc.cpu().numpy()

    def _load_file(self, filename):
        file = osp.join(self.args.checkpoints_dir, self.PREFIX + filename)
        if not osp.exists(file):
            return None

        return np.load(file)

    def _save_file(self, filename, array):
        file = osp.join(self.args.checkpoints_dir, self.PREFIX + filename)
        np.save(file, array)
