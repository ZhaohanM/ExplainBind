import json
import sys
import os
import json
import sys
import os
import torch
import logging
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

sys.path.append("../")

LOGGER = logging.getLogger(__name__)


class DatabaseProcessor:
    """
    DatabaseProcessor
    -----------------
    Utility for loading and preparing datasets for
    protein–ligand interaction (PLI) prediction tasks.
    """

    def __init__(self, args):
        # Identify the dataset name based on the provided argument
        if args.dataset == "BindingDB":
            self.name = "BindingDB"
        elif args.dataset == "Biosnap":
            self.name = "Biosnap"
        elif args.dataset == "Human":
            self.name = "Human"
        elif args.dataset == "DUDE":
            self.name = "DUDE"
        elif args.dataset == "Lit_PCBA":
            self.name = "Lit_PCBA"
        elif args.dataset == "InteractBind":
            self.name = "InteractBind"
        else:
            raise ValueError(
                "Invalid dataset name provided. Please choose from "
                "'InteractBind', 'BindingDB', 'Human', or 'Biosnap'."
            )

        # Mapping from split identifier to directory name on disk
        # Note: keys here define the ONLY valid split options.
        split_map = {
            "S0": "random",
            "S1": "novel",
            "25": "25",
            "28": "28",
            "31": "31",
            "33": "33",
            "006": "006",
            "043": "043",
            "062": "062",
            "088": "088",
        }

        # Validate split argument against the actual supported keys
        if args.split not in split_map:
            raise ValueError(
                f"Invalid split '{args.split}'. Choose one of: {', '.join(split_map.keys())}."
            )

        split_dir = split_map[args.split]
        dataset_dir = os.path.join(args.data_path, args.dataset, split_dir)

        # Load CSV partitions
        self.train_dataset_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
        self.val_dataset_df = pd.read_csv(os.path.join(dataset_dir, "val.csv"))
        self.test_dataset_df = pd.read_csv(os.path.join(dataset_dir, "test.csv"))

        # Wrap into Dataset objects (named '*_data_loader' for historical compatibility)
        self.train_data_loader = PLI_Dataset(
            (
                self.train_dataset_df["fasta"].values,
                self.train_dataset_df["selfies"].values,
                self.train_dataset_df["att_site"].values,
                self.train_dataset_df["label"].values,
            )
        )
        self.val_data_loader = PLI_Dataset(
            (
                self.val_dataset_df["fasta"].values,
                self.val_dataset_df["selfies"].values,
                self.val_dataset_df["att_site"].values,
                self.val_dataset_df["label"].values,
            )
        )
        self.test_data_loader = PLI_Dataset(
            (
                self.test_dataset_df["fasta"].values,
                self.test_dataset_df["selfies"].values,
                self.test_dataset_df["att_site"].values, # Activity att_site target
                self.test_dataset_df["label"].values,
            )
        )

    def get_train_examples(self, test: int | bool = False):
        """
        Retrieve training examples.

        Parameters
        ----------
        test : int or bool
            If 1, returns a small subset for quick functional tests.
            If >1, returns exactly that many examples.
            Otherwise, returns the full training Dataset.
        """
        if test == 1:
            return (
                self.train_dataset_df["fasta"].values[:4096],
                self.train_dataset_df["selfies"].values[:4096],
                self.train_dataset_df["att_site"].values[:4096],
                self.train_dataset_df["label"].values[:4096],
            )
        elif isinstance(test, int) and test > 1:
            return (
                self.train_dataset_df["fasta"].values[:test],
                self.train_dataset_df["selfies"].values[:test],
                self.train_dataset_df["att_site"].values[:test],
                self.train_dataset_df["label"].values[:test],
            )
        else:
            return self.train_data_loader

    def get_val_examples(self, test: int | bool = False):
        """
        Retrieve validation examples.

        Parameters
        ----------
        test : int or bool
            If 1, returns a small subset for quick functional tests.
            If >1, returns exactly that many examples.
            Otherwise, returns the full validation Dataset.
        """
        if test == 1:
            return (
                self.val_dataset_df["fasta"].values[:1024],
                self.val_dataset_df["selfies"].values[:1024],
                self.val_dataset_df["att_site"].values[:1024],
                self.val_dataset_df["label"].values[:1024],
            )
        elif isinstance(test, int) and test > 1:
            return (
                self.val_dataset_df["fasta"].values[:test],
                self.val_dataset_df["selfies"].values[:test],
                self.val_dataset_df["att_site"].values[:test],
                self.val_dataset_df["label"].values[:test],
            )
        else:
            return self.val_data_loader

    def get_test_examples(self, test: int | bool = False):
        """
        Retrieve test examples.

        Parameters
        ----------
        test : int or bool
            If 1, returns a small subset for quick functional tests.
            If >1, returns exactly that many examples.
            Otherwise, returns the full test Dataset.
        """
        if test == 1:
            return (
                self.test_dataset_df["fasta"].values[:1024],
                self.test_dataset_df["selfies"].values[:1024],
                self.test_dataset_df["att_site"].values[:1024],
                self.test_dataset_df["label"].values[:1024],
            )
        elif isinstance(test, int) and test > 1:
            return (
                self.test_dataset_df["fasta"].values[:test],
                self.test_dataset_df["selfies"].values[:test],
                self.test_dataset_df["att_site"].values[:test],
                self.test_dataset_df["label"].values[:test],
            )
        else:
            return self.test_data_loader


class BatchFileDataset(Dataset):
    """
    BatchFileDataset
    ----------------
    A simple wrapper dataset for batch processing a list of file paths.
    """

    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        """Return the total number of files."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """Retrieve a single file path by index."""
        return self.file_list[idx]
