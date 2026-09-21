"""
data_loader.py
==============
FakeNewsNet Dataset Loader with graph construction,
tokenisation, and controlled synthetic demo generation.

Group 31-11 | SOA University | FRP-2026
"""

import os
import random
import logging
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GraphData
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================
# SEED
# ==========================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# CSV LOADER
# ==========================================================

def load_raw_csv(
        real_path,
        fake_path,
        dataset_name="politifact"
):

    if not os.path.exists(real_path) or not os.path.exists(fake_path):

        logger.warning(
            f"[DataLoader] CSV not found for {dataset_name}. "
            f"Using synthetic demo data."
        )

        return _generate_demo_data()

    df_real = pd.read_csv(real_path)
    df_fake = pd.read_csv(fake_path)

    df_real["label"] = 0
    df_fake["label"] = 1

    df = pd.concat(
        [df_real, df_fake],
        ignore_index=True
    )

    df = df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    logger.info(
        f"[DataLoader] Loaded {len(df)} samples."
    )

    return df


# ==========================================================
# SYNTHETIC DATA
# ==========================================================

def _generate_demo_data(n=800):

    set_seed(SEED)

    real_headlines = [

        "Government announces infrastructure bill",

        "Scientists discover new medical treatment",

        "Tech company releases AI platform",

        "Climate summit signs carbon agreement",

        "Local election sees record turnout"

    ]

    fake_headlines = [

        "SECRET cure doctors hide from public",

        "Celebrity exposed in shocking scandal",

        "Government hiding alien evidence",

        "Miracle diet pill melts fat instantly",

        "Economy collapsing next month experts warn"

    ]

    neutral_words = [
        "official",
        "report",
        "analysis",
        "study",
        "confirmed",
        "policy",
        "evidence",
        "research"
    ]

    sensational_words = [
        "SHOCKING",
        "EXPOSED",
        "MUST SHARE",
        "WAKE UP",
        "HIDDEN TRUTH",
        "UNBELIEVABLE"
    ]

    texts = []
    labels = []

    for _ in range(n // 2):

        base = random.choice(real_headlines)

        text = base + " " + " ".join(
            random.choices(
                neutral_words,
                k=15
            )
        )

        label = 0

        # 10% noise
        if random.random() < 0.10:
            label = 1

        texts.append(text)
        labels.append(label)

    for _ in range(n // 2):

        base = random.choice(fake_headlines)

        text = base + " " + " ".join(
            random.choices(
                sensational_words,
                k=15
            )
        )

        label = 1

        # 10% noise
        if random.random() < 0.10:
            label = 0

        texts.append(text)
        labels.append(label)

    df = pd.DataFrame({

        "title": texts,
        "label": labels

    })

    df = df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    logger.info(
        f"[DataLoader] Generated {len(df)} synthetic samples."
    )

    return df


# ==========================================================
# GRAPH CREATION
# ==========================================================

def build_propagation_graph(
        num_users,
        news_node_feat
):

    feat_dim = news_node_feat.shape[-1]

    user_feats = torch.randn(
        num_users,
        feat_dim
    ) * 0.05

    x = torch.cat(

        [
            news_node_feat.unsqueeze(0),
            user_feats
        ],

        dim=0
    )

    edge_src = [0]
    edge_dst = [0]

    for u in range(1, num_users + 1):

        edge_src += [0, u]
        edge_dst += [u, 0]

    edge_index = torch.tensor(
        [edge_src, edge_dst],
        dtype=torch.long
    )

    return GraphData(
        x=x,
        edge_index=edge_index
    )


# ==========================================================
# DATASET
# ==========================================================

class FakeNewsDataset(Dataset):

    def __init__(
            self,
            df,
            tokenizer,
            max_length=MAX_SEQ_LENGTH,
            num_users_range=(5,15)
    ):

        self.df = df.reset_index(drop=True)

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.num_users_range = num_users_range

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = str(
            row.get("title","")
        )

        label = int(row["label"])

        enc = self.tokenizer(

            text,

            max_length=self.max_length,

            padding="max_length",

            truncation=True,

            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)

        attention_mask = enc[
            "attention_mask"
        ].squeeze(0)

        num_users = random.randint(
            *self.num_users_range
        )

        return {

            "input_ids": input_ids,

            "attention_mask": attention_mask,

            "num_users": torch.tensor(
                num_users,
                dtype=torch.long
            ),

            "label": torch.tensor(
                label,
                dtype=torch.long
            )
        }


# ==========================================================
# DATAMODULE
# ==========================================================

class FakeNewsDataModule:

    def __init__(
            self,
            df,
            model_name=BERT_MODEL_NAME,
            batch_size=BATCH_SIZE
    ):

        self.df = df

        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self._split()

    def _split(self):

        train_df, temp_df = train_test_split(

            self.df,

            test_size=(VAL_SPLIT+TEST_SPLIT),

            random_state=SEED,

            stratify=self.df["label"]
        )

        val_df, test_df = train_test_split(

            temp_df,

            test_size=TEST_SPLIT/(VAL_SPLIT+TEST_SPLIT),

            random_state=SEED,

            stratify=temp_df["label"]
        )

        self.train_dataset = FakeNewsDataset(
            train_df,
            self.tokenizer
        )

        self.val_dataset = FakeNewsDataset(
            val_df,
            self.tokenizer
        )

        self.test_dataset = FakeNewsDataset(
            test_df,
            self.tokenizer
        )

        logger.info(

            f"[DataModule] "
            f"Train={len(train_df)} "
            f"Val={len(val_df)} "
            f"Test={len(test_df)}"
        )

    def train_loader(self):

        return DataLoader(

            self.train_dataset,

            batch_size=self.batch_size,

            shuffle=True
        )

    def val_loader(self):

        return DataLoader(

            self.val_dataset,

            batch_size=self.batch_size,

            shuffle=False
        )

    def test_loader(self):

        return DataLoader(

            self.test_dataset,

            batch_size=self.batch_size,

            shuffle=False
        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    set_seed()

    df = load_raw_csv(

        POLITIFACT_REAL_PATH,

        POLITIFACT_FAKE_PATH
    )

    dm = FakeNewsDataModule(df)

    batch = next(
        iter(dm.train_loader())
    )

    print(
        batch["input_ids"].shape
    )