from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import csv

class CVSSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_cvss_txt(split_dir, list_classes):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["LOW", "HIGH"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            for i in range(len(list_classes)):
                if list_classes[i] == label_dir:
                    labels.append(i)
                else:
                    continue

    return texts, labels

def read_cvss_csv(file_name, num_label, list_classes):
    texts  = []
    labels = []
    with open(file_name, "r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",", quotechar='"')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue  # skip header
            texts.append(row[1])  # <-- description column
            # map label string to index
            label_str = row[num_label]
            try:
                labels.append(list_classes.index(label_str))
            except ValueError:
                # If an unexpected label slips in, skip it (or raise)
                continue
    return texts, labels


def read_cvss_csv_with_ids(file_name, num_label, list_classes):
    """
    Same as read_cvss_csv, but also returns CVE IDs.
    Returns: (ids, texts, labels)
    """
    ids    = []
    texts  = []
    labels = []
    with open(file_name, "r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",", quotechar='"')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue  # skip header
            cve_id = row[0]           # CVE ID column
            desc   = row[1]           # description column
            label_str = row[num_label]
            try:
                label_idx = list_classes.index(label_str)
            except ValueError:
                continue  # skip unknown labels

            ids.append(cve_id)
            texts.append(desc)
            labels.append(label_idx)
    return ids, texts, labels



def read_cvss_data(data, num_label, list_classes):
    texts      = []
    labels     = []

    for row in data.itertuples(): #row[0] is the index
        texts.append(row[2]) # description

        for i in range(len(list_classes)):
            if list_classes[i] == row[num_label+1]: #offset by 1 because of the index
                labels.append(i)
            else:
                continue
    
    return texts, labels