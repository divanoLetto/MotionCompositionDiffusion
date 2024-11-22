import os
import codecs as cs
import orjson  # loading faster than json
import json

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch 
from .collate import collate_tensor_with_padding, collate_text_motion


from src.stmc import TextInterval

def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class MTTDataset(Dataset):
    def __init__(
        self,
        name: str,
        motion_loader,
        text_encoder,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
        # only during training
        drop_motion_perc: float = 0.15,
        drop_cond: float = 0.10,
        drop_trans: float = 0.5,
    ):
        if tiny:
            split = split + "_tiny"

        name = "mtt_2"
        self.path = f"{name}/"
        self.collate_fn = collate_text_motion # non ho capito cosa fa
        self.split = split
        self.keyids = read_split(self.path, split)

        self.text_encoder = text_encoder
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        # self.annotations = load_annotations(path)

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        # if "test" not in split:
        #     self.annotations = self.filter_annotations(self.annotations)

        self.is_training = "train" in split
        self.drop_motion_perc = drop_motion_perc
        self.drop_cond = drop_cond
        self.drop_trans = drop_trans

        # self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.nfeats = self.motion_loader.nfeats
        self.fps = 20

        # if preload:
        #     for _ in tqdm(self, desc="Preloading the dataset"):
        #         continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        print(f"Index - {index}")
        if index == 62:
            pass
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        texts = []
        lengths_frame = []
        paths = [] 
        starts_sec, starts_frame = [], []
        ends_sec, ends_frame = [], []
        xs = []

        with open(f"{self.path}/MTT/{keyid}", "r") as f:
            for line in f:
                text = line.split("#")[0].strip()
                start_s, end_s = float(line.split("#")[1].strip()), float(line.split("#")[2].strip())
                start_f, end_f = int(start_s*self.fps), int(end_s*self.fps)
                filename = line.split("#")[3].strip().replace("\n","")
                __ = np.load(f"{self.path}/movements/{filename}.npy")

                texts.append(text)
                lengths_frame.append(end_f-start_f)
                paths.append(filename)
                starts_sec.append(start_s)
                starts_frame.append(start_f)
                ends_sec.append(end_s)
                ends_frame.append(end_f)

        tx_emb = self.text_encoder(texts)
        tx_emb_uncond = self.text_encoder(["" for _ in range(len(texts))])
        # Aggiungo stringhe vuote pari al numero di testi invece delle righe sopra 
        # perchè il metodo badabim_2 è fatto in modo che ci siano le transizioni tra azioni e quindi il doppio dii roba

        # tx_emb = self.text_encoder(texts + ["" for _ in range(len(texts))])
        # tx_emb_uncond = self.text_encoder(texts + ["" for _ in range(len(texts))])      

        tx_emb = {
            "x": torch.stack([e["x"] for e in tx_emb]),
            "length": torch.tensor([1 for _ in range(len(tx_emb))])#.to(c.device),
        }
        tx_emb_uncond = {
            "x": torch.stack([e["x"] for e in tx_emb_uncond]),
            "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))])#.to(c.device),
        }
        tx_emb_core = self.text_encoder([""])[0]
        tx_emb_core["length"] = torch.tensor([1])#.to(c.devic
        infos = {"tx_emb_core":tx_emb_core}
        
        infos["n_frames"] = int(max(ends_sec)*self.fps)
        infos["n_seq"] = 1
        
        # Problema, non posso passare alla GPU qualcosa che è un Interval e non un tensore o stringa
        # infos["all_intervals"] = [[TextInterval(text=texts[i], start=starts_sec[i]*20, end=ends_sec[i]*20, bodyparts="spine") for i in range(num_actions)]]
        infos["all_intervals"] = {}
        infos["all_intervals"]["starts"] = torch.tensor(starts_frame)
        infos["all_intervals"]["ends"] = torch.tensor(ends_frame)

        for i in range(len(paths)):
            motion_x_dict = self.motion_loader(
                path=paths[i],
                start=0, # non starts_sec[i], altrimenti croppa
                end=lengths_frame[i]/self.fps, # non ends_sec[i], altrimenti croppa
                drop_motion_perc=None,
                load_transition=False, # altrimenti croppa una segmento casuale dalla transizione
            )
            xs.append(motion_x_dict["x"])
            assert lengths_frame[i] == int(motion_x_dict["length"])

        infos["all_lengths"] = lengths_frame
        infos["n_uncond"] = [len(lengths_frame)]
        xs = collate_tensor_with_padding(xs).squeeze() 

        output = {
            "x": xs,
            "text": texts,
            "tx_emb": tx_emb,
            "tx_emb_uncond": tx_emb_uncond,
            "keyid": keyid,
            "infos":infos
        }
        return output

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            path = val["path"]

            # remove humanact12
            # buggy left/right + no SMPL
            if "humanact12" in path:
                continue

            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
