from pathlib import Path
import jsonlines
from torch.utils.data import Dataset


def get_train_test_dataset(save_dir,*args, **kwargs):
    env_dir = Path(__file__).parent.parent
    if 'GPQA' in save_dir:
        test_ds = JsonlMathDataset(env_dir / "MATH/dataset/gpqa/test.jsonl",'GPQA') 
    elif 'MATH500' in save_dir:
        test_ds = JsonlMathDataset(env_dir / "MATH/dataset/test500.jsonl",'MATH500')
    elif 'AIME24' in save_dir:
        test_ds = JsonlMathDataset(env_dir / "MATH/dataset/AIME24/data/test.jsonl",'AIME24')
    return None, test_ds


class JsonlMathDataset(Dataset):
    def __init__(self, data_path, dataname):
        super().__init__()
        self.data = []
        self.dataname=dataname
        with jsonlines.open(data_path, "r") as reader:
            for obj in reader:
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if 'GPQA' in self.dataname:
            return {"question": x["question"], "answer": x["answer"]} # gpqa
        elif 'MATH500' in self.dataname:
            return {"question": x["problem"], "answer": x["solution"], "level": x["level"]}
        elif 'AIME24' in self.dataname:
            return {"question": x["problem"], "answer": x["answer"]} # aime24