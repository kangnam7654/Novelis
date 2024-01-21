import argparse
import torch
from pipelines.pipeline import Pipeline
from models.regressor import Regressor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()
    return args


def main(args):
    model = Regressor()
    state_dict = torch.load(args.ckpt)["state_dict"]
    pipeline = Pipeline(model=model, lr=1e-3)
    pipeline.load_state_dict(state_dict)

    to_save = pipeline.model
    torch.save(to_save.state_dict(), args.out_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
