import torch
from pipelines.pipeline import Pipeline
from models.regressor import Regressor

def main():
    model = Regressor()
    state_dict = torch.load("/home/kangnam/project/Novelis/checkpoints/epoch=4-step=15000.ckpt")["state_dict"]
    pipeline = Pipeline(model=model, lr=1e-3)
    pipeline.load_state_dict(state_dict)
    
    to_save = pipeline.model
    torch.save(to_save.state_dict(), "./weights/regressor.pt")
    
    
if __name__ == "__main__":
    main()