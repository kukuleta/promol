import torch
import numpy as np

def get_predictions(model, dataset, callback_fn=None):

    preds = []
    model.eval()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with torch.no_grad():
        for idx, args in enumerate(dataset):
            model_preds = model(*[x.to(dev) for x in args[:-1]])

            if callback_fn:
              model_preds = callback_fn(model_preds)

            preds.append(model_preds.detach().cpu().numpy())

    return np.concatenate(preds)