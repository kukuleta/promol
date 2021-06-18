import torch
import numpy as np

def get_predictions(model, dataset, callback_fn=None):

    preds = []
    model.eval()

    with torch.no_grad():
        for idx, args in enumerate(dataset):
            model_preds = model(*args[:-1])

            if callback_fn:
              model_preds = callback_fn(model_preds)

            preds.append(model_preds.detach().cpu().numpy())

    return np.concatenate(preds)