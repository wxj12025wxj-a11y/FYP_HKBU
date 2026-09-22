import traceback
import numpy as np
import torch
print("torch", torch.__version__)
import openunmix
print("openunmix", getattr(openunmix, "__version__", "?"))
print("openunmix path", openunmix.__file__)

from openunmix import predict as umx_predict
print("has separate:", hasattr(umx_predict, "separate"))

# 先看模型能不能下载
try:
    from openunmix import model as umx_model
    import inspect
    print(inspect.signature(umx_model.load_target_models) if hasattr(umx_model, "load_target_models") else "no load_target_models")
except Exception as e:
    print("model import err", e)

y = np.random.randn(2, 44100).astype(np.float32) * 0.01
try:
    out = umx_predict.separate(
        audio=torch.as_tensor(y), rate=44100, targets=["vocals"],
        model_str_or_path="umxhq", device="cpu")
    print("OK umxhq", type(out), np.shape(out))
except Exception:
    traceback.print_exc()
