import numpy as np
import onnxruntime as ort
mp = "./results/models/forward_model.onnx"
session = ort.InferenceSession(mp)
indat = np.arange(36).reshape(1, 36).astype(np.float32)
inn = session.get_inputs()[0].name
outn = session.get_outputs()[0].name
print(session.run([outn], {inn: indat}))