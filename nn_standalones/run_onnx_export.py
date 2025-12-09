import onnxruntime as ort
mp = "D:\\forward_model.onxx"
session = ort.InferenceSession(mp)
import numpy as np
indat = np.arange(36).reshape(1, 36).astype(np.float32)
inn = session.get_inputs()[0].name
outn = session.get_outputs()[0].name
print(session.run([outn], {inn: indat}))