from inference.predictor import LLMFingerprintPredictor

predictor = LLMFingerprintPredictor()

sample = "This article provides a deep explanation of neural network architectures and attention mechanisms in transformer models."
res = predictor.predict(sample)

print("====================================")
print("Predicted Model")
print(res["predicted_model"])
print("\nConfidence")
print(f"{res['confidence']:.2f} %")
print("------------------------------------")
for m, p in res["probabilities"].items():
    print(f"{m:<10} {p:.1f} %")
print("------------------------------------")
print(f"Inference Time : {res['processing_time']:.2f} sec")
print("====================================")
