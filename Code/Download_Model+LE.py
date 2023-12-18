import joblib
import torch
from google.colab import files

# Save the LabelEncoder
label_encoder_path = "label_encoder.joblib"
joblib.dump(label_encoder, label_encoder_path)

# Save the trained Camembert model
model_path = "camembert_model.pth"
torch.save(model.state_dict(), model_path)

# Download the saved model and LabelEncoder
files.download(label_encoder_path)
files.download(model_path)
