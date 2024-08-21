# from models.yolo import Model
# import torch

# # Load the model
# model_path = r"E:\TRAFFIC LIGHT MANAGEMENT\Object_Detection\yolov5\runs\train\exp2\weights\best.pt"
# model = torch.load(model_path)
# model.eval()

# # Load your test image
# from PIL import Image
# from torchvision import transforms

# transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
# for i in range(1, 500, 10):
#     image = Image.open(
#         r"E:\TRAFFIC LIGHT MANAGEMENT\Object_Detection\data\images\test\image (i).jpg"
#     )
#     image = transform(image).unsqueeze(0)  # Add batch dimension

#     # Run inference
#     with torch.no_grad():
#         predictions = model(image)

#     # Process predictions
#     print(predictions)


import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Path to your saved model checkpoint
model_path = r"E:\TRAFFIC LIGHT MANAGEMENT\Object_Detection\yolov5\runs\train\exp2\weights\best.pt"

# Load the model
model = torch.load(model_path)["model"].float()  # 'model' key contains the model
model.eval()


# Load and preprocess your test image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((640, 640)),  # Resize to match YOLOv5 input size
            transforms.ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0)  # Add batch dimension


# Path to your test image
image_path = (
    r"E:\TRAFFIC LIGHT MANAGEMENT\Object_Detection\data\images\test\image (1).jpg"
)
image = preprocess_image(image_path)

# Run inference
with torch.no_grad():
    predictions = model(image)

# Print or process the predictions
print(predictions)
