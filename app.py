import gradio as gr
import spaces
import torch
import torch.nn as nn
from torchvision import transforms

# Define the same CNN architecture used during training
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Load the trained model on CPU initially
checkpoint = torch.load(
    "plant_disease_cnn.pth",
    map_location="cpu",
    weights_only=False
)

class_mapping = {
    int(idx): name
    for idx, name in checkpoint["class_mapping"].items()
}

num_classes = len(class_mapping)
class_names = [class_mapping[i] for i in range(num_classes)]
image_size = checkpoint.get("image_size", 224)

model = PlantCNN(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


@spaces.GPU(duration=30)
def predict_disease(image):
    if image is None:
        return {"Please upload a plant leaf image": 1.0}

    image = image.convert("RGB")
    image_tensor = test_transform(image).unsqueeze(0)

    model.to("cuda")
    try:
        image_tensor = image_tensor.to("cuda")

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

            top_probs, top_indices = torch.topk(
                probabilities,
                k=min(3, num_classes)
            )

            results = {
                class_names[idx.item()]: float(prob.item())
                for prob, idx in zip(top_probs, top_indices)
            }

        return results

    finally:
        model.to("cpu")


demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(
        type="pil",
        label="Upload a Plant Leaf Image"
    ),
    outputs=gr.Label(
        num_top_classes=3,
        label="Predicted Disease"
    ),
    title="Plant Disease Classification",
    description=(
        "Upload a plant leaf image to get the top 3 "
        "predictions and their confidence scores."
    )
)

if __name__ == "__main__":
    demo.launch()