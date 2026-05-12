import base64
import io
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, render_template, request
from PIL import Image


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.1, use_dropout=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))

        layers.extend(
            [
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        )
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class EfficientNetV2SUNet(nn.Module):
    """
    U-Net with a pretrained EfficientNetV2-S encoder.
    - Classification head branches from the bottleneck for damage type (6 classes)
    - Segmentation decoder predicts the defect mask
    """
    def __init__(self, in_channels=3, num_classes=6, dropout_rate=0.15):
        super().__init__()

        try:
            backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        except Exception as exc:
            print(f"Warning: pretrained EfficientNetV2-S weights unavailable ({exc}); using random init.")
            backbone = models.efficientnet_v2_s(weights=None)

        self.features = backbone.features

        for param in self.parameters():
            param.requires_grad = False

        self.bottleneck = DoubleConv(1280, 512, dropout_rate=0.2, use_dropout=True)

        self.avg_pool_cls = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.skip4 = nn.Conv2d(128, 128, kernel_size=1)
        self.skip3 = nn.Conv2d(64, 64, kernel_size=1)
        self.skip2 = nn.Conv2d(48, 64, kernel_size=1)
        self.skip1 = nn.Conv2d(24, 32, kernel_size=1)

        self.up4 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128 + 128, 128, dropout_rate, use_dropout=True)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(64 + 64, 64, dropout_rate, use_dropout=True)

        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64, dropout_rate, use_dropout=True)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32 + 32, 32, dropout_rate, use_dropout=True)

        self.up0 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

        for module in [self.bottleneck, self.class_head, self.skip4, self.skip3, self.skip2, self.skip1,
                       self.up4, self.up3, self.up2, self.up1, self.up0,
                       self.dec1, self.dec2, self.dec3, self.dec4, self.final_conv]:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x):
        feats = []
        out = x
        for layer in self.features:
            out = layer(out)
            feats.append(out)

        e0 = feats[0]
        e1 = feats[1]
        e2 = feats[2]
        e3 = feats[3]
        e4 = feats[4]
        e7 = feats[7]

        b = self.bottleneck(e7)

        class_logits = self.avg_pool_cls(b)
        class_logits = class_logits.view(class_logits.size(0), -1)
        class_logits = self.class_head(class_logits)

        d4 = self.up4(b)
        d4 = torch.cat([d4, self.skip4(e4)], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, self.skip3(e3)], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, self.skip2(e2)], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, self.skip1(e1)], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        seg_logits = self.final_conv(d0)

        return seg_logits, class_logits


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "Models", "efficientnetv2", "best_multitask_efficientnetv2s.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (256, 256)

# Class names for damage classification
CLASS_NAMES = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Free', 'MT_Uneven']

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model():
    model = EfficientNetV2SUNet(in_channels=3, num_classes=6, dropout_rate=0.15)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def pil_to_base64(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_overlay(original_pil, mask_arr):
    original_arr = np.array(original_pil).astype(np.uint8)
    overlay_arr = original_arr.copy()

    mask_bool = mask_arr > 0
    overlay_arr[mask_bool] = (
        0.65 * overlay_arr[mask_bool] + 0.35 * np.array([255, 35, 35], dtype=np.float32)
    ).astype(np.uint8)

    return Image.fromarray(overlay_arr)


app = Flask(__name__)

model = None
model_error = None
if not os.path.isfile(MODEL_PATH):
    model_error = f"Model file not found: {MODEL_PATH}"
else:
    try:
        model = load_model()
    except Exception as exc:
        model_error = f"Failed to load model: {exc}"


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "title": "Segmentation of Magnetic Tile Defects",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "model_error": model_error,
        "orig_b64": None,
        "mask_b64": None,
        "overlay_b64": None,
        "defect_pixels": None,
        "total_pixels": None,
        "defect_ratio": None,
        "predicted_class": None,
        "predicted_confidence": None,
        "error": None,
    }

    if request.method == "POST":
        if model is None:
            context["error"] = "Model is not loaded. Check model file and server logs."
            return render_template("index.html", **context)

        file = request.files.get("image")
        if file is None or file.filename.strip() == "":
            context["error"] = "Please upload an image file first."
            return render_template("index.html", **context)

        try:
            original = Image.open(file.stream).convert("RGB")
            orig_w, orig_h = original.size

            x = TRANSFORM(original).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                seg_logits, class_logits = model(x)
                probs = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
                class_probs = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
                predicted_class_idx = int(np.argmax(class_probs))
                predicted_class_name = CLASS_NAMES[predicted_class_idx]
                predicted_confidence = float(class_probs[predicted_class_idx]) * 100

            mask_small = (probs >= 0.5).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_small, mode="L").resize((orig_w, orig_h), Image.NEAREST)
            overlay_pil = build_overlay(original, np.array(mask_pil))

            mask_arr = np.array(mask_pil) > 0
            defect_pixels = int(mask_arr.sum())
            total_pixels = int(orig_w * orig_h)
            defect_ratio = (defect_pixels / max(total_pixels, 1)) * 100.0

            context["orig_b64"] = pil_to_base64(original)
            context["mask_b64"] = pil_to_base64(mask_pil)
            context["overlay_b64"] = pil_to_base64(overlay_pil)
            context["defect_pixels"] = defect_pixels
            context["total_pixels"] = total_pixels
            context["defect_ratio"] = round(defect_ratio, 2)
            context["predicted_class"] = predicted_class_name
            context["predicted_confidence"] = round(predicted_confidence, 2)
        except Exception as exc:
            context["error"] = f"Segmentation failed: {exc}"

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
