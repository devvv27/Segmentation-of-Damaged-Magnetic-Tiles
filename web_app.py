import base64
import io
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
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


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None, dropout_rate=0.15):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.enc1 = DoubleConv(in_channels, features[0], dropout_rate, use_dropout=False)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(features[0], features[1], dropout_rate, use_dropout=False)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(features[1], features[2], dropout_rate, use_dropout=False)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(features[2], features[3], dropout_rate=0.15, use_dropout=False)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[3], features[3] * 2, dropout_rate=0.2, use_dropout=True)

        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[3] * 2, features[3], dropout_rate, use_dropout=True)

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[2] * 2, features[2], dropout_rate, use_dropout=True)

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[1] * 2, features[1], dropout_rate, use_dropout=True)

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[0] * 2, features[0], dropout_rate, use_dropout=True)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "best_unet_mt.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (256, 256)

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model():
    model = UNet(in_channels=3, out_channels=1, features=[32, 64, 128, 256], dropout_rate=0.15)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
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
                logits = model(x)
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

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
        except Exception as exc:
            context["error"] = f"Segmentation failed: {exc}"

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
