# SiteSight — Construction Safety AI

An interactive portfolio demo that runs a trained YOLO11 construction detector **in the visitor's browser**. Drop an image or video, or use the included construction samples. The page draws the model's boxes and reports possible PPE concerns. No upload server or Python service is needed for the published site.

**Live site:** [SiteSight on GitHub Pages](https://abdulrahmanakh.github.io/YOLO_SiteSight_Project/)

## Project files

| Path | Purpose |
| --- | --- |
| `notebooks/Construction_Safety_AI.ipynb` | Executed training and evaluation notebook |
| `models/best.pt` | Trained PyTorch checkpoint for further training or export |
| `site/models/best.onnx` | Browser inference model |
| `site/app.js` | Image preprocessing, ONNX inference, box decoding, NMS, video loop, and report |
| `site/assets/` | Credited raw image and construction video samples |
| `.github/workflows/deploy-pages.yml` | Publishes `site/` to GitHub Pages |

## How it works

The ONNX model expects RGB float pixels in a `1 × 3 × 640 × 640` tensor and returns `1 × 21 × 8400` candidate predictions. The page letterboxes each image or video frame, runs ONNX Runtime Web, applies confidence filtering and class-aware non-maximum suppression, and draws the kept boxes on a canvas. Uploaded files stay in the browser. Video output plays as annotated analyzed frames, and supported browsers offer a recorded WebM download after playback.

The source dataset contains **bounding-box annotations only**. This model performs object detection, not construction-specific instance segmentation.

## Evaluation and limits

The attached notebook trained for 40 epochs on 307 training images, validated on 57 images, and evaluated on a held-out 34-image test set. Overall test **mAP50 is 0.465** and **mAP50–95 is 0.301**. The test split contains few examples of several safety classes. For example, `no-mask` has only two instances and mAP50 of 0.014; `no-hardhat` has seven instances. These numbers do not support a claim that the model detects every site activity or violation.

Only explicit `no-hardhat`, `no-mask`, and `no-safety vest` detections are highlighted as **possible** PPE concerns. A missing positive PPE detection is not treated as a violation. All results need human review. Video counts are detections across analyzed frames, not unique people; frames may be skipped when the model runs slower than video playback.

## Run locally

The site is static. Serve the **repository root** locally and open `/site/`. The bundled helper gives the ONNX runtime module the correct MIME type:

```powershell
python scripts/serve.py
```

Then open `http://127.0.0.1:8000/site/`.

## Deploy

In the GitHub repository, set **Settings → Pages → Build and deployment → Source: GitHub Actions**. A push to `main` deploys the `site/` directory. The published URL is `https://abdulrahmanakh.github.io/YOLO_SiteSight_Project/`.

## Credits

See [ASSET_CREDITS.md](ASSET_CREDITS.md) for image, video, runtime, and dataset attribution.
