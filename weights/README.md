# Weights

Place model weights in this directory.

Do not commit model files to Git. Add download links and checksum notes here when a model is introduced.

Use the helper script to populate the local model caches:

```bash
python scripts/download_models.py
python scripts/download_models.py --models all
python scripts/download_models.py --models birefnet
```

`all` prepares the RVM MobileNetV3, RVM ResNet50, and MiDaS caches. BiRefNet is opt-in because it is larger.

## RVM

The MVP runner uses:

```python
torch.hub.load("PeterL1n/RobustVideoMatting:master", "mobilenetv3", pretrained=True, trust_repo=True)
```

This may download model code and weights into the local Torch cache on first use.

If a local checkpoint is used later, place it here:

```text
weights/rvm_mobilenetv3.pth
```

Then pass it through the CLI:

```bash
python app.py matte --model rvm_mobilenetv3 --checkpoint weights/rvm_mobilenetv3.pth --input data/input/smoke_demo.mp4 --output data/output/rvm_green.mp4 --mode green --max-frames 24
```

In the current environment the first RVM download has already populated the Torch cache:

```text
C:\Users\chen'xiao'de\.cache\torch\hub\PeterL1n_RobustVideoMatting_master
```
