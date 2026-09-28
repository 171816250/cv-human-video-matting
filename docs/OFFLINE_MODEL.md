# RVM离线模型包与部署检查

离线模型包将**本机已有的RVM源码、对应预训练权重和原始LICENSE**复制到一个独立目录，并记录逐文件SHA256。复制成功后，可将整个目录搬到答辩电脑，在不访问全局torch hub缓存的情况下加载RVM。

它不包含Python、PyTorch、TorchVision、CUDA、显卡驱动或其他依赖，也不包含MiDaS和BiRefNet。仍需提前准备目标机器的运行环境和测试视频，并完成断网试运行。

## 从现有缓存导出

在项目根目录执行：

```powershell
python scripts/prepare_offline_model.py --model mobilenetv3 --output weights/offline/rvm-mobilenetv3
```

ResNet50使用 `--model resnet50`。脚本默认从 `torch.hub.get_dir()` 对应目录读取缓存，也可以明确指定已经存在的缓存：

```powershell
python scripts/prepare_offline_model.py --model mobilenetv3 --hub-dir D:/model-cache/torch/hub --output weights/offline/rvm-mobilenetv3
```

参数含义：

| 参数 | 说明 |
| --- | --- |
| `--output` | 新模型包目录；已有目录会报错，避免覆盖 |
| `--model` | `mobilenetv3`或`resnet50`，默认前者 |
| `--hub-dir` | 可选；现有torch hub缓存目录，不是项目权重目录 |

缓存中需要存在 `PeterL1n_RobustVideoMatting_master` 源码目录，以及 `checkpoints/rvm_mobilenetv3.pth` 或 `checkpoints/rvm_resnet50.pth`。缺失文件会明确报错，导出脚本不会联网补齐。若显式提供 `--hub-dir`，复制和校验本身只需Python标准库。

生成目录结构为：

```text
rvm-mobilenetv3/
├── manifest.json
├── code/
│   ├── hubconf.py
│   ├── LICENSE
│   └── model/...
└── weights/
    └── rvm_mobilenetv3.pth
```

源码中的 `.git`、`__pycache__` 和字节码缓存不纳入模型包。导出先在同级临时目录中完成复制与校验，成功后发布整个目录。清单记录的上游引用是可变的`master`分支，不冒充固定的上游提交；本次源码与权重以实际文件哈希标识。

## 配置和运行

在所用YAML配置中填写：

```yaml
matting:
  model: rvm_mobilenetv3
  checkpoint: null
  bundle: weights/offline/rvm-mobilenetv3
  downsample_ratio: auto

depth:
  enabled: false
```

模型名称必须与包中的模型一致。`matting.bundle`与自定义`matting.checkpoint`互斥。相对路径按运行命令时的当前工作目录解释，建议在项目根目录运行。

也可以用命令行明确选择：

```powershell
python app.py matte --model rvm_mobilenetv3 --model-bundle weights/offline/rvm-mobilenetv3 --input data/input/demo.mp4 --output data/output/offline-demo.mp4 --report data/output/offline-demo.json
```

加载前会校验清单、模型名和全部记录文件的大小与SHA256，拒绝缺失、额外或已变更的文件。随后使用本地源码目录构建模型，并加载包内权重。校验失败会报错，不回退到联网加载。

程序也可以单独校验而不执行模型代码：

```python
from cv_human_video_matting.utils.model_bundle import verify_model_bundle

identity = verify_model_bundle("weights/offline/rvm-mobilenetv3", expected_model="mobilenetv3")
print(identity["content_sha256"])
```

`content_sha256`是模型包内容身份，同一内容重新导出或搬移目录后保持一致；`manifest_sha256`标识具体清单文件，会受到清单生成时间等字段影响。

**哈希一致只能说明文件符合这份清单，不能证明代码可信。**模型包会执行其中的Python源码；应只使用项目团队已核实来源、检查并试运行过的包。源码包附带的清单不是上游签名，模型包验证也不等于模型推理成功或抠像质量达标。

## 在答辩电脑上检查

让配置中的`matting.bundle`指向搬移后的模型目录，然后执行：

```powershell
python app.py doctor --report data/output/doctor-offline.json
```

Doctor不会下载或初始化模型。它会检查配置、模型包哈希、依赖导入、安装版本约束、设备、FFmpeg的H.264/AAC编码器，以及OpenCV的FFV1/Matroska真实写读。

依赖版本默认按`requirements-demo.txt`及其引用的`requirements.txt`合并核对。导入成功但版本低于声明下限、超过上限、与固定版本不符或无法获取版本时，检查仍会失败，JSON中的`ready`为`false`。程序化调用`check_environment(requirements_path=...)`可以检查另一份明确的环境声明；这不会改变默认部署要求。

现有开发环境可能已经通过测试或真实模型试运行，同时仍使用低于声明的NumPy、OpenCV、Pillow或PyTorch版本。**这些运行记录不能证明该环境满足当前干净安装要求。**保留实际诊断结果，在独立环境中安装和验证发布时使用的依赖组合，不要据此把失败项改成通过。

模型包可以在缺少PyTorch时单独完成文件完整性校验，但完整Doctor仍会因运行依赖缺失而失败。开启距离标注时，Doctor会提示MiDaS不在RVM模型包中；应另外准备并验证MiDaS，或者在离线演示配置中关闭深度功能。

## FFV1中间视频与磁盘空间

视频处理先写入FFV1编码的Matroska中间文件，再导出H.264/AAC MP4。FFV1是无损编码，可避免先经过有损中间编码再导出带来的额外画质损失。代价是中间文件通常明显大于最终MP4，占用更多临时磁盘空间，也可能增加磁盘I/O时间。

Doctor只写读3帧、24×16的小文件，并确认解码像素与输入一致，结束后清理临时文件。这项检查验证编码器和容器能否实际工作，不验证长视频磁盘容量、持续吞吐或GPU性能。答辩前应在目标机器上用计划展示的分辨率和视频长度跑完整流程，确认可用空间、输出文件、音轨和运行报告。
