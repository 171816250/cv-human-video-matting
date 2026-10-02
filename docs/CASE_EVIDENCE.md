# 真实试用与教学案例证据

这个流程把一次 CLI 运行的原片、结果、参数和同帧截图连起来，供私下复核。收集工具只在本机运行，不下载素材、不重新推理，也不复制整段原片或结果视频。默认把截图和清单写入用户主目录下的 `cv-human-video-matting-case-evidence/`；仓库内的路径即使被 `.gitignore` 忽略也会被拒绝。截图本身可能识别人物，必须按私有素材管理。

## 准备三组场景

建议至少各准备一组已获授权的素材：

| `--scenario` | 建议内容 | 重点观察 |
| --- | --- | --- |
| `clear_single` | 光线充足、背景简洁的单人镜头 | 主体边缘和背景残留 |
| `complex_background` | 人物与背景颜色接近，或有细线、树叶等纹理 | 发丝、背景穿透、边缘污染 |
| `moving_person` | 人物快速转身、挥手或进出画面 | 闪烁、拖影、短暂漏抠 |

弱光、半透明衣物、发丝等困难点可以纳入上述场景，也可以用 `other` 单独记录。每组先保存授权凭据，确认其覆盖本次本地处理、截图保存和预定的教学/试用用途。`--authorization-record` 可以是文本、PDF 或其他已有文件；工具只检查它非空并记录哈希，**无法判断签名、授权人身份或使用范围是否有效**。`--authorized-use` 是操作者对已确认用途的文字记录，不会自动扩大授权。

## 生成运行报告

在项目根目录运行 CLI，并保存原片、实际结果和 JSON 报告。以下是普通视频的示例；实际试用应选择真实模型与适当参数：

```powershell
python app.py matte --input D:\private-media\person.mp4 --output D:\private-media\person-result.mp4 --model rvm_mobilenetv3 --device cpu --report D:\private-media\run.json
```

报告必须是 `schema_version=3`、`kind=run_report`。其中 `input` 和 `output` 的 SHA256 用于确认收集时使用的文件与运行时一致，`options` 记录参数，`result.frames` 约束可取的帧号。移动文件后仍可用 `--source` 和 `--result` 指向内容相同的副本。`--result` 要指向报告中的**实际输出**：普通模式是视频，`transparent` 模式是 `frame_000000.png` 等文件所在的 PNG 序列目录。

## 收集与复核

下面以零起始的第 0、50、100 帧为例。只选小于 `result.frames` 的帧，建议包含开头、中间和一个困难/失败位置；若总帧数较少，改用实际存在的序号。

```powershell
python scripts/collect_case_evidence.py collect --case-id lesson_01 --scenario moving_person --authorization-record D:\private-media\permission.pdf --authorized-use "private teaching review" --report D:\private-media\run.json --source D:\private-media\person.mp4 --result D:\private-media\person-result.mp4 --frame 0 --frame 50 --frame 100
```

可通过 `--private-dir D:\private-cases` 指定其他仓库外目录。输出是 `<private-dir>/<case-id>/manifest.json` 及逐帧的 `*_source.png`、`*_result.png`；透明 PNG 结果还会生成实际 alpha 通道的 `*_alpha.png`。工具从原片和结果的**相同零起始帧序号**顺序解码，再保存截图。清单记录授权文件、运行报告、原片和结果的哈希、运行参数以及每张截图的哈希；人工质量评价始终为 `null`。已有案例目录不会被覆盖；重复收集时使用新的私有目录，然后比较清单和截图。

```powershell
python scripts/collect_case_evidence.py verify --case-dir D:\private-cases\lesson_01
```

复核命令需要原片、结果、授权文件和报告仍可访问。它重新核对文件哈希，并逐像素比较每张截图与所标记帧的解码结果。哈希证明文件是否变化，不能证明授权真实、拍摄条件完整或报告未被有意伪造；同一帧序号也不能单独证明音画时间戳完全同步。

## 人工结论与公开材料

在私有案例记录中逐帧写明观察者、日期、设备/显示条件、选帧依据、成功与失败现象，并保留不利于结论的帧。发丝、边缘污染、拖影等应分别描述。别人说“效果还可以”只是反馈线索，不能据此生成评分、通过率或人工验收结论。没有真实 alpha 标注时也不能把视觉观察写成像素准确率；CLI 报告的 `quality_metrics: null` 表示未测量。

公开教学材料应另行筛选，只在授权范围内导出经过审查的截图和必要参数。此工具生成的整个案例目录、原片、授权文件以及含本地路径的 JSON 报告都应保持私有，不要直接提交仓库或上传公开发布包。
