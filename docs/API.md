# 核心接口

## 模型接口

`MattingRunner.predict(frame_bgr) -> MattingResult` 接受 H×W×3、uint8 的BGR图像。
结果包含同尺寸的alpha（uint8 0—255或float 0—1）及可选的BGR前景图。
合成优先使用模型前景；不提供前景时使用原始帧。有记忆状态的模型实现 `reset()`，用于隔离不同视频。

## 单帧与视频处理

`render_matting_frame(frame_bgr, runner, mode, ...)` 返回合成图像及后处理后的alpha。
`process_video(input_path, output_path, runner, mode, ...)` 负责读取、处理和写出，返回 `PipelineStats`。
`max_frames=None` 表示处理到末尾，其他值必须为正整数。输入输出不得指向同一个文件。
透明模式输出 `输出名_frames` 目录的BGRA PNG序列；已有帧时拒绝覆盖，需换一个输出名。
底层 `process_video` 只写无音轨中间视频，直接调用会替换同名文件。交付请使用CLI，或将管线输出写到临时路径后调用 `export_video_for_delivery(processed_path, output_path, source_path=原视频, preserve_audio=True)`。
交付接口输出H.264/yuv420p和可选AAC，通过临时文件原子替换，失败保留原结果；禁止覆盖显式传入的原视频。透明PNG不含声音。

视频接口新增 `resize_short=None`（原尺寸）、`depth_model`、`depth_near`、`depth_mid`、`background_color_bgr`。颜色同样适用于单帧接口，在green/side_by_side生效；`parse_background_color('#RRGGBB')` 返回BGR元组。

## 深度与状态

`DepthEstimator.person_distance(depth, alpha)` 返回相对距离值和标签，兼容两种alpha表示；无足够有效人物像素时返回空距离。
网页摄像头通过 `gr.State` 为每个浏览器会话保存独立模型、时序和平面深度缓存。
深度默认关闭，开启后按帧间隔刷新，间隔越大算力开销越小，但标注更新会更滞后。

## 性能报告

`write_run_report(path, input_path, stats, options, total_seconds, export_seconds=0)` 保存JSON运行记录（schema_version=2）。计时口径详见EVALUATION.md。

`check_environment(config_path)` 返回本地诊断，不下载模型。ready只表示未发现阻断项，不代表画质与速度验收通过。
