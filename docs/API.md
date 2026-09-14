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
普通MP4输出会替换同名文件，暂不保留音轨。

## 深度与状态

`DepthEstimator.person_distance(depth, alpha)` 返回相对距离值和标签，兼容两种alpha表示；无足够有效人物像素时返回空距离。
网页摄像头通过 `gr.State` 为每个浏览器会话保存独立模型、时序和平面深度缓存。
深度默认关闭，开启后按帧间隔刷新，间隔越大算力开销越小，但标注更新会更滞后。

## 性能报告

`write_run_report(path, input_path, stats, options, total_seconds)` 保存JSON格式的运行条件与结果。计时口径详见EVALUATION.md。
