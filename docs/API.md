# 核心接口

## 模型接口

`MattingRunner.predict(frame_bgr) -> MattingResult` 接受 H×W×3、uint8 的BGR图像。
结果包含同尺寸的alpha（uint8 0—255或float 0—1）及可选的BGR前景图。
合成优先使用模型前景；不提供前景时使用原始帧。有记忆状态的模型实现 `reset()`，用于隔离不同视频。

## 单帧与视频处理

`validate_video_request(input_path, output_path, mode, ...)`共用预检在模型加载前检查输入、背景、输出和处理参数，返回`PreparedVideoRequest`。不写输出，不初始化模型；管线内部仍会重复验证。

`render_matting_frame(frame_bgr, runner, mode, ...)` 返回合成图像及后处理后的alpha。
`process_video(input_path, output_path, runner, mode, ...)` 负责读取、处理和写出，返回 `PipelineStats`。
`max_frames=None` 表示处理到末尾，其他值必须为正整数。输入输出不得指向同一个文件。
透明模式输出 `输出名_frames` 目录的BGRA PNG序列；已有帧时拒绝覆盖，需换一个输出名。
底层 `process_video` 只写无音轨中间视频，直接调用会替换同名文件。交付请使用CLI，或将管线输出写到临时路径后调用 `export_video_for_delivery(processed_path, output_path, source_path=原视频, preserve_audio=True)`。
交付接口输出H.264/yuv420p和可选AAC，通过临时文件原子替换，失败保留原结果；禁止覆盖显式传入的原视频。透明PNG不含声音。

视频接口新增 `resize_short=None`（原尺寸）、`depth_model`、`depth_near`、`depth_mid`、`background_color_bgr`。颜色同样适用于单帧接口，在green/side_by_side生效；`parse_background_color('#RRGGBB')` 返回BGR元组。

`process_video(..., temporal_mode='ema', temporal_motion_threshold=0.15)`控制平滑方法；直接API默认保留EMA语义，CLI和网页默认方法为adaptive，平滑开关仍为关闭。`TemporalAlphaSmoother(blend=.75, mode='adaptive', motion_threshold=.15)`按相邻原始alpha变化对历史权重进行四次幂减权。`.mkv`输出选择FFV1，其余路径保留原mp4v底层行为；CLI/UI使用FFV1后再交付转码。

已知源帧数时，解码不足`min(帧数, max_frames)`会报错；元数据未知时允许自然结束。未知输入总帧数在报告记为null，不进入严格的同条件评测组。输入帧率和宽高必须有效。

`process_video(..., progress_callback=None, should_cancel=None)` 的进度回调接收 `(processed_frames, total_frames)`，总数不可靠时为None。取消回调无参数，返回true时抛出 `utils.progress.ProcessingCancelled`。透明序列通过独立暂存目录生成，失败清理本任务文件；已有帧仍拒绝覆盖。`export_video_for_delivery(..., should_cancel=None)`同样支持取消，并回收编码子进程。

## 深度与状态

`DepthEstimator.person_distance(depth, alpha)` 返回相对距离值和标签，兼容两种alpha表示；无足够有效人物像素时返回空距离。
网页摄像头通过 `gr.State` 为每个浏览器会话保存独立模型、时序和平面深度缓存。
深度默认关闭，开启后按帧间隔刷新，间隔越大算力开销越小，但标注更新会更滞后。

## 性能报告

`write_run_report(path, input_path, stats, options, total_seconds, export_seconds=0, published_output_path=None)` 保存schema3运行记录。可选发布路径只改变报告中展示的输出路径，哈希和媒体检查仍读取stats.output_path暂存文件，供事务式发布使用；调用方必须在报告成功后发布对应产物。计时口径详见EVALUATION.md。

`evaluation.summarize_reports(paths)`返回分组统计，`write_evaluation_summary(paths, output_path)`保存自包含HTML或JSON。汇总只读取传入的运行报告，不访问其中引用的媒体路径，也不计算未测量的画质指标。

`check_environment(config_path)` 返回本地诊断，不下载模型。ready只表示未发现阻断项，不代表画质与速度验收通过。

## 离线模型与发布保护

`create_matting_runner(..., bundle_path=None)`与`RVMRunner(..., bundle_path=None)`接受RVM离线包，不能同时传checkpoint。`utils.model_bundle.prepare_model_bundle(output_path, model_name='mobilenetv3', hub_dir=None)`只复制现有缓存；`verify_model_bundle(path, expected_model=None)`校验全部文件并返回稳定内容身份。文件哈希不是上游签名。

`write_run_report(..., model_bundle_info=None)`可接收runner.bundle_info；报告前重新验证文件，若与模型加载时内容身份不同则拒绝保存。`validate_artifact_paths(report_path, input_path, output_path, asset_paths=())`防止输出或报告覆盖输入、权重、背景及离线包子目录。

CLI先准备所有结果，再发布媒体与报告；多路径发布对普通异常进行回滚，不承诺进程被强杀或断电时具有文件系统事务语义。网页继续整体重命名任务目录发布。低层管线直接调用的写入语义保持不变。
