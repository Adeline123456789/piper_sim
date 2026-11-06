# 双臂 Piper 实采场景

工作区根目录为 `/media/midea/F496AC8096AC4548/piper_simulation`。本次以仓库的可移植 MuJoCo 工作站为底座，使用 Python 3.11.15、MuJoCo 3.3.7、Gymnasium 1.3.0 和 EGL。两个新增任务均以 30 Hz 工作。实采文件只读，两个场景在训练时只需本项目内的资源。

## 启动与采集

在项目根目录执行以下任一条命令。GUI 的 Control 面板可以分别驱动两臂。

```bash
bash scripts/run_mujoco.sh --task-scene configs/tasks/real_connector.yaml --interactive --enable_cameras
bash scripts/run_mujoco.sh --task-scene configs/tasks/real_straw.yaml --interactive --enable_cameras
```

无窗口采集举例。将文件名换成新的路径可再次采集。默认保持初始姿态，可用本地策略驱动 Gym 环境，或用上游的 HTTP 接口驱动采集。

```bash
source .venv-mujoco/bin/activate
bash scripts/run_mujoco.sh --task-scene configs/tasks/real_straw.yaml --task real_straw \
  --headless --enable_cameras --steps 30 --record outputs/my_straw.h5
python scripts/check_episode.py outputs/my_straw.h5
```

在一台新机器复现环境时，先运行 `bash scripts/install_mujoco.sh`，再运行 `uv pip install --python .venv-mujoco/bin/python -e '.[recorded-scenes]'`。视频对照和 LeRobot 导出另外使用 PATH 中的 ffmpeg/ffprobe。本工作区的虚拟环境通过符号链接使用本机 tote_insert 环境中的 FFmpeg 7.1.1。无需连接 CAN。

## 接入训练

```python
from piper_sim.recorded_env import RecordedPiperEnv

env = RecordedPiperEnv("real_connector", cameras=True, randomize=True, max_steps=900)
try:
    obs, info = env.reset(seed=42)
    # 以当前姿态为例。训练时用策略输出的 14 维绝对目标替换 target。
    target = obs["state"].astype("float64")
    obs, reward, terminated, truncated, info = env.step(target)
finally:
    env.close()
```

在无显示器环境中先设置 `export MUJOCO_GL=egl`。另一任务名为 `real_straw`。观测含三路 HWC uint8 RGB（`overhead`、`left_wrist`、`right_wrist`）及 14 维 `state`。动作顺序为左臂六关节弧度、左夹爪总开口米、右臂六关节弧度、右夹爪总开口米。动作是绝对目标。稀疏奖励在插入条件连续保持 0.30 秒后为 1，并终止该集。超时单独返回 truncated。默认使用标称布局，开启 randomize 后用 seed 复现工装或杯管的轻微摆放扰动。位置和阈值均可在两个任务 YAML 中调整。

## 数据与场景对应

| | 插接 | 吸管入杯 |
|---|---|---|
| 原始根目录 | `/media/midea/F496AC8096AC4548/act_td3/connector_plugin/` | `/media/midea/61500ece-df8d-46ce-be73-e04b2378079c/datasets/piper_insert_straw_lerobot0630/` |
| 本地清点 | 210 集、54,364 帧、630 个视频、420 个力传感 CSV | 102 集、54,566 帧、306 个视频 |
| 参考初态 | 20260908/000001，右臂工作，灰色工装内有黄色三脚插头及黑色充电器 | episode_000019，左右夹爪张开，杯和吸管分放于棕色桌垫 |
| 操作 | 右爪从灰板取黄插头，插到白色插排最右侧蓝色模块 | 左爪扶带孔盖的杯，右爪拾起蓝色吸管并穿过盖孔 |
| 场景几何 | 灰色工装为 170 × 60 × 33 mm，白色插座为 212 × 76 × 30 mm。前者沿用 CAD 孔位并补齐封闭壳体；后者从左到右有双 USB-A 和三组五孔交流插座，右端为铭牌 | 近似刚性吸管长 216 mm、直径 9 mm，杯高 112 mm，盖孔直径 13 mm。杯体和杯盖用分段凸碰撞保留内部空间 |
| 数据映射 | HDF5 左右臂分字段，角度转弧度，夹爪反馈按 95 mm 标称行程换算。示例回放把全程静止且指令恒定的臂保持在首帧反馈位置 | Parquet 的 `state/actions` 各 15 维。按照取值和抓取画面，将前 14 维解释为两组“六关节弧度 + 归一化夹爪”，以 95 mm 为暂定行程。第 15 维 state 全零，action 在每集末 8 帧为 1，回放时单独排除 |
| 相机对应 | cam_external_rgb / cam_left_wrist_rgb / cam_right_wrist_rgb | image / wrist_image_l / wrist_image_r |

吸管集含从已经抓持或对孔阶段开始的片段。当前 reset 复现有完整拾取过程的桌面初态，换用其他 episode 只更换机械臂记录。对于混合训练，需要先统一原始数据和仿真数据的字段、图像布局及单位。上游 LeRobot 导出器生成 connector 风格的角度字段，吸管集使用上述独立映射。

初始物体位置结合中央画面中的桌边、桌垫四角、灰板长度、腕部近景和抓取时关节反馈估算。两组视频各抽看 6 集的初、中、末段。桌垫纹理取自吸管第 19 集可见布面。采集背景模式用同场景静止左腕画面构成固定背景平面，现场人物、线缆和房间深度均有差异。中央相机在 CAD 位姿和标称裁剪内参基础上分别使用 -42 / -30 像素的纵向投影偏移匹配桌边。插接任务的右腕 RGB 光轴采用第 1 集第 0、145、330 帧的人工对应点估算方向，关节命令仍使用原 SI 接口。有效投影偏移仅用于本模拟场景。

## 插座实物照片修订（2026-11-05）

- 根据用户提供的两组长、宽、高，从编译后所有几何体的局部坐标极值复测，灰色工装外包为 **170 × 60 × 33 mm**，白色插座外包为 **212 × 76 × 30 mm**。白色按总高包含突出按键处理，面板高度取 26 mm，按键凸起取 4 mm；这两项拆分来自外观估算。夹爪、插头和孔隙尺寸仍需实测。
- 对照 `/home/midea/桌面/tu/plug/` 的 15 张照片。143357、143401、143406 用于白色插座的面板排布、开关、指示灯、铭牌、侧贴与端部结构；143553、143601 用于灰色侧壁、纹理和孔型。其余近景用于黄色插头的圆肩、扁握柄、顶部进线孔、侧螺钉和黑色适配器的 USB-A 凹槽、侧面铭文。几何尺寸与纹理来自不同依据，产品近照未用于重定现场摆放。
- 操作窗口默认聚焦桌面，可在 MuJoCo 的 **Rendering → Camera** 中选择 `white_strip_detail` 或 `gray_holder_detail` 检查局部。策略观测仍为工作站的三路 RGB。
- [白、灰插座实物与仿真近景](../outputs/recorded_scenes/connector_photo/实物与仿真.jpg) · [带两件插头的灰色工装](../outputs/recorded_scenes/connector_photo/fixture.jpg) · [实采与仿真三视角](../outputs/recorded_scenes/real_connector-compare.jpg)。

### 补充工位照片与插头装配

插接场景中，左右机械臂和各自的开槽底板按用户补充的尺寸分别向外平移 10 mm。以中央支架的纵向中心面为基准，两臂基座的世界坐标分别为 (0, +0.310, 0.036) m 和 (0, −0.310, 0.036) m。夹爪和腕相机随臂平移，桌面工装仍采用采集布局。配置项为 `arm_outward_offset_m`。

补充照片中的办公桌、显示器和地面与 9 月视频中的背景不同。插接配置默认使用 `workstation_background: photos` 展示新照片中的工位。复核 9 月视频时可设为 `recording`。桌边六只夹具、腕部引线、桌腿及办公区陈设按照片估算，仅用于视觉。底板上表面与基座安装面重合。两件插头在灰色工装内由重力和孔壁接触支撑。沿工装长轴从空矩形孔开始，依次为黄色三脚插头和黑色充电器。黄色外壳的螺钉朝空孔端，充电器顶部 USB-A 口朝上。

查看器内可选 `station_front` 检查左右安装位置，用 `assembly_side` 和 `assembly_end` 查看装配。下图的全景使用近似照片的双臂折叠姿态，训练重置使用数据集首帧姿态。对照图中两件插头与灰工装的配合来自模拟器的实际几何。[工位与装配对照图](../outputs/recorded_scenes/connector_photo/工位与装配对照.jpg)

两段 11 月 5 日的现场环拍补充了右侧办公桌的朝向、显示器支座、后桌小件、臂上线束和桌下供电走线。可选 `workbench_walkthrough` 从视频近似视角查看。两段视频均为空桌，任务工件继续使用 9 月实采首帧布局。[视频与仿真同角度近似对照](../outputs/recorded_scenes/connector_photo/视频工位对照.jpg)

按视频交叉核对了桌缝和支座。右侧红色垫座屏和联想屏同桌；Piper 对桌放 Acer，紫色键盘在屏幕后方。左侧近桌可看到银色薄屏和黑屏，更远处是华为屏所在桌。每组显示器从属于相应桌体。办公桌尺寸、屏幕距离及朝向按视频估算。[三个视角的桌面核对](../outputs/recorded_scenes/connector_photo/办公桌分组核对.jpg) · [仿真俯视](../outputs/recorded_scenes/connector_photo/桌面分布俯视.jpg)

## 验证与真机迁移

激活环境后，可在本机复查参考帧、真实动作的仿真回放及孔洞测试。

```bash
export MUJOCO_GL=egl
python scripts/compare_recorded_scene.py real_connector --output outputs/connector_pair.jpg
python scripts/compare_recorded_scene.py real_straw --output outputs/straw_pair.jpg
python scripts/replay_recorded_scene.py real_connector --episode 1
python scripts/replay_recorded_scene.py real_straw --episode 19
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_recorded_scenes.py
```

- 两个场景分别完成 30 步三路相机 HDF5 采集与校验，Gymnasium 的 reset/step 环境检查通过。工作区回归测试 72 项通过，5 项依赖 USD 或生成的 CAD/URDF 资源而跳过。
- 在有重力的仿真中，初始任务物体静置 2 秒的位置漂移小于 1 mm。部件从正确孔位上方释放可进入并触发成功；横向偏置 12 mm 释放会接触工件，两项均未误报成功。这组测试覆盖接触与判定，机械臂全程抓取另用真实动作回放检查。
- 对插接第 75 集的第 140、180 帧，用反馈关节位置投影三枚交流按键和右端铭牌中心。在相同新版工件几何下，双基座外移后，原始光轴和经方向修正的光轴分别与 8 个人工对应点平均相差 74.8 和 26.3 像素（640×480）。该诊断同时包含工装位置、按键尺寸、采集同步和相机模型误差，逐点结果见 `outputs/recorded_scenes/connector-wrist-residual.json`。
- 插接第 1 集回放 446 帧，黄插头从工装上脱落，未送入白色插座。吸管第 19 集回放 485 帧，左爪移动了杯子，右爪未稳定抓起吸管。两段的成功帧数都为零。三路对照图、回放结果及 HDF5 位于 `outputs/recorded_scenes/`。
- 这两个环境可用于仿真训练。上真机前的优先测量项为三台相机 640×480 的设备内参及腕部手眼外参、夹爪实际行程与指尖接触面、工装孔位和杯盖净间隙。接触摩擦、抓持力、控制延迟仍是迁移误差来源。插接数据的六维力 CSV 已清点，便于后续辨识，当前训练接口不输出与实机标定一致的六维力。应先在相同起始位姿复核拾取与插入轨迹，再做低速单步真机验证。本次未训练策略、未连接或驱动真机。
