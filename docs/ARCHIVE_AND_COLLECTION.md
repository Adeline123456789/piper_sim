# 数据恢复、运行与采集

先读[失败结论](FAILURE_REPORT.md)。本分支是失败任务的交接版本，不是已经完成标定的真机部署版本。下面的采集入口按现有代码说明，归档期间没有再次执行硬件。

## 1. 代码与完整数据

```bash
git clone --branch codex/failed-sim-real-archive-20261111 https://github.com/Adeline123456789/piper_sim.git
cd piper_sim
mkdir -p downloads restored
gh release download failed-sim-real-20261111 --repo Adeline123456789/piper_sim --dir downloads
(cd downloads && sha256sum -c SHA256SUMS)
cat downloads/sim_real_plug.tar.zst.part-* | zstd -d | tar -xf - -C restored
cat downloads/original_outputs.tar.zst.part-* | zstd -d | tar -xf - -C restored
mkdir -p restored/original_workspace
tar --zstd -xf downloads/original_workspace_source.tar.zst -C restored/original_workspace
tar --zstd -xf downloads/piper_use_source.tar.zst -C restored
```

分卷需全部下载，顺序为三位数字后缀。每个分卷512MiB以内，校验值在SHA256SUMS；压缩包不含虚拟环境、Git数据库和Python缓存。它们不是采集数据；依赖快照见`requirements/*archive-freeze.txt`。源码、RGB-D原帧、视频、CAN/指令记录、轨迹、候选参数、报告和对比视频均保留。

恢复后的关键目录：

| 目录 | 含义 |
|---|---|
| `restored/sim_real_plug/complete_real/failed/` | 真实完整任务失败归档 |
| `restored/sim_real_plug/complete_real/successed/` | 用户约定的真实成功目录名；不能用仿真成功填入 |
| `restored/sim_real_plug/complete_sim/succeed/` | 历史模型内成功 |
| `restored/sim_real_plug/complete_sim/failed/` | 历史模型内失败 |
| `restored/sim_real_plug/calibration/` | 全部试验、标定、诊断、原始采集和离线源码 |
| `restored/outputs/` | 原工作区早期控制/仿真输出 |
| `restored/original_workspace/` | 原工作区源码快照，和后来离线修改分开保存 |
| `restored/piper_use/` | 现有遥操作/回零控制依赖源码 |

历史文件保存原机器绝对路径。新运行时用以下命令生成本机任务配置；不要批量覆盖原始记录。

## 2. 安装纯仿真环境

```bash
python3.11 -m venv .venv-mujoco
.venv-mujoco/bin/python -m pip install -e '.[mujoco,recorded-scenes,mixed-recording,test]'
export PIPER_DATA="$PWD/restored/sim_real_plug"
.venv-mujoco/bin/python - <<'PY'
import os
from pathlib import Path
import yaml
p=Path('configs/tasks/real_connector_sysid_candidate.yaml')
cfg=yaml.safe_load(p.read_text())
cfg['camera_calibration']=str(Path(os.environ['PIPER_DATA'])/'calibration/sysid_strip_contact_20261110/metric_camera_candidate.json')
Path('configs/tasks/local_archive.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
```

MuJoCo固定3.3.7。`assets/mujoco/`提供可移植模型和网格，正常运行不需要重新转换STEP。无显示器渲染用`MUJOCO_GL=egl`，需要可用的EGL驱动。4.0官方sysid独立于此环境，不能直接升级后宣称历史结果不变。

## 3. 纯仿真生成一条取插轨迹并录制

```bash
export MUJOCO_GL=egl
.venv-mujoco/bin/python scripts/try_plug_pick_sim.py \
  --task configs/tasks/local_archive.yaml --output restored/new_sim_trial \
  --speed 25 --smooth-timing --grasp-height .012 --grasp-tilt-deg 0 \
  --grasp-roll-deg 30 --align-corrections 4 --grasp-opening 0 \
  --start-right-deg 0 0 0 0 0 0
.venv-mujoco/bin/python scripts/collect_plug_sim_episode.py \
  --trial restored/new_sim_trial --root "$PIPER_DATA" \
  --session new_sim_attempt --episode 000001
```

这组参数来自历史模型内成功尝试，但后来几何已调整，**不保证新运行仍成功**。试验输出目录必须新建；采集器重新物理执行并检查轨迹一致性，不能直接把历史动作配上新模型视频标成成功。每集包含三路MP4、HDF5、元数据及模拟器特权信息。采集器根据本次结果归档到`succeed`或`failed`，失败/不完整报告也应保留。

历史成功示例是`20261111_sim_calibrated_pick_insert_000001`（1019帧，33.97秒）；它是旧模型中的专家控制，不代表学习策略或真机部署能力。

## 4. 真实失败指令的纯仿真回放

```bash
.venv-mujoco/bin/python scripts/replay_plug_real_in_sim.py \
  --capture "$PIPER_DATA/calibration/sysid_housing_shape_20261110/mapped_capture" \
  --trial "$PIPER_DATA/calibration/complete_fold_20261109_01" \
  --task configs/tasks/local_archive.yaml \
  --output restored/new_failure_replay
```

只发送到MuJoCo，不操作CAN。使用零阶保持的实发目标，物体由物理积分运动；输出`trajectory.jsonl`、初态、场景配置与报告。`mapped_capture`夹爪开度包含历史图像拟合映射，不是已独立校准的指尖开口。

## 5. 真机数据采集接口（历史流程，未验证可成功插入）

真机依赖`piper_use`中的`replay_connector.py`、`replay_controller.py`和Piper SDK。`continue_plug_task.py`目前硬编码原路径`/media/midea/F496AC8096AC4548/act_td3/piper_use`；换机器需将该行改为恢复后的目录。相机序列号在`capture_plug_calibration.py`的`CAMERAS`，CAN左右映射要对应实际硬件。不能沿用历史使能状态。

- 采集程序：`capture_plug_calibration.py`，三路640×480/30Hz RGB-D与只读CAN；支持START/STOP触发。
- 控制程序：`continue_plug_task.py`，不带`--execute`只打印计划；带该参数才控制右臂。既有流程包含退开和折叠，末态仍需按程序提示管理。
- 原生MuJoCo Control：`sync_mujoco_home.py --side both --piper-use <恢复目录>`；只有显式`--execute`才请求硬件控制。控制单位rad，反馈叠影不是相机实测姿态。

原机器采集用法如下。只有准备实际采集时才启动硬件段，白色插排必须保持断开市电；旧失败轨迹未经修正不能视为有效插入路径。

```bash
# 终端1：只读采集，等待控制程序写START；最多约180秒
.venv-mujoco/bin/python scripts/capture_plug_calibration.py \
  --output restored/new_real_capture --frames 5400 --wait-trigger

# 终端2：先查看计划（没有硬件命令）
.venv-mujoco/bin/python scripts/continue_plug_task.py \
  --trial restored/new_sim_trial --capture restored/new_real_capture
# 实际控制入口是同一命令加 --execute；不在归档流程自动执行。
```

使用硬件还需安装`.[hardware]`及恢复的Piper SDK，并根据现场重新核对设备映射。摄像头READY后控制程序写START；结束写STOP。核对现场结果后，将`execution_report.json`中的`physical_task_success`填为实物观察的true/false，不能把`completed_sequence`当作成功。

```bash
.venv-mujoco/bin/python scripts/export_plug_calibration.py \
  --capture restored/new_real_capture --root "$PIPER_DATA" \
  --session new_real_attempt --episode 000001 \
  --commands restored/new_real_capture/commanded_probe.csv \
  --purpose yellow_pick_insert_attempt \
  --execution-report restored/new_real_capture/execution_report.json
```

导出之后，现有`piper_sim.task_archive.archive_episode(root, session, episode)`按现场结果放入`complete_real/successed`或`failed`。它复制标准数据和视频；原始RGB-D/CAN目录还需一并保留为`raw/task/`，不能只留下压缩视频。

## 6. 数据单位与限制

HDF5关节角为度、末端位置为毫米、四元数xyzw；MuJoCo控制是弧度/米。`gripper_open`是历史开口/95mm格式，另有原始宽度。无效动作字段可能NaN，不能填零冒充动作。CAN末端位置是控制器估计，不是外部绝对定位。没有真实六维力/力矩传感器数据；`effort_Nm`历史字段按未标定代理量解释。

更多历史细节：[混合采集说明](sim_real_plug_collection.md)、[任务记忆](task_memory/plug_insertion.md)。若与历史成功措辞冲突，以本次失败报告和`TASK_STATUS.md`为准。
