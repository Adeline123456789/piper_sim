# 插接场景运行与采集

按首页安装MuJoCo3.3.7。assets/mujoco包含可移植模型和网格，无需大数据下载或STEP重建。使用configs/tasks/real_connector_final.yaml，相机配置已随仓库保存。

## 生成纯仿真轨迹

```bash
export MUJOCO_GL=egl
.venv-mujoco/bin/python scripts/try_plug_pick_sim.py \
  --task configs/tasks/real_connector_final.yaml --output outputs/plug_trial_new \
  --speed 25 --smooth-timing --grasp-height .012 --grasp-roll-deg 30 \
  --align-corrections 4 --grasp-opening 0 --start-right-deg 0 0 0 0 0 0
```

输出目录须新建。动作包括展开、抓取、搬运、对准、插入、释放、退开和折叠。参数来自历史模型内尝试，不保证当前模型或新布局成功。以本次report.json与物体状态判断，不以命令执行完毕判断成功。

## 录制三路视频和状态

```bash
.venv-mujoco/bin/python scripts/collect_plug_sim_episode.py \
  --trial outputs/plug_trial_new \
  --root /mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug \
  --session plug_new --episode 000001
```

重新物理执行并输出三路MP4、HDF5、元数据和仿真特权状态。按结果归入complete_sim/succeed或failed。只采集需要的数据，不再批量保存调参产物。success仅表示模型内成功。

HDF5关节角为度、末端位置为毫米、四元数xyzw；MuJoCo控制是弧度/米。gripper_open为历史开口/95mm格式，另有原始宽度。NaN动作不可补零冒充实录。无真实六维力/力矩数据。

## 真机采集入口

依赖本机/media/midea/F496AC8096AC4548/act_td3/piper_use中的ReplayController和Piper SDK；换机器需提供依赖并更新路径。相机序列号在capture_plug_calibration.py的CAMERAS中，CAN左右映射须对应现场设备。

- `capture_plug_calibration.py --output <新采集目录> --frames 5400 --wait-trigger`：三路640×480/30Hz RGB-D和只读CAN，等待START触发。
- `continue_plug_task.py --trial <轨迹目录> --capture <采集目录>`：只打印计划；加`--execute`才执行。旧失败轨迹未解决对孔，不应直接重复当作有效插入方案。
- `sync_mujoco_home.py --side both --piper-use <依赖路径>`：MuJoCo原生Control入口，显式`--execute`才请求硬件控制。
- `export_plug_calibration.py`：将已有采集导出标准格式，现场确认physical_task_success，不得用completed_sequence代替。

真实成功/失败仍归complete_real/successed和complete_real/failed，本机原数据不随场景源码上传。白排须断电；本次整理未操作真机。

## 纯仿真回放真实指令

```bash
.venv-mujoco/bin/python scripts/replay_plug_real_in_sim.py \
  --capture <含commanded_probe.csv的目录> \
  --trial <含匹配initial_state.npz和task.yaml的目录> \
  --task configs/tasks/real_connector_final.yaml --output outputs/replay_new
```

只控制MuJoCo，不操作CAN。初态、夹爪映射和模型必须匹配；旧试验目录已清理，不再提供失效绝对路径命令。
