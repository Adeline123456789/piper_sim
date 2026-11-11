# 黄色插头真实/仿真混合采集

用户确认的任务：从固定灰色底座取出黄色插头，插入已断电的白色插排。
黑色插头保留为障碍物，不采集它的插接任务。

输出根目录：`/mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug`。
目录沿用参考数据集的 `observations/<session>/<episode>/videos/` 和
`proprio_stats/<session>/<episode>/proprio_stats.h5`，不创建六维力 `sensors/`。

## 数据语义

三个视频名称为 `cam_external_rgb.mp4`、`cam_left_wrist_rgb.mp4`、
`cam_right_wrist_rgb.mp4`。HDF5 的关节位置为度，末端位置为毫米、相对各臂底座，
四元数为 xyzw。`gripper_open` 保留旧格式的开口/95 mm，另存原始
`gripper_width_m`。真实夹爪的编码器长度与物理指尖开口尚未完成标定，不截断
超过 95 mm 的反馈值，也不把它当成已验证的物理尺寸。

每条数据均有 `domain`、`purpose`、`success` 和 `policy_training_eligible`。
真实标定记录没有动作时，action 为 NaN 且 `valid/action=false`，不能将观测角度
复制为示范动作。真实 CAN 末端姿态来自控制器，不是外部相机的绝对位置测量。
仿真 GT 写入同目录 `privileged.jsonl`，包含物体位姿、相机位姿、接触和任务指标。
真实样本不能被标成具有仿真真值；后续真实位姿估计应独立保留来源和误差。

## 本次已得到的证据

- `calibration/rgbd_20261108_01`：三路各 45 帧 RGB-D、设备内参、曝光时间戳、
  同期只读 CAN。导出为 `20261108_real_calibration/000001`，共同覆盖 35 帧；
  不作为动作策略示范。该早期导出的 CAN 状态对应图像接收时间，场景为静止状态。
- `calibration/table_registration.json` 实际位于对应 RGB-D capture 子目录内。
  桌面深度平面用于估计外部相机高度和倾角，XY/yaw 仍依赖 CAD 假设，
  不能作为毫米级手眼标定结果。
- `20261108_sim_contact_probes/000001..000002`：对准和横向错开 12 mm 的
  物理接触检查，各 45 帧；这是重力落入/阻挡检查，不是完整抓取插入示范。
- `calibration/pick_insert_trial_02`：真实物理步进完成抓取、抬升、搬运、插入和松爪。
  无附着焊接，无中途改 qpos。656 帧、约 21.87 s，逐子步最大压入约 0.002154 mm。
- `20261108_sim_yellow_insert/000001`：上述完整流程重新物理执行并记录三视角。
  与预试轨迹状态差为 0；成功保持插入，标为可用于仿真策略训练。
- `calibration/wrist_probe_01`：右腕 +8° 后返回的真实控制日志。相机先达到
  60 秒上限，仅覆盖约 1.64–1.86 秒控制区间，不能用其证明完整往返录像。
  详见 `coverage_audit.json`，不应当导出为完整成功操作示范。
- `20261108_real_wrist_probe_partial/000001`：只导出三相机与指令日志共同覆盖的
  49 帧，视频最大时间偏差约 10.49 ms。保留已记录的右臂关节请求目标，
  `valid/right_arm_joint_action=true`；左臂和双夹爪动作仍缺失，
  `valid/action=false`，不用于完整任务策略训练。指令时间不是 CAN 实际发出时间。
- `calibration/right_motion_check_02`：追踪底座表面 36 个特征点，与 CAN/FK
  相机运动预测对照。末帧中位误差 1.85 px；已记录的指令区间最大中位误差
  2.78 px，仅覆盖 J6 约 1.24°变化。启动阶段第 1725 帧出现 85.64 px 不一致，
  保留该异常，不将整段判为标定通过；也不能据小幅运动证明绝对定位精度。
- `calibration/trigger_smoke_01`：三相机只读触发测试，触发前 0 张图像，
  触发后每路 90 帧，首曝光延迟约 6.3–22.7 ms。没有发送 CAN 指令。
- `calibration/wrist_probe_02`：三路各 349 帧，完整覆盖右腕 8° 往返运动；
  反馈 J6 约 18.123–26.144°，最后为 18.168°。结束保持的最后 11–27 ms
  未被录像覆盖，因此严格的完整指令区间覆盖仍为 false。后续程序已增加 150 ms
  录像尾段，未因此重新执行硬件动作。
- `20261108_real_wrist_probe/000001`：共同时间窗口导出 335 帧，三路最大偏差
  17.30 ms，保留真实右臂请求指令，不含虚构的左臂/夹爪动作。
  这是腕部标定样本，仍不是黄色插头插接示范。
- `calibration/right_motion_check_03`：整个 8° 往返中，旧相机安装参数的中位投影
  误差峰值约 9.85 px，不能沿用小运动检查的结果声称标定已通过。
- `calibration/wrist_axis_fit_01`：以真实 RGB-D 初始特征深度拟合 J6 轴相关的
  4 个可观测参数，只用去程拟合、回程验证。回程 P95 像素误差从 5.25 降至
  1.03，但候选参数在原静态空孔中心处误差为 33.64 px；尝试静态约束其剩余
  自由度后仍为 20.38 px。候选未应用到默认场景，也未用于真机定位。
  下一步需要多轴运动、机器人基座与固定插座坐标校准，不能只优化这一次往返。

## 物理采集配置

`configure_rigid_collection()` 使用 1/3840 s 子步、30 Hz 观测，并检查每个子步的
所有活动接触。超过 0.2 mm 压入即拒绝该轨迹，不删掉接触，也不把物体移出障碍。
该容差是数值接触验收门槛，不是实体标定精度。当前插接摩擦/刚度仍是估计值。
指垫使用六维接触模型，使已有扭转/滚动摩擦参与计算，避免两点夹持自由翻转。
机器人重力补偿在 MJCF 编译前启用，物体仍受重力。旧任务默认配置未改变。

试验默认仿真速度为 10°/s，真实标定为 2°/s；不能把两者当成已经辨识一致的控制器。
仿真相机和布局目前仍为参考配置，真实 RGB-D 标定尚未全部应用到训练场景。

## 固定插座的视觉对齐版本

`calibration/visual_alignment_01/task.yaml` 是单独的对齐场景，不覆盖参考场景。
`cameras.json` 保存设备内参和暂定外参，`comparison.png` 对比实拍、原场景及新场景。
灰色底座和白色插排分别拟合位置及偏航角；右腕相机外参由同一底座在两路图像中的
对应关系估计，随 `right/link6` 运动。左腕目前只更新设备内参，外参仍采用 CAD。
外部相机世界 XY/yaw 仍依赖 CAD 锚点，因此不能把本次对齐当成独立手眼标定。

未用于拟合的空插孔中心检查见
`calibration/rgbd_20261108_01/heldout_aperture_check.json`：外部视角误差约 4.20 px，
右腕约 1.40 px（人工点选不确定度约 2 px）。这些是单姿态投影误差，不是真机末端
毫米级精度，也不证明移动后的相机外参准确。

新布局的离线试验 `calibration/pick_insert_aligned_03` 已完成完整物理插接。
夹爪中心在插头底面上方 20 mm 处抓取，插入前底面距插排 25 mm；
未修改关节限位。最大数值接触压入约 0.03315 mm。之前的 `_01` 受 J5 限位限制，
`_02` 倾斜抓取后插接失败，两者只保留为调试记录，不标成成功示范。
这些仿真动作尚未在实体机械臂上执行。
已重新物理重放并采集为 `20261108_sim_yellow_aligned/000001`：679 帧，
三路视频、HDF5 和特权信息齐全，重放关节状态误差为 0；仅标为仿真成功示范。

## 复现

在项目目录用 `.venv-mujoco-hardware/bin/python` 运行脚本，清除 ROS 的 PYTHONPATH。
该环境基于原有 piper-teleop 环境；项目的通用 Python/NumPy 要求不代表这个硬件兼容环境。

```bash
# 只读 RGB-D/CAN。输出目录必须是新的。
env -u PYTHONPATH .venv-mujoco-hardware/bin/python scripts/capture_plug_calibration.py \
  --output /absolute/new_capture --frames 45

# 真实动作采集：相机预热后等待 START；控制程序在使能前触发，结束后写 STOP。
env -u PYTHONPATH .venv-mujoco-hardware/bin/python scripts/capture_plug_calibration.py \
  --output /absolute/new_probe --frames 1800 --wait-trigger
# 另一终端，确认三路 READY 后使用原版控制流程，只动右腕，不动夹爪。
env -u PYTHONPATH -u LD_LIBRARY_PATH .venv-mujoco-hardware/bin/python \
  scripts/probe_plug_right_wrist.py --capture /absolute/new_probe --execute
# 终端确认词 PROBE right。完成后保持使能，扶稳后输入 DISABLE。

# 纯离线物理抓取插入试验，无硬件命令。
env -u PYTHONPATH MUJOCO_GL=egl .venv-mujoco-hardware/bin/python \
  scripts/try_plug_pick_sim.py --output /absolute/new_sim_trial

# 使用新布局；任务配置和相机参数会快照到试验目录。
env -u PYTHONPATH MUJOCO_GL=egl .venv-mujoco-hardware/bin/python \
  scripts/try_plug_pick_sim.py \
  --task /absolute/dataset/calibration/visual_alignment_01/task.yaml \
  --grasp-height .020 --preinsert-gap .045 --smooth-timing --speed 30 \
  --output /absolute/new_aligned_trial

# 将验证通过的试验重新通过物理执行，采集三路视频和特权信息。
env -u PYTHONPATH MUJOCO_GL=egl .venv-mujoco-hardware/bin/python \
  scripts/collect_plug_sim_episode.py --trial /absolute/new_sim_trial \
  --root /absolute/dataset --session new_sim_session --episode 000001
```

2026-11-09 更新：预插间隙大于 30 mm 时，搬运目标采用 5° 倾斜以满足 J5 限位，
随后在距插排表面 25 mm 处摆正，再插入；`align` 追加在阶段索引末尾，保留旧索引含义。
以上 `30°/s` 是纯仿真参数，未修改真机速度。当前固定布局、抓取高度 20 mm 的验证
记录为 `20261109_sim_clearance/000001`，391 帧、13.03 秒；不含从折叠状态展开。

尚未完成：完整真实动作录像、多姿态手眼/夹爪校准、固定插座相对真机基座的独立定位、
实体黄色插头抓取插入示范、多个初始状态下的混合数据。当前不能声称整项采集完成。

## 下一轮多轴标定（仅准备，尚未执行）

`scripts/probe_plug_right_wrist.py --multi-axis` 复用现有控制流程，依次执行
J6 +8°、J5 −5°、J4 +5°，每轴往返至入口姿态，速度 2°/s，含停留约 25 秒。
夹爪和左臂均不控制。三路相机仍需先预热等待 START；记录上限可用 1800 帧。
导出器识别 `j6_outbound` 等分段名称，不把它们作为完整插接示范。

准备结果在数据根目录 `calibration/multiaxis_plan_01/`：`state.json` 为只读 CAN
快照，`plan.json` 为待执行目标。`sweep.json` 记录被放弃的 J5 +5° 方案，右臂
模型最大自碰撞重叠 1.525 mm；`sweep_j5_negative.json` 对最终方案检查 1806
个静态关节姿态，未发现涉及右臂的负距离接触。静止左臂模型原有折叠重叠仍保留。
这些是暂定模型的几何诊断，不是动力学训练轨迹，也不证明实物间隙。
真实夹爪宽度尚未标定，检查使用模型 95 mm 开口。

旧程序仍在使能保持并等待扶稳下使能，必须正常退出后才能启动下一轮，不能同时
启动第二个 CAN 发送程序。下一轮开始前要重新读取当前姿态；不能直接执行旧快照。

## 2026-11-09：完整任务执行与实际失败

新增 `20261109_sim_full_task/000001`：1136 帧，从右臂折叠入口姿态通过物理积分
走完接近、抓取、抬起、搬运、插入、松爪及沿插入路径向上退开 25 mm。
最大数值压入 0.03315 mm；三路录像及仿真特权信息已保存。左臂为仿真默认静止姿态，
右臂初始关节来自实测入口；并非完整双臂实物姿态标定。

新增真实多轴标定 `20261109_real_multiaxis/000001`（762 帧），以及
`20261109_real_task_approach/000001`（915 帧）。它们均不标记为完整任务示范。

当时用户明确要求保持使能并连续走完流程后，旧版 `scripts/continue_plug_task.py` 在确认原
任务起步发送者静止且持有对应 CAN 锁后，结束原发送进程，取得该锁，以新鲜实测
关节目标恢复保持。接管不发送使能、下使能、复位或回零指令，沿用 ReplayController
的限位、反馈与电机故障检查。之后连续执行剩余取插动作，包括夹爪控制。

`20261109_real_full_attempt/000001` 保存 3426 帧同步实物数据：原始各相机
3432 帧，已知右臂关节和夹爪动作；左臂动作未知，保留 NaN 与无效掩码。没有
六维力信息或伪造的实物特权状态。运动段录像完整，但开头夹爪准备的指令窗口
未完全被三相机公共时间窗覆盖，`complete_command_video_coverage=false` 保留。

**本次实物结果失败**：用户确认“插头掉落或未放到插排上”。程序走完序列与
物理任务成功是分开的字段，本条 `success=false`、`policy_training_eligible=false`。
录像确认从灰色底座提起并搬运，最终精确掉落/错位位置仍待确认。
`calibration/full_real_task_20261109_01/real_sim_comparison.json` 给出首轮比较：
关节指令跟踪最大误差 0.366°；闭合末端实物夹爪反馈 51.4 mm，仿真 41.7 mm；
抬起后实物变为 38.5 mm，仿真 41.5 mm。夹持阶段存在显著差异，但这些编码器
数值本身不足以确定掉落原因，不能把它归因于已证明的摩擦或标定错误。

## 连续执行器更新（离线验证，尚未执行新真机任务）

当前 `scripts/continue_plug_task.py` 直接复用 `home_zero` 使用的 `Hardware.connect`、
原控制器及退出保持流程，一次启动覆盖展开至退开；不再接管另一个起步进程。
旧交接版本保留在 `outputs/connector_reference/continue_plug_task_handoff_20261109.py`。
旧参数 `--handoff-pid`、`--start-frame` 已移除，只接受完整成功的仿真轨迹。

先离线查看计划（不导入 SDK、不连接 CAN）：

```bash
cd /media/midea/F496AC8096AC4548/piper_simulation
env -u PYTHONPATH -u LD_LIBRARY_PATH .venv-mujoco-hardware/bin/python scripts/continue_plug_task.py \
  --trial /mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug/calibration/from_current_20261109_01 \
  --capture /tmp/plug_dry_run_no_capture
```

611 个源帧按原时间轴平滑路径重采样到 100 Hz，关节目标限速 12°/s，
归一化夹爪速度不超过 0.5/s，整段轨迹 50.618 秒（包含展开）。
准备夹爪及结束保持另需约 2 秒，控制开销及到位等待可能延长实际耗时。
未采用参考数据集的姿态或抖动，也没有将插入动作强行压缩到参考的 8 秒。

实际执行需三路录制器 READY 后显式加 `--execute`，终端输入 `TASK right`。
在使能前核对当前折叠入口与轨迹入口；其后连续执行，不在阶段之间切换进程。
此任务控制器在入口到位后设为 SDK 15%，其他调用仍默认 5%；夹爪力矩仍为 0.3 N·m。
沿用终端空格暂停、q 停止推进、e 急停及退出时扶稳后输入 DISABLE 的原有操作。
`completed_sequence` 仅表示指令序列结束，`physical_task_success` 保持 null，必须另行核验。

**实物孔位及抓取后插头变换仍未验证。此离线计划不代表可直接可靠插入，
不能用计划完成或关节反馈接近目标来认定实物成功。**

上述计划随后已执行，用户接受速度并确认失败，归档为
`complete_real/failed/20261109_real_smooth_attempt_000001`（1584 同步帧）。
用户要求收尾折叠，当前执行器已在退开后默认增加五次平滑回零，保持夹爪开度；
同程序连续录制，无阶段交接。该轨迹预计再增加约 17.6 秒，计划总时长约 70.3 秒。
本轮已经单独执行并录像折叠，六关节实测 0°；由于是在任务完成后补做，
两段之间的等待未录像，归档 segments.json 明确标记此间隔。
