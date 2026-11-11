# 真实 Piper 双臂连接接口

接口位于 `src/piper_sim/piper_connection.py`，命令行入口为 `scripts/connect_piper.py`。
直接使用提供的 `piper_sdk.C_PiperInterface_V2`，不加载 Isaac、MuJoCo 或旧遥操作控制线程。
本机 `can_arm1` 对应左臂、`can_arm2` 对应右臂。最新实测已通过原生 Control
分别驱动左右腕 J6，结束录制后的相机画面确认对应实体转动。
证据位于 `outputs/live_piper/native_both_live_01/verification.json` 和同目录的
`native_control_physical_proof.mp4`。先前从尚未结束录制的 AVI 跳帧取得的对照图不可靠，
不能用这些旧图判断实体是否运动。

## 直接使用 MuJoCo 原生 Control

### 原版 home_zero 控制流程入口（推荐）

新版入口直接调用 `piper_use/replay_connector.py` 的 `Hardware.connect()`、
`Hardware.send()` 和 `Hardware.shutdown()`；原生 Control 修改模型关节位置，
读取该模型的 qpos 后以 100 Hz 更新目标。最大目标变化速度为 3°/秒，
SDK 速度仍是原版 5%。运行环境 `.venv-mujoco-hardware` 基于原版
`piper-teleop` 的 Python 3.10 / NumPy 2.2.6，单独安装 MuJoCo，未修改原环境。

```bash
cd /media/midea/F496AC8096AC4548/piper_simulation
bash scripts/sync_mujoco_home.sh --side both --execute --output outputs/live_piper/native_$(date +%Y%m%d_%H%M%S)
```

输出目录必须不存在。终端核对当前姿态后输入 `SYNC both`；程序先用当前姿态与限位
设置滑条范围、创建窗口，然后执行原版使能握手。展开右侧 Control，逐个拖动
`left/joint1`～`left/joint6` 或 `right/joint1`～`right/joint6`。
只控制一条臂时用 `--side left` 或 `--side right`，终端确认词对应改为 `SYNC left/right`。
单位是 rad，目标是绝对角度；本次启动姿态附近 ±15° 且同时满足模型/硬件限位。
夹爪不受控；一次调整一个关节，其他关节保持原目标。灰色为 MuJoCo 姿态，青色为 CAN 反馈姿态；
此模式是关节位姿同步，不是动力学精度验证。

`F9` 或空格停止发送；扶稳后 `F10` 下使能并退出。关闭窗口会停止发送，
终端等待扶稳后输入 `DISABLE`。省略 `--execute` 可只读连接并预览滑条，不能使能。
左右腕 J6 的约 +10° 操作已有实体录像验证，其他轴尚未逐轴上机实测。
原生单位为弧度，10° = 0.174533 rad；滑条目标为绝对角度，鼠标离散步长会使
此次 +10° 点击实际落在约 +9.9°。不要把 CAN 跟踪差值当作绝对位置标定误差。

`piper_use/replay_connector.py` 的 `Hardware.connect()` 新增可选 `before_enable`
回调，用于在电机未使能时初始化窗口。原版回零/回放不传此参数，默认流程保持原样。

### 早期同步入口

此入口只有一个 MuJoCo 窗口；不使用额外的角度控制面板。

```bash
cd /media/midea/F496AC8096AC4548/piper_simulation
.venv-mujoco/bin/python scripts/sync_mujoco_joint.py --can can_arm2 --side right
```

该入口已直接复用 `piper_use/replay_controller.py` 的 `ReplayController`，与
`home_zero.sh` 相同，以 100 Hz 在独立线程发令。连接校验同一份
`piper_use/example_agilex/scripts/piper_vision_ports.json` 中的主机、USB 接口和左右映射。
启用前按该脚本处理复位过渡：仅电机全关闭时等待短暂状态 5/错误码低六位消退，
正常状态连续稳定 0.5 秒后才开始使能，运动期间仍拒绝这些错误。

启动不自动使能。**未使能时原生 Control 可以控制灰色仿真模型，青色模型持续显示 CAN
反馈；不会把滑条复位到真机角度。** 当前 `can_arm2` 配置为右臂。
`--preview-only` 明确禁止所有使能请求，适用于只检验窗口输入。

如需沿用 `home_zero.sh` 的终端确认方式，添加 `--execute`：

```bash
.venv-mujoco/bin/python scripts/sync_mujoco_joint.py --can can_arm2 --side right --execute
```

确认已松开机械臂并避开范围后，在终端输入 `SYNC right`；程序先对齐实时姿态再使能。
预览阶段留下的目标会被丢弃，不会在使能时突然追赶预览位置。

在 MuJoCo 三维画面上取得键盘焦点后：

1. 松开机械臂并离开活动范围，按 **F8** 启用。六关节先进入保持姿态；若 J2/J3
   在折叠零位外不超过 3°，会先移到合法零边界，实际初值和进入值写入事件记录。
2. 等待画面显示 `active`，直接拖动 **Control → right/joint1 … right/joint6**。
   当前版本的滑条使用鼠标拖动，不支持直接输入数值；原生单位为弧度，5° = 0.0872665 rad，10° = 0.174533 rad。滑条存在离散步长，应以日志中的实际目标值为准。
   目标为绝对角度；只调整一个关节，其他关节保持。最大速度默认 0.5°/s，
   本次连接相对进入姿态最多 ±15°，同时受真机和模型限位约束。
3. 灰色机械臂为虚拟控制姿态，青色半透明模型为 CAN 反馈的正运动学姿态。
   左上角同时显示 `GOAL`、`SIM`、`CAN`、`CAN-SIM`，单位为度。
   `FK tip gap` 是使用同一模型计算的末端位置差，不是相机测量的真机绝对位置误差。
4. **F9 或空格**请求停止，保持电机使能；扶稳后按 **F10** 下使能。
   **G** 切换反馈模型显示。活动时关闭窗口会请求停止，终端等待扶稳后输入 `DISABLE`。

未连接侧和夹爪滑条不允许发送实体命令；`Clear all` 也不作为真机回零命令。
连接保持独占锁；反馈失效、超过跟踪误差、主循环超过 300 ms 未更新均停止。

该窗口采用限速的关节位置遥操作预览，使用 MuJoCo 的 qpos 和正运动学显示位置；
不积分自由动力学，不等同于训练策略或接触动力学的仿真到真机验证。
训练场景和其物理参数未因此修改。青色模型来自 CAN，不能代替对实体运动的现场确认。

逐帧日志保存在 `outputs/live_piper/native_sync_时间/trajectory.csv`，包括输入、
仿真姿态、实际发送角度、CAN 反馈，以及驱动器高速反馈的原始位置、速度和电流字段。
状态切换、按键事件、原生滑条目标（注明是否发往真机）和错误保存在同目录的 `events.jsonl`。

## 与原真机控制的对应关系

核对资料为 `example_agilex/piper_controller.py` 的 `_send_gripper_command()`、
`piper_sdk/piper_sdk/interface/piper_interface_v2.py` 的 `GripperCtrl()`，以及SDK的
`demo/V2/piper_ctrl_gripper.py`。原代码每臂一次 `JointCtrl` 加一次 `GripperCtrl`，
没有分别控制左右手指的硬件指令。

| 项目API | SDK调用 | 换算 |
|---|---|---|
| 每臂6个 `joints_rad` | `JointCtrl(j1, ..., j6)` | rad → 0.001° |
| 每臂1个 `gripper_width_m` | `GripperCtrl(width, effort, 0x01, 0)` | m × 1,000,000 → 0.001 mm |
| `read().position` | 关节反馈及 `GetArmGripperMsgs()` | 相反方向换算，仍为14维 |

例如0.04 m开口发送 `GripperCtrl(40000, ...)`，只发送一次，不分别给手指发送20000。
原控制器把夹爪字段叫作degrees，但SDK定义是长度；旧归一化0–1对应0–95 mm。
仿真现在使用0–95 mm总开口。原控制器开启了SDK限位，但随附SDK的默认范围为0–70 mm，
因此旧控制器请求95 mm仍可能被SDK裁到70 mm。真机接口通过 `GetSDKGripperRangeParam()`
读取实际软件范围；`robot.limits()` 和HTTP `/metadata` 返回左右范围，超限目标明确拒绝，读反馈不截断。
`hardware.gripper_max_width_m: null` 保留SDK当前范围；设备可用行程确认后可显式配置 `0.095`，
通过 `SetSDKGripperRangeParam(0, .095)` 修改当前SDK实例的软件限位，不修改固件或机械行程。
仿真范围与实际指尖开口的标定仍需真机测量。

MuJoCo使用一个总开口驱动及双指等量约束，Control只提供每臂一个gripper滑条；
Isaac使用 `joint8` mimic `joint7`，只给主指设置驱动目标。
物理模型仍需要两根可移动手指来计算几何与接触，独立控制自由度为1。
对应的MuJoCo约束机制见 [官方MJCF说明](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#equality-joint)。

## 连接与读取

在项目根目录执行，`can0/can1` 必须替换为实际左右臂通道。接口不修改系统CAN配置。
先接入并按原设备流程配置USB-CAN，确认左右映射和设备要求的bitrate。

```bash
# 只列出现有CAN接口，既不连接机器人，也不配置网络。
.venv-mujoco/bin/python scripts/connect_piper.py --list-can

# 连接双臂，读取一次完整的新鲜反馈，然后断开。
.venv-mujoco/bin/python scripts/connect_piper.py \
  --can-left can0 --can-right can1

# 持续读取，Ctrl+C退出。
.venv-mujoco/bin/python scripts/connect_piper.py \
  --can-left can0 --can-right can1 --samples 0 --rate 10
```

SDK路径沿用 `configs/workcell.yaml` 中的 `hardware.sdk_path`。
连接检查通道存在且UP，再以 `ConnectPort(piper_init=False)` 接收数据，等待两臂全部
关节对及夹爪反馈新鲜后才返回；默认5秒超时。不会自动上电、回零或切换固件主从模式。

```python
from piper_sim.piper_connection import PiperConnection, state_dict

with PiperConnection(can_left="can0", can_right="can1") as robot:
    state = robot.read()
    print(state_dict(state))
    print(robot.limits())
    # state.position: 左6关节、左夹爪、右6关节、右夹爪
```

## 显式控制API

物理电机需已由现有设备流程使能。`allow_actuation=True` 加 `arm()` 才允许发指令；
首个目标需与反馈对齐，持续命令仍经过新鲜度、顺序、限位、跟踪误差和速度检查。
下面示例只保持连接时读到的位置。控制源通常以30 Hz更新目标。

```python
import time
from piper_sim.piper_connection import PiperConnection

with PiperConnection(can_left="can0", can_right="can1", allow_actuation=True) as robot:
    q = robot.read().position.copy()
    robot.arm()
    for _ in range(30):
        accepted = robot.send(
            left_joints_rad=q[:6], left_gripper_width_m=q[6],
            right_joints_rad=q[7:13], right_gripper_width_m=q[13],
        )
        time.sleep(1/30)
```

只改变夹爪时，持续调用 `robot.set_gripper_widths(left_m=..., right_m=...)`；
机械臂关节保持上一条已接受的目标。返回值是经过速度限制的实际发送目标。
已有 `Command` 可用 `robot.apply(command)`，其时间戳和sequence不会被接口刷新。
停止发送超过默认150 ms、发送失败或反馈过期都会进入故障停止流程，需要重新连接。
SDK原本只记录日志的发送错误也会转为异常，交给双臂停止逻辑处理。
执行过控制的上下文退出时会发送停止请求，纯读取连接退出仅断开。

## 本机HTTP接口与仿真联动

```bash
# 只读状态服务：GET /state；POST /target会返回403。
.venv-mujoco/bin/python scripts/connect_piper.py \
  --can-left can0 --can-right can1 --command-port 8765

# 显式启用目标接收：收到首个有效指令后才进入控制。
.venv-mujoco/bin/python scripts/connect_piper.py \
  --can-left can0 --can-right can1 --command-port 8765 --allow-actuation
```

服务仅监听127.0.0.1。请求格式与现有仿真控制入口一致：
`POST /target` 的JSON为 `{"position": [14个SI值], "sequence": 递增整数}`；
客户端先读 `/state` 对齐，然后持续更新目标。重复旧命令不会延长控制源的有效期。

需要真实主臂带动仿真，或仿真带动真机时，使用已有 `run_mujoco.sh --mode real-to-sim`
与 `--mode sim-to-real`，并在配置中填写相同CAN映射；Isaac使用 `run.sh`。
同一双臂应由一个控制进程负责，不同时启动独立真机服务和仿真主从服务。

离线测试覆盖只读连接、单位换算、每臂一次夹爪命令、超时清理、陈旧命令、CAN发送失败及只读HTTP拒绝。
这验证软件行为，不代表物理控制已在当前设备验收。
