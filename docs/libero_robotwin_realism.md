# 插接场景外观改进：LIBERO / RoboTwin 参考

本轮在现有 MuJoCo 场景内实现，无新增真机运动。只处理黄色插头插入白色插排。

## 参考与落地范围

- [LIBERO TableArena](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/arenas/table_arena.py)
  分开设置桌面视觉和碰撞几何，并支持环境纹理风格。这里将外观配置单独放入
  `connector_appearance.py`，不借外观修改改变孔隙、碰撞形状或材料摩擦。
- [RoboTwin 场景初始化](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/_base_task.py)
  使用 SAPIEN 光追、环境及多方向光源。
  当前实现仍是 MuJoCo OpenGL：四个低强度顶灯近似扩展光源，加侧向补光，
  4096 阴影图、4 倍多重采样。不是路径追踪或真正面光源。
- [RoboTwin 域随机化](https://robotwin-platform.github.io/doc/usage/domain-randomization.html)
  包括背景、灯光、桌高和视角变化。这里先实现与固定工位相容的灯光、颜色和相机
  小扰动；没有随机更换整张桌子、改变桌高或向插接区域塞入新障碍。

没有引入上游资源包或复制其机器人资产；沿用本项目的照片纹理和 Piper 模型。

## 当前入口

- `configs/tasks/real_connector.yaml`：参考布局，默认启用 office 外观。
- `configs/tasks/real_connector_calibrated.yaml`：RGB-D 暂定布局 + office 外观。
- `configs/calibration/plug_rgbd_20261108.json`：设备内参与暂定相机外参快照。

后者保留原有标定限制：左腕外参未标定，外部相机世界 XY/yaw 仍有 CAD 假设，
单姿态底座对应不能代替多姿态手眼标定。

在项目根目录打开纯仿真原生 Control：

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src .venv-mujoco-hardware/bin/python scripts/run_scene.py \
  --backend mujoco --mode sim --interactive \
  --task-scene configs/tasks/real_connector_calibrated.yaml
```

重新输出静态对照图（输出目录应为新的）：

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv-mujoco-hardware/bin/python scripts/preview_plug_realism.py \
  --task configs/tasks/real_connector_calibrated.yaml \
  --capture /mnt/a2c7e415-58c1-4de8-be3d-8374e8e93d3e/sim_real_plug/calibration/rgbd_20261108_01 \
  --output outputs/realism/new_preview
```

`appearance.randomization.enabled: true` 开启每回合外观扰动。
`sim.reset(seed=23)` 可复现：灯光增益 0.9–1.1、颜色轻微变化、每轴相机安装位置
±2 mm、旋转总角度 ±0.4°。这些范围是训练超参数，未测得真实硬件误差分布。
相机每次 reset 从基线恢复，无累计漂移；实际姿态及采样参数在观测 extras 中提供，
混合采集器也写入 `privileged.jsonl`。默认关闭随机化以便做实拍对齐。

## 验证与交付

- 14 项测试通过，包括碰撞/质量/摩擦/限位不变、物理状态一致、随机种子复现与相机重置。
- `outputs/realism/libero_robotwin_final/real_before_office.png`：实拍 / 原版 / 新版，
  外部和右腕两种视角；`comparison.png` 另含左腕、插头和插排细节及随机化示例。
- 同一空桌面区域 `[220,160,420,220]` 的平均 RGB：实拍约 136/139/134，
  原版 234/232/228，新版 176/177/175。只说明该区域的亮度差缩小，
  不代表全场景相似度或定位精度。
- `20261108_sim_yellow_office/000001` 已保存到既定混合数据目录：679 帧，
  三路视频帧数与 HDF5 一致，完整插接成功，最大数值接触压入 0.03315 mm。
  这是原 `pick_insert_aligned_03` 动作轨迹的外观变体，重放状态误差为 0，
  不能算新的独立动作示范或真实成功次数。

仍有差距：玻璃折射和间接照明、物体细节及缺失的桌面杂物、精确相机外参、
夹爪开口和接触参数。没有完成一比一重建，也未证明训练后可直接真机部署。
