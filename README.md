# Piper 双臂插接仿真与实录归档

**任务结论：失败。** 截至2026-11-11，尚未把真实抓持滑移、插入失败、翻转掉落过程近一比一复现；没有证明仿真训练后能直接部署真机。[归档分支](https://github.com/Adeline123456789/piper_sim/tree/codex/failed-sim-real-archive-20261111)保存最终代码、候选模型、失败证据和采集说明，停止本轮继续调参。

> **数据纠正：没有六维力/力矩传感器。** 历史 `effort_Nm` 是按SDK描述缩放的未标定反馈字段，不是实测力矩。不得把它当作接触力、指尖法向力或经过验证的力矩轨迹。

- [失败结论、已完成与未完成项](https://github.com/Adeline123456789/piper_sim/blob/codex/failed-sim-real-archive-20261111/docs/FAILURE_REPORT.md)
- [安装、恢复数据、仿真采集与真机采集入口](https://github.com/Adeline123456789/piper_sim/blob/codex/failed-sim-real-archive-20261111/docs/ARCHIVE_AND_COLLECTION.md)
- [完整数据与旧工作区输出下载](https://github.com/Adeline123456789/piper_sim/releases/tag/failed-sim-real-20261111)
- [任务记忆与历史试验索引](https://github.com/Adeline123456789/piper_sim/blob/codex/failed-sim-real-archive-20261111/docs/task_memory/plug_insertion.md)
- [原项目首页及其他场景说明](https://github.com/Adeline123456789/piper_sim/blob/codex/failed-sim-real-archive-20261111/README_PRE_ARCHIVE.md)（历史说明，不代表本任务验收通过）

以下代码路径位于归档分支，main原有源码保持历史版本。

## 保存了什么

| 内容 | 位置 |
|---|---|
| MuJoCo场景、碰撞几何、控制与采集代码 | `src/`、`scripts/`、`configs/`、`assets/` |
| 最新离线改动 | `codex/failed-sim-real-archive-20261111`分支源码；原始快照另在Release |
| 真实失败录像、RGB-D、关节与CAN记录 | Release中的`sim_real_plug/complete_real/failed/`及`calibration/` |
| 历史模型内成功/失败仿真数据 | `complete_sim/succeed/`、`complete_sim/failed/` |
| 所有辨识候选、否决结果、对比视频 | `sim_real_plug/calibration/` |
| 早期控制和仿真输出 | Release中的`original_outputs`分卷 |
| 原工作区源码、遥操作控制依赖 | Release的独立源码压缩包 |

Piper每臂6个旋转关节和1个夹爪开合控制通道。主要仿真版本为MuJoCo3.3.7；官方sysid试验使用隔离的4.0.0环境，两者结果不混称同一个基准。黄色插头实测质量40.9g，内侧夹爪垫是用户确认的硬质橡胶；摩擦、COM、惯量和接触刚度仍未独立标定。

## 最短使用路径

1. 克隆`codex/failed-sim-real-archive-20261111`分支，按[恢复说明](https://github.com/Adeline123456789/piper_sim/blob/codex/failed-sim-real-archive-20261111/docs/ARCHIVE_AND_COLLECTION.md)下载Release并校验分卷。
2. 安装MuJoCo3.3.7及采集依赖。
3. 先查看已有视频和`TASK_STATUS.md`；要生成新数据，再运行文档中的纯仿真命令。
4. 仿真成功只表示该模型内成功。真实成功必须由实物结果确认，不能由命令发送完毕或反馈变化代替。

没有在归档阶段操作真机，也没有补造成功数据。原数据和历史结论保留，通过顶层失败说明纠正解释。
