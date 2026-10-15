# 墨镜

[![English](https://img.shields.io/badge/Readme-English-777777?style=flat-square)](README.md)
[![简体中文](https://img.shields.io/badge/点击看我-中文-444444?style=flat-square)](README.zh-CN.md)

游戏太闪？给屏幕戴副墨镜。

直接看处理效果，强度均为 **50%**，附亮度条和曲线。

为避免不适，演示中的原始闪光画面已完全遮挡。想看原片，请主动点击下方附件链接（含闪光）。

**防止单次强光**

瞬间白光亮起时压暗刺眼高光，同时看清暗处细节。

![单次强光处理结果：左侧原始画面完全遮挡，右侧为处理后画面，下方保留亮度条和曲线](assets/demos/single-burst.gif)

[原片附件：单次强光（含一次闪光）](assets/demos/original-single-burst.gif)

<details>
<summary>连续闪光处理对比：处理后仍有闪动，点击展开</summary>

**防止单次强光与防止连续闪光**

左侧原片已遮挡，中间为“防止单次强光”，右侧为“防止连续闪光”。亮度条和曲线保留三路的实际数据。

![连续闪光处理结果：原始画面完全遮挡，保留两种模式的处理画面与三路亮度数据](assets/demos/repeated-flashes.gif)

</details>

[原片附件：连续闪光（含每秒 5 次的高对比度闪光）](assets/demos/original-repeated-flashes.gif)

## 模式

| 模式 | 效果 |
| --- | --- |
| 防止单次强光 | 画面突然变亮时短暂压暗高光，随后恢复；稳定的亮画面不会持续触发。 |
| 整屏调暗 | 直接降低整块屏幕亮度，无需屏幕捕获。 |
| 防止连续闪光 | 画面反复闪烁时自动调低亮度与对比度；也会影响整幅画面的明暗。 |

支持选择指定显示器与中英文切换。画面仅在本地实时处理，不保存、不上传，也不注入游戏进程。

## 使用方法

若有便携包，解压后直接运行 `Sunglasses.exe`。

- **鼠标操作**：点击左镜片开关滤镜，拖动右镜片调节强度，打开眼镜盒切换显示器、模式或语言。
- **快捷键**：`Ctrl+Alt+S` 开关滤镜，`Ctrl+Alt+Esc` 紧急关闭。
- 启动默认关闭（主显示器、防止单次强光模式、50% 强度），退出程序即移除滤镜。

## 从源码构建

环境要求：Windows 11 x64、Visual Studio（含 C++ 桌面开发工具与 Windows SDK）、CMake 3.24+、Ninja、Python 3.12。

```powershell
.\build.cmd
.\run.cmd
```

运行离线测试与代码检查（需额外安装 uv、Node.js、FFmpeg）：

```powershell
.\ci.cmd check
```

## 已知限制

- **原理限制**：采用“捕获屏幕 + 叠加遮罩”实现，无法拦截闪光的第一帧；主要针对大面积亮度突变，不解决小范围特效、镜头晃动或显示器硬件频闪。
- **环境与兼容性**：目前针对 Windows 11 SDR 画面；HDR、独占全屏及反作弊兼容性尚未验证。
- **定位**：仅用于改善视觉舒适度，不构成医疗保护承诺。

## 致谢与开源协议

- **防止连续闪光模式**移植自 Apple 开源的 [VideoFlashingReduction](https://github.com/apple-aiml-research/VideoFlashingReduction/tree/6e9604d92446c8e6763bc53c5e12ccc4e0046796) 参考算法，详见[源码对应关系与 MIT 声明](sunglasses/data/APPLE_VIDEO_FLASHING_REDUCTION_NOTICE.md)。
- 离线评估参考了 [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP/tree/v0.5.7)（[SIGGRAPH 2024](https://doi.org/10.1145/3658144)）与 [WCAG 2.2 闪光定义](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)。
- 本项目采用 [GPL-3.0-only](LICENSE) 协议开源。
