# Nyx 宇宙学模拟可视化分析平台

基于 [Nyx](https://amrex-astro.github.io/Nyx/) 宇宙学模拟数据的交互式可视化分析平台。

## 项目简介

本项目基于 Nyx（AMReX 框架下的并行自适应网格细化宇宙学模拟代码）生成的 100 个时间步长、$128^3$ 分辨率的宇宙密度场数据，完成了从体数据渲染、统计演化分析到交互式相空间刷选的完整可视化研究链条。

## 仓库结构

```
DV_Final/
├── Document/                 # LaTeX 报告与插图生成代码
│   ├── report.tex           # 报告源文件
│   ├── report.pdf           # 编译好的 PDF
│   ├── figures/             # 报告插图
│   │   ├── task1_mip_sequence.png
│   │   ├── task1_contours.png
│   │   ├── task2_evolution_stats.png
│   │   ├── task2_quantile_shifts.png
│   │   ├── task3_log_histogram_sequence.png
│   │   ├── task3_log_histogram_overlay.png
│   │   ├── task3_cdf_sequence.png
│   │   ├── task4_histogram_brush.jpg
│   │   ├── task4_top1_3d.jpg
│   │   ├── task4_t50_p95.png
│   │   ├── task4_t99_p99.png
│   │   ├── task4_t50_aip.png
│   │   ├── task4_slice_comparison.png
│   │   └── system_overview.jpg
│   └── generate_*.py        # 各任务插图生成脚本
│       ├── generate_task1_figures.py
│       ├── generate_task2_figures.py
│       ├── generate_task3_figures.py
│       ├── generate_task3_cdf.py
│       └── generate_task4_figs.py
│
├── data/                     # 原始数据（未提交到 git）
│   └── Nyx/                 # 100 个时间步 .dat 文件
│
├── code/                     # 数据预处理脚本
│   ├── preprocess.py        # 统计量计算与 JSON 导出
│   ├── preprocess_web_data.py   # 生成 Web 预渲染数据
│   ├── preprocess_all_visuals.py # 批量生成投影图
│   ├── generate_task1_figures.py # 任务一插图
│   ├── generate_report_figures.py  # 报告辅助图
│   ├── yt_volume_render.py  # yt 科学级体渲染
│   └── fix_histograms.py    # 直方图数据修正
│
└── web/                      # 可视化 Web 程序（前后端不分离）
    ├── app.py               # Flask 后端 API
    ├── start.sh             # 启动脚本
    ├── README.md            # Web 程序说明
    ├── templates/
    │   └── index.html       # 单页面仪表板
    └── static/
        ├── css/style.css    # 宇宙主题样式
        ├── js/app.js        # 前端逻辑（Three.js + Plotly）
        ├── images/          # 背景图
        └── data/            # 运行时数据
            ├── stats.json       # 100 步统计量
            └── histograms.json  # 100 步直方图
```

## 技术栈

- **后端**：Flask（Python），提供投影/体数据/统计 API
- **前端**：Three.js（WebGL 2.0 Ray Marching）+ Plotly
- **数据处理**：NumPy, Matplotlib, SciPy
- **报告**：LaTeX (XeLaTeX)

## 四个核心任务

### 任务一：体数据渲染
- **三维体渲染**：WebGL 2.0 + `DataTexture3D` + Ray Marching 实时光线投射
- **传递函数**：Hot / Cool / Fire / Rainbow / yt Log-Layers 五种预设
- **投影方式**：MIP（最大强度投影）与 AIP（平均强度投影）互补
- **报告插图**：MIP 演化序列 + 等值线叠加

### 任务二：密度演化特征分析
- 统计量时序演化：均值、标准差、极差、变异系数四面板
- 分位数"剪刀差"：P1/P5 下降、P95/P99/P999 上升，三层填充带展示
- 演化四阶段：均匀期 → 线性增长期 → 非线性坍缩期 → 成熟网络期

### 任务三：时序统计特征分析
- 密度对数直方图演化序列（2×3 面板）
- 六条曲线叠加对比（从窄峰到宽平台）
- CDF 时序演化 + P1/P5/P50/P95/P99 分位数竖线标注

### 任务四：相空间交互式刷选
- **直方图框选**：鼠标框选密度区间，实时联动 3D 体渲染
- **密度筛选预设**：全部 / Top 10% / Top 5% / Top 1%
- **阈值对比**：P99 见节点、P95 见纤维丝
- **投影对比**：MIP 骨架视图 vs AIP 气体填充全貌
- **报告插图**：Web 界面截图 + 切片对比

## 运行方式

### 前置依赖
```bash
pip install flask numpy matplotlib scipy
```

### 启动 Web 平台
```bash
cd web
python3 app.py
# 或
./start.sh
```

默认监听 `0.0.0.0:5000`，浏览器访问 `http://localhost:5000`。

### 生成报告插图
```bash
cd Document
python3 generate_task1_figures.py
python3 generate_task2_figures.py
python3 generate_task3_figures.py
python3 generate_task3_cdf.py
python3 generate_task4_figs.py
```

### 编译报告
```bash
cd Document
xelatex -interaction=nonstopmode report.tex
xelatex -interaction=nonstopmode report.tex
```

## 数据说明

原始 Nyx 模拟数据为 100 个时间步的 $128^3$ 宇宙密度场，每个文件约 8.4 MB（`float32` 小端序）。预处理脚本 `code/preprocess.py` 提取统计量并导出 JSON；`code/preprocess_web_data.py` 生成 Web 所需的降采样体数据（$64^3$ `uint8`）与投影 PNG。
