# Nyx 宇宙学模拟可视化分析平台 — Web 仪表盘

## 启动方式

```bash
cd /home/zhb/DV_Final/web
python3 app.py
```

或直接使用：

```bash
./start.sh
```

默认监听 `0.0.0.0:5000`，浏览器访问 `http://localhost:5000` 即可。

## 功能说明

- **体数据渲染**：WebGL 2.0 实时光线投射（Ray Marching）体渲染，支持 MIP/AIP 投影图对比，支持时间步切换
- **传递函数系统**：Hot / Cool / Fire / Rainbow / yt Log-Layers 五种预设，支持不透明度与亮度实时调节
- **对数映射模式**：勾选「对数映射」后，密度值经 `log10` 变换归一化，有效拉开低密度区域动态范围，增强宇宙空洞结构的可分辨性
- **面板透明度**：UI 面板采用毛玻璃效果（`backdrop-filter: blur`），背景宇宙星空图可透过面板，提升沉浸感
- **时序统计特征**：交互式直方图、对数直方图与累积分布函数（CDF），支持播放动画
- **相空间交互仪表盘**：
  - Three.js 三维体数据实时渲染（光线投射，支持连续介质光学积分）
  - Plotly 直方图框选刷选
  - 密度区间滑块与快捷预设（Top 1%/5%/10%）
  - 统计-空间双向联动：框选直方图区间实时联动 3D 视图

## 技术栈

- 后端：Flask（前后端不分离，模板渲染）
- 前端：Three.js + Plotly + 原生 JavaScript
- 数据：Nyx $128^3$ 宇宙密度场，100 个时间步，降采样至 $64^3$ `uint8` 体数据

## 文件结构

```
web/
├── app.py              # Flask 后端：/api/stats/{ts} /api/volume/{ts}
├── start.sh            # 一键启动脚本
├── templates/
│   └── index.html      # 单页仪表板布局（三面板 + 控制栏）
└── static/
    ├── css/style.css   # 深空暗色主题 + 玻璃态 UI
    ├── js/app.js       # 前端核心：Three.js Ray Marching + Plotly 图表
    ├── images/         # 星空背景图
    └── data/
        ├── stats.json      # 100 步统计量（均值/标准差/分位数）
        └── histograms.json # 100 步直方图（edges + hist + log_hist）
```

## 预渲染数据说明

Web 前端运行时依赖 `static/data/` 下的预计算数据：

- **stats.json**：每个时间步的均值、标准差、最值、中位数、P1/P5/P50/P95/P99/P999
- **histograms.json**：每个时间步的直方图分箱边界（edges）、频数（hist）、对数直方图（log_hist）与累积分布（cdf）

这些数据由根目录 `code/preprocess.py` 和 `code/preprocess_web_data.py` 生成。
