# 增强型排行榜前端

为 ModernTSF 排行榜创建的交互式前端界面，具有丰富的筛选和显示选项。

## 功能特性

### 数据集分类

排行榜自动将数据集分为以下类别：

#### 静态数据集（8个核心基准）
- ETTh1, ETTh2, ETTm1, ETTm2
- Weather（天气）
- Electricity（电力）
- Traffic（交通）
- Solar（太阳能）

#### 动态数据集
1. **股票预测**
   - Exchange（汇率）
   - FRED-MD（宏观经济）
   - NN5（现金提取）

2. **交通预测**
   - METR-LA
   - PEMS-BAY
   - PEMS03, PEMS04, PEMS07, PEMS08

3. **空气质量预测**
   - cauair（CauAir数据集）
   - Beijing-Air（北京空气质量）
   - AQShunyi（顺义区空气质量）
   - AQWan（万柳区空气质量）

每个数据集卡片显示彩色标签，指示其所属类别。

### 筛选功能

#### 数据集类别筛选
- 独立切换每个类别的可见性
- 实时显示每个类别中的数据集数量
- 每个类别都有说明文字

#### 搜索功能
- 实时文本搜索数据集名称
- 大小写不敏感匹配
- 无结果时显示清晰的视觉反馈

#### 显示选项
六个交互式开关控制可视化：

1. **显示奖牌** — 为前3名显示 🥇🥈🥉
2. **性能条** — 在数值下方显示可视化指标条
3. **高亮前3名** — 为领先模型添加强调背景色
4. **紧凑模式** — 减少间距以实现密集视图
5. **提交ID** — 显示运行标识符（详细视图中）
6. **按排名着色** — 渐变着色（前3名为金/银/铜色）

### 视图模式

三种视图模式优化不同使用场景：

- **表格** — 完整的电子表格样式表格，显示所有指标
- **紧凑** — 基于卡片的布局，针对移动设备优化
- **详细** — 带有可展开行的表格，显示提交元数据

### 排序功能

所有列都可排序：
- **模型** — 按字母顺序
- **MSE** — 升序（越低越好）或降序
- **MAE** — 升序（越低越好）或降序
- **运行次数** — 每个模型的提交数量

点击列标题进行排序；再次点击反转方向。激活的排序显示 ↑/↓ 指示器。

### 赛道切换

顶部标签页在预测赛道之间切换：
- 时间序列（Time Series）
- 时空（Spatiotemporal）
- 协变量（Covariate）
- 实时（RealTime）

每个赛道保持独立的筛选和排序状态。

### 预测步长选择

每个数据集卡片显示可用的预测步长（例如：96, 192, 336, 720步）。点击可在保留筛选器的同时切换步长。

## 文件结构

```
HomePageSourceCode/
├── components/
│   ├── leaderboard-enhanced.tsx    # 主组件（710行）
│   ├── tseval-page.tsx              # 页面包装器
│   ├── LEADERBOARD.md               # 英文文档
│   └── leaderboard.tsx              # 原始简单版本（保留）
├── content/
│   └── tseval/
│       ├── leaderboard.json         # 实际数据
│       └── leaderboard-sample.json  # 示例数据
├── lib/
│   └── dictionaries.ts              # 国际化文本
└── app/
    └── (en)/tseval/
        └── page.tsx                 # 路由入口
```

## 技术实现

### 核心技术栈
- **React 19** — 使用 hooks（useState, useMemo）
- **TypeScript** — 完整类型安全
- **Tailwind CSS 4** — 响应式样式
- **Next.js 15** — 服务端渲染

### 状态管理
组件使用15个独立的状态变量，精细控制每个功能：
- track, sortKey, sortDir, viewMode
- searchQuery
- showStatic, showDynamicStock, showDynamicTraffic, showDynamicAir, showOther
- showMedals, showVisualization, highlightTop3, compactMode, showSubmissionIds, colorByPerformance

### 性能优化
- 使用 `useMemo` 缓存筛选和统计结果
- 避免不必要的重新渲染
- 高效的列表渲染

### 响应式设计
- 移动优先设计
- 筛选面板在小屏幕上自动调整布局
- 窄视口下表格可横向滚动
- 触摸优化的点击目标

## 使用方法

### 基本集成

组件导入和基本使用：

```tsx
import { LeaderboardEnhanced } from "@/components/leaderboard-enhanced";
import type { LeaderboardData } from "@/components/leaderboard-enhanced";

export function MyPage() {
  return (
    <LeaderboardEnhanced 
      data={leaderboardData} 
      dict={dictionary} 
    />
  );
}
```

### 数据格式

期望的数据结构：

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-06-13T12:00:00Z",
  "primary_metric": "mse",
  "n_submissions": 48,
  "tracks": {
    "time_series": {
      "datasets": {
        "ETTh1": {
          "horizons": {
            "96": [
              {
                "model": "DLinear",
                "mse": 0.375,
                "mae": 0.398,
                "n_runs": 3,
                "submission_ids": ["run_1", "run_2"],
                "rank": 1
              }
            ]
          }
        }
      }
    }
  }
}
```

## 自定义扩展

### 添加新的数据集类别

编辑 `leaderboard-enhanced.tsx` 中的常量：

```typescript
const STATIC_DATASETS = {
  core: ["ETTh1", "ETTh2", ...],
  // 添加新类别
  new_category: ["Dataset1", "Dataset2"],
};

const DYNAMIC_DATASETS = {
  stock: ["Exchange", ...],
  traffic: ["METR-LA", ...],
  air_quality: ["cauair", ...],
  // 添加新的动态类别
  energy: ["Solar", "Wind"],
};
```

### 修改类别颜色

在 `DatasetCard` 组件中修改 `categoryBadge` 对象：

```typescript
const categoryBadge = {
  static: { 
    label: "静态", 
    color: "bg-blue-500/10 text-blue-600 dark:text-blue-400" 
  },
  "dynamic-stock": { 
    label: "股票", 
    color: "bg-green-500/10 text-green-600 dark:text-green-400" 
  },
  // 添加新颜色方案
};
```

## 可访问性

- ✅ 语义化 HTML 结构
- ✅ ARIA 标签
- ✅ 键盘导航支持
- ✅ 所有交互元素的焦点指示器
- ✅ 颜色对比度符合 WCAG AA 标准
- ✅ 屏幕阅读器友好

## 浏览器兼容性

支持所有现代浏览器：
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- 移动端 Safari/Chrome

## 开发与测试

### 本地开发

```bash
cd HomePageSourceCode
npm install  # 或 bun install
npm run dev
```

访问 http://localhost:3000/tseval 查看排行榜。

### 构建生产版本

```bash
npm run build
npm start
```

### 部署到 Cloudflare Pages

```bash
npm run deploy
```

## 更新日志

### 2026-06-13
- ✨ 创建增强型排行榜组件
- ✨ 添加数据集分类（静态/动态）
- ✨ 实现15种交互式控制选项
- ✨ 支持3种视图模式
- ✨ 完整的搜索和筛选功能
- ✨ 响应式设计和暗色模式支持
- 📝 添加完整文档

## 相关资源

- [英文文档](./components/LEADERBOARD.md)
- [ModernTSF 主仓库](https://github.com/Diaugeia/ModernTSF)
- [Diaugeia 官网](https://diaugeia.ai)

## 许可证

与 ModernTSF 主项目保持一致。
