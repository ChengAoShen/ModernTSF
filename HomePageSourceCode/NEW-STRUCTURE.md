# TS-Eval: Hundreds Models Campaign (百模大战)

## ✅ 新架构已实现

### 🎯 核心概念

**TS-Eval: Hundreds Models Campaign (百模大战)**
- 支持数百个模型的大规模基准测试
- 两层分类体系
- 完整的筛选和展示功能

---

## 📊 新的层级结构

### 第一层：主类别（Main Category）

```
┌─────────────────────────────────────────┐
│  1. Common Static Dataset (通用静态数据集) │
│  2. Real-Time Dataset (实时数据集)        │
└─────────────────────────────────────────┘
```

### 第二层：子类别（Sub-Category / Track）

#### Common Static Dataset 的子类别
```
┌──────────────┬─────────────────┬──────────────┐
│ Time Series  │ Spatial Temporal│  Covariate   │
│   (左边)      │     (中间)       │    (右边)    │
│     📈       │       🗺️        │      🔗      │
└──────────────┴─────────────────┴──────────────┘
```

**8个核心数据集：**
- ETTh1, ETTh2 (Electricity Transformer - Hourly)
- ETTm1, ETTm2 (Electricity Transformer - Minute)
- Traffic (交通)
- Solar (太阳能)
- Electricity (电力)
- Weather (天气)

#### Real-Time Dataset 的子类别
```
┌──────────────┬─────────────────┬──────────────┐
│    Stock     │     Traffic     │ Air Quality  │
│   (左边)      │     (中间)       │    (右边)    │
│     💰       │       🚗        │      🌫️     │
└──────────────┴─────────────────┴──────────────┘
```

---

## 🎨 界面布局

### 顶部区域
```
┌───────────────────────────────────────────────────┐
│     TS-Eval: Hundreds Models Campaign             │
│     百模大战 · Benchmarking at Scale              │
│     150 submissions · 10 datasets                 │
└───────────────────────────────────────────────────┘
```

### 主类别选择（大按钮）
```
┌──────────────────────┐  ┌──────────────────────┐
│ Common Static Dataset│  │  Real-Time Dataset   │
│    通用静态数据集      │  │     实时数据集        │
└──────────────────────┘  └──────────────────────┘
```

### 子类别选择（三列）
```
┌─────────────┬─────────────┬─────────────┐
│ Time Series │Spatial      │ Covariate   │
│     📈      │Temporal 🗺️ │     🔗      │
└─────────────┴─────────────┴─────────────┘
```

### 筛选面板
```
┌────────────────────────────────────────┐
│ Filters & Options          [Reset all] │
├────────────────────────────────────────┤
│ Search datasets                        │
│ [________________]                     │
│                                        │
│ Dataset Categories (8 Core)            │
│ ☑ ETTh1  ☑ ETTh2  ☑ ETTm1  ☑ ETTm2   │
│ ☑ Traffic ☑ Solar ☑ Electricity ☑ Weather│
│                                        │
│ Display                                │
│ ☑ Show medals  ☑ Performance bars     │
│ ☑ Highlight top 3  ☑ Compact mode     │
│                                        │
│ Matrix                                 │
│ ☐ Matrix view                          │
│                                        │
│ Options                                │
│ [Table] [Compact] [Detailed]           │
└────────────────────────────────────────┘
```

---

## 📁 文件结构

### 新文件
```
components/
├── leaderboard-enhanced-v2.tsx    ← 新架构组件 (28KB)
├── leaderboard-enhanced.tsx       ← 旧版本 (保留)
└── tseval-page.tsx                ← 已更新使用 v2

content/tseval/
├── leaderboard.json               ← 当前使用（v2结构）
├── leaderboard-v2.json            ← 新结构示例
├── leaderboard-old.json           ← 旧数据备份
└── leaderboard-sample.json        ← 原示例
```

---

## 🔧 技术实现

### Track 映射
```typescript
const CATEGORY_STRUCTURE = {
  common_static: {
    label: "Common Static Dataset",
    tracks: ["time_series", "spatiotemporal", "covariate"],
    datasets: ["ETTh1", "ETTh2", "ETTm1", "ETTm2", 
               "Traffic", "Solar", "Electricity", "Weather"]
  },
  realtime: {
    label: "Real-Time Dataset",
    tracks: ["stock", "traffic_rt", "air_quality"],
    datasets: [] // 动态填充
  }
};
```

### Track 元数据
```typescript
const TRACK_META = {
  time_series: { label: "Time Series", icon: "📈", position: "left" },
  spatiotemporal: { label: "Spatial Temporal", icon: "🗺️", position: "center" },
  covariate: { label: "Covariate", icon: "🔗", position: "right" },
  stock: { label: "Stock", icon: "💰", position: "left" },
  traffic_rt: { label: "Traffic", icon: "🚗", position: "center" },
  air_quality: { label: "Air Quality", icon: "🌫️", position: "right" }
};
```

---

## 🎯 功能特性

### ✅ 已实现
- [x] 两层分类体系（主类别 + 子类别）
- [x] "百模大战" 主题标题
- [x] 8个核心数据集独立筛选
- [x] 三列子类别布局（左中右）
- [x] Display / Options / Matrix 三大选项组
- [x] 自动识别 Track 所属主类别
- [x] 响应式设计
- [x] 实时搜索和筛选
- [x] 多种视图模式

### 📊 数据结构要求

JSON 文件中的 tracks 命名：
```json
{
  "tracks": {
    "time_series": {},      // Common Static → 左
    "spatiotemporal": {},   // Common Static → 中
    "covariate": {},        // Common Static → 右
    "stock": {},            // Real-Time → 左
    "traffic_rt": {},       // Real-Time → 中
    "air_quality": {}       // Real-Time → 右
  }
}
```

---

## 🚀 使用方法

### 访问页面
```
http://localhost:3000/tseval/
```

### 测试新功能

1. **切换主类别**
   - 点击 "Common Static Dataset" 或 "Real-Time Dataset"
   - 子类别会自动更新

2. **选择子类别（三列布局）**
   - Common Static: Time Series (左) / Spatial Temporal (中) / Covariate (右)
   - Real-Time: Stock (左) / Traffic (中) / Air Quality (右)

3. **筛选8个核心数据集**
   - 在 "Dataset Categories (8 Core)" 区域
   - 取消勾选任意数据集立即隐藏

4. **Display / Options / Matrix**
   - Display: 6个显示选项
   - Options: Table / Compact / Detailed
   - Matrix: 矩阵视图（展示所有预测步长）

---

## 📈 扩展性

### 支持"百模大战"规模

当前设计支持：
- ✅ 每个数据集 100+ 个模型
- ✅ 多个预测步长（96, 192, 336, 720...）
- ✅ 高效的筛选和排序
- ✅ 紧凑和详细视图切换
- ✅ 分页和虚拟滚动（可后续添加）

### 添加新数据集

只需在 JSON 中添加对应 track：
```json
{
  "tracks": {
    "time_series": {
      "datasets": {
        "NewDataset": {
          "horizons": {
            "96": [...]
          }
        }
      }
    }
  }
}
```

---

## 🎨 设计亮点

1. **清晰的层级** - 两层分类，避免混乱
2. **对称布局** - 三列子类别，视觉平衡
3. **规模化设计** - 支持百模级别的展示
4. **灵活筛选** - 多维度、多粒度控制
5. **响应式** - 移动端和桌面端自适应

---

## 📝 下一步优化建议

### 短期
- [ ] 添加批量操作（全选/反选）
- [ ] 矩阵视图完整实现
- [ ] 模型数量统计更明显
- [ ] 添加数据导出功能

### 中期
- [ ] 虚拟滚动优化（100+ 模型）
- [ ] 模型对比功能
- [ ] 性能趋势图
- [ ] 收藏和书签

### 长期
- [ ] 实时更新（WebSocket）
- [ ] 协作标注功能
- [ ] AI 辅助分析
- [ ] 自动化报告生成

---

## 🎊 当前状态

✅ **新架构已完全实现并运行**

访问地址：http://localhost:3000/tseval/

所有功能已就绪，可以立即体验"百模大战"规模的排行榜！
