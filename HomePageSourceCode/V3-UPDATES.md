# V3 更新：横向切换 + Metrics（度量指标）

## ✅ 修复的问题

### 1. Matrix → Metrics ✅
**问题**: 之前显示 "Matrix View"（矩阵视图）是不对的  
**修复**: 改为 "Metrics"（度量指标），用于选择显示哪些评估指标

### 2. 横向切换数据集 ✅
**问题**: 之前竖着堆叠所有数据集，当每个数据集有100+模型时会非常长  
**修复**: 改为横向切换，一次只显示一个数据集，使用左右按钮切换

---

## 🎯 新的界面设计

### 横向数据集导航

```
┌─────────────────────────────────────────────────────┐
│  [← Previous]     ETTh1 (Dataset 1 of 8)    [Next →]│
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  ETTh1                            120 models         │
│  [pred 96] [pred 192] [pred 336] [pred 720]         │
├─────────────────────────────────────────────────────┤
│  #  Model         MSE      MAE     Runs              │
│  🥇 DLinear      0.3750   0.3980   3×                │
│  🥈 PatchTST     0.3820   0.4050   2×                │
│  🥉 iTransformer 0.3910   0.4120   1×                │
│  4  TimesNet     0.4050   0.4210   1×                │
│  5  FEDformer    0.4120   0.4280   1×                │
│  ... (115 more models)                               │
└─────────────────────────────────────────────────────┘
```

### Metrics 选项（新增）

```
┌─────────────────────────────────────────┐
│ Metrics                                 │
├─────────────────────────────────────────┤
│ ☑ MSE    - Mean Squared Error           │
│ ☑ MAE    - Mean Absolute Error          │
│ ☑ Runs   - Number of runs               │
└─────────────────────────────────────────┘
```

**功能**:
- 动态显示/隐藏表格中的列
- 取消勾选 MSE → MSE 列消失
- 取消勾选 MAE → MAE 列消失
- 取消勾选 Runs → Runs 列消失

---

## 🎨 完整的筛选面板结构

```
┌────────────────────────────────────────┐
│ Filters & Options      [Reset all]     │
├────────────────────────────────────────┤
│ ▼ Search datasets                      │
│   [_________________________]          │
│                                        │
│ ▼ Dataset Categories (8 Core)         │
│   ☑ ETTh1   ☑ ETTh2   ☑ ETTm1  ☑ ETTm2│
│   ☑ Traffic ☑ Solar   ☑ Elec.  ☑ Weather│
│                                        │
│ ▼ Display (6 options)                 │
│   ☑ Show medals                        │
│   ☑ Performance bars                   │
│   ☑ Highlight top 3                    │
│   ☑ Compact mode                       │
│   ☑ Submission IDs                     │
│   ☑ Color by rank                      │
│                                        │
│ ▼ Metrics (3 options) ⭐NEW           │
│   ☑ MSE    - Mean Squared Error        │
│   ☑ MAE    - Mean Absolute Error       │
│   ☑ Runs   - Number of runs            │
│                                        │
│ ▼ Options (3 view modes)              │
│   [Table] [Compact] [Detailed]         │
└────────────────────────────────────────┘
```

---

## 🔄 横向导航的优势

### 问题场景
假设有这样的规模：
- 8 个数据集
- 每个数据集 100+ 个模型
- 每个模型占一行

**旧设计（竖向堆叠）**:
```
ETTh1
  - Model 1
  - Model 2
  ...
  - Model 120    ← 太长了！

ETTh2
  - Model 1
  ...
  - Model 150    ← 继续变长！

... (6 more datasets)

总计：800+ 行，需要滚动很久
```

**新设计（横向切换）**:
```
显示当前数据集 (例如 ETTh1)
  - Model 1
  - Model 2
  ...
  - Model 120

[← Previous] [Next →] 切换到其他数据集
总计：每次只显示 ~120 行
```

### 优势
✅ **更专注** - 一次只看一个数据集  
✅ **更快速** - 渲染速度快（少100+个DOM节点）  
✅ **更清晰** - 避免长页面迷失方向  
✅ **更灵活** - 支持百模大战规模  

---

## 💡 使用方式

### 切换数据集

1. **下一个数据集**
   - 点击右侧 "Next →" 按钮
   - 或者使用键盘快捷键（可添加）

2. **上一个数据集**
   - 点击左侧 "← Previous" 按钮
   - 到达第一个时按钮禁用

3. **快速导航**
   - 顶部显示 "Dataset X of Y"
   - 中间显示当前数据集名称

### 选择显示的度量指标

**场景 1: 只关注 MSE**
```
☑ MSE
☐ MAE     ← 取消勾选
☐ Runs    ← 取消勾选

结果：表格只显示 Model 和 MSE 列
```

**场景 2: 只看模型名称和运行次数**
```
☐ MSE
☐ MAE
☑ Runs

结果：表格显示 # / Model / Runs
```

**场景 3: 全部显示（默认）**
```
☑ MSE
☑ MAE
☑ Runs

结果：显示所有列
```

---

## 📊 对比表

| 功能 | V2（旧版） | V3（新版） |
|------|-----------|-----------|
| 数据集显示 | 全部竖向堆叠 | 横向切换 ✅ |
| 页面长度 | 800+ 行（8×100） | ~120 行 ✅ |
| 度量选项 | Matrix View ❌ | Metrics ✅ |
| MSE/MAE 切换 | 不支持 | 支持 ✅ |
| 导航方式 | 滚动 | 按钮切换 ✅ |
| 性能 | 慢（大DOM） | 快 ✅ |

---

## 🎯 技术实现

### 状态管理

```typescript
// 当前显示的数据集索引
const [currentDatasetIndex, setCurrentDatasetIndex] = useState(0);

// 当前数据集
const currentDataset = filteredDatasets[currentDatasetIndex];

// 切换函数
const goToNextDataset = () => {
  setCurrentDatasetIndex(prev => Math.min(filteredDatasets.length - 1, prev + 1));
};

const goToPrevDataset = () => {
  setCurrentDatasetIndex(prev => Math.max(0, prev - 1));
};
```

### Metrics 状态

```typescript
const [showMSE, setShowMSE] = useState(true);
const [showMAE, setShowMAE] = useState(true);
const [showRuns, setShowRuns] = useState(true);
```

### 动态列显示

```tsx
<thead>
  <tr>
    <Th>#</Th>
    <SortTh label="Model" ... />
    {showMSE && <SortTh label="MSE" ... />}
    {showMAE && <SortTh label="MAE" ... />}
    {showRuns && <SortTh label="Runs" ... />}
  </tr>
</thead>
```

---

## 📁 文件更新

```
components/
├── leaderboard-enhanced-v3.tsx    ← 新版本 (28KB) ✨
├── leaderboard-enhanced-v2.tsx    ← 旧版本 (保留)
└── tseval-page.tsx                ← 已更新使用 v3 ✅

文档/
└── V3-UPDATES.md                  ← 本文件
```

---

## 🚀 立即体验

访问: http://localhost:3000/tseval/

### 测试清单

□ 1. 看到横向导航
   - ✓ [← Previous] 按钮在左侧
   - ✓ 数据集名称在中间
   - ✓ [Next →] 按钮在右侧
   - ✓ 显示 "Dataset X of Y"

□ 2. 切换数据集
   - ✓ 点击 Next → 切换到下一个
   - ✓ 点击 Previous ← 回到上一个
   - ✓ 第一个时 Previous 禁用
   - ✓ 最后一个时 Next 禁用

□ 3. Metrics 选项
   - ✓ 看到 "Metrics" 标题（不是 Matrix）
   - ✓ 3个复选框：MSE / MAE / Runs
   - ✓ 每个都有描述文字

□ 4. 动态列显示
   - ✓ 取消 MSE → MSE 列消失
   - ✓ 取消 MAE → MAE 列消失
   - ✓ 取消 Runs → Runs 列消失
   - ✓ 重新勾选 → 列重新出现

□ 5. 性能测试
   - ✓ 页面加载快速
   - ✓ 切换数据集流畅
   - ✓ 没有长时间的滚动

---

## 🎊 更新总结

✅ **问题 1 已修复**: Matrix → Metrics（度量指标）  
✅ **问题 2 已修复**: 竖向堆叠 → 横向切换  
✅ **性能优化**: 减少 DOM 节点，提升渲染速度  
✅ **用户体验**: 更专注，更清晰  
✅ **扩展性**: 支持百模大战规模  

---

## 📝 下一步建议

### 短期优化
- [ ] 添加键盘快捷键（← → 切换数据集）
- [ ] 添加数据集快速跳转下拉菜单
- [ ] 显示当前数据集的模型统计信息

### 中期功能
- [ ] 数据集对比模式（并排显示2个数据集）
- [ ] 收藏/书签特定数据集
- [ ] 导出当前数据集结果

### 长期规划
- [ ] 虚拟滚动（优化100+模型的渲染）
- [ ] 数据集缩略图预览
- [ ] 数据集之间的性能对比图表

---

🎉 V3 版本已完全部署并运行！
访问 http://localhost:3000/tseval/ 立即体验新功能！
