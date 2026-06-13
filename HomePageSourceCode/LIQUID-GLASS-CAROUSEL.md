# 液态玻璃轮播导航 - Liquid Glass Carousel

## 🎨 设计灵感

Mac 风格的液态玻璃效果 + 无限循环轮播 + 拖动交互

## ✨ 核心特性

### 1. 液态玻璃选择器（中央高亮）

```
     [Dataset1]  [Dataset2]  [Dataset3]  [Dataset4]
              ┌──────────────┐
              │ Liquid Glass │  ← 中央选择器
              │   Selector   │
              └──────────────┘
```

**视觉效果**:
- ✨ 半透明磨砂玻璃（backdrop-filter blur）
- 💎 渐变反光（内部高光）
- ✨ 闪光动画（shimmer effect）
- 🌟 柔和阴影（多层阴影叠加）
- 🎯 边缘光晕（rgba边框）

### 2. 无限循环轮盘

```
... Dataset6 Dataset7 Dataset8 Dataset1 Dataset2 Dataset3 Dataset1 Dataset2 ...
                                  ↑
                            中央选择器
```

**特点**:
- 🔄 三倍重复数据集列表
- 🔁 循环归一化算法
- ∞ 真正的无限滚动体验

### 3. 拖动交互

**鼠标/触摸拖动**:
```
用户拖动 → 背景条移动 → 实时检测中心项 → 自动切换内容
```

**动量滚动**:
```
快速滑动 → 计算速度 → 应用惯性 → 逐渐减速 → 吸附到最近项
```

**吸附效果**:
```
停止拖动 → 平滑动画 → 吸附到中心 → 确保对齐
```

---

## 🎯 交互逻辑

### 拖动流程

1. **开始拖动** (onMouseDown / onTouchStart)
   - 记录起始位置
   - 设置拖动状态
   - 初始化速度跟踪

2. **拖动中** (onMouseMove / onTouchMove)
   - 计算位移
   - 更新滚动偏移
   - 计算拖动速度
   - 检测中心数据集
   - 自动切换内容

3. **结束拖动** (onMouseUp / onTouchEnd)
   - 判断是否有足够速度
   - 有速度 → 应用动量滚动
   - 无速度 → 吸附到最近项

### 动量滚动算法

```javascript
let velocity = initialVelocity;
const friction = 0.92; // 摩擦系数

function animate() {
  velocity *= friction; // 每帧减速
  
  if (Math.abs(velocity) < 0.3) {
    snapToNearest(); // 速度过小，吸附
    return;
  }

  offset += velocity; // 应用速度
  updateContent(); // 更新内容
  
  requestAnimationFrame(animate); // 下一帧
}
```

### 吸附算法

```javascript
function snapToNearest() {
  const targetOffset = -currentIndex * ITEM_WIDTH;
  
  function animate() {
    const diff = targetOffset - currentOffset;
    
    if (Math.abs(diff) < 0.5) {
      // 到达目标
      return;
    }

    // 缓动函数 (easing)
    currentOffset += diff * 0.15;
    requestAnimationFrame(animate);
  }
}
```

### 循环归一化

```javascript
function normalizeOffset(offset) {
  const cycleWidth = datasets.length * ITEM_WIDTH;
  let normalized = offset % cycleWidth;
  
  // 保持在 [-cycleWidth/2, cycleWidth/2] 范围
  if (normalized < -cycleWidth / 2) normalized += cycleWidth;
  if (normalized > cycleWidth / 2) normalized -= cycleWidth;
  
  return normalized;
}
```

---

## 🎨 液态玻璃 CSS

### 主容器

```css
background: linear-gradient(135deg, 
  rgba(255,255,255,0.15) 0%, 
  rgba(255,255,255,0.05) 100%
);

backdrop-filter: blur(20px) saturate(180%);
-webkit-backdrop-filter: blur(20px) saturate(180%);

box-shadow: 
  0 0 0 1px rgba(255,255,255,0.1),           /* 外边框 */
  0 8px 32px rgba(0,0,0,0.12),               /* 外阴影 */
  inset 0 1px 0 rgba(255,255,255,0.3),      /* 内顶部高光 */
  inset 0 -1px 0 rgba(0,0,0,0.1);           /* 内底部阴影 */

border: 1px solid rgba(255,255,255,0.18);
border-radius: 16px;
```

### 内部光晕

```css
background: radial-gradient(
  circle at 50% 0%, 
  rgba(255,255,255,0.3), 
  transparent 70%
);
opacity: 0.6;
```

### 闪光动画

```css
@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

background: linear-gradient(90deg, 
  transparent, 
  rgba(255,255,255,0.2), 
  transparent
);

animation: shimmer 3s infinite;
```

---

## 📏 尺寸参数

```javascript
const ITEM_WIDTH = 120;      // 每个数据集宽度
const ITEM_SPACING = 20;     // 数据集间距
const TOTAL_ITEM_WIDTH = 140; // 总宽度（包含间距）
```

---

## 🎭 视觉效果

### 距离衰减

```javascript
// 根据与中心的距离调整
const distance = Math.abs(currentIndex - index);
const opacity = Math.max(0.3, 1 - distance * 0.2);
const scale = Math.max(0.75, 1 - distance * 0.1);
```

**效果**:
```
远端      中间      中心      中间      远端
0.3       0.7       1.0       0.7       0.3    ← 透明度
0.75      0.85      1.0       0.85      0.75   ← 缩放
```

### 中心项特殊样式

- 透明度：100%
- 缩放：1.0
- 文字颜色：accent-fg（强调色）
- 背景：透明（让液态玻璃显示）

### 非中心项

- 透明度：30% - 80%
- 缩放：0.75 - 0.95
- 文字颜色：muted → hover:ink
- 背景：轻微灰色

---

## 🖱️ 交互状态

### 默认状态
- 光标：`cursor: grab`
- 过渡：150ms cubic-bezier

### 拖动状态
- 光标：`cursor: grabbing`
- 过渡：0ms（跟随手指）

### 动量滚动
- 光标：`cursor: grab`
- 过渡：requestAnimationFrame（60fps）

### 吸附状态
- 光标：`cursor: grab`
- 过渡：缓动动画（easing）

---

## 🎬 动画性能

### 优化措施

1. **使用 transform**
   - `transform: translateX()` 代替 `left`
   - GPU 加速，60fps 流畅

2. **requestAnimationFrame**
   - 与浏览器刷新率同步
   - 避免掉帧

3. **条件渲染**
   - 只渲染可见项 + 左右缓冲
   - 减少 DOM 节点

4. **防抖优化**
   - 速度计算防抖
   - 避免频繁触发

---

## 🎮 用户体验

### 直观性
✅ 拖动条带，而不是拖动选择器  
✅ 中心项自动高亮  
✅ 实时内容切换  

### 流畅性
✅ 60fps 动画  
✅ 平滑的动量滚动  
✅ 柔和的吸附  

### 反馈性
✅ 拖动光标变化  
✅ 距离衰减视觉反馈  
✅ 底部显示位置信息  

### 可访问性
✅ 触摸设备支持  
✅ 鼠标操作支持  
✅ 点击数据集直接跳转  

---

## 🔧 集成到主组件

### 替换内容

**旧版（按钮导航）**:
```tsx
<button onClick={goPrev}>← Previous</button>
<div>{currentDataset}</div>
<button onClick={goNext}>Next →</button>
```

**新版（液态玻璃）**:
```tsx
<DatasetCarousel
  datasets={filteredDatasets.map(([name]) => name)}
  currentIndex={currentDatasetIndex}
  onIndexChange={setCurrentDatasetIndex}
/>
```

### 自动内容切换

```tsx
// 在 DatasetCarousel 内部
const newIndex = getCurrentDatasetIndex(offset);
if (newIndex !== currentIndex) {
  onIndexChange(newIndex); // 触发父组件更新
}
```

---

## 🎨 品牌定制

### 颜色主题

修改 `dataset-carousel.tsx` 中的样式：

```css
/* 液态玻璃颜色 */
background: "linear-gradient(...)" 

/* 可改为品牌色 */
background: "linear-gradient(135deg, 
  rgba(YOUR_COLOR, 0.15), 
  rgba(YOUR_COLOR, 0.05)
)"
```

### 尺寸调整

```javascript
const ITEM_WIDTH = 150;  // 更宽的数据集项
const ITEM_SPACING = 30; // 更大的间距
```

### 动画速度

```javascript
const friction = 0.95;  // 更慢的减速
const easing = 0.20;    // 更快的吸附
```

---

## 🐛 故障排查

### 拖动不流畅
- 检查是否使用了 `transform`
- 确认 `transition` 在拖动时为 `0ms`
- 使用 Chrome DevTools Performance 分析

### 循环不正常
- 验证 `normalizeOffset` 逻辑
- 检查三倍数据集是否正确生成
- 确认 `getCurrentDatasetIndex` 取模运算

### 液态玻璃不显示
- 确认浏览器支持 `backdrop-filter`
- 检查是否有背景内容（需要模糊对象）
- 验证 CSS 层级（z-index）

---

## 🚀 浏览器支持

| 浏览器 | backdrop-filter | 拖动 | 触摸 |
|--------|----------------|------|------|
| Chrome 90+ | ✅ | ✅ | ✅ |
| Safari 14+ | ✅ | ✅ | ✅ |
| Firefox 103+ | ✅ | ✅ | ✅ |
| Edge 90+ | ✅ | ✅ | ✅ |

### Fallback

对于不支持 `backdrop-filter` 的浏览器：
```css
@supports not (backdrop-filter: blur(20px)) {
  background: rgba(255, 255, 255, 0.8);
}
```

---

## 📱 移动端优化

### 触摸优化
- `touch-action: pan-x` 防止垂直滚动干扰
- 触摸反馈动画
- 更大的点击区域

### 性能优化
- 减少同时渲染的项数
- 使用 `will-change: transform`
- 避免复杂的阴影

---

## 🎊 最终效果

访问 http://localhost:3000/tseval/

**体验**:
1. 看到中央的液态玻璃选择器
2. 拖动条带左右移动
3. 感受动量滚动和平滑吸附
4. 观察实时内容切换

**Mac 风格特征**:
- ✨ 磨砂玻璃效果
- 💎 细腻的高光和阴影
- 🌟 优雅的动画过渡
- 🎯 精确的吸附对齐

---

🎉 享受流畅的 Mac 风格交互体验！
