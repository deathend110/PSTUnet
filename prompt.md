
---

# **Figure 1：QAG-PST Fusion Cell（子模块结构图）**

## **布局结构（从左到右）**

```
[x_curr] ----\
               ---> [Concat] --> [Fusion Conv] --> [Output]
[h_prev] --> [Shift] --> [Crop Align] --> [Gate] --> [Pad] --/
```

---

## **详细分块（必须按这个拆）**

### **① 输入层（左侧）**

* 两个 3D block：

  * `Current Feature x_t (C×H×W)`（蓝色）
  * `Previous Hidden h_{t-1}`（橙色）
* 下方附：

  * `conf_t`
  * `conf_{t-1}`（灰色标量或小方块）

---

### **② Spatial Shift + Alignment（核心创新点之一）**

放在 h_prev 路径：

```
h_prev → [Spatial Shift (→ or ←, Δ pixels)] → [Valid Region Cropping]
```

视觉要点：

* 用**横向错位的block**表示 shift
* Crop 用“裁剪框”或虚线区域表示
* 标注：

  ```
  Shift Δ
  Direction: forward / backward
  ```

---

### **③ Feature Difference + Quality Injection**

中间区域：

```
|h - x|  +  (conf_{t-1} - conf_t)
```

画法：

* 两条输入：

  * `|h - x|`
  * `Quality Diff`
* 汇合到：

```
[Concat] → [Conv → ReLU → Conv → Sigmoid]
```

标注：

```
Quality-Aware Gate
```

---

### **④ Soft Gating（关键视觉重点）**

```
Gate Map (H×W)
        ↓
h_aligned × Gate
```

画法：

* 用“mask覆盖feature”的形式（半透明绿色）
* 标注：

```
Element-wise Modulation
```

---

### **⑤ Padding / Re-alignment**

```
[Filtered h] → [Padding] → h_aligned
```

视觉：

* 左/右补零（画空白块）
* 标注：

```
Spatial Padding (restore size)
```

---

### **⑥ Final Fusion**

```
Concat(x_curr, h_aligned)
        ↓
Conv + ReLU
        ↓
Output h_t
```

---

## **Figure 1 Prompt（精确版）**

*A detailed module-level diagram of a Quality-Aware Gated Spatiotemporal Fusion Cell. The cell takes current feature x_t and previous hidden state h_{t-1} with confidence scores. The previous feature undergoes spatial shift and valid-region cropping for alignment. Feature difference and confidence difference are combined to generate a soft gating map through a small convolutional network (Conv-ReLU-Conv-Sigmoid). The gate modulates the aligned feature via element-wise multiplication. The filtered feature is padded back to original resolution and fused with the current feature through convolution. Highlight spatial shift, cropping, quality-aware gating, element-wise modulation, and final fusion. Use clean vector style, blue/orange feature blocks, green gating map, gray confidence signals.*



