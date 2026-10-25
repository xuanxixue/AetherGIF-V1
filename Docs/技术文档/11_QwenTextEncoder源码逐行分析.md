# 11 · QwenTextEncoder 源码逐行分析

源文件：`src/qwen_encoder/qwen_encoder.py`（85行）
调用点：`aether_gif_model.py:99-104`（use_qwen=True 时）

---

## 1. 文件头声明（第1-7行）

```python
"""
Qwen 文本编码器：冻结 Qwen2.5-0.5B-Instruct。

该编码器仅用于提取文本 Embedding，作为 PMNN 的物理条件注入。
核心原则：Qwen 只输出连续嵌入向量，不修改物理核的任何算子结构。
"""
```

---

## 2. 导入（第8-10行）

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
```

---

## 3. `QwenTextEncoder` 类（第12-77行）

### 3.1 `__init__`（第19-37行）

```python
def __init__(self, model_path: str = "Qwen2.5-0.5B-Instruct/snapshots/master", device="cuda", freeze=True):
    super().__init__()
    self.device = device
    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    self.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    if freeze:
        self.requires_grad_(False)
        for p in self.parameters():
            p.requires_grad = False
        self.model.eval()
    # 确保需要冻结的权重被冻结
    self.model.eval()
    self.hidden_size = self.model.config.hidden_size
    print(f"Qwen 文本编码器已初始化: {model_path} (hidden_size={self.hidden_size})")
```

关键点：
- `torch_dtype=torch.float16`：fp16 加载
- `device_map="auto"`：自动设备分配
- `freeze=True`：`requires_grad_(False)` + `eval()`
- `hidden_size` 从 `model.config.hidden_size` 读取

Qwen2.5-0.5B-Instruct 的 hidden_size 实际值：896（与 `aether_gif_model.py:97` 的 `self.text_dim = 896` 一致）。

**版本兼容问题**（历史记录）：
- transformers 5.19.0 与 torch 2.5.1 不兼容，`AutoModelForCausalLM` 导入失败
- v4200 降级到 transformers 4.38.2
- tokenizers 需要匹配版本（4.38.2 要求 tokenizers>=0.19,<0.20，曾装 0.15.2 冲突）

### 3.2 `encode` 方法（第39-68行）

```python
@torch.no_grad()
def encode(self, texts, max_length=128, return_pooling=True):
    """
    编码文本为向量。
    return: [B, D] 向量
    """
    # 使用 tokenizer 分词
    inputs = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(self.device)

    outputs = self.model(**inputs, output_hidden_states=True)
    # Qwen 输出: logits 或 hidden_states
    # 取最后一个 token 的隐藏状态作为表征
    # 使用 pooling 得到固定长度
    # 注意：Qwen 模型的隐藏层维度是 512
    hidden_states = outputs.hidden_states[-1]
    # 使用 mean pooling
    attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
    if hidden_states.dim() == 3:
        # [B, L, H]
        pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-8)
    else:
        # 只有 single token，直接返回
        pooled = hidden_states.squeeze(1)
    return pooled
```

- `@torch.no_grad()`：无梯度
- `output_hidden_states=True`：取最后一层 hidden states
- mean pooling：`(hidden_states × attention_mask).sum(dim=1) / attention_mask.sum(dim=1)`
- 注释说「Qwen 模型的隐藏层维度是 512」，但 Qwen2.5-0.5B 实际 hidden_size=896。注释错误。

**注意**：`return_pooling` 参数定义了但未使用。

### 3.3 `encode_batch` 方法（第70-77行）

```python
def encode_batch(self, texts, batch_size=32):
    """批量编码"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = self.encode(batch)
        results.append(emb)
    return torch.cat(results, dim=0)
```

分批编码，默认 batch_size=32。

---

## 4. v4200 中的替代方案

v4200 不用 Qwen，改用 jina-clip-v2 双塔。根据交接文档：

**CLIP 模型**：`D:\AetherGIF-V1\models\jina-clip-v2`
- XLM-R 文本塔（561M）
- EVA-02 视觉塔（304M）
- 1024 维共享空间

**`clip_condition.py` 中的 `ClipConditionEncoder`**：
- 塔级分裂：视觉上 GPU fp16、文本留 CPU fp32
- `CondProjector`：text/image 双 proj + 可学模态门
- `build_cond_vector`：构造 2049 维条件向量

**预编码缓存**（`data/clip_cache/`）：
- `text_emb.npy`: [204, 1024]
- `image_emb.npy`: [14145, 1024]
- `image_ok.npy`
- `group_list.json`

训练时 `use_qwen=False`，不加载 Qwen 权重（500MB），也不加载 CLIP（865M 参数，fp32 权重 3.5GB，4GB 卡放不下）。CLIP 特征完全离线预编码。

---

## 5. use_qwen=False 的旁路

`aether_gif_model.py:105-107`：
```python
else:
    self.qwen = None
    self.text_param = nn.Parameter(torch.randn(1, self.text_dim, device=device) * 0.02)
```

用一个可学习条件向量 `text_param` [1, 896] 代替文本嵌入。`aether_gif_model.py:167`：
```python
text_embs = self.text_param.expand(len(texts), -1).to(self.device)
```

这是「验证阶段可旁路」的设计：跳过 500MB 权重加载与前向，专注验证物理核心能否生成 GIF。

---

## 6. Qwen 编码器的已知问题

| 问题 | 详情 | 状态 |
|---|---|---|
| 版本兼容 | transformers 5.19.0 与 torch 2.5.1 不兼容 | v4200 降级到 4.38.2 |
| tokenizers 冲突 | 4.38.2 要求 tokenizers>=0.19,<0.20 | 曾装 0.15.2，已卸载重装 0.19.1 |
| fp16 dtype | 权重 fp16，输入需匹配 | 需确保 dtype 一致 |
| hidden_size 注释错误 | 注释说 512，实际 896 | 未修正 |
| return_pooling 未使用 | 参数定义未使用 | 未清理 |
| device_map="auto" | 自动分配可能到 CPU | 4GB 卡上 500MB fp16 模型可放 GPU |

---

## 7. 未验证项

1. Qwen 编码的文本特征质量（mean pooling vs last token）无对比实验。
2. `text_param` 旁路模式下，物理核的生成能力无完整验证（v4200 走 CLIP 预编码，不用 text_param）。
3. jina-clip-v2 的 1024 维特征与 Qwen 的 896 维特征的物理条件注入效果无对比实验。
4. CLIP 预编码的完整性（14145 张首帧图像全部成功编码）无验证记录。
5. `ClipConditionEncoder` 的「视觉上 GPU fp16、文本留 CPU fp32」塔级分裂策略的实现在 v4200 代码中，本文档未展开分析（见 `src/data/clip_condition.py`）。

---

## 8. 代码事实汇总

| 项 | 值 | 来源 |
|---|---|---|
| 模型 | Qwen2.5-0.5B-Instruct | qwen_encoder.py:19 |
| torch_dtype | float16 | qwen_encoder.py:25 |
| device_map | "auto" | qwen_encoder.py:27 |
| freeze | True | qwen_encoder.py:21, 29-33 |
| max_length | 128 | qwen_encoder.py:40 |
| pooling | mean（attention_mask 加权） | qwen_encoder.py:61-64 |
| hidden_size | model.config.hidden_size（实际896） | qwen_encoder.py:36 |
| encode_batch | 32 | qwen_encoder.py:70 |
| no_grad | @torch.no_grad() | qwen_encoder.py:39 |
| text_dim（旁路） | 896 | aether_gif_model.py:97 |
| text_param 初始化 | randn × 0.02 | aether_gif_model.py:107 |
