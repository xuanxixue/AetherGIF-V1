# -*- coding: utf-8 -*-
"""
Qwen 文本编码器：冻结 Qwen2.5-0.5B-Instruct。

该编码器仅用于提取文本 Embedding，作为 PMNN 的物理条件注入。
核心原则：Qwen 只输出连续嵌入向量，不修改物理核的任何算子结构。
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenTextEncoder(nn.Module):
    """
    冻结的 Qwen 编码模型。
    输入：文本字符串列表
    输出：[B, L, D] 或 [B, D] 固定长度向量
    """

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

    def encode_batch(self, texts, batch_size=32):
        """批量编码"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self.encode(batch)
            results.append(emb)
        return torch.cat(results, dim=0)

if __name__ == "__main__":
    # 快速测试
    enc = QwenTextEncoder()
    texts = ["一个线条表情包", "简单的动画gif", " minimalist style"]
    emb = enc.encode(texts)
    print(emb.shape)
