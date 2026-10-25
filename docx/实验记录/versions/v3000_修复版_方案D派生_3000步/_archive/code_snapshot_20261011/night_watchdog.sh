#!/bin/bash
# 【夜间自动看护】等待 3000 步训练结束 → 自动跑验收 → 写报告
# 用法：bash scripts/night_watchdog.sh <train_task_marker>
# 不依赖 sleep 轮询训练进程，而是等待 ckpt 出现且稳定后再验收。

BASE="D:/AetherGIF-V1/docx/实验记录/versions/v3000_修复版_方案D派生_3000步"
OUT="$BASE/outputs_real/v3000J"
LOG="$BASE/outputs_real/auto_accept_report.txt"
PY="C:/Python312/python.exe"

echo "[WATCHDOG] 开始看护，等待 $OUT/overfit_ckpt.pt ..." > "$LOG"
date >> "$LOG"

# 等训练结束标志：[CMP] 行出现 = 训练+出图+保存全部完成
while true; do
  if grep -q "\[CMP\]" "$BASE/outputs_real/train_v3000J_full.txt" 2>/dev/null; then
    echo "[WATCHDOG] 训练已完成（检测到 [CMP] 标记）" >> "$LOG"
    break
  fi
  # 兜底：若训练进程消失但有 best ckpt，也视为结束
  if ! tasklist 2>/dev/null | grep -qi "python"; then
    echo "[WATCHDOG] python 进程已退出，执行验收" >> "$LOG"
    break
  fi
  sleep 60
done

sleep 5
echo "" >> "$LOG"
echo "=================== 自动验收报告 ===================" >> "$LOG"
date >> "$LOG"

# 优先用 best ckpt（对比度最接近 target 的那一步）
CKPT="$OUT/overfit_ckpt_best.pt"
[ -f "$CKPT" ] || CKPT="$OUT/overfit_ckpt.pt"

"$PY" "$BASE/scripts/auto_accept.py" --ckpt "$CKPT" --gif "$OUT/overfit_honest.gif" \
  --json "$OUT/accept_result.json" >> "$LOG" 2>&1
RC=$?

echo "" >> "$LOG"
echo "验收退出码 = $RC  (0=PASS, 1=FAIL)" >> "$LOG"
echo "[WATCHDOG] 报告路径：$LOG" >> "$LOG"
exit $RC