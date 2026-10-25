# -*- coding: utf-8 -*-
"""studio.py — 生成「录屏模式」播放器：给科普视频录制用"""
import os, io, json

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

# 旁白/字幕提示：每张卡的建议口播与节奏
NARR = {
 1:  ("开场。抛出反常识：做视频模型，主流是用 self-attention；这里我们用 6,144 个耦合振子。",
      "先给结论数字：相位余弦 0.9991，PSNR 19.84 dB，参数量只有 202 万。", 6),
 2:  ("讲清楚研究问题：注意力回答「谁该关注谁」，Kuramoto 回答「谁的相位该被谁拉动」。",
      "右边每一块都有对应的物理替代物。这不是比喻，是真的换掉了。", 8),
 3:  ("全景图走一遍。GIF 进，编解码器（冻结的，零参数）把像素变成复数场。",
      "12 层物理块在场上演化，再原路变回像素。深度 = 物理时间。", 9),
 4:  ("每个振子只有两个数：振幅 a 和相位 θ，组合成复数。",
      "6,144 个振子 × 125 个复通道 = 153 万个自由度，可训练参数才 202 万。", 8),
 5:  ("五块物理快速过一遍，这是全片的骨架。", "记住这张图，后面每一块都会展开。", 7),
 6:  ("第一块：漂移场。初始是噪声，ω 是确定的，所以目标信息只剩这一条入口。",
      "两个关键设计：距离必须 L2 归一化，力必须用 sin(θ_tar − θ)。", 10),
 7:  ("第二块：Kuramoto 同步，顶掉 self-attention。四个振子从乱转变成对齐，序参量 r 涨到 0.94。",
      "四种耦合通路：均场、Toeplitz、漂移、朗之万噪声。", 11),
 8:  ("第三块：驻波 FFN。通道混合不再靠矩阵乘，靠一条洛伦兹共振曲线——扫频穿过谁，谁被放大。",
      "驻波调制替代位置编码。必须两个残差头，否则相位不收敛。", 10),
 9:  ("第四块：OU 恢复力。振幅在 log 域乘性演化。为什么不能用加性？因为会压平重尾频谱。",
      "顺带说方案 D 的两个已知弱点：γ 一身二职、log 和 log1p 域不一致。", 10),
 10: ("第五块：TimeField。深度就是物理时间。",
      "三条课程曲线：耦合强度、温度退火、驱动频率扫频——一条热力学时间线。", 8),
 11: ("编解码器：零参数双向变换。编码 5 步，解码逆着来。",
      "NORM=10 是为了让两侧尺度对齐。往返误差 6e-7，FFT 是清白的。", 9),
 12: ("v3600 只改了一件事：codec 宽度 64 → 125，全谱。物理核心一行没动。",
      "所以这是干净的消融。天花板抬了 21 dB。", 7),
 13: ("Bug 故事：v3500 一直对着「LED 网格」伪影微调模型，进展甚微。",
      "真凶是截断不是物理。而且反直觉：低通角保留 88% 能量，伪影反而比截断更差——形状比数量致命。", 11),
 14: ("训练协议。损失是被数据统计逼出来的。",
      "69.4% 的系数是数值噪声，所以相位按功率加权、振幅用 log1p 加根号权重软化。", 9),
 15: ("结果。wcos 0.9991，频谱轮廓 0.9967。",
      "注意一个反常识：未加权平均余弦反而降了——因为模型在抬高弱系数振幅，它们的噪声相位进了平均。这是好事。", 9),
 16: ("八条承重结论，每条都由一个被记录在案的失败模式换来。",
      "这是这条线最有价值的部分。", 9),
 17: ("诚实边界：单样本过拟合，泛化未验证。",
      "但它证明了「噪声→振子→像素」这条纯物理通路走得通，而且吃掉了那 21 dB 额外余量。", 9),
 18: ("收尾：三个下一步方向 + 复现命令。",
      "把「注意」换成「共振」，把「层」换成「时间」——这条路已经走通了第一段。", 8),
}

def build():
    cards = json.load(io.open(os.path.join(BASE, "_cards.json"), encoding="utf-8"))
    items = []
    for i, name, title in cards:
        n = NARR.get(i, ("", "", 8))
        items.append({"i": i, "src": f"cards/{name}", "title": title,
                      "n1": n[0], "n2": n[1], "dur": n[2]})
    data = json.dumps(items, ensure_ascii=False)

    html = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AetherGIF v3600 · 录屏模式</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#05070E;color:#EDF3FF;overflow:hidden;
 font-family:'Inter','HarmonyOS Sans SC','PingFang SC','Microsoft YaHei',system-ui,sans-serif}
#app{display:flex;height:100%}
/* 舞台 */
#stage{flex:1;min-width:0;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;
 background:radial-gradient(ellipse at 50% 0%,#0C1224 0%,#05070E 70%)}
#frame{width:100%;aspect-ratio:16/9;position:relative;max-height:100%;
 box-shadow:0 0 0 1px #1E2A44,0 30px 90px rgba(0,0,0,.7);background:#04060D}
#frame img{width:100%;height:100%;display:block;object-fit:contain}
/* 侧栏 */
#side{width:330px;background:#080C16;border-left:1px solid #1A2438;display:flex;flex-direction:column}
#side.hidden{display:none}
#list{flex:1;overflow-y:auto;padding:10px}
.row{display:flex;gap:10px;align-items:flex-start;padding:9px 10px;border-radius:9px;cursor:pointer;
 transition:background .15s}
.row:hover{background:#111A2A}
.row.on{background:#132338;box-shadow:inset 2px 0 0 #2DD4F0}
.row .n{font-family:ui-monospace,monospace;font-size:12px;color:#2DD4F0;min-width:22px;padding-top:2px}
.row .t{font-size:12.5px;line-height:1.45;color:#A9B7D2}
.row.on .t{color:#EDF3FF}
/* 控制条 */
#ctl{padding:14px 16px;border-top:1px solid #1A2438;background:#0A0F1C}
.btn{background:#132033;border:1px solid #24314C;color:#D6E2F7;border-radius:8px;padding:7px 12px;
 font-size:12.5px;cursor:pointer;transition:.15s;font-family:inherit}
.btn:hover{background:#1B2C46;border-color:#2DD4F0;color:#fff}
.btn.pri{background:#2DD4F0;color:#05070E;font-weight:700;border-color:#2DD4F0}
.row2{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}
.lbl{font-size:11px;color:#6E7C99;margin-top:11px;letter-spacing:.4px}
select{background:#0E1626;color:#D6E2F7;border:1px solid #24314C;border-radius:7px;padding:5px 8px;
 font-size:12px;width:100%;font-family:inherit}
/* 字幕 */
#cap{position:absolute;left:50%;transform:translateX(-50%);bottom:26px;max-width:78%;
 background:rgba(4,6,13,.86);border:1px solid #24314C;border-radius:10px;padding:12px 20px;
 font-size:17px;line-height:1.6;color:#EAF1FF;backdrop-filter:blur(8px);text-align:center}
#cap b{color:#2DD4F0}
/* 角标 */
#meta{position:absolute;top:16px;left:18px;display:flex;gap:8px;align-items:center;
 background:rgba(4,6,13,.7);border:1px solid #1E2A44;border-radius:8px;padding:6px 11px;
 font-size:12px;color:#8FA3C4;font-family:ui-monospace,monospace}
#meta .rec{width:9px;height:9px;border-radius:50%;background:#FB7185;animation:blink 1.4s steps(1) infinite}
@keyframes blink{0%,50%{opacity:1}51%,100%{opacity:.15}}
#fs{position:fixed;inset:0;background:#04060D;display:none;align-items:center;justify-content:center;z-index:50}
#fs.on{display:flex}
#fs img{width:100%;height:100%;object-fit:contain}
</style></head><body>
<div id="app">
  <div id="stage">
    <div id="frame">
      <img id="cur" alt="">
      <div id="meta"><span class="rec"></span><span id="mtxt">01 / 18</span></div>
      <div id="cap"></div>
    </div>
  </div>
  <div id="side">
    <div style="padding:13px 16px;border-bottom:1px solid #1A2438">
      <div style="font-size:14px;font-weight:800;letter-spacing:-.2px">录屏模式</div>
      <div style="font-size:11.5px;color:#6E7C99;margin-top:3px">AetherGIF v3600 · 科普视频素材</div>
    </div>
    <div id="list"></div>
    <div id="ctl">
      <div class="row2">
        <button class="btn pri" id="play">▶ 播放</button>
        <button class="btn" id="prev">◀ 上一张</button>
        <button class="btn" id="next">下一张 ▶</button>
        <button class="btn" id="fsb">全屏</button>
      </div>
      <div class="lbl">每张停留（秒）</div>
      <select id="dur">
        <option value="4">4 秒</option><option value="6">6 秒</option>
        <option value="8" selected>8 秒</option><option value="10">10 秒</option>
        <option value="12">12 秒</option><option value="15">15 秒</option>
        <option value="20">20 秒</option>
      </select>
      <div class="lbl">旁白提示</div>
      <select id="narrsel">
        <option value="0">关闭字幕</option>
        <option value="1" selected>显示字幕（推荐录屏时用）</option>
      </select>
      <div class="row2" style="margin-top:11px">
        <button class="btn" id="sidetoggle">隐藏侧栏</button>
        <button class="btn" id="loop">循环本张</button>
      </div>
    </div>
  </div>
</div>
<div id="fs"><img id="fsimg" alt=""></div>
<script>
const DATA = __DATA__;
let idx = 0, playing = false, timer = null, loopOne = false, fsMode = false;

const $ = s => document.querySelector(s);
const cur = $('#cur'), cap = $('#cap'), mtxt = $('#mtxt'), list = $('#list');

DATA.forEach((d, i) => {
  const el = document.createElement('div');
  el.className = 'row';
  el.innerHTML = `<div class="n">${String(d.i).padStart(2,'0')}</div><div class="t">${d.title}</div>`;
  el.onclick = () => { go(i); };
  list.appendChild(el);
});

function go(i, keepTimer){
  idx = (i + DATA.length) % DATA.length;
  const d = DATA[idx];
  cur.src = d.src;
  if (fsMode) $('#fsimg').src = d.src;
  mtxt.textContent = `${String(d.i).padStart(2,'0')} / ${DATA.length}`;
  showCap();
  [...list.children].forEach((el,k)=>el.classList.toggle('on', k===idx));
  const act = list.children[idx]; if (act) act.scrollIntoView({block:'nearest'});
  if (!keepTimer) schedule();
}
function showCap(){
  if ($('#narrsel').value === '0'){ cap.style.display='none'; return; }
  const d = DATA[idx];
  cap.style.display='block';
  cap.innerHTML = `<b>口播：</b>${d.n1}<br><span style="color:#9AA8C4;font-size:15px">${d.n2}</span>`;
}
function schedule(){
  clearTimeout(timer);
  if (!playing) return;
  const d = DATA[idx];
  const base = +$('#dur').value;
  const dur = (d.dur && !loopOne) ? d.dur : base;
  timer = setTimeout(()=>{ if(loopOne){ schedule(); } else { go(idx+1); } }, dur*1000);
}
function setPlay(on){
  playing = on;
  $('#play').textContent = on ? '❚❚ 暂停' : '▶ 播放';
  if (on) schedule(); else clearTimeout(timer);
}
$('#play').onclick = ()=> setPlay(!playing);
$('#prev').onclick = ()=> go(idx-1);
$('#next').onclick = ()=> go(idx+1);
$('#dur').onchange = ()=> schedule();
$('#narrsel').onchange = showCap;
$('#loop').onclick = (e)=>{ loopOne=!loopOne; e.target.style.color = loopOne? '#2DD4F0':''; schedule(); };
$('#sidetoggle').onclick = ()=>{ $('#side').classList.toggle('hidden'); };
$('#fsb').onclick = ()=>{
  fsMode = !fsMode;
  $('#fs').classList.toggle('on', fsMode);
  if (fsMode) $('#fsimg').src = DATA[idx].src;
};
document.addEventListener('keydown', e=>{
  if (e.key==='ArrowRight') go(idx+1);
  if (e.key==='ArrowLeft') go(idx-1);
  if (e.key===' '){ e.preventDefault(); setPlay(!playing); }
  if (e.key==='f') $('#fsb').click();
});
go(0);
</script></body></html>"""
    html = html.replace("__DATA__", data)
    p = os.path.join(BASE, "studio.html")
    io.open(p, "w", encoding="utf-8").write(html)
    return p

if __name__ == "__main__":
    print(build())
