# -*- coding: utf-8 -*-
"""Fix merged ComfyUI workflow: restore widget_ue_connectable for T8 nodes + full structural self-check."""
import json
import os
from collections import OrderedDict, deque, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCHIVE = os.path.join(_HERE, "tools", "comfyui_workflow")

# 原始工作流（Downloads 下已不在），脚本内已归档一份可复核副本
SRC_CANDIDATES = [
    os.path.join(_ARCHIVE, "deepseek_json_20261015_8e1698_fixed.json"),
    r"C:\Users\玄曦雪\Downloads\deepseek_json_20261015_8e1698.json",
]
SRC = next((p for p in SRC_CANDIDATES if os.path.exists(p)), SRC_CANDIDATES[0])
OUT = os.path.join(_ARCHIVE, "deepseek_json_20261015_8e1698_fixed.json")

with open(SRC, "r", encoding="utf-8") as f:
    wf = json.load(f, object_pairs_hook=OrderedDict)

# widget_ue_connectable copied verbatim from the verified deepseek workflow (file B)
fix_wue = {
    1101: ["scheduler", "shift_audio", "shift_video", "sampler_name", "steps"],
    1102: ["base_steps", "coarse_steps", "refine_steps"],
    1103: ["noise_seed"],
    1104: [],
    1105: [],
    1106: ["max_anisotropy", "model_name", "target_height", "precision", "size_mode",
           "scale_by", "target_megapixels", "release_policy", "aspect_policy", "target_width"],
    1108: ["stg_double_blocks", "shift_audio", "enable_model_time_bias", "extra_tail_steps",
           "shift_video", "stg_start_progress", "restart_video_sigma", "enable_restart",
           "bias_domain", "restart_seed", "tail_spacing", "bias_end_progress", "stg_scale",
           "bias", "enable_stg", "restart_steps", "bias_start_progress", "enable_tail",
           "stg_end_progress"],
    1109: [],
    1110: [],
    1111: [],
    1112: ["save_output", "filename_prefix", "loop_count", "format", "frame_rate", "pingpong"],
}

nodes = {n["id"]: n for n in wf["nodes"]}
report = []

# --- fix dangling latent_image reference on PASS1 (1105): add a real link 233 ---
# 1105.inputs[4].latent_image currently claims link 211, but the links array only has
# one entry for 211 targeting 1101.in[1]. B had two distinct links (7 and 15) feeding
# from the same source output; C only kept one. Recreate the second one.
link_ids = {l[0] for l in wf["links"]}
if 233 in link_ids:
    # 已是修复过的文件：幂等处理，不重复插入 link 233
    report.append("1105.latent_image: link 233 已存在，跳过重复插入")
else:
    new_link = [233, 1009, 1, 1105, 4, "LATENT"]
    wf["links"].append(new_link)
    in4 = nodes[1105]["inputs"][4]
    report.append("1105.latent_image: dangling link 211 -> new link 233 (was claiming 211)")
    in4["link"] = 233
    # source output bookkeeping
    latent_out = nodes[1009]["outputs"][1]
    if 233 not in (latent_out.get("links") or []):
        latent_out.setdefault("links", []).append(233)
    wf["last_link_id"] = max(l[0] for l in wf["links"])
    report.append("last_link_id -> %d" % wf["last_link_id"])

for nid, keys in fix_wue.items():
    n = nodes[nid]
    props = n.setdefault("properties", {})
    props["widget_ue_connectable"] = OrderedDict((k, True) for k in keys)
    report.append("node %d (%s): widget_ue_connectable <- %s" % (nid, n["type"], sorted(keys)))

errs, warns = [], []

link_map = {}
for l in wf["links"]:
    lid = l[0]
    if lid in link_map:
        errs.append("duplicate link id %s" % lid)
    link_map[lid] = l

# outputs[].links <-> links bidirectional
for n in wf["nodes"]:
    for oi, o in enumerate(n.get("outputs") or []):
        for lid in (o.get("links") or []):
            if lid not in link_map:
                errs.append("node %s output %d lists unknown link %s" % (n["id"], oi, lid))
            else:
                l = link_map[lid]
                if l[1] != n["id"] or l[2] != oi:
                    errs.append("node %s output %d lists link %s but link is [src=%s,%s]" % (n["id"], oi, lid, l[1], l[2]))

# inputs[].link <-> links bidirectional + index alignment
for n in wf["nodes"]:
    for ii, i in enumerate(n.get("inputs") or []):
        lid = i.get("link")
        if lid is None:
            continue
        if lid not in link_map:
            errs.append("node %s input '%s' references unknown link %s" % (n["id"], i.get("name"), lid))
        else:
            l = link_map[lid]
            if l[3] != n["id"] or l[4] != ii:
                errs.append("node %s input idx %d ('%s') refs link %s but it targets [%s,%s]" % (n["id"], ii, i.get("name"), lid, l[3], l[4]))

# type matching on every link
for l in wf["links"]:
    lid, s_id, s_slot, t_id, t_slot, typ = l
    if s_id not in nodes or t_id not in nodes:
        errs.append("link %s: missing node (%s -> %s)" % (lid, s_id, t_id)); continue
    s, t = nodes[s_id], nodes[t_id]
    if s_slot >= len(s.get("outputs") or []):
        errs.append("link %s: source slot %s out of range on node %s" % (lid, s_slot, s_id)); continue
    if t_slot >= len(t.get("inputs") or []):
        errs.append("link %s: target slot %s out of range on node %s" % (lid, t_slot, t_id)); continue
    so, ti = s["outputs"][s_slot], t["inputs"][t_slot]
    ot, it = so.get("type"), ti.get("type")
    ok = ("*" in (ot, it, typ)) or (ot == it == typ)
    if not ok:
        errs.append("link %s TYPE MISMATCH: %s.%s(%s) -> %s.%s(%s) link=%s" % (lid, s_id, so.get("name"), ot, t_id, ti.get("name"), it, typ))
    if typ == "LATENT" and "widget" in ti:
        errs.append("link %s: LATENT lands on WIDGET input '%s' of node %s" % (lid, ti.get("name"), t_id))

# LATENT link audit: where does every LATENT line actually land
latent_lines = []
for l in wf["links"]:
    lid, s_id, s_slot, t_id, t_slot, typ = l
    if typ == "LATENT":
        so = nodes[s_id]["outputs"][s_slot]
        ti = nodes[t_id]["inputs"][t_slot]
        latent_lines.append("  link %s: %d.%s(%s) -> %d.in[%d]='%s'" % (lid, s_id, so.get("name"), so.get("type"), t_id, t_slot, ti.get("name")))

# no LATENT line may resolve onto shift_video/shift_audio/steps etc.
for line in latent_lines:
    if "shift_" in line or "steps" in line:
        errs.append("LATENT misroute: %s" % line)

# widget_ue_connectable completeness for T8 nodes
for nid in (1101, 1102, 1106, 1108):
    n = nodes[nid]
    wue = n["properties"].get("widget_ue_connectable", {})
    widget_inputs = [i["name"] for i in n["inputs"] if "widget" in i]
    missing = [w for w in widget_inputs if w not in wue]
    if missing:
        errs.append("node %s: widget inputs missing from widget_ue_connectable: %s" % (nid, missing))

# id consistency
if wf.get("last_link_id") != max(l[0] for l in wf["links"]):
    warns.append("last_link_id %s != max %s" % (wf.get("last_link_id"), max(l[0] for l in wf["links"])))
if wf.get("last_node_id") != max(n["id"] for n in wf["nodes"]):
    warns.append("last_node_id %s != max %s" % (wf.get("last_node_id"), max(n["id"] for n in wf["nodes"])))
orders = [n.get("order") for n in wf["nodes"]]
if len(set(orders)) != len(orders):
    warns.append("duplicate order values present")

# topological order (Kahn) - ensure no cycles and execution is feasible
indeg = {n["id"]: 0 for n in wf["nodes"]}
adj = defaultdict(list)
for l in wf["links"]:
    indeg[l[3]] += 1
    adj[l[1]].append(l[3])
q = deque(sorted([nid for nid, d in indeg.items() if d == 0]))
seen = 0
while q:
    x = q.popleft(); seen += 1
    for y in sorted(adj[x]):
        indeg[y] -= 1
        if indeg[y] == 0:
            q.append(y)
if seen != len(nodes):
    errs.append("cycle detected: ordered %d/%d nodes" % (seen, len(nodes)))

print("== PROPERTIES PATCHES ==")
print("\n".join(report))
print("\n== LATENT LINK AUDIT ==")
print("\n".join(latent_lines))
print("\n== ERRORS ==")
print("\n".join(errs) if errs else "NONE")
print("\n== WARNINGS ==")
print("\n".join(warns) if warns else "NONE")

if not errs:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent="\t")
    print("\nWROTE %s" % OUT)
else:
    print("\nNOT WRITTEN due to errors")
