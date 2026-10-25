# -*- coding: utf-8 -*-
"""Diff audit: fixed file must only contain the intended changes vs the original."""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCHIVE = os.path.join(_HERE, "tools", "comfyui_workflow")

SRC_CANDIDATES = [
    r"C:\Users\玄曦雪\Downloads\deepseek_json_20261015_8e1698.json",
    os.path.join(_ARCHIVE, "deepseek_json_20261015_8e1698_original.json"),
]
FIX_CANDIDATES = [
    os.path.join(_ARCHIVE, "deepseek_json_20261015_8e1698_fixed.json"),
    r"C:\Users\玄曦雪\Downloads\deepseek_json_20261015_8e1698_fixed.json",
]
SRC = next((p for p in SRC_CANDIDATES if os.path.exists(p)), SRC_CANDIDATES[0])
FIX = next((p for p in FIX_CANDIDATES if os.path.exists(p)), FIX_CANDIDATES[0])

if not os.path.exists(SRC):
    # 原始工作流已不在（Downloads 已清理）。改为对归档的修复产物做结构自检，
    # 校验它满足 fix_h3_merge.py 的全部完整性约束。
    print("[INFO] 原始工作流已不存在：%s" % SRC)
    print("[INFO] 转为对修复产物做结构自检：%s\n" % FIX)
    with open(FIX, "r", encoding="utf-8") as f:
        wf = json.load(f)

    errs = []
    nodes = {n["id"]: n for n in wf["nodes"]}
    link_map = {}
    for l in wf["links"]:
        if l[0] in link_map:
            errs.append("duplicate link id %s" % l[0])
        link_map[l[0]] = l

    for n in wf["nodes"]:
        for oi, o in enumerate(n.get("outputs") or []):
            for lid in (o.get("links") or []):
                if lid not in link_map:
                    errs.append("node %s output %d lists unknown link %s" % (n["id"], oi, lid))
                else:
                    l = link_map[lid]
                    if l[1] != n["id"] or l[2] != oi:
                        errs.append("node %s output %d link %s mismatch" % (n["id"], oi, lid))

    for n in wf["nodes"]:
        for ii, i in enumerate(n.get("inputs") or []):
            lid = i.get("link")
            if lid is None:
                continue
            if lid not in link_map:
                errs.append("node %s input '%s' unknown link %s" % (n["id"], i.get("name"), lid))
            else:
                l = link_map[lid]
                if l[3] != n["id"] or l[4] != ii:
                    errs.append("node %s input idx %d link %s mismatch" % (n["id"], ii, lid))

    for l in wf["links"]:
        lid, s_id, s_slot, t_id, t_slot, typ = l
        if s_id not in nodes or t_id not in nodes:
            errs.append("link %s: missing node" % lid); continue
        s, t = nodes[s_id], nodes[t_id]
        if s_slot >= len(s.get("outputs") or []):
            errs.append("link %s: source slot out of range" % lid); continue
        if t_slot >= len(t.get("inputs") or []):
            errs.append("link %s: target slot out of range" % lid); continue
        so, ti = s["outputs"][s_slot], t["inputs"][t_slot]
        ot, it = so.get("type"), ti.get("type")
        if not (("*" in (ot, it, typ)) or (ot == it == typ)):
            errs.append("link %s TYPE MISMATCH: %s vs %s" % (lid, ot, it))
        if typ == "LATENT" and "widget" in ti:
            errs.append("link %s: LATENT lands on WIDGET input" % lid)
        if typ == "LATENT" and ("shift_" in str(ti.get("name")) or "steps" in str(ti.get("name"))):
            errs.append("link %s: LATENT misroute onto %s" % (lid, ti.get("name")))

    for nid in (1101, 1102, 1106, 1108):
        if nid not in nodes:
            errs.append("T8 node %d missing" % nid); continue
        wue = nodes[nid].get("properties", {}).get("widget_ue_connectable", {})
        widget_inputs = [i["name"] for i in nodes[nid].get("inputs", []) if "widget" in i]
        missing = [w for w in widget_inputs if w not in wue]
        if missing:
            errs.append("node %d: widget inputs missing from widget_ue_connectable: %s" % (nid, missing))

    if wf.get("last_link_id") != max(l[0] for l in wf["links"]):
        errs.append("last_link_id != max link id")
    if wf.get("last_node_id") != max(n["id"] for n in wf["nodes"]):
        errs.append("last_node_id != max node id")

    print("== ERRORS ==")
    print("\n".join(errs) if errs else "NONE")
    print("\n== RESULT ==")
    print("PASS" if not errs else "FAIL")
    raise SystemExit(0 if not errs else 1)

with open(SRC, "r", encoding="utf-8") as f:
    a = json.load(f)
with open(FIX, "r", encoding="utf-8") as f:
    b = json.load(f)

a_nodes = {n["id"]: n for n in a["nodes"]}
b_nodes = {n["id"]: n for n in b["nodes"]}
assert set(a_nodes) == set(b_nodes), "node id set changed!"

allowed = []
unexpected = []

for nid in a_nodes:
    na, nb = a_nodes[nid], b_nodes[nid]
    for key in set(list(na.keys()) + list(nb.keys())):
        if na.get(key) == nb.get(key):
            continue
        if key == "properties":
            added = {k: nb["properties"].get(k) for k in set(nb.get("properties", {})) - set(na.get("properties", {}))}
            removed = {k: na["properties"].get(k) for k in set(na.get("properties", {})) - set(nb.get("properties", {}))}
            changed = {k: (na["properties"].get(k), nb["properties"].get(k)) for k in set(na.get("properties", {})) & set(nb.get("properties", {})) if na["properties"].get(k) != nb["properties"].get(k)}
            if removed or changed:
                unexpected.append("node %s properties removed=%s changed=%s" % (nid, removed, changed))
            else:
                allowed.append("node %s properties + %s" % (nid, list(added)))
        elif key == "inputs":
            for i, (ia, ib) in enumerate(zip(na["inputs"], nb["inputs"])):
                if ia != ib:
                    d = {k: (ia.get(k), ib.get(k)) for k in set(list(ia.keys()) + list(ib.keys())) if ia.get(k) != ib.get(k)}
                    if set(d) == {"link"}:
                        allowed.append("node %s inputs[%d].%s: %s -> %s" % (nid, i, "link", ia.get("link"), ib.get("link")))
                    else:
                        unexpected.append("node %s inputs[%d] diff %s" % (nid, i, d))
        elif key == "outputs":
            for i, (oa, ob) in enumerate(zip(na["outputs"], nb["outputs"])):
                if oa != ob:
                    d = {k: (oa.get(k), ob.get(k)) for k in set(list(oa.keys()) + list(ob.keys())) if oa.get(k) != ob.get(k)}
                    if set(d) == {"links"}:
                        allowed.append("node %s outputs[%d].links: %s -> %s" % (nid, i, oa.get("links"), ob.get("links")))
                    else:
                        unexpected.append("node %s outputs[%d] diff %s" % (nid, i, d))
        else:
            unexpected.append("node %s field '%s' changed" % (nid, key))

if len(a["links"]) != len(b["links"]):
    added_l = [l for l in b["links"] if l[0] not in {x[0] for x in a["links"]}]
    removed_l = [l for l in a["links"] if l[0] not in {x[0] for x in b["links"]}]
    allowed.append("links array: +%d / -%d" % (len(added_l), len(removed_l)))
    for l in added_l:
        allowed.append("  added link: %s" % l)
    if removed_l:
        unexpected.append("links removed: %s" % removed_l)
else:
    for la, lb in zip(a["links"], b["links"]):
        if la != lb:
            unexpected.append("link mutated: %s -> %s" % (la, lb))

top = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
for k, (va, vb) in top.items():
    if k in ("last_link_id", "last_node_id"):
        allowed.append("%s: %s -> %s" % (k, va, vb))
    elif k in ("nodes", "links"):
        # already audited item-by-item above (node-level diff + links array +N/-0 check)
        continue
    else:
        unexpected.append("top-level '%s' changed: %s -> %s" % (k, str(va)[:80], str(vb)[:80]))

print("== ALLOWED CHANGES ==")
print("\n".join(allowed))
print("\n== UNEXPECTED CHANGES ==")
print("\n".join(unexpected) if unexpected else "NONE")
print("\n== RESULT ==")
print("PASS" if not unexpected else "FAIL")
