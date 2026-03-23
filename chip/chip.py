"""
CHIP: Cross-modal Hierarchy-Informed Projection for Multimodal Unlearning.

A training-free, hierarchy-aware multimodal unlearning method that:
    1. Collects cross-modal activations   (Eq. 1)
    2. Selects forget-relevant neurons    (s_i = I_forget,i - I_retain,i)
    3. Computes sibling-differential directions  (d = normalize(mu_target - mu_siblings))
    4. Aggregates directions via SVD
    5. Applies orthogonal weight projection      (W_S <- (I - QQ^T) W_S)

Reference:
    Wu et al., "Hierarchy-Aware Multimodal Unlearning for Medical AI", 2025.
    arXiv: 2512.09867
"""

import gc
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================================
# Section 3.1 — Hierarchy Graph
# ============================================================================

LEVEL_HIERARCHY = {
    "institution": "patient",
    "patient": "study",
    "study": "section",
    "section": None,
}


@dataclass
class HierarchyNode:
    """A node in the four-level clinical hierarchy."""
    node_id: str
    level: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    qa_pairs: List[Dict] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)


class HierarchyGraph:
    """
    Hierarchy graph modelling nested medical data:
        Institution ⊃ Patient ⊃ Study ⊃ Section
    """

    def __init__(self):
        self.nodes: Dict[str, HierarchyNode] = {}
        self.level_to_nodes: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, node: HierarchyNode):
        self.nodes[node.node_id] = node
        self.level_to_nodes[node.level].append(node.node_id)

    def get_node(self, nid: str) -> Optional[HierarchyNode]:
        return self.nodes.get(nid)

    def get_descendants(self, nid: str) -> List[HierarchyNode]:
        out, q = [], [nid]
        while q:
            cur = q.pop(0)
            node = self.get_node(cur)
            if node and cur != nid:
                out.append(node)
            if node:
                q.extend(node.children_ids)
        return out

    def get_all_qa_pairs(self, nid: str, include_descendants=True) -> List[Dict]:
        pairs = list(self.get_node(nid).qa_pairs) if self.get_node(nid) else []
        if include_descendants:
            for d in self.get_descendants(nid):
                pairs.extend(d.qa_pairs)
        return pairs


def _classify_question(question: str, hint: str = None) -> str:
    if hint and hint in ("institution", "patient", "study", "section"):
        return hint
    q = question.lower()
    if re.search(r"\binstitution\b|\bhospital\b|\bfacility\b", q):
        return "institution"
    if re.search(r"\bpatient\b|\bmedical record\b|\bpatient's\b", q):
        return "patient"
    if re.search(r"\bimaging study\b|\bstudy identifier\b|\bstudy id\b", q):
        return "study"
    return "section"


def build_hierarchy_from_parquet(df: pd.DataFrame) -> HierarchyGraph:
    """
    Build a hierarchy graph from a MedForget parquet DataFrame.

    Expected columns: image, image_path, metadata, hierarchy_metadata, ID.
    """
    g = HierarchyGraph()

    for idx, row in df.iterrows():
        path_parts = row.get("image_path", "").split("/")
        image_id = row.get("ID", f"img_{idx}")

        hm = {}
        if "hierarchy_metadata" in row.index and row["hierarchy_metadata"]:
            try:
                hm = json.loads(row["hierarchy_metadata"]) if isinstance(
                    row["hierarchy_metadata"], str) else row["hierarchy_metadata"]
            except (json.JSONDecodeError, TypeError):
                pass

        inst = hm.get("institution_id") or (path_parts[0] if len(path_parts) > 0 else f"inst_{idx}")
        pat  = hm.get("patient_id")     or (path_parts[1] if len(path_parts) > 1 else f"pat_{idx}")
        stu  = hm.get("study_id")       or (path_parts[2] if len(path_parts) > 2 else f"study_{idx}")
        sec_name = hm.get("section_name", "")
        h_level  = hm.get("hierarchy_level", "")

        img_data = row["image"].get("bytes") if isinstance(row["image"], dict) else row["image"]

        try:
            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        except (json.JSONDecodeError, TypeError):
            metadata = []

        # Build nodes
        if inst not in g.nodes:
            g.add_node(HierarchyNode(inst, "institution"))
        if pat not in g.nodes:
            g.add_node(HierarchyNode(pat, "patient", parent_id=inst))
            g.nodes[inst].children_ids.append(pat)
        if stu not in g.nodes:
            g.add_node(HierarchyNode(stu, "study", parent_id=pat))
            g.nodes[pat].children_ids.append(stu)
        if "image_bytes" not in g.nodes[stu].attributes:
            g.nodes[stu].attributes.update(image_bytes=img_data, image_path=row.get("image_path", ""), image_id=image_id)

        # Assign QA pairs
        for qi, qa in enumerate(metadata):
            question, answer = qa.get("Question", ""), qa.get("Answer", "")
            if not question or not answer:
                continue
            ql = _classify_question(question, h_level)
            rec = dict(question=question, answer=answer, image_id=image_id,
                       image_bytes=img_data, level=ql, section_name=sec_name)

            if ql == "institution":
                g.nodes[inst].qa_pairs.append(rec)
            elif ql == "patient":
                g.nodes[pat].qa_pairs.append(rec)
            elif ql == "study":
                g.nodes[stu].qa_pairs.append(rec)
            else:
                det = sec_name or next(
                    (s for s in ["EXAMINATION", "INDICATION", "FINDINGS", "IMPRESSION"]
                     if s in question.upper()), f"section_{qi}")
                sid = f"{stu}_{det}"
                if sid not in g.nodes:
                    g.add_node(HierarchyNode(sid, "section", parent_id=stu,
                                             attributes={"section_name": det}))
                    g.nodes[stu].children_ids.append(sid)
                g.nodes[sid].qa_pairs.append(rec)
    return g


# ============================================================================
# Section 4 — Cross-modal Activation Collection  (Eq. 1)
# ============================================================================

class _VQADataset(Dataset):
    def __init__(self, qa_pairs): self.qa_pairs = qa_pairs
    def __len__(self): return len(self.qa_pairs)
    def __getitem__(self, i):
        qa = self.qa_pairs[i]
        try:    img = Image.open(BytesIO(qa["image_bytes"])).convert("RGB")
        except: img = Image.new("RGB", (224, 224), "white")
        return dict(image=img, question=qa["question"], answer=qa["answer"],
                    level=qa.get("level", "section"))


def _make_collate(processor, model_type):
    """Build a collate function that also returns vision-token ranges."""
    if model_type == "Lingshu":
        from qwen_vl_utils import process_vision_info

        def collate(examples):
            msgs_list = [[
                {"role": "user", "content": [
                    {"type": "image", "image": ex["image"]},
                    {"type": "text", "text": ex["question"]}]},
                {"role": "assistant", "content": ex["answer"]}
            ] for ex in examples]

            texts, imgs = [], []
            for m in msgs_list:
                texts.append(processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False))
                ii, _ = process_vision_info(m)
                imgs.extend(ii or [])

            batch = processor(text=texts, images=imgs or None, padding=True, return_tensors="pt")
            labels = batch["input_ids"].clone()
            mk = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            vs_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            ve_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

            vranges = []
            for i in range(len(batch["input_ids"])):
                ids = batch["input_ids"][i].tolist()
                for j in range(len(ids) - len(mk) + 1):
                    if ids[j:j+len(mk)] == mk:
                        labels[i, :j+len(mk)] = -100; break
                vs = ve = 0
                for j, t in enumerate(ids):
                    if t == vs_id and vs == 0: vs = j + 1
                    elif t == ve_id and vs > 0: ve = j; break
                vranges.append((vs, ve))
            labels[labels == processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            return batch, vranges
        return collate

    else:  # Llava
        def collate(examples):
            imgs = [ex["image"] for ex in examples]
            texts = [f"USER: <image>\n{ex['question']}\nASSISTANT: {ex['answer']}" for ex in examples]
            batch = processor(text=texts, images=imgs, padding=True, truncation=True, return_tensors="pt")
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            iid = processor.tokenizer.convert_tokens_to_ids("<image>")
            vranges = []
            for i in range(len(batch["input_ids"])):
                pos = next((j for j, t in enumerate(batch["input_ids"][i].tolist()) if t == iid), None)
                vranges.append((pos, pos + 576) if pos is not None else (0, 0))
            return batch, vranges
        return collate


class _ActivationCollector:
    """
    Hooks into language MLP layers and VL merger layers.

    Language layers use cross-modal aggregation (Eq. 1):
        a = alpha * a_vision + (1 - alpha) * a_text

    Merger layers use global average pooling.
    """

    def __init__(self, model, model_type, alpha=0.3,
                 vision_text_sep=True, lang_layers=None):
        self.model, self.model_type = model, model_type
        self.alpha, self.vis_sep = alpha, vision_text_sep
        self.lang_layers = lang_layers or [22, 23, 24, 25, 26, 27]
        self.hooks, self.acts = [], defaultdict(list)
        self.vranges = None

    def _lang(self):
        if self.model_type == "Llava":
            return self.model.language_model.model.layers
        # transformers >=4.52: layers moved to model.model.language_model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            lm = self.model.model.language_model
            return lm.layers if hasattr(lm, "layers") else lm.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        return self.model.language_model.model.layers

    def _targets(self):
        layers = {}
        lang = self._lang()
        for i in self.lang_layers:
            if i < len(lang):
                m = lang[i]
                layers[f"lang_down_{i}"] = m.mlp.down_proj
                layers[f"lang_up_{i}"]   = m.mlp.up_proj
                layers[f"lang_gate_{i}"] = m.mlp.gate_proj
        if self.model_type == "Lingshu" and hasattr(self.model, "visual") \
                and hasattr(self.model.visual, "merger"):
            mlp = self.model.visual.merger.mlp
            if len(mlp) >= 3:
                layers["merger_fc1"] = mlp[0]
                layers["merger_fc2"] = mlp[2]
        return layers

    def _hook(self, name):
        def fn(_, __, out):
            if isinstance(out, tuple): out = out[0]
            if name.startswith("lang_") and out.dim() == 3:
                B, S, _ = out.shape
                if self.vis_sep and self.vranges:
                    a_list = []
                    for b in range(B):
                        vs, ve = self.vranges[b] if b < len(self.vranges) else (0, 0)
                        if 0 < vs < ve <= S:
                            a_v = out[b, vs:ve].mean(0)
                            mask = torch.ones(S, dtype=torch.bool, device=out.device)
                            mask[vs:ve] = False
                            a_t = out[b, mask].mean(0) if mask.sum() > 0 else out[b, -1]
                            a_list.append(self.alpha * a_v + (1 - self.alpha) * a_t)
                        else:
                            a_list.append(out[b].mean(0))
                    act = torch.stack(a_list).detach().cpu()
                else:
                    act = out.mean(1).detach().cpu()
            elif out.dim() == 3:
                act = out.mean(1).detach().cpu()
            else:
                act = out.detach().cpu()
            self.acts[name].append(act)
        return fn

    def register(self):
        for n, m in self._targets().items():
            self.hooks.append(m.register_forward_hook(self._hook(n)))

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def clear(self):
        self.acts = defaultdict(list); self.vranges = None

    def get(self, name):
        return torch.cat(self.acts[name], 0) if name in self.acts else None


def _collect(model, processor, graph, levels, collector, model_type,
             batch_size=4, max_per_node=500, device="cuda"):
    """Collect per-node activations at the given hierarchy levels."""
    nids = []
    for lv in levels:
        nids.extend(graph.level_to_nodes.get(lv, []))

    model.eval()
    collector.register()
    result = {}
    collate = _make_collate(processor, model_type)

    for nid in tqdm(nids, desc="  Collecting activations"):
        node = graph.get_node(nid)
        if not node: continue
        qas = graph.get_all_qa_pairs(nid, include_descendants=True)
        if not qas: continue
        if len(qas) > max_per_node:
            qas = random.Random(hash(nid) % 2**32).sample(qas, max_per_node)

        loader = DataLoader(_VQADataset(qas), batch_size=batch_size,
                            shuffle=False, collate_fn=collate)
        collector.clear()
        with torch.no_grad():
            for batch, vr in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                collector.vranges = vr
                try: model(**batch)
                except Exception as e: print(f"  Warning: {nid}: {e}")

        result[nid] = {n: collector.get(n) for n in collector.acts
                       if collector.get(n) is not None and collector.get(n).shape[0] > 0}
        if len(result) % 10 == 0:
            torch.cuda.empty_cache(); gc.collect()

    collector.remove()
    torch.cuda.empty_cache()
    return result


# ============================================================================
# Section 4 — CHIP Pipeline
# ============================================================================

class CHIP:
    """
    Cross-modal Hierarchy-Informed Projection for multimodal unlearning.

    Args:
        model: The multimodal LLM to unlearn from.
        processor: Corresponding processor / tokenizer.
        model_type: ``"Lingshu"`` (Qwen2.5-VL) or ``"Llava"``.
        top_k_percent: Percentage of neurons to select (k).
        variance_threshold: SVD variance threshold (tau, e.g. 0.95).
        alpha: Vision weight in language layer activations (Eq. 1).
        vision_text_separation: Enable cross-modal activation separation.
        lang_layers: Language layer indices for weight projection.
    """

    def __init__(self, model, processor, model_type="Lingshu", *,
                 top_k_percent=10.0, variance_threshold=0.95,
                 alpha=0.3, vision_text_separation=True, lang_layers=None):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.top_k = top_k_percent
        self.tau = variance_threshold
        self.lang_layers = lang_layers or [22, 23, 24, 25, 26, 27]
        self.device = str(next(model.parameters()).device)
        self._col = _ActivationCollector(
            model, model_type, alpha, vision_text_separation, self.lang_layers)
        self.sel: Dict[str, torch.Tensor] = {}
        self.dirs: Dict[str, List[torch.Tensor]] = defaultdict(list)

    # ----- public API -----

    def run(self, forget_graph, retain_graph, level="patient",
            batch_size=4, max_samples_per_node=500, max_targets=None):
        """Execute the full CHIP unlearning pipeline."""
        print(f"\n{'='*60}\nCHIP: Cross-modal Hierarchy-Informed Projection\n{'='*60}")
        print(f"  Level: {level}  |  top-k: {self.top_k}%  |  tau: {self.tau}"
              f"  |  alpha: {self._col.alpha}")

        child = LEVEL_HIERARCHY.get(level)
        lvls = [level] + ([child] if child else [])

        # Step 1 — activation collection
        print("\n[1/4] Collecting cross-modal activations...")
        f_act = _collect(self.model, self.processor, forget_graph, lvls,
                         self._col, self.model_type, batch_size,
                         max_samples_per_node, self.device)
        r_act = _collect(self.model, self.processor, retain_graph, lvls,
                         self._col, self.model_type, batch_size,
                         max_samples_per_node, self.device)
        print(f"  Forget nodes: {len(f_act)}, Retain nodes: {len(r_act)}")
        if not f_act:
            raise RuntimeError("No forget activations collected.")

        # Step 2 — neuron selection
        print(f"\n[2/4] Selecting top-{self.top_k}% neurons...")
        self.sel = self._select_neurons(f_act, r_act)

        # Step 3 — sibling-differential directions
        print("\n[3/4] Computing sibling-differential directions...")
        tids = [n for n in forget_graph.level_to_nodes.get(level, [])
                if n in f_act or any(c in f_act for c in
                    (forget_graph.get_node(n).children_ids
                     if forget_graph.get_node(n) else []))]
        if max_targets:
            tids = tids[:max_targets]
        print(f"  Processing {len(tids)} targets")
        for tid in tqdm(tids, desc="  Directions"):
            for nm, ds in self._directions(
                    tid, level, forget_graph, retain_graph, f_act, r_act).items():
                self.dirs[nm].extend(ds)
        print(f"  Total directions: {sum(len(v) for v in self.dirs.values())}")

        # Step 4 — SVD + projection
        print("\n[4/4] SVD aggregation and weight projection...")
        self._project()
        print(f"\n{'='*60}\nCHIP unlearning complete.\n{'='*60}")

    def save(self, save_dir):
        """Save unlearned model and metadata."""
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        state = os.path.join(save_dir, "surgery_state")
        os.makedirs(state, exist_ok=True)
        torch.save(self.sel, os.path.join(state, "selected_neurons.pt"))
        torch.save(dict(self.dirs), os.path.join(state, "all_directions.pt"))
        cfg = dict(method="CHIP", top_k_percent=self.top_k,
                   variance_threshold=self.tau, alpha=self._col.alpha,
                   vision_text_separation=self._col.vis_sep,
                   lang_layers=self.lang_layers)
        with open(os.path.join(save_dir, "chip_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Saved to {save_dir}")

    # ----- Step 2: neuron selection -----

    def _select_neurons(self, f_act, r_act):
        fa, ra = defaultdict(list), defaultdict(list)
        for d in f_act.values():
            for n, a in d.items(): fa[n].append(a)
        for d in r_act.values():
            for n, a in d.items(): ra[n].append(a)
        sel = {}
        for n in fa:
            if n not in ra: continue
            f = torch.cat(fa[n], 0); r = torch.cat(ra[n], 0)
            D = f.shape[1]; k = max(1, int(D * self.top_k / 100))
            scores = f.abs().mean(0) - r.abs().mean(0)
            sel[n] = torch.topk(scores, k).indices
            print(f"    {n}: {k}/{D} neurons")
        return sel

    # ----- Step 3: sibling-differential directions -----

    def _directions(self, tid, level, fg, rg, fa, ra):
        dirs = defaultdict(list)
        target = fg.get_node(tid)
        if not target: return dirs
        child_lv = LEVEL_HIERARCHY.get(level)

        if not target.children_ids or child_lv is None:
            if tid not in fa: return dirs
            sibs = self._sibs(tid, target, level, rg, ra)
            self._add_dirs(tid, sibs, fa, ra, dirs)
            return dirs

        for cid in target.children_ids:
            if cid not in fa: continue
            sibs = self._child_sibs(target, child_lv, rg, ra)
            if not sibs:
                sibs = [n for n, nd in rg.nodes.items()
                        if nd.level == child_lv and n in ra]
            if sibs:
                self._add_dirs(cid, sibs, fa, ra, dirs)
        return dirs

    def _sibs(self, tid, target, level, rg, ra):
        if target.parent_id:
            p = rg.get_node(target.parent_id)
            if p:
                ids = [c for c in p.children_ids if c in ra and c != tid]
                if ids: return ids
        return [n for n, nd in rg.nodes.items() if nd.level == level and n in ra]

    def _child_sibs(self, target, child_lv, rg, ra):
        ids = []
        if target.parent_id:
            gp = rg.get_node(target.parent_id)
            if gp:
                for s in gp.children_ids:
                    sn = rg.get_node(s)
                    if sn:
                        ids.extend(c for c in sn.children_ids if c in ra)
        return ids

    def _add_dirs(self, nid, sibs, fa, ra, dirs):
        for name, idx in self.sel.items():
            if name not in fa.get(nid, {}): continue
            t = fa[nid][name][:, idx]
            ss = [ra[s][name][:, idx] for s in sibs if s in ra and name in ra[s]]
            if not ss: continue
            mu_t = t.mean(0).float()
            mu_s = torch.cat(ss, 0).mean(0).float()
            d = mu_t - mu_s
            n = d.norm()
            if n > 1e-8:
                dirs[name].append(d / n)

    # ----- Step 4: SVD aggregation & orthogonal projection -----

    def _project(self):
        for name, directions in self.dirs.items():
            if not directions: continue
            idx = self.sel[name]; ns = len(idx)
            D = torch.stack(directions, 0).float()
            try:
                _, S, Vh = torch.linalg.svd(D, full_matrices=False)
                tv = (S**2).sum()
                r = (torch.cumsum(S**2, 0) < self.tau * tv).sum().item() + 1
                r = max(1, min(r, len(S)))
                Q = Vh[:r].T
                print(f"    {name}: {len(directions)} dirs -> {r} components "
                      f"({self.tau*100:.0f}% var)")
            except Exception as e:
                print(f"    {name}: SVD failed ({e}), using mean")
                Q = D.mean(0, keepdim=True).T; Q = Q / (Q.norm() + 1e-8)

            P = torch.eye(ns, dtype=torch.float32) - Q @ Q.T
            layer = self._resolve(name)
            if layer is None: continue
            P = P.to(device=layer.weight.device, dtype=layer.weight.dtype)
            with torch.no_grad():
                W = layer.weight.data
                orig = W[idx].norm().item()
                W[idx] = P @ W[idx]
                print(f"    Applied: norm {orig:.4f} -> {W[idx].norm().item():.4f}")

    def _resolve(self, name):
        lang = self._col._lang()
        if name.startswith("lang_"):
            i = int(name.split("_")[-1])
            if i >= len(lang): return None
            m = lang[i]
            if "down" in name: return m.mlp.down_proj
            if "up"   in name: return m.mlp.up_proj
            if "gate" in name: return m.mlp.gate_proj
        if name.startswith("merger_") and self.model_type == "Lingshu":
            if hasattr(self.model, "visual") and hasattr(self.model.visual, "merger"):
                mlp = self.model.visual.merger.mlp
                if name == "merger_fc1" and len(mlp) >= 1: return mlp[0]
                if name == "merger_fc2" and len(mlp) >= 3: return mlp[2]
        return None
