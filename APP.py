# APP2.py — Carton Designer + Paper Spec + Pallet + PDF Export (Thai font fixed)
# Run: streamlit run APP2.py

import math, os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# === ReportLab for PDF ===
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ──────────────────────────────────────────────────────────────────────────────
# Thai Font Register (auto-detect NotoSansThai/Sarabun/TH Sarabun New)
def _find_font(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def register_thai_font():
    """
    Tries to register a Thai font (regular + bold). Returns (regular_name, bold_name).
    Put TTF files in the same folder if system paths are not available.
    """
    candidates = [
        # Noto Sans Thai
        (["NotoSansThai-Regular.ttf",
          "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
          "/Library/Fonts/NotoSansThai-Regular.ttf",
          os.path.expanduser("~/Library/Fonts/NotoSansThai-Regular.ttf")],
         ["NotoSansThai-Bold.ttf",
          "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
          "/Library/Fonts/NotoSansThai-Bold.ttf",
          os.path.expanduser("~/Library/Fonts/NotoSansThai-Bold.ttf")],
         "NotoSansThai", "NotoSansThai-Bold"),

        # Sarabun
        (["Sarabun-Regular.ttf",
          "/usr/share/fonts/truetype/sarabun/Sarabun-Regular.ttf",
          "/Library/Fonts/Sarabun-Regular.ttf",
          os.path.expanduser("~/Library/Fonts/Sarabun-Regular.ttf")],
         ["Sarabun-Bold.ttf",
          "/usr/share/fonts/truetype/sarabun/Sarabun-Bold.ttf",
          "/Library/Fonts/Sarabun-Bold.ttf",
          os.path.expanduser("~/Library/Fonts/Sarabun-Bold.ttf")],
         "Sarabun", "Sarabun-Bold"),

        # TH Sarabun New
        (["THSarabunNew.ttf",
          "/usr/share/fonts/truetype/thai/THSarabunNew.ttf",
          "/Library/Fonts/THSarabunNew.ttf",
          os.path.expanduser("~/Library/Fonts/THSarabunNew.ttf")],
         ["THSarabunNew Bold.ttf",
          "/usr/share/fonts/truetype/thai/THSarabunNew Bold.ttf",
          "/Library/Fonts/THSarabunNew Bold.ttf",
          os.path.expanduser("~/Library/Fonts/THSarabunNew Bold.ttf")],
         "THSarabunNew", "THSarabunNew-Bold"),
    ]

    for reg_paths, bold_paths, reg_name, bold_name in candidates:
        reg_file = _find_font(reg_paths)
        bold_file = _find_font(bold_paths)
        if reg_file and bold_file:
            try:
                pdfmetrics.registerFont(TTFont(reg_name, reg_file))
                pdfmetrics.registerFont(TTFont(bold_name, bold_file))
                return reg_name, bold_name
            except Exception:
                pass

    # fallback (will show tofu for Thai, but app will warn)
    return "Helvetica", "Helvetica-Bold"

THAI_REG, THAI_BOLD = register_thai_font()
THAI_FONT_OK = (THAI_REG != "Helvetica")

# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib (Thai-capable) — just for on-screen plots
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans Thai", "Sarabun", "TH Sarabun New",
    "Tahoma", "Arial Unicode MS", "Segoe UI", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

# Palette
PALETTE = {
    "frame":  "#d9a362",
    "inside": (0.42, 0.67, 0.86, 0.95),
    "grid":   "#ffffff",
    "edge":   "#4a4a4a",
    "pallet": "#9c6a3a",
}

# ──────────────────────────────────────────────────────────────────────────────
# Units helper
class Unit:
    factor_to_mm = {"mm": 1.0, "cm": 10.0, "in": 25.4}
    @staticmethod
    def to_mm(x, unit): return x * Unit.factor_to_mm[unit]
    @staticmethod
    def to_cm(x, unit): return x * (Unit.factor_to_mm[unit] / 10.0)

# Models
@dataclass
class BasePlan:
    pattern: str
    per_layer: int
    nx: int
    ny: int
    spacing_x: float
    spacing_y: float
    shape: str = "rect"
    item_w: float = 0.0
    item_l: float = 0.0

@dataclass
class PackResult:
    orientation: Tuple[float, float, float]  # (iw, il, ih)
    layers: int
    per_layer: int
    total: int
    plan: BasePlan
    meta: dict

# Basic geometry
def surface_area_box(W, L, H):
    return 2 * (W * L + W * H + L * H)

def _permutations_with_height_labels(w: float, l: float, h: float):
    return [
        (w,l,h,"h"), (w,h,l,"l"),
        (l,w,h,"h"), (l,h,w,"w"),
        (h,w,l,"l"), (h,l,w,"w"),
    ]

def rect_design_min_carton(qty, w, l, h,
                           locked_axis: Optional[str]=None,
                           force_layers: Optional[int]=None,
                           max_per_carton: Optional[int]=None):
    """เลือก layout ที่ SA ต่ำสุด (และจุครบ) พร้อมจำกัดชิ้น/กล่องตามน้ำหนัก"""
    perms = _permutations_with_height_labels(w,l,h)
    if locked_axis and locked_axis != "auto":
        perms = [p for p in perms if p[3] == locked_axis]
    if not perms: return None

    best = None
    layer_candidates = [force_layers] if force_layers else list(range(1, qty+1))
    for (iw, il, ih, hfrom) in perms:
        for layers in layer_candidates:
            need = math.ceil(qty / layers)
            for nx in range(1, need+1):
                ny = math.ceil(need / nx)
                plan = BasePlan(f"{ny}×{nx}", nx*ny, nx, ny, iw, il, "rect", iw, il)
                W, L, H = plan.ny*iw, plan.nx*il, layers*ih
                total_per_carton = nx*ny*layers
                if max_per_carton is not None and total_per_carton > max_per_carton:
                    continue
                if total_per_carton < qty:
                    continue
                SA = surface_area_box(W, L, H)
                res = PackResult((iw, il, ih), layers, nx*ny, total_per_carton, plan,
                                 {"SA": SA, "height_from": hfrom})
                if (best is None) or (SA < best.meta["SA"]) or (SA == best.meta["SA"] and total_per_carton < best.total):
                    best = res
            if force_layers: break
    return best

# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers (2D/3D)
def make_top_view(W, L, plan, unit, show_fill=False, show_index=False,
                  scale=0.55):
    from matplotlib.patches import Rectangle
    base = 3.6 * scale
    fig_w = base
    fig_h = max(2.4, fig_w * (W / L) if L > 0 else fig_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    frame_pad = max(L, W) * 0.02
    ax.add_patch(Rectangle((-frame_pad, -frame_pad),
                           L + 2*frame_pad, W + 2*frame_pad,
                           facecolor=PALETTE["frame"], edgecolor=PALETTE["frame"]))
    ax.add_patch(Rectangle((0, 0), L, W,
                           facecolor=PALETTE["inside"] if show_fill else (1,1,1,1),
                           edgecolor=PALETTE["edge"], linewidth=1.5))
    bw, bl = plan.spacing_x, plan.spacing_y
    idx = 1
    for iy in range(plan.ny):
        for ix in range(plan.nx):
            x, y = ix*bl, iy*bw
            from matplotlib.patches import Rectangle as R2
            ax.add_patch(R2((x, y), bl, bw,
                            fill=False, edgecolor=PALETTE["grid"], linewidth=1.6))
            if show_index:
                ax.text(x + bl/2, y + bw/2, str(idx), ha="center", va="center", fontsize=7, color="#333")
            idx += 1
    ax.set_xlim(-frame_pad, L + frame_pad)
    ax.set_ylim(-frame_pad, W + frame_pad)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig

def make_side_view(W, L, H, layers, plan, ih, scale=0.55):
    from matplotlib.patches import Rectangle
    base = 3.0 * scale
    fig, ax = plt.subplots(figsize=(base, max(2.0, base*(H/max(W,1e-6)))))
    frame_pad = max(W, H) * 0.02
    ax.add_patch(Rectangle((-frame_pad, -frame_pad),
                           W + 2*frame_pad, H + 2*frame_pad,
                           facecolor=PALETTE["frame"], edgecolor=PALETTE["frame"]))
    ax.add_patch(Rectangle((0, 0), W, H,
                           facecolor=PALETTE["inside"], edgecolor=PALETTE["edge"],
                           linewidth=1.5))
    for z in range(1, layers):
        y = z * ih
        ax.plot([0, W], [y, y], color=PALETTE["grid"], linewidth=1.6)
    x_pos = np.cumsum([plan.spacing_x] * plan.ny)
    for x in x_pos[:-1]:
        ax.plot([x, x], [0, H], color=PALETTE["grid"], linewidth=1.6)
    ax.set_xlim(-frame_pad, W + frame_pad)
    ax.set_ylim(-frame_pad, H + frame_pad)
    ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig

def make_3d_stack(W, L, plan, layers, ih, scale=0.8):
    fig = plt.figure(figsize=(4.2*scale, 3.1*scale))
    ax = fig.add_subplot(111, projection="3d")
    try: ax.set_proj_type('ortho')
    except Exception: pass
    z_unit = ih
    for z in range(layers):
        for iy in range(plan.ny):
            for ix in range(plan.nx):
                ax.bar3d(ix*plan.spacing_y, iy*plan.spacing_x, z*z_unit,
                         plan.spacing_y, plan.spacing_x, z_unit,
                         shade=True, alpha=.43, edgecolor="#213a5a")
    ax.set_xlim(0, L); ax.set_ylim(0, plan.ny*plan.spacing_x); ax.set_zlim(0, layers*z_unit)
    ax.view_init(elev=22, azim=-60); ax.set_axis_off()
    fig.tight_layout(pad=0.05)
    return fig

def render_wlh_diagram_oriented(w, l, h, up="h", figsize=(2.6,2.0)):
    up = (up or "h").lower()
    if up == "h":    x_len,y_len,z_len = w,l,h
    elif up == "w":  x_len,y_len,z_len = l,h,w
    else:            x_len,y_len,z_len = w,h,l
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    try: ax.set_proj_type('ortho')
    except Exception: pass
    verts = [
        [(0,0,0),(x_len,0,0),(x_len,y_len,0),(0,y_len,0)],
        [(0,0,z_len),(x_len,0,z_len),(x_len,y_len,z_len),(0,y_len,z_len)],
        [(0,0,0),(x_len,0,0),(x_len,0,z_len),(0,0,z_len)],
        [(0,y_len,0),(x_len,y_len,0),(x_len,y_len,z_len),(0,y_len,z_len)],
        [(0,0,0),(0,y_len,0),(0,y_len,z_len),(0,0,z_len)],
        [(x_len,0,0),(x_len,y_len,0),(x_len,y_len,z_len),(x_len,0,z_len)],
    ]
    box = Poly3DCollection(verts, facecolors=(0.75,0.88,1.0,0.45),
                           edgecolors="navy", linewidths=1.2)
    ax.add_collection3d(box); ax.set_box_aspect((x_len,y_len,z_len))
    ax.view_init(elev=20, azim=-60); ax.set_axis_off()
    plt.subplots_adjust(0,0,1,1); fig.tight_layout(pad=0.02)
    return fig

# Summary card
def render_summary_box_cm(L_cm,W_cm,H_cm,layers,per_layer,
                          sa_cm2=None,tol_w_cm=None,tol_l_cm=None,tol_h_cm=None,
                          cartons_needed=None,weight_kg=None,limit_kg=18.0,
                          area_m2=None, cbm_per_carton=None, cbm_total=None):
    total = layers*per_layer
    st.markdown("""
    <style>
      .sumcard{background:#e7f7ef;padding:12px 14px;border-radius:10px;
               border:1px solid #c5ead9;box-shadow:0 4px 10px rgba(0,0,0,.04);font-size:15px}
      .sumh{font-size:18px;font-weight:800;color:#0d6832;margin-bottom:8px}
      .sumrow{display:grid;grid-template-columns:auto 1fr;gap:6px 10px}
      .sumk{color:#466c57}.sumv{font-weight:700}
      .dim{display:flex;flex-direction:column;gap:2px;margin-top:4px}
      .dim span{display:block}
      .ok{color:#0d6832;font-weight:800}
      .bad{color:#b00020;font-weight:800}
    </style>
    """, unsafe_allow_html=True)
    size_line = f"{L_cm:.2f}×{W_cm:.2f}×{H_cm:.2f} cm"
    dims_html = (
        f"<div class='dim'>"
        f"<span>กว้าง (W): <b>{W_cm:.2f} cm</b></span>"
        f"<span>ยาว (L): <b>{L_cm:.2f} cm</b></span>"
        f"<span>สูง (H): <b>{H_cm:.2f} cm</b></span>"
        f"</div>"
    )
    rows = [("ขนาดกล่อง", size_line + dims_html),
            ("จำนวนชั้น", f"{layers} ชั้น"),
            ("ชั้นละ", f"{per_layer} ชิ้น"),
            ("รวมทั้งกล่อง", f"{total} ชิ้น")]
    if cartons_needed is not None: rows.append(("จำนวนกล่อง", f"{cartons_needed} กล่อง"))
    if sa_cm2 is not None: rows.append(("Surface Area", f"{sa_cm2:.2f} cm²"))
    if any(v is not None for v in (tol_w_cm,tol_l_cm,tol_h_cm)):
        rows.append(("Tolerance", f"W {tol_w_cm or 0:.2f} cm · L {tol_l_cm or 0:.2f} cm · H {tol_h_cm or 0:.2f} cm"))
    if area_m2 is not None: rows.append(("พื้นที่ฐาน L×W", f"{area_m2:.4f} m²"))
    if cbm_per_carton is not None: rows.append(("ปริมาตร (CBM) ต่อกล่อง", f"{cbm_per_carton:.4f} m³"))
    if cbm_total is not None: rows.append(("CBM รวมทั้งหมด", f"{cbm_total:.4f} m³"))
    if weight_kg is not None:
        status = "✅" if weight_kg <= limit_kg + 1e-9 else "⛔"
        color_cls = "ok" if weight_kg <= limit_kg + 1e-9 else "bad"
        rows.append(("น้ำหนักรวม/กล่อง", f"<span class='{color_cls}'>{status} {weight_kg:.2f} kg</span> (จำกัด ≤ {limit_kg:.2f} kg)"))
    html = "".join(f"<div class='sumk'>{k}</div><div class='sumv'>{v}</div>" for k, v in rows)
    st.markdown(f"<div class='sumcard'><div class='sumh'>กล่องแนะนำ</div><div class='sumrow'>{html}</div></div>",
                unsafe_allow_html=True)

# 3D blocks helper
def _faces(x, y, z, dx, dy, dz):
    return [
        [(x,y,z), (x+dx,y,z), (x+dx,y+dy,z), (x,y+dy,z)],
        [(x,y,z+dz), (x+dx,y,z+dz), (x+dx,y+dy,z+dz), (x,y+dy,z+dz)],
        [(x,y,z), (x+dx,y,z), (x+dx,y,z+dz), (x,y,z+dz)],
        [(x,y+dy,z), (x+dx,y+dy,z), (x+dx,y+dy,z+dz), (x,y+dy,z+dz)],
        [(x,y,z), (x,y+dy,z), (x,y+dy,z+dz), (x,y,z+dz)],
        [(x+dx,y,z), (x+dx,y+dy,z), (x+dx,y+dy,z+dz), (x+dx,y,z+dz)],
    ]

def draw_carton_stack_only(carton_L_cm, carton_W_cm, carton_H_cm,
                           nx, ny, layers, gap=0.0, scale=0.90):
    fig = plt.figure(figsize=(6.0*scale, 4.4*scale))
    ax  = fig.add_subplot(111, projection="3d")
    try: ax.set_proj_type('ortho')
    except Exception: pass
    used_L = nx*carton_L_cm + max(0, nx-1)*gap
    used_W = ny*carton_W_cm + max(0, ny-1)*gap
    used_H = layers*carton_H_cm
    face_c = (0.30, 0.50, 0.75, 0.95); edge_c = "#1a3a5a"
    for k in range(layers):
        z0 = k*carton_H_cm
        for j in range(ny):
            for i in range(nx):
                x = i*(carton_L_cm + gap); y = j*(carton_W_cm + gap)
                ax.add_collection3d(Poly3DCollection(
                    _faces(x, y, z0, carton_L_cm, carton_W_cm, carton_H_cm),
                    facecolors=face_c, edgecolors=edge_c, linewidths=0.9, zsort="max"
                ))
    pad_L = max(6, 0.10*used_L); pad_W = max(6, 0.10*used_W); pad_H = max(6, 0.10*used_H)
    ax.set_xlim(-pad_L, used_L + pad_L); ax.set_ylim(-pad_W, used_W + pad_W); ax.set_zlim(0, used_H + pad_H)
    ax.set_box_aspect((used_L + 2*pad_L, used_W + 2*pad_W, used_H + pad_H))
    ax.view_init(elev=26, azim=-42); ax.set_axis_off(); fig.tight_layout(pad=0.1)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Paper DB
LINER_GRADES = [
    "KA125","KA150","KA185","KA230","KA280",
    "KI125","KI150","KI185","KI230","KI280",
    "KL150","KL185","KL205","KL230",
    "KW140","KW185","KW230",
    "CA150","CA185","CA230",
    "CS150","CS185","CS230",
]
MEDIUM_GRADES = ["Medium115","Medium125","Medium150","Medium170"]
GRADE_INDEX: Dict[str, float] = {
    "KA125":1.00, "KA150":1.10, "KA185":1.22, "KA230":1.35, "KA280":1.50,
    "KI125":0.95, "KI150":1.03, "KI185":1.15, "KI230":1.28, "KI280":1.40,
    "KL150":1.05, "KL185":1.16, "KL205":1.22, "KL230":1.30,
    "KW140":0.98, "KW185":1.10, "KW230":1.22,
    "CA150":0.92, "CA185":1.02, "CA230":1.12,
    "CS150":0.90, "CS185":1.00, "CS230":1.08,
    "Medium115":0.85, "Medium125":0.92, "Medium150":1.00, "Medium170":1.08,
}
FLUTE_BASE_ECT = {"E":4.2,"B":5.3,"C":6.0,"BE":7.6,"BC":8.8}

def estimate_ect(flute: str, liners: Tuple[str, ...], mediums: Tuple[str, ...]) -> float:
    idx_vals = [GRADE_INDEX[g] for g in liners + mediums]
    avg_idx = sum(idx_vals) / len(idx_vals)
    base = FLUTE_BASE_ECT.get(flute, 5.3)
    return base * avg_idx

def mckee_bct_estimate(ect_kn_per_m: float, P_cm: float, t_mm: float = 4.0) -> float:
    k = 5.876
    ect_n_per_mm = ect_kn_per_m  # 1 kN/m = 1 N/mm
    P_mm = P_cm * 10.0
    bct = k * ((ect_n_per_mm * t_mm * P_mm) ** 0.746)
    return bct

# Derating factors
def factor_stacking_method(method: str) -> float:  return 1.00 if method == "column" else 0.85
def factor_base_surface(base: str) -> float:      return 1.00 if base == "pallet" else 0.90
def factor_humidity(rh: str) -> float:            return {"<=50%":1.00,"50–70%":0.95,"70–85%":0.85,">85%":0.75}[rh]
def factor_storage_time(t: str) -> float:         return {"< 1 เดือน":1.00,"1–3 เดือน":0.92,"> 3 เดือน":0.85}[t]
def factor_handling(times: str) -> float:         return {"น้อย (≤3 ครั้ง)":1.00,"ปานกลาง (4–10)":0.95,"บ่อย (>10)":0.90}[times]
def total_derating(method, base, rh, stg, hd, safety: float) -> float:
    f = (factor_stacking_method(method) *
         factor_base_surface(base) *
         factor_humidity(rh) * factor_storage_time(stg) * factor_handling(hd) * safety)
    return min(f, 1.0)

# Flute preview (SVG)
def _svg_wave_points(width, amp, period, offset_y, steps=200):
    import math
    pts = []
    for i in range(steps + 1):
        x = width * i / steps
        y = offset_y + amp * math.sin(2 * math.pi * (x / period))
        pts.append(f"{x:.2f},{y:.2f}")
    return " ".join(pts)

def flute_svg(flute: str, wall: str):
    W, H = 360, 80; PADX = 16; RADIUS = 10
    BG = "#c89b6b"; LINER = "#7c5936"; FLUTE1 = "#6f4c2b"; FLUTE2 = "#5f4126"
    conf = {"E":{"amp":5,"period":26,"t":26},
            "B":{"amp":7,"period":34,"t":32},
            "C":{"amp":9,"period":44,"t":40},
            "BC":{"amp1":8,"period1":36,"amp2":11,"period2":52,"t":64}}
    if wall == "double": flute = "BC"
    clip_id = f"clip_{flute}_{wall}".replace(" ","_")
    innerW = W - 2*PADX
    if flute != "BC":
        p = conf[flute]; top=(H-p["t"])//2; bottom=top+p["t"]; mid=(top+bottom)//2
        svg=[f"<svg viewBox='0 0 {W} {H}' width='100%' height='60' xmlns='http://www.w3.org/2000/svg'>"]
        svg.append(f"<rect x='{PADX}' y='{top}' width='{innerW}' height='{p['t']}' rx='{RADIUS}' ry='{RADIUS}' fill='{BG}'/>")
        svg.append(f"<clipPath id='{clip_id}'><rect x='{PADX}' y='{top}' width='{innerW}' height='{p['t']}' rx='{RADIUS}' ry='{RADIUS}'/></clipPath>")
        svg.append(f"<line x1='{PADX+10}' y1='{top+6}' x2='{W-PADX-10}' y2='{top+6}' stroke='{LINER}' stroke-width='2' opacity='.85'/>")
        svg.append(f"<line x1='{PADX+10}' y1='{bottom-6}' x2='{W-PADX-10}' y2='{bottom-6}' stroke='{LINER}' stroke-width='2' opacity='.85'/>")
        pts=_svg_wave_points(innerW-20, p["amp"], p["period"], mid)
        svg.append(f"<g clip-path='url(#{clip_id})'><polyline points=' {PADX+10:.2f},0 {pts} ' transform='translate(10,0)' fill='none' stroke='{FLUTE1}' stroke-width='2.2'/></g>")
        svg.append("</svg>"); return "".join(svg)
    # BC
    p=conf["BC"]; top=(H-p["t"])//2; bottom=top+p["t"]; midliner=(top+bottom)//2; y1=top+16; y2=bottom-16
    svg=[f"<svg viewBox='0 0 {W} {H}' width='100%' height='60' xmlns='http://www.w3.org/2000/svg'>"]
    svg.append(f"<rect x='{PADX}' y='{top}' width='{innerW}' height='{p['t']}' rx='{RADIUS}' ry='{RADIUS}' fill='{BG}'/>")
    svg.append(f"<clipPath id='{clip_id}'><rect x='{PADX}' y='{top}' width='{innerW}' height='{p['t']}' rx='{RADIUS}' ry='{RADIUS}'/></clipPath>")
    svg.append(f"<line x1='{PADX+10}' y1='{top+6}' x2='{W-PADX-10}' y2='{top+6}' stroke='{LINER}' stroke-width='2' opacity='.85'/>")
    svg.append(f"<line x1='{PADX+10}' y1='{midliner}' x2='{W-PADX-10}' y2='{midliner}' stroke='{LINER}' stroke-width='2' opacity='.85'/>")
    svg.append(f"<line x1='{PADX+10}' y1='{bottom-6}' x2='{W-PADX-10}' y2='{bottom-6}' stroke='{LINER}' stroke-width='2' opacity='.85'/>")
    pts1=_svg_wave_points(innerW-20,p["amp1"],p["period1"],y1); pts2=_svg_wave_points(innerW-20,p["amp2"],p["period2"],y2)
    svg.append(f"<g clip-path='url(#{clip_id})'>"
               f"<polyline points=' {PADX+10:.2f},0 {pts1} ' transform='translate(10,0)' fill='none' stroke='{FLUTE1}' stroke-width='2.2'/>"
               f"<polyline points=' {PADX+10:.2f},0 {pts2} ' transform='translate(10,0)' fill='none' stroke='{FLUTE2}' stroke-width='2.2'/>"
               f"</g></svg>")
    return "".join(svg)

# ──────────────────────────────────────────────────────────────────────────────
# PDF builder (uses Thai fonts)
def build_paper_spec_pdf(summary: dict) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # Header bar
    c.setFillColorRGB(0.07, 0.41, 0.24)
    c.rect(0, H-2.0*cm, W, 2.0*cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(THAI_BOLD, 16)
    title_text = summary.get("title", "Paper Spec & Strength Sheet")
    c.drawString(1.5*cm, H-1.3*cm, title_text)

    # Simple box illustration
    c.setFillColorRGB(0.78, 0.61, 0.41)
    bx, by = 1.7*cm, H-8.5*cm
    c.rect(bx, by, 6.5*cm, 4.5*cm, fill=1, stroke=0)
    c.setFillColorRGB(0.65, 0.49, 0.33)
    c.rect(bx+0.3*cm, by+0.3*cm, 5.9*cm, 3.9*cm, fill=1, stroke=0)
    c.setFillColorRGB(0.86, 0.70, 0.50)
    # polygon might not exist in older reportlab; wrap in try
    try:
        c.polygon([bx, by+4.5*cm, bx+2.8*cm, by+6.0*cm, bx+5.6*cm, by+4.5*cm], fill=1, stroke=0)
        c.polygon([bx+6.5*cm, by+4.5*cm, bx+3.7*cm, by+6.0*cm, bx+0.9*cm, by+4.5*cm], fill=1, stroke=0)
    except Exception:
        pass

    # Info column
    xL = 9.0*cm; y = H-3.2*cm
    c.setFillColor(colors.black)

    def label(t):
        nonlocal y
        c.setFont(THAI_BOLD, 11)
        c.drawString(xL, y, t); y -= 0.55*cm

    def kv(k, v):
        nonlocal y
        c.setFont(THAI_REG, 10)
        c.setFillColor(colors.HexColor("#333333"))
        c.drawString(xL+0.5*cm, y, f"• {k}: {v}"); y -= 0.48*cm
        c.setFillColor(colors.black)

    label("ข้อมูลกล่อง")
    kv("ขนาดกล่อง (L×W×H, cm)", summary.get("carton_size","-"))
    kv("ชิ้น/กล่อง", summary.get("pieces_per_carton","-"))
    kv("CBM ต่อกล่อง", summary.get("cbm_per_carton","-"))
    kv("น้ำหนัก/กล่อง", summary.get("weight_per_carton","-"))

    y -= 0.2*cm
    label("โครงสร้างกระดาษ")
    kv("แบบ", summary.get("paper_mode","-"))
    kv("ลอน", summary.get("flute","-"))
    kv("ความหนา (mm)", summary.get("t_mm","-"))
    if summary.get("liner_mid"):
        kv("Liner (นอก/กลาง/ใน)", f"{summary.get('liner_out','-')} / {summary.get('liner_mid','-')} / {summary.get('liner_in','-')}")
        kv("Medium (1/2)", f"{summary.get('medium1','-')} / {summary.get('medium2','-')}")
    else:
        kv("Liner (นอก/ใน)", f"{summary.get('liner_out','-')} / {summary.get('liner_in','-')}")
        kv("Medium", summary.get("medium1","-"))

    y -= 0.2*cm
    label("ความแข็งแรง (ประมาณ)")
    kv("ECT", f"{summary.get('ect_knm','-')} kN/m")
    kv("BCT", f"{summary.get('bct_kgf','-')} kgf")

    y -= 0.2*cm
    label("Derating & เงื่อนไขใช้งาน")
    kv("วิธีเรียงซ้อน / พื้นวาง", f"{summary.get('stacking_method','-')} / {summary.get('base_surface','-')}")
    kv("RH / ระยะเวลาเก็บ", f"{summary.get('rh','-')} / {summary.get('storage_time','-')}")
    kv("การเคลื่อนย้าย / Safety", f"{summary.get('handling','-')} / {summary.get('safety_factor','-')}")
    kv("กำลังรับหลังเดอเรต", f"{summary.get('derated_capacity','-')} kgf (ประมาณ)")

    c.setStrokeColor(colors.HexColor("#dddddd"))
    c.line(1.5*cm, 2.5*cm, W-1.5*cm, 2.5*cm)
    c.setFont(THAI_REG, 8)
    c.setFillColor(colors.HexColor("#888888"))
    foot = "Generated by Carton Designer"
    if not THAI_FONT_OK:
        foot += " • (Warning: Thai font not embedded)"
    c.drawRightString(W-1.5*cm, 2.1*cm, foot)

    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
st.set_page_config(page_title="Carton Designer", layout="wide")
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;
padding:1.0rem;border-radius:12px;text-align:center;margin-bottom:0.8rem}
.card{background:#fff;border:1px solid #ebedf0;border-radius:12px;padding:14px 16px;
box-shadow:0 5px 14px rgba(0,0,0,.04);margin-bottom:12px}
</style>
<div class="main-header"><h1>📦 ตัวช่วยออกแบบขนาดกล่อง & เลือกกระดาษ & วางพาเลต</h1>
<p>Giftbox ➜ Carton · 📄 Paper Spec · 🧱 Carton ➜ Pallet</p></div>
""", unsafe_allow_html=True)

tab_carton, tab_paper, tab_pallet = st.tabs(["🎁 Giftbox ➜ Carton", "📄 Paper Spec", "🧱 Carton ➜ Pallet"])

# ── Tab 1: Giftbox ➜ Carton
with tab_carton:
    col_main, col_side = st.columns([8,4], gap="large")
    with col_main:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧩 ลักษณะของสินค้า")
        st.selectbox("เลือกหมวดสินค้า",
                     ["ตลับ/พาเลตเมคอัพ","แท่งลิป/บาล์ม","ดินสอ/อายไลเนอร์","มาสคาร่า/คอนซีลเลอร์",
                      "กระปุกครีม","ขวดดรอปเปอร์","ปั๊ม/เออร์เลส","โทนเนอร์/สเปรย์",
                      "แอมพูล/ไวอัล","ซองครีม/ซองเจล","ชีทมาสก์","สแตนด์อัพพาวช์",
                      "บลิสเตอร์/การ์ด","กิฟต์เซ็ต/มัลติแพ็ก","อื่น ๆ"], index=0)
        st.caption("ทรงเหลี่ยม (Rectangular)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📦 ตั้งค่าการวาง Gift box")
        unit = st.selectbox("📏 หน่วยที่ใช้", ["mm","cm","in"], index=0)
        c1,c2,c3 = st.columns(3)
        gb_w = c1.number_input("ขนาด Giftbox — กว้าง (W)", 0.1, 10000.0, 120.0, 1.0)
        gb_l = c2.number_input("ขนาด Giftbox — ยาว (L)",   0.1, 10000.0, 200.0, 1.0)
        gb_h = c3.number_input("ขนาด Giftbox — สูง (H)",   0.1, 10000.0, 80.0,  1.0)
        qty = st.number_input("จำนวน giftbox ที่ต้องการ (รวม)", 1, 999999, 20, 1)
        cwg1, cwg2 = st.columns([1,1])
        gb_weight_g = cwg1.number_input("น้ำหนัก giftbox ต่อชิ้น (g)", 0.0, 100000.0, 0.0, 1.0,
                                        help="ใส่ 0 ถ้าไม่ต้องการจำกัดด้วยน้ำหนัก")
        cwg2.markdown("**น้ำหนักสูงสุด/กล่อง:** ≤ 18.00 kg")
        auto_layers = st.checkbox("คำนวณจำนวนชั้นอัตโนมัติ", value=True)
        layers_for_one = None if auto_layers else st.number_input("จำนวนชั้นที่ต้องการวาง (ต่อ 1 กล่อง)", 1, 50, 1, 1)
        st.markdown("### 🔁 เลือกทิศทางการวาง")
        lock_opt = st.radio("ทิศทาง (แกนที่ตั้งอยู่ด้านบน)",
                            ["ไม่ล็อก – ให้ระบบลองทุกแบบ","วางแบบเดิม – H ขึ้น","พลิกให้ด้านกว้างขึ้น – W ขึ้น","พลิกให้ด้านยาวขึ้น – L ขึ้น"],
                            horizontal=False, index=0)
        lock_map = {"ไม่ล็อก – ให้ระบบลองทุกแบบ":"auto","วางแบบเดิม – H ขึ้น":"h","พลิกให้ด้านกว้างขึ้น – W ขึ้น":"w","พลิกให้ด้านยาวขึ้น – L ขึ้น":"l"}
        locked_axis = lock_map[lock_opt]
        st.caption("รูปช่วยจำอัตราส่วน W/L/H")
        st.pyplot(render_wlh_diagram_oriented(gb_w, gb_l, gb_h,
                  up=(locked_axis if locked_axis!="auto" else "h")), use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_side:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🛠️ ระยะเผื่อ (Tolerance)")
        t1,t2,t3 = st.columns(3)
        tol_w = t1.number_input(f"เผื่อกว้าง W ({unit})", 0.0, 200.0, 0.0, 0.5)
        tol_l = t2.number_input(f"เผื่อยาว L ({unit})",   0.0, 200.0, 0.0, 0.5)
        tol_h = t3.number_input(f"เผื่อสูง H ({unit})",   0.0, 200.0, 0.0, 0.5)
        try_bruteforce = st.checkbox("🔎 ลองเพิ่มเผื่อเท่ากันทั้งสามแกน (0–0.5)", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    calc = st.button("🚀 คำนวณขนาดกล่องที่เหมาะสม", use_container_width=True)

    def apply_tol(w,l,h, a,b,c): return w+a, l+b, h+c

    if calc:
        max_per_carton = None
        if gb_weight_g > 0:
            max_per_carton = int(18000 // gb_weight_g)
            if max_per_carton < 1:
                st.error("⛔ น้ำหนักต่อชิ้นมากเกินไป (แม้ 1 กล่องก็เกิน 18 kg)"); st.stop()
        deltas = [round(x,2) for x in np.arange(0,0.51,0.05)] if try_bruteforce else [0.0]
        best = None; best_tol = None
        for d in deltas:
            iw, il, ih = apply_tol(gb_w, gb_l, gb_h, tol_w+d, tol_l+d, tol_h+d)
            axis_forced = None if locked_axis=="auto" else locked_axis
            res = rect_design_min_carton(qty, iw, il, ih, axis_forced, layers_for_one, max_per_carton)
            if res and (best is None or res.meta["SA"] < best.meta["SA"]):
                best = res; best_tol = (tol_w+d, tol_l+d, tol_h+d)
        if not best: st.error("ไม่พบวิธีจัดวางที่เหมาะสมภายใต้เงื่อนไขปัจจุบัน")
        else:
            st.session_state.result = best
            st.session_state.tol_w, st.session_state.tol_l, st.session_state.tol_h = best_tol
            st.session_state.unit = unit
            st.session_state.qty = qty
            st.session_state.gb_weight_g = gb_weight_g
            st.success("✅ คำนวณสำเร็จ! ➜ ไปหน้า 'Paper Spec' ต่อ")

    res = st.session_state.get("result")
    if res:
        base_unit = st.session_state.get("unit", "mm")
        tol_w_sv, tol_l_sv, tol_h_sv = (st.session_state.get("tol_w",0.0),
                                        st.session_state.get("tol_l",0.0),
                                        st.session_state.get("tol_h",0.0))
        qty_sv  = st.session_state.get("qty", 1)
        gb_wt_g = st.session_state.get("gb_weight_g", 0.0)
        iw,il,ih = res.orientation
        W = res.plan.ny * res.plan.spacing_x; L = res.plan.nx * res.plan.spacing_y; H = res.layers * ih
        W_cm = Unit.to_cm(W, base_unit); L_cm = Unit.to_cm(L, base_unit); H_cm = Unit.to_cm(H, base_unit)
        SA_cm2 = 2*(W_cm*L_cm + W_cm*H_cm + L_cm*H_cm)
        tol_w_cm = Unit.to_cm(tol_w_sv, base_unit); tol_l_cm = Unit.to_cm(tol_l_sv, base_unit); tol_h_cm = Unit.to_cm(tol_h_sv, base_unit)
        per_carton = res.layers * res.per_layer
        need_cartons = math.ceil(qty_sv / per_carton)
        area_m2 = (L_cm * W_cm) / 1e4
        cbm_per_carton = (L_cm * W_cm * H_cm) / 1e6
        cbm_total = cbm_per_carton * need_cartons
        weight_per_carton_kg = (per_carton * gb_wt_g)/1000.0 if gb_wt_g>0 else None

        st.divider(); st.header("🖼️ Visualization")
        tab2d, tab3d = st.tabs(["📐 มุมบน/ข้าง (2D)", "📦 ซ้อนชั้น (3D)"])
        with tab2d:
            left, right = st.columns([3,2])
            with right:
                fill = st.checkbox("🟩 แสดงสี", True)
                idx  = st.checkbox("🔢 หมายเลข", False)
                st.markdown("### สรุป (หน่วย cm / m² / m³)")
                render_summary_box_cm(L_cm, W_cm, H_cm, res.layers, res.per_layer,
                                      SA_cm2, tol_w_cm, tol_l_cm, tol_h_cm,
                                      need_cartons, weight_per_carton_kg, 18.0,
                                      area_m2, cbm_per_carton, cbm_total)
                if weight_per_carton_kg and weight_per_carton_kg > 18.0:
                    st.error("⛔ น้ำหนักรวมต่อกล่องเกิน 18 kg — กรุณาปรับจำนวน/ชั้น")
            with left:
                st.pyplot(make_top_view(W,L,res.plan,base_unit, show_fill=fill, show_index=idx), use_container_width=False)
            st.markdown("---")
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**มุมบน**")
                st.pyplot(make_top_view(W,L,res.plan,base_unit, True, False, 0.45), use_container_width=False)
            with cB:
                st.markdown("**มุมข้าง**")
                st.pyplot(make_side_view(W,L,H,res.layers,res.plan,ih, 0.45), use_container_width=False)
        with tab3d:
            st.pyplot(make_3d_stack(W,L,res.plan,res.layers,ih, scale=0.7), use_container_width=False)

# Helper to get carton dims from session
def get_carton_from_session():
    res = st.session_state.get("result")
    if not res: return None
    base_unit = st.session_state.get("unit","mm")
    iw, il, ih = res.orientation
    W = res.plan.ny*res.plan.spacing_x; L = res.plan.nx*res.plan.spacing_y; H = res.layers*ih
    L_cm = Unit.to_cm(L, base_unit); W_cm = Unit.to_cm(W, base_unit); H_cm = Unit.to_cm(H, base_unit)
    per_carton = res.layers * res.per_layer
    gb_weight_g = st.session_state.get("gb_weight_g", 0.0)
    weight_kg = (per_carton * gb_weight_g / 1000.0) if gb_weight_g > 0 else None
    return (L_cm, W_cm, H_cm, per_carton, weight_kg)

# ── Tab 2: Paper Spec
with tab_paper:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 กระดาษลูกฟูก & ความแข็งแรง")
    data = get_carton_from_session()
    if not data:
        st.warning("ยังไม่มีผลจากหน้า Giftbox ➜ Carton กรุณาคำนวณก่อน")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        L_cm, W_cm, H_cm, per_carton, weight_kg = data
        cbm = (L_cm*W_cm*H_cm)/1e6
        c1,c2,c3 = st.columns(3)
        c1.metric("ขนาด Carton (cm)", f"{L_cm:.1f} × {W_cm:.1f} × {H_cm:.1f}")
        c2.metric("ชิ้น/กล่อง", f"{per_carton}")
        c3.metric("CBM ต่อกล่อง", f"{cbm:.4f} m³")
        if weight_kg is not None: st.info(f"น้ำหนักรวมต่อกล่อง ≈ **{weight_kg:.2f} kg**")

        st.markdown("---"); st.markdown("### 1) เลือกความหนากระดาษ/ลอน")
        st.markdown("<style>.cardPick{border:1px solid #e6e8ec;border-radius:12px;padding:12px 14px;margin:8px 0 16px 0;box-shadow:0 3px 10px rgba(0,0,0,.03)}</style>", unsafe_allow_html=True)

        if "paper_mode" not in st.session_state: st.session_state.paper_mode = "5 ชั้น"
        if "paper_flute3" not in st.session_state: st.session_state.paper_flute3 = "E"

        col3, col5 = st.columns(2)
        with col3:
            st.markdown("<div class='cardPick'>", unsafe_allow_html=True)
            st.markdown("**กระดาษหนา 3 ชั้น**")
            pick3 = st.radio("ลอน (3 ชั้น)", ["B","C","E"],
                             index=["B","C","E"].index(st.session_state.paper_flute3),
                             horizontal=True, label_visibility="collapsed")
            st.session_state.paper_flute3 = pick3
            st.markdown(flute_svg(pick3, "single"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col5:
            st.markdown("<div class='cardPick'>", unsafe_allow_html=True)
            st.markdown("**กระดาษหนา 5 ชั้น**")
            st.radio("ลอน (5 ชั้น)", ["BC"], index=0, horizontal=True, label_visibility="collapsed")
            st.markdown(flute_svg("BC", "double"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        mode = st.radio("โหมดที่ใช้คำนวณ", ["3 ชั้น","5 ชั้น"], horizontal=True,
                        index=(0 if st.session_state.paper_mode=="3 ชั้น" else 1))
        st.session_state.paper_mode = mode
        T_DEFAULT = {"E":2.5,"B":3.0,"C":4.0,"BC":6.5}
        if mode == "3 ชั้น":
            wall="single"; flute = st.session_state.paper_flute3; t_mm=T_DEFAULT[flute]
        else:
            wall="double"; flute = "BC"; t_mm=T_DEFAULT["BC"]
        t_mm = st.number_input("ความหนาบอร์ด (mm)", 2.5, 12.0, float(t_mm), 0.1)

        st.caption("เลือกเกรดกระดาษ (ปรับได้ตามโรงงาน)")
        if wall == "single":
            colL, colM, colR = st.columns(3)
            liner_out = colL.selectbox("Liner - ผิวนอก", LINER_GRADES, index=LINER_GRADES.index("KA185"))
            medium1   = colM.selectbox("Medium - แผ่นลอน", MEDIUM_GRADES, index=MEDIUM_GRADES.index("Medium150"))
            liner_in  = colR.selectbox("Liner - ผิวใน", LINER_GRADES, index=LINER_GRADES.index("CA185"))
            ect_knm = estimate_ect(flute, (liner_out, liner_in), (medium1,))
            liner_mid=None; medium2=None
        else:
            colL, colM1, colC, colM2, colR = st.columns(5)
            liner_out = colL.selectbox("Liner - ผิวนอก", LINER_GRADES, index=LINER_GRADES.index("KA230"))
            medium1   = colM1.selectbox("Medium 1 - ลอนแรก", MEDIUM_GRADES, index=MEDIUM_GRADES.index("Medium150"))
            liner_mid = colC.selectbox("Liner - ชั้นกลาง", LINER_GRADES, index=LINER_GRADES.index("KI185"))
            medium2   = colM2.selectbox("Medium 2 - ลอนสอง", MEDIUM_GRADES, index=MEDIUM_GRADES.index("Medium150"))
            liner_in  = colR.selectbox("Liner - ผิวใน", LINER_GRADES, index=LINER_GRADES.index("CA185"))
            ect_knm = estimate_ect(flute, (liner_out, liner_mid, liner_in), (medium1, medium2))

        P_cm = 2.0*(L_cm + W_cm)
        bct_N = mckee_bct_estimate(ect_knm, P_cm, t_mm)
        bct_kgf = bct_N / 9.81

        st.markdown("### 2) เงื่อนไขการกองเก็บ/ใช้งาน")
        cS1, cS2 = st.columns(2)
        method = cS1.radio("วิธีเรียงซ้อน", ["column","interlocking"], index=0, horizontal=True)
        base   = cS2.radio("พื้นวาง", ["pallet","floor"], index=0, horizontal=True)
        cS3, cS4, cS5 = st.columns(3)
        rh  = cS3.selectbox("ความชื้นสัมพัทธ์ (RH)", ["<=50%","50–70%","70–85%",">85%"], index=1)
        stg = cS4.selectbox("ระยะเวลาจัดเก็บ", ["< 1 เดือน","1–3 เดือน","> 3 เดือน"], index=1)
        hd  = cS5.selectbox("จำนวนครั้งการเคลื่อนย้าย", ["น้อย (≤3 ครั้ง)","ปานกลาง (4–10)","บ่อย (>10)"], index=1)
        safety = st.slider("Safety factor", 0.50, 0.90, 0.60, 0.01)
        F = total_derating(method, base, rh, stg, hd, safety)

        st.markdown("### 3) คำนวณชั้นซ้อนที่ปลอดภัย / เช็กผ่าน-ไม่ผ่าน")
        if weight_kg:
            layers_try = st.number_input("ลองคำนวณซ้อนกี่ชั้น", 1, 50, 5, 1)
            load_bottom_kg = weight_kg * layers_try
            capacity_kg = bct_kgf * F
            c1, c2, c3 = st.columns(3)
            c1.metric("ECT ประมาณ", f"{ect_knm:.2f} kN/m")
            c2.metric("BCT ประมาณ", f"{bct_kgf:.0f} kgf")
            c3.metric("ความสามารถหลัง Derating", f"{capacity_kg:.0f} kgf")
            if capacity_kg >= load_bottom_kg:
                st.success(f"ผ่าน: ซ้อนได้ {layers_try} ชั้น (capacity≈{capacity_kg:.0f} ≥ load≈{load_bottom_kg:.0f})")
            else:
                st.error(f"อาจไม่ผ่าน: capacity≈{capacity_kg:.0f} < load≈{load_bottom_kg:.0f}")
                max_layers = max(1, int(capacity_kg // max(weight_kg, 1e-6)))
                st.info(f"ชั้นซ้อนแนะนำสูงสุด ≈ {max_layers} ชั้น")
        else:
            st.warning("ยังไม่มีน้ำหนักรวมต่อกล่อง — ใส่น้ำหนัก/ชิ้นในแท็บแรก")

        # Export PDF button
        summary = {
            "title": "กระดาษลูกฟูก & ความแข็งแรง — ตัวอย่างสรุป",
            "carton_size": f"{L_cm:.1f} × {W_cm:.1f} × {H_cm:.1f}",
            "pieces_per_carton": f"{per_carton}",
            "cbm_per_carton": f"{(L_cm*W_cm*H_cm)/1e6:.4f} m³",
            "weight_per_carton": f"{(weight_kg or 0):.2f} kg" if weight_kg else "-",
            "paper_mode": ("5 ชั้น" if wall=="double" else "3 ชั้น"),
            "flute": flute, "t_mm": f"{t_mm:.1f}",
            "liner_out": liner_out, "liner_mid": (liner_mid if wall=="double" else None),
            "liner_in": liner_in, "medium1": medium1, "medium2": (medium2 if wall=="double" else None),
            "ect_knm": f"{ect_knm:.2f}", "bct_kgf": f"{(bct_N/9.81):.0f}",
            "derated_capacity": f"{(bct_N/9.81)*F:.0f}",
            "stacking_method": method, "base_surface": base,
            "rh": rh, "storage_time": stg, "handling": hd, "safety_factor": f"{safety:.2f}",
        }
        pdf_bytes = build_paper_spec_pdf(summary)
        st.download_button("📥 ดาวน์โหลดไฟล์ตัวอย่าง Paper Spec (PDF)",
                           data=pdf_bytes, file_name="PaperSpec_Example.pdf", mime="application/pdf",
                           use_container_width=True)
        if not THAI_FONT_OK:
            st.warning("ไม่พบฟอนต์ไทยในระบบ/โฟลเดอร์แอป — แนะนำวาง NotoSansThai หรือ Sarabun (Regular/Bold) .ttf ไว้โฟลเดอร์เดียวกับ APP2.py เพื่อให้ PDF แสดงภาษาไทยสมบูรณ์")

        st.markdown("</div>", unsafe_allow_html=True)

# ── Tab 3: Carton ➜ Pallet
def auto_fit_per_layer(pL, pW, bL, bW, overhang_each_side_cm=0.0):
    capL = pL + 2*overhang_each_side_cm
    capW = pW + 2*overhang_each_side_cm
    nx0 = int(capL // bL); ny0 = int(capW // bW); per0 = max(0, nx0)*max(0, ny0)
    nx9 = int(capL // bW); ny9 = int(capW // bL); per9 = max(0, nx9)*max(0, ny9)
    return (nx9, ny9, 90) if per9 > per0 else (nx0, ny0, 0)

def draw_stack_with_alignment(carton_L_cm, carton_W_cm, carton_H_cm,
                              nx, ny, layers, align_mode="parallel",
                              gap=0.0, scale=0.95,
                              show_pallet=True, pallet_W_cm=None, pallet_L_cm=None):
    """วาด 3D: parallel / rotated(ชั้นคี่สลับและจัดกลาง) / h- & v-mirrored"""
    fig = plt.figure(figsize=(6.0*scale, 4.8*scale))
    ax  = fig.add_subplot(111, projection="3d")
    try: ax.set_proj_type('ortho')
    except Exception: pass
    face_c = (0.30, 0.50, 0.75, 0.95); edge_c = "#1a3a5a"
    base_L = nx * carton_L_cm + max(0, nx-1) * gap
    base_W = ny * carton_W_cm + max(0, ny-1) * gap

    def dims_for_layer(k):
        if align_mode == "rotated" and (k % 2 == 1):
            return carton_W_cm, carton_L_cm
        return carton_L_cm, carton_W_cm

    used_L = used_W = 0.0; used_H = layers * carton_H_cm

    if show_pallet and (pallet_L_cm and pallet_W_cm):
        px = - (pallet_L_cm - base_L) / 2; py = - (pallet_W_cm - base_W) / 2
        pal = [[(px,py,0),(px+pallet_L_cm,py,0),(px+pallet_L_cm,py+pallet_W_cm,0),(px,py+pallet_W_cm,0)]]
        ax.add_collection3d(Poly3DCollection(pal, facecolors=(0.65,0.55,0.40,0.35),
                                             edgecolors="#5e3d22", linewidths=0.8, zsort="min"))

    for k in range(layers):
        bL, bW = dims_for_layer(k)
        layer_L = nx * bL + max(0, nx-1) * gap
        layer_W = ny * bW + max(0, ny-1) * gap
        x0 = (base_L - layer_L) / 2.0
        y0 = (base_W - layer_W) / 2.0
        if align_mode == "h-mirrored" and (k % 2 == 1): x0 += bL * 0.15
        if align_mode == "v-mirrored" and (k % 2 == 1): y0 += bW * 0.15
        z0 = k * carton_H_cm
        for j in range(ny):
            for i in range(nx):
                x = x0 + i * (bL + gap); y = y0 + j * (bW + gap)
                ax.add_collection3d(Poly3DCollection(
                    _faces(x, y, z0, bL, bW, carton_H_cm),
                    facecolors=face_c, edgecolors=edge_c, linewidths=0.9, zsort="max"
                ))
        used_L = max(used_L, base_L); used_W = max(used_W, base_W)

    pad_L = max(6, 0.10*used_L); pad_W = max(6, 0.10*used_W); pad_H = max(6, 0.10*used_H)
    ax.set_xlim(-pad_L, used_L + pad_L); ax.set_ylim(-pad_W, used_W + pad_W); ax.set_zlim(0, used_H + pad_H)
    ax.set_box_aspect((used_L + 2*pad_L, used_W + 2*pad_W, used_H + pad_H))
    ax.view_init(elev=26, azim=-42); ax.set_axis_off(); fig.tight_layout(pad=0.1)
    return fig

with tab_pallet:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧱 Carton ➜ Pallet (Step-by-step)")

    st.markdown("#### 1) Unit information")
    use_prev = st.checkbox("📥 ใช้ขนาด Carton จากหน้าแรก (ถ้ามีผลคำนวณ)", value=True)
    if use_prev and st.session_state.get("result"):
        res = st.session_state["result"]
        iw, il, ih = res.orientation
        cart_W = res.plan.ny * res.plan.spacing_x
        cart_L = res.plan.nx * res.plan.spacing_y
        cart_H = ih
        base_unit = st.session_state.get("unit","mm")
        cart_W_cm = Unit.to_cm(cart_W, base_unit)
        cart_L_cm = Unit.to_cm(cart_L, base_unit)
        cart_H_cm = Unit.to_cm(cart_H, base_unit)
        st.info(f"Carton: **{cart_L_cm:.2f} × {cart_W_cm:.2f} × {cart_H_cm:.2f} cm**")
    else:
        c1,c2,c3 = st.columns(3)
        cart_W_cm = c1.number_input("กว้าง (W) ของกล่อง (cm)", 1.0, 200.0, 30.0, 0.5)
        cart_L_cm = c2.number_input("ยาว (L) ของกล่อง (cm)",   1.0, 200.0, 40.0, 0.5)
        cart_H_cm = c3.number_input("สูง (H) ของกล่อง (cm)",   1.0, 200.0, 20.0, 0.5)

    per_carton = (st.session_state.get("result").layers * st.session_state.get("result").per_layer) if st.session_state.get("result") else None
    gb_weight_g = st.session_state.get("gb_weight_g", 0.0)
    weight_per_carton_kg = (per_carton * gb_weight_g / 1000.0) if (per_carton and gb_weight_g>0) else None

    st.divider()
    st.markdown("#### 2) Pallet data")
    presets = {"Custom (กำหนดเอง)":(0,0),"EU 1200×800 mm":(120.0,80.0),"EU 1200×1000 mm":(120.0,100.0),
               "US 48×40 in (121.9×101.6 cm)":(121.9,101.6),"Half 800×600 mm":(80.0,60.0)}
    preset = st.selectbox("เลือกพาเลต", list(presets.keys()), index=2)
    pW_cm, pL_cm = presets[preset]
    cpal1, cpal2, cpal3, cpal4 = st.columns(4)
    pal_W_cm = cpal1.number_input("กว้างพาเลต (cm)", 50.0, 200.0, pW_cm if pW_cm else 100.0, 0.5, disabled=(preset!="Custom (กำหนดเอง)" and pW_cm>0))
    pal_L_cm = cpal2.number_input("ยาวพาเลต (cm)",   60.0, 200.0, pL_cm if pL_cm else 120.0, 0.5, disabled=(preset!="Custom (กำหนดเอง)" and pL_cm>0))
    pal_Hmax = cpal3.number_input("สูงรวมไม่เกิน (cm)", 60.0, 300.0, 180.0, 1.0)
    pal_H    = cpal4.number_input("ความสูงพาเลต (cm)",  5.0,  40.0,  12.0, 0.5)

    st.divider()
    st.markdown("#### 3) Layer configuration")
    cOver1, cOver2 = st.columns([1,3])
    allow_ov = cOver1.checkbox("Allow overhang", value=False)
    overhang_cm = cOver2.slider("ระยะยื่นขอบ (ต่อด้าน, cm)", 0.0, 15.0, 0.0, 0.5, disabled=not allow_ov)
    oh = overhang_cm if allow_ov else 0.0

    mode_auto = st.radio("โหมดจำนวนต่อชั้น", ["Auto fit","กำหนดเอง"], horizontal=True, index=0)
    if mode_auto == "Auto fit":
        nx_auto, ny_auto, ori = auto_fit_per_layer(pal_L_cm, pal_W_cm, cart_L_cm, cart_W_cm, overhang_each_side_cm=oh)
        st.info(f"Auto: ต่อชั้นได้ **{nx_auto} × {ny_auto} = {nx_auto*ny_auto}** (orientation {'0°' if ori==0 else '90°'})")
        nx = st.number_input("แก้ไข nx (ยาวพาเลต)", 0, 200, nx_auto, 1)
        ny = st.number_input("แก้ไข ny (กว้างพาเลต)", 0, 200, ny_auto, 1)
        orientation = st.radio("ทิศกล่องต่อชั้น", ["0° (L // Lpallet)","90° (W // Lpallet)"], index=(0 if ori==0 else 1), horizontal=True)
    else:
        nx = st.number_input("nx (ยาวพาเลต)", 0, 200, 2, 1)
        ny = st.number_input("ny (กว้างพาเลต)", 0, 200, 2, 1)
        orientation = st.radio("ทิศกล่องต่อชั้น", ["0° (L // Lpallet)","90° (W // Lpallet)"], index=0, horizontal=True)

    layer_L_box, layer_W_box = (cart_W_cm, cart_L_cm) if "90°" in orientation else (cart_L_cm, cart_W_cm)
    used_L_layer = nx * layer_L_box; used_W_layer = ny * layer_W_box
    ok_L = (used_L_layer <= pal_L_cm + 2*oh + 1e-9); ok_W = (used_W_layer <= pal_W_cm + 2*oh + 1e-9)
    st.caption("ผู้ผลิตมักไม่แนะนำให้ยื่นเกิน ~15 cm ต่อด้านเพื่อความเสถียร")
    if not ok_L or not ok_W:
        st.error(f"ยังล้นพาเลต (รวม overhang): L={used_L_layer:.1f}/{pal_L_cm+2*oh:.1f}, W={used_W_layer:.1f}/{pal_W_cm+2*oh:.1f} cm")

    layers = st.number_input("จำนวนชั้น (layers)", 0, 100, 5, 1)

    st.divider()
    st.markdown("#### 4) Layer alignment")
    align_mode = st.radio("การสลับชั้น", ["parallel","rotated","h-mirrored","v-mirrored"], index=0, horizontal=True)

    st.divider()
    st.markdown("#### 5) Results")
    problems = []
    total_height = pal_H + layers * cart_H_cm
    if total_height > pal_Hmax + 1e-9: problems.append(f"ความสูงรวมเกินกำหนด: {total_height:.1f} > {pal_Hmax:.1f} cm")
    if nx==0 or ny==0 or layers==0: problems.append("ค่า nx/ny/layers ต้องมากกว่า 0")
    footprint_L = nx * layer_L_box; footprint_W = ny * layer_W_box
    total_boxes = nx * ny * layers; stack_H = layers * cart_H_cm
    area_pallet_m2 = (footprint_L * footprint_W) / 1e4
    cbm_per_pallet = (footprint_L * footprint_W * stack_H) / 1e6
    overhang_note = f"(ยื่นได้ด้านละ {oh:.1f} cm)" if allow_ov and oh>0 else "(ไม่ยื่นขอบ)"
    if problems:
        st.error("ยังคำนวณไม่ได้:\n\n- " + "\n- ".join(problems))
    else:
        left, right = st.columns([2.1,1], gap="large")
        with left:
            st.pyplot(draw_stack_with_alignment(layer_L_box, layer_W_box, cart_H_cm,
                                                nx, ny, layers, align_mode=align_mode,
                                                gap=0.0, scale=0.95,
                                                show_pallet=True, pallet_W_cm=pal_W_cm, pallet_L_cm=pal_L_cm),
                      use_container_width=False)
        with right:
            st.markdown("### สรุปการจัดวาง")
            st.markdown(f"- **รูปแบบต่อชั้น**: {nx} × {ny} ({orientation.split()[0]})")
            st.markdown(f"- **การเรียงชั้น**: {align_mode}")
            st.markdown(f"- **ขนาดกองรวม**: {footprint_L:.0f} × {footprint_W:.0f} × {stack_H:.1f} cm {overhang_note}")
            st.markdown(f"- **รวมต่อพาเลต**: **{total_boxes} กล่อง**")
            st.markdown(f"- **พื้นที่ฐานที่ใช้ (L×W)**: {area_pallet_m2:.4f} m²")
            st.markdown(f"- **CBM ต่อพาเลต**: {cbm_per_pallet:.4f} m³")
            if weight_per_carton_kg is not None:
                st.markdown(f"- **น้ำหนัก/กล่อง** ≈ {weight_per_carton_kg:.2f} kg → **น้ำหนักรวม/พาเลต** ≈ {weight_per_carton_kg*total_boxes:.1f} kg")
    st.markdown("</div>", unsafe_allow_html=True)
