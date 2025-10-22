# APP2.py — Carton Designer + Paper Spec + Pallet (full rev with polished SVG flutes)
# Run: streamlit run APP2.py

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- Matplotlib Thai ----------
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Tahoma", "Sarabun", "TH Sarabun New", "Noto Sans Thai",
    "Arial Unicode MS", "Segoe UI", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

# ---------- Palette ----------
PALETTE = {
    "frame":  "#d9a362",
    "inside": (0.42, 0.67, 0.86, 0.95),
    "grid":   "#ffffff",
    "edge":   "#4a4a4a",
    "pallet": "#9c6a3a",
}

# ---------- Unit helper ----------
class Unit:
    factor_to_mm = {"mm": 1.0, "cm": 10.0, "in": 25.4}
    @staticmethod
    def to_mm(x, unit): return x * Unit.factor_to_mm[unit]
    @staticmethod
    def from_mm(x, unit): return x / Unit.factor_to_mm[unit]
    @staticmethod
    def to_cm(x, unit):  # -> cm
        return x * (Unit.factor_to_mm[unit] / 10.0)

# ---------- Models ----------
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

# ---------- Packing/Geometry ----------
def surface_area_box(W, L, H):
    return 2 * (W * L + W * H + L * H)

def _permutations_with_height_labels(w: float, l: float, h: float):
    return [
        (w,l,h,"h"), (w,h,l,"l"),
        (l,w,h,"h"), (l,h,w,"w"),
        (h,w,l,"l"), (h,l,w,"w"),
    ]

def rect_design_min_carton(
    qty, w, l, h,
    locked_axis: Optional[str]=None,
    force_layers: Optional[int]=None,
    max_per_carton: Optional[int]=None
):
    """หา layout กล่องที่ SA ต่ำสุด (จุครบ qty) + จำกัด per_carton ตามน้ำหนัก"""
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
                plan = BasePlan(
                    pattern=f"{ny}×{nx}", per_layer=nx*ny, nx=nx, ny=ny,
                    spacing_x=iw, spacing_y=il, shape="rect",
                    item_w=iw, item_l=il
                )
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
            if force_layers:
                break
    return best

# ---------- Drawing helpers ----------
def make_top_view(W, L, plan, unit, show_fill=False, show_index=False,
                  scale=0.55, title_text: str = "", title_size: int = 12):
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
    ax.set_aspect("equal")
    if title_text:
        ax.set_title(title_text, fontsize=title_size, fontweight="bold")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    return fig

def make_side_view(W, L, H, layers, plan, ih, scale=0.55, title_text: str = "", title_size: int = 11):
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

    if title_text:
        ax.set_title(title_text, fontsize=title_size, fontweight="bold")

    ax.set_xlim(-frame_pad, W + frame_pad)
    ax.set_ylim(-frame_pad, H + frame_pad)
    ax.set_aspect("auto")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
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
    ax.view_init(elev=22, azim=-60)
    ax.set_axis_off()
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
    ax.add_collection3d(box)
    ax.set_box_aspect((x_len,y_len,z_len))
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    plt.subplots_adjust(0,0,1,1)
    fig.tight_layout(pad=0.02)
    return fig

# ---------- Summary card ----------
def render_summary_box_cm(
    L_cm,W_cm,H_cm,layers,per_layer,
    sa_cm2=None,
    tol_w_cm=None,tol_l_cm=None,tol_h_cm=None,
    cartons_needed=None,
    weight_kg=None,limit_kg=18.0,
    area_m2=None, cbm_per_carton=None, cbm_total=None
):
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
    rows = [
        ("ขนาดกล่อง", size_line + dims_html),
        ("จำนวนชั้น", f"{layers} ชั้น"),
        ("ชั้นละ",     f"{per_layer} ชิ้น"),
        ("รวมทั้งกล่อง", f"{total} ชิ้น"),
    ]
    if cartons_needed is not None:
        rows.append(("จำนวนกล่อง", f"{cartons_needed} กล่อง"))
    if sa_cm2 is not None:
        rows.append(("Surface Area", f"{sa_cm2:.2f} cm²"))
    if any(v is not None for v in (tol_w_cm,tol_l_cm,tol_h_cm)):
        rows.append(("Tolerance",
                     f"W {tol_w_cm or 0:.2f} cm · L {tol_l_cm or 0:.2f} cm · H {tol_h_cm or 0:.2f} cm"))
    if area_m2 is not None:
        rows.append(("พื้นที่ฐาน L×W", f"{area_m2:.4f} m²"))
    if cbm_per_carton is not None:
        rows.append(("ปริมาตร (CBM) ต่อกล่อง", f"{cbm_per_carton:.4f} m³"))
    if cbm_total is not None:
        rows.append(("CBM รวมทั้งหมด", f"{cbm_total:.4f} m³"))
    if weight_kg is not None:
        status = "✅" if weight_kg <= limit_kg + 1e-9 else "⛔"
        color_cls = "ok" if weight_kg <= limit_kg + 1e-9 else "bad"
        rows.append(("น้ำหนักรวม/กล่อง", f"<span class='{color_cls}'>{status} {weight_kg:.2f} kg</span> (จำกัด ≤ {limit_kg:.2f} kg)"))

    html = "".join(f"<div class='sumk'>{k}</div><div class='sumv'>{v}</div>" for k, v in rows)
    st.markdown(f"<div class='sumcard'><div class='sumh'>กล่องแนะนำ</div><div class='sumrow'>{html}</div></div>",
                unsafe_allow_html=True)

# ---------- 3D utils ----------
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

    face_c = (0.30, 0.50, 0.75, 0.95)
    edge_c = "#1a3a5a"

    for k in range(layers):
        z0 = k*carton_H_cm
        for j in range(ny):
            for i in range(nx):
                x = i*(carton_L_cm + gap)
                y = j*(carton_W_cm + gap)
                ax.add_collection3d(Poly3DCollection(
                    _faces(x, y, z0, carton_L_cm, carton_W_cm, carton_H_cm),
                    facecolors=face_c, edgecolors=edge_c, linewidths=0.9, zsort="max"
                ))

    pad_L = max(6, 0.10*used_L)
    pad_W = max(6, 0.10*used_W)
    pad_H = max(6, 0.10*used_H)

    ax.set_xlim(-pad_L, used_L + pad_L)
    ax.set_ylim(-pad_W, used_W + pad_W)
    ax.set_zlim(0, used_H + pad_H)
    ax.set_box_aspect((used_L + 2*pad_L, used_W + 2*pad_W, used_H + pad_H))

    ax.view_init(elev=26, azim=-42)
    ax.set_axis_off()
    fig.tight_layout(pad=0.1)
    return fig

# ---------- Paper DB (example) ----------
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

FLUTE_BASE_ECT = {
    "E":  4.2,
    "B":  5.3,
    "C":  6.0,
    "BE": 7.6,
    "BC": 8.8,
}

def estimate_ect(flute: str,
                 liners: Tuple[str, ...],
                 mediums: Tuple[str, ...]) -> float:
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

# ---------- Derating factors ----------
def factor_stacking_method(method: str) -> float:  # column better
    return 1.00 if method == "column" else 0.85
def factor_base_surface(base: str) -> float:       # pallet better
    return 1.00 if base == "pallet" else 0.90
def factor_humidity(rh: str) -> float:
    return {"<=50%":1.00,"50–70%":0.95,"70–85%":0.85,">85%":0.75}[rh]
def factor_storage_time(t: str) -> float:
    return {"< 1 เดือน":1.00,"1–3 เดือน":0.92,"> 3 เดือน":0.85}[t]
def factor_handling(times: str) -> float:
    return {"น้อย (≤3 ครั้ง)":1.00,"ปานกลาง (4–10)":0.95,"บ่อย (>10)":0.90}[times]
def total_derating(method, base, rh, stg, handling, safety: float) -> float:
    f = (factor_stacking_method(method) *
         factor_base_surface(base) *
         factor_humidity(rh) *
         factor_storage_time(stg) *
         factor_handling(handling) *
         safety)
    return min(f, 1.0)

# ---------- SVG flutes (polished) ----------
def _svg_wave_points(width, amp, period, offset_y, steps=200):
    """คืนจุด polyline ของรูปคลื่น (อยู่ในช่วง 0..width)"""
    import math
    pts = []
    for i in range(steps + 1):
        x = width * i / steps
        y = offset_y + amp * math.sin(2 * math.pi * (x / period))
        pts.append(f"{x:.2f},{y:.2f}")
    return " ".join(pts)

def flute_svg(flute: str, wall: str):
    """
    SVG แสดงหน้าตัดกระดาษลอน:
    - single-wall: liner 2 เส้น + ลอน 1 ชุด
    - double-wall (BC): liner 3 เส้น + ลอน 2 ชุด
    ใช้ clipPath กันคลื่นล้นกรอบ และ padding ซ้าย/ขวา
    """
    W, H = 360, 80
    PADX = 16
    RADIUS = 10
    BG = "#c89b6b"
    LINER = "#7c5936"
    FLUTE1 = "#6f4c2b"
    FLUTE2 = "#5f4126"

    conf = {
        "E":  {"amp": 5, "period": 26, "t": 26},
        "B":  {"amp": 7, "period": 34, "t": 32},
        "C":  {"amp": 9, "period": 44, "t": 40},
        "BC": {"amp1": 8, "period1": 36, "amp2": 11, "period2": 52, "t": 64},
    }

    if wall == "double":
        flute = "BC"

    clip_id = f"clip_{flute}_{wall}".replace(" ", "_")
    innerW = W - 2 * PADX

    if flute != "BC":
        p = conf[flute]
        top = (H - p["t"]) // 2
        bottom = top + p["t"]
        mid = (top + bottom) // 2

        svg = [f"<svg viewBox='0 0 {W} {H}' width='100%' height='60' xmlns='http://www.w3.org/2000/svg'>"]
        svg.append(f"<rect x='{PADX}' y='{top}' width='{innerW}' height='{p['t']}' rx='{RADIUS}' ry='{RADIUS}' fill='{BG}'/>")
        svg.append(f"<clipPath id='{clip_id}'>"
                   f"<rect x='{PADX}' y='{top}' width='{innerW}' height='{p['t']}' rx='{RADIUS}' ry='{RADIUS}'/>"
                   f"</clipPath>")
        svg.append(f"<line x1='{PADX+10}' y1='{top+6}' x2='{W-PADX-10}' y2='{top+6}' stroke='{LINER}' stroke-width='2' opacity='.85'/>")
        svg.append(f"<line x1='{PADX+10}' y1='{bottom-6}' x2='{W-PADX-10}' y2='{bottom-6}' stroke='{LINER}' stroke-width='2' opacity='.85'/>")
        pts = _svg_wave_points(innerW-20, p["amp"], p["period"], mid)
        svg.append(f"<g clip-path='url(#{clip_id})'>"
                   f"<polyline points=' {PADX+10:.2f},0 {pts} ' transform='translate(10,0)' "
                   f"fill='none' stroke='{FLUTE1}' stroke-width='2.2'/>"
                   f"</g>")
        svg.append("</svg>")
        return "".join(svg)

    # BC (double wall)
    p = conf["BC"]
    top = (H - p["t"]) // 2
    bottom = top + p["t"]
    midliner = (top + bottom) // 2
    y1 = top + 16
    y2 = bottom - 16

    svg = [f"<svg viewBox='0 0 {W} {H}' width='100%' height='60' xmlns='http://www.w3.org/2000/svg'>"]
    svg.append(f"<rect x='{PADX}' y='{top}' width='{innerW}' height='{p['t']}' rx='{RADIUS}' ry='{RADIUS}' fill='{BG}'/>")
    svg.append(f"<clipPath id='{clip_id}'>"
               f"<rect x='{PADX}' y='{top}' width='{innerW}' height='{p['t']}' rx='{RADIUS}' ry='{RADIUS}'/>"
               f"</clipPath>")
    svg.append(f"<line x1='{PADX+10}' y1='{top+6}' x2='{W-PADX-10}' y2='{top+6}' stroke='{LINER}' stroke-width='2' opacity='.85'/>")
    svg.append(f"<line x1='{PADX+10}' y1='{midliner}' x2='{W-PADX-10}' y2='{midliner}' stroke='{LINER}' stroke-width='2' opacity='.85'/>")
    svg.append(f"<line x1='{PADX+10}' y1='{bottom-6}' x2='{W-PADX-10}' y2='{bottom-6}' stroke='{LINER}' stroke-width='2' opacity='.85'/>")
    pts1 = _svg_wave_points(innerW-20, p["amp1"], p["period1"], y1)
    pts2 = _svg_wave_points(innerW-20, p["amp2"], p["period2"], y2)
    svg.append(f"<g clip-path='url(#{clip_id})'>"
               f"<polyline points=' {PADX+10:.2f},0 {pts1} ' transform='translate(10,0)' fill='none' stroke='{FLUTE1}' stroke-width='2.2'/>"
               f"<polyline points=' {PADX+10:.2f},0 {pts2} ' transform='translate(10,0)' fill='none' stroke='{FLUTE2}' stroke-width='2.2'/>"
               f"</g>")
    svg.append("</svg>")
    return "".join(svg)

# ---------- UI shell ----------
st.set_page_config(page_title="Carton Designer", layout="wide")
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;
padding:1.0rem;border-radius:12px;text-align:center;margin-bottom:0.8rem}
.card{background:#fff;border:1px solid #ebedf0;border-radius:12px;padding:14px 16px;
box-shadow:0 5px 14px rgba(0,0,0,.04);margin-bottom:12px}
</style>
<div class="main-header"><h1>📦 ตัวช่วยออกแบบขนาดกล่อง & เลือกกระดาษ & วางพาเลต</h1>
<p>Giftbox ➜ Carton + 📄 Paper Spec + Carton ➜ Pallet</p></div>
""", unsafe_allow_html=True)

tab_carton, tab_paper, tab_pallet = st.tabs([
    "🎁 Giftbox ➜ Carton",
    "📄 Paper Spec",
    "🧱 Carton ➜ Pallet"
])

# ---------- PAGE 1: Giftbox ➜ Carton ----------
with tab_carton:
    col_main, col_side = st.columns([8,4], gap="large")

    with col_main:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧩 ลักษณะของสินค้า")
        product_type = st.selectbox(
            "เลือกหมวดสินค้า",
            [
                "ตลับ/พาเลตเมคอัพ", "แท่งลิป/บาล์ม", "ดินสอ/อายไลเนอร์",
                "มาสคาร่า/คอนซีลเลอร์", "กระปุกครีม", "ขวดดรอปเปอร์",
                "ปั๊ม/เออร์เลส", "โทนเนอร์/สเปรย์", "แอมพูล/ไวอัล",
                "ซองครีม/ซองเจล", "ชีทมาสก์", "สแตนด์อัพพาวช์",
                "บลิสเตอร์/การ์ด", "กิฟต์เซ็ต/มัลติแพ็ก", "อื่น ๆ"
            ],
            index=0
        )
        st.caption("ทรงเหลี่ยม (Rectangular)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📦 ตั้งค่าการวาง Gift box")
        unit = st.selectbox("📏 หน่วยที่ใช้", ["mm","cm","in"], index=0)
        st.divider()

        c1,c2,c3 = st.columns(3)
        gb_w = c1.number_input("ขนาด Giftbox — กว้าง (W)", min_value=0.1, value=120.0, step=1.0)
        gb_l = c2.number_input("ขนาด Giftbox — ยาว (L)",   min_value=0.1, value=200.0, step=1.0)
        gb_h = c3.number_input("ขนาด Giftbox — สูง (H)",   min_value=0.1, value=80.0,  step=1.0)

        qty = st.number_input("จำนวน giftbox ที่ต้องการ (รวม)", 1, 999999, 20, 1)

        cwg1, cwg2 = st.columns([1,1])
        gb_weight_g = cwg1.number_input("น้ำหนัก giftbox ต่อชิ้น (g)", min_value=0.0, value=0.0, step=1.0,
                                        help="ใส่ 0 ถ้าไม่ต้องการจำกัดด้วยน้ำหนัก")
        weight_limit_kg = 18.0
        cwg2.markdown(f"**น้ำหนักสูงสุด/กล่อง:** ≤ {weight_limit_kg:.2f} kg")

        auto_layers = st.checkbox("คำนวณจำนวนชั้นอัตโนมัติ", value=True)
        layers_for_one = None
        if not auto_layers:
            layers_for_one = st.number_input("จำนวนชั้นที่ต้องการวาง (ต่อ 1 กล่อง)", 1, 50, 1, 1)

        st.markdown("### 🔁 เลือกทิศทางการวาง")
        lock_opt = st.radio(
            "ทิศทาง (แกนที่ตั้งอยู่ด้านบน)",
            ["ไม่ล็อก – ให้ระบบลองทุกแบบ", "วางแบบเดิม – H ขึ้น",
             "พลิกให้ด้านกว้างขึ้น – W ขึ้น", "พลิกให้ด้านยาวขึ้น – L ขึ้น"],
            horizontal=False, index=0
        )
        lock_map = {"ไม่ล็อก – ให้ระบบลองทุกแบบ":"auto","วางแบบเดิม – H ขึ้น":"h","พลิกให้ด้านกว้างขึ้น – W ขึ้น":"w","พลิกให้ด้านยาวขึ้น – L ขึ้น":"l"}
        locked_axis = lock_map[lock_opt]

        st.caption("รูปช่วยจำอัตราส่วน W/L/H")
        st.pyplot(render_wlh_diagram_oriented(gb_w, gb_l, gb_h,
                  up=(locked_axis if locked_axis!="auto" else "h")),
                  use_container_width=False)
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

    st.markdown("<br>", unsafe_allow_html=True)
    calc = st.button("🚀 คำนวณขนาดกล่องที่เหมาะสม", use_container_width=True)

    def apply_tolerance_wlh(w,l,h, add_w, add_l, add_h):
        return w + add_w, l + add_l, h + add_h

    if calc:
        max_per_carton = None
        if gb_weight_g > 0:
            max_per_carton = int(18000 // gb_weight_g)
            if max_per_carton < 1:
                st.error("⛔ น้ำหนักต่อชิ้นมากเกินไป (แม้ 1 กล่องก็เกิน 18 kg)")
                st.stop()

        deltas = [round(x,2) for x in np.arange(0,0.51,0.05)] if try_bruteforce else [0.0]
        best = None; best_tol = None
        for d in deltas:
            iw, il, ih = apply_tolerance_wlh(gb_w, gb_l, gb_h, tol_w + d, tol_l + d, tol_h + d)
            axis_forced = None if locked_axis=="auto" else locked_axis
            res = rect_design_min_carton(
                qty, iw, il, ih, locked_axis=axis_forced, force_layers=layers_for_one,
                max_per_carton=max_per_carton
            )
            if res and (best is None or res.meta["SA"] < best.meta["SA"]):
                best = res; best_tol = (tol_w + d, tol_l + d, tol_h + d)

        if not best:
            st.error("ไม่พบวิธีจัดวางที่เหมาะสมภายใต้เงื่อนไขปัจจุบัน")
        else:
            st.session_state.result = best
            st.session_state.tol_w, st.session_state.tol_l, st.session_state.tol_h = best_tol
            st.session_state.unit = unit
            st.session_state.qty = qty
            st.session_state.gb_weight_g = gb_weight_g
            st.success("✅ คำนวณสำเร็จ! ➜ ไปหน้า 'Paper Spec' เพื่อเลือกกระดาษ/ลอน แล้วค่อยไปวางพาเลต")

    res = st.session_state.get("result")
    if res:
        tol_w_sv = st.session_state.get("tol_w", 0.0)
        tol_l_sv = st.session_state.get("tol_l", 0.0)
        tol_h_sv = st.session_state.get("tol_h", 0.0)
        base_unit = st.session_state.get("unit", "mm")
        qty_saved  = st.session_state.get("qty", 1)
        gb_weight_g_saved = st.session_state.get("gb_weight_g", 0.0)

        iw,il,ih = res.orientation
        W = res.plan.ny * res.plan.spacing_x
        L = res.plan.nx * res.plan.spacing_y
        H = res.layers * ih

        W_cm = Unit.to_cm(W, base_unit)
        L_cm = Unit.to_cm(L, base_unit)
        H_cm = Unit.to_cm(H, base_unit)
        SA_cm2 = 2*(W_cm*L_cm + W_cm*H_cm + L_cm*H_cm)
        tol_w_cm = Unit.to_cm(tol_w_sv, base_unit)
        tol_l_cm = Unit.to_cm(tol_l_sv, base_unit)
        tol_h_cm = Unit.to_cm(tol_h_sv, base_unit)

        per_carton = res.layers * res.per_layer
        need_cartons = math.ceil(qty_saved / per_carton)

        area_m2 = (L_cm * W_cm) / 1e4
        cbm_per_carton = (L_cm * W_cm * H_cm) / 1e6
        cbm_total = cbm_per_carton * need_cartons if need_cartons else None

        weight_per_carton_kg = None
        if gb_weight_g_saved and gb_weight_g_saved > 0:
            weight_per_carton_kg = (per_carton * gb_weight_g_saved) / 1000.0

        st.divider()
        st.header("🖼️ Visualization")

        tab2d, tab3d = st.tabs(["📐 มุมบน/ข้าง (2D)", "📦 ซ้อนชั้น (3D)"])
        with tab2d:
            left, right = st.columns([3,2])
            with right:
                fill = st.checkbox("🟩 แสดงสี", True)
                idx  = st.checkbox("🔢 หมายเลข", False)
                st.markdown("### สรุป (หน่วย cm / m² / m³)")
                render_summary_box_cm(
                    L_cm, W_cm, H_cm, res.layers, res.per_layer,
                    SA_cm2,
                    tol_w_cm, tol_l_cm, tol_h_cm,
                    need_cartons,
                    weight_kg=weight_per_carton_kg, limit_kg=18.0,
                    area_m2=area_m2, cbm_per_carton=cbm_per_carton, cbm_total=cbm_total
                )
                if weight_per_carton_kg is not None and weight_per_carton_kg > 18.0 + 1e-9:
                    st.error("⛔ น้ำหนักรวมต่อกล่องเกิน 18 kg — กรุณาลดจำนวนชั้นหรือจำนวนชิ้นต่อชั้น")
            with left:
                st.pyplot(make_top_view(W,L,res.plan,base_unit, show_fill=fill, show_index=idx,
                                        scale=0.55, title_text="", title_size=12),
                          use_container_width=False)
            st.markdown("---")
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**มุมบน**")
                st.pyplot(make_top_view(W,L,res.plan,base_unit, True, False, 0.45, "", 10),
                          use_container_width=False)
            with cB:
                st.markdown("**มุมข้าง**")
                st.pyplot(make_side_view(W,L,H,res.layers,res.plan,ih, 0.45, "", 10),
                          use_container_width=False)
        with tab3d:
            st.pyplot(make_3d_stack(W,L,res.plan,res.layers,ih, scale=0.7),
                      use_container_width=False)

# ---------- Helper: pull from session ----------
def get_carton_from_session():
    res = st.session_state.get("result")
    if not res: return None
    base_unit = st.session_state.get("unit", "mm")
    iw, il, ih = res.orientation
    W = res.plan.ny * res.plan.spacing_x
    L = res.plan.nx * res.plan.spacing_y
    H = res.layers * ih
    L_cm = Unit.to_cm(L, base_unit)
    W_cm = Unit.to_cm(W, base_unit)
    H_cm = Unit.to_cm(H, base_unit)
    per_carton = res.layers * res.per_layer
    gb_weight_g = st.session_state.get("gb_weight_g", 0.0)
    weight_kg = (per_carton * gb_weight_g / 1000.0) if gb_weight_g > 0 else None
    return (L_cm, W_cm, H_cm, per_carton, weight_kg)

# ---------- PAGE 1.5: Paper Spec ----------
with tab_paper:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 กระดาษลูกฟูก & ความแข็งแรง")

    data = get_carton_from_session()
    if not data:
        st.warning("ยังไม่มีผลจากหน้า Giftbox ➜ Carton กรุณาคำนวณก่อน แล้วค่อยกลับมาหน้านี้ครับ")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        L_cm, W_cm, H_cm, per_carton, weight_kg = data
        qty = st.session_state.get("qty", 1)
        cbm = (L_cm * W_cm * H_cm) / 1e6

        c1, c2, c3 = st.columns(3)
        c1.metric("ขนาด Carton (cm)", f"{L_cm:.1f} × {W_cm:.1f} × {H_cm:.1f}")
        c2.metric("ชิ้น/กล่อง", f"{per_carton}")
        c3.metric("CBM ต่อกล่อง", f"{cbm:.4f} m³")
        if weight_kg is not None:
            st.info(f"น้ำหนักรวมต่อกล่อง ≈ **{weight_kg:.2f} kg**")

        st.markdown("---")
        st.markdown("### 1) เลือกความหนากระดาษ/ลอน")

        st.markdown("""
        <style>
        .cardPick{border:1px solid #e6e8ec;border-radius:12px;padding:12px 14px;margin:8px 0 16px 0;
                  box-shadow:0 3px 10px rgba(0,0,0,.03)}
        </style>
        """, unsafe_allow_html=True)

        if "paper_mode" not in st.session_state:
            st.session_state.paper_mode = "5 ชั้น"
        if "paper_flute3" not in st.session_state:
            st.session_state.paper_flute3 = "E"

        col3, col5 = st.columns(2)
        with col3:
            st.markdown("<div class='cardPick'>", unsafe_allow_html=True)
            st.markdown("**กระดาษหนา 3 ชั้น**")
            pick3 = st.radio("ลอน (3 ชั้น)", ["B","C","E"],
                             index=["B","C","E"].index(st.session_state.paper_flute3),
                             horizontal=True, label_visibility="collapsed", key="flute3_radio")
            st.session_state.paper_flute3 = pick3
            st.markdown(flute_svg(st.session_state.paper_flute3, "single"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col5:
            st.markdown("<div class='cardPick'>", unsafe_allow_html=True)
            st.markdown("**กระดาษหนา 5 ชั้น**")
            st.radio("ลอน (5 ชั้น)", ["BC"], index=0, horizontal=True, label_visibility="collapsed", key="flute5_radio")
            st.markdown(flute_svg("BC", "double"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        mode = st.radio("โหมดที่ใช้คำนวณ", ["3 ชั้น", "5 ชั้น"], horizontal=True,
                        index=(0 if st.session_state.paper_mode=="3 ชั้น" else 1), key="paper_mode_radio")
        st.session_state.paper_mode = mode

        T_DEFAULT = {"E":2.5, "B":3.0, "C":4.0, "BC":6.5}
        if st.session_state.paper_mode == "3 ชั้น":
            wall  = "single"
            flute = st.session_state.paper_flute3
            t_mm  = T_DEFAULT[flute]
        else:
            wall  = "double"
            flute = "BC"
            t_mm  = T_DEFAULT["BC"]

        t_mm = st.number_input("ความหนาบอร์ด (mm)", 2.5, 12.0, float(t_mm), 0.1)

        # ---- Liner/Medium by wall ----
        st.caption("เลือกเกรดกระดาษ (ตัวอย่าง ปรับตารางได้)")
        if wall == "single":
            colL, colM, colR = st.columns(3)
            liner_out = colL.selectbox("Liner - ผิวนอก", LINER_GRADES, index=LINER_GRADES.index("KA185"))
            medium1   = colM.selectbox("Medium - แผ่นลอน", MEDIUM_GRADES, index=MEDIUM_GRADES.index("Medium150"))
            liner_in  = colR.selectbox("Liner - ผิวใน", LINER_GRADES, index=LINER_GRADES.index("CA185"))
            ect_knm = estimate_ect(flute, (liner_out, liner_in), (medium1,))
        else:
            colL, colM1, colCen, colM2, colR = st.columns(5)
            liner_out = colL.selectbox("Liner - ผิวนอก", LINER_GRADES, index=LINER_GRADES.index("KA230"))
            medium1   = colM1.selectbox("Medium 1 - ลอนแรก", MEDIUM_GRADES, index=MEDIUM_GRADES.index("Medium150"))
            liner_mid = colCen.selectbox("Liner - ชั้นกลาง", LINER_GRADES, index=LINER_GRADES.index("KI185"))
            medium2   = colM2.selectbox("Medium 2 - ลอนสอง", MEDIUM_GRADES, index=MEDIUM_GRADES.index("Medium150"))
            liner_in  = colR.selectbox("Liner - ผิวใน", LINER_GRADES, index=LINER_GRADES.index("CA185"))
            ect_knm = estimate_ect(flute, (liner_out, liner_mid, liner_in), (medium1, medium2))

        P_cm = 2.0 * (L_cm + W_cm)
        bct_N = mckee_bct_estimate(ect_knm, P_cm, t_mm)
        bct_kgf = bct_N / 9.81

        st.markdown("### 2) เงื่อนไขการกองเก็บ/ใช้งาน")
        cS1, cS2 = st.columns(2)
        method = cS1.radio("วิธีเรียงซ้อน", ["column","interlocking"], index=0, horizontal=True)
        base   = cS2.radio("พื้นวาง", ["pallet","floor"], index=0, horizontal=True)

        cS3, cS4, cS5 = st.columns(3)
        rh = cS3.selectbox("ความชื้นสัมพัทธ์ (RH)", ["<=50%","50–70%","70–85%",">85%"], index=1)
        stg= cS4.selectbox("ระยะเวลาจัดเก็บ", ["< 1 เดือน","1–3 เดือน","> 3 เดือน"], index=1)
        hd = cS5.selectbox("จำนวนครั้งการเคลื่อนย้าย", ["น้อย (≤3 ครั้ง)","ปานกลาง (4–10)","บ่อย (>10)"], index=1)

        safety = st.slider("Safety factor", min_value=0.50, max_value=0.90, value=0.60, step=0.01)
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
                st.info(f"ชั้นซ้อนแนะนำสูงสุด ≈ {max_layers} ชั้น ด้วยสเปก/ปัจจัยปัจจุบัน")
        else:
            st.warning("ยังไม่มีน้ำหนักรวมต่อกล่อง — กรุณาใส่น้ำหนัก/ชิ้นในแท็บแรกเพื่อคำนวณความแข็งแรง")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- PAGE 2: Carton ➜ Pallet ----------
with tab_pallet:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧱 วาง Carton")

    use_from_prev = st.checkbox("📥 ดึงขนาด Carton จากหน้า Giftbox ➜ Carton (ถ้ามีผลคำนวณ)", value=True)
    if use_from_prev and st.session_state.get("result") is not None:
        res = st.session_state["result"]
        iw, il, ih = res.orientation
        cart_W = res.plan.ny * res.plan.spacing_x
        cart_L = res.plan.nx * res.plan.spacing_y
        cart_H = ih
        base_unit = st.session_state.get("unit","mm")
        cart_W_cm = Unit.to_cm(cart_W, base_unit)
        cart_L_cm = Unit.to_cm(cart_L, base_unit)
        cart_H_cm = Unit.to_cm(cart_H, base_unit)
        st.info(f"ขนาดจากผลคำนวณ: W={cart_W_cm:.2f} cm, L={cart_L_cm:.2f} cm, H={cart_H_cm:.2f} cm")
    else:
        c1,c2,c3 = st.columns(3)
        cart_W_cm = c1.number_input("กว้าง (W) ของกล่อง (cm)", 1.0, 200.0, 30.0, 0.5)
        cart_L_cm = c2.number_input("ยาว (L) ของกล่อง (cm)",   1.0, 200.0, 40.0, 0.5)
        cart_H_cm = c3.number_input("สูง (H) ของกล่อง (cm)",   1.0, 200.0, 20.0, 0.5)

    p1, p2, p3, p4 = st.columns(4)
    pal_W_cm = p1.number_input("กว้างพาเลต (cm)", 50.0, 200.0, 100.0, 1.0)
    pal_L_cm = p2.number_input("ยาวพาเลต (cm)",   60.0, 200.0, 120.0, 1.0)
    pal_Hmax = p3.number_input("สูงรวมไม่เกิน (cm)", 60.0, 300.0, 180.0, 1.0)
    pal_H    = p4.number_input("ความสูงพาเลต (cm)",  5.0,  40.0,  12.0, 0.5)

    st.markdown("### การจัดวาง")
    n1, n2, n3 = st.columns([1.1,1.1,1])
    nx = n1.number_input("จำนวนคอลัมน์ตามแนวยาวพาเลต (nx)", 0, 100, 2, 1)
    ny = n2.number_input("จำนวนแถวตามแนวกว้างพาเลต (ny)", 0, 100, 2, 1)
    per_layer = nx * ny
    n3.metric("รวมต่อชั้น", per_layer)

    layers = st.number_input("จำนวนชั้น (layers) — วางกี่ชั้น", 0, 50, 5, 1)

    problems = []
    if nx * cart_L_cm > pal_L_cm:
        problems.append(f"กล่องแนวยาวล้นพาเลต: nx×L = {nx*cart_L_cm:.1f} > {pal_L_cm:.1f} cm")
    if ny * cart_W_cm > pal_W_cm:
        problems.append(f"กล่องแนวกว้างล้นพาเลต: ny×W = {ny*cart_W_cm:.1f} > {pal_W_cm:.1f} cm")
    total_height = pal_H + layers * cart_H_cm
    if total_height > pal_Hmax:
        problems.append(f"ความสูงรวมเกินกำหนด: {total_height:.1f} > {pal_Hmax:.1f} cm")

    if problems:
        st.error("ไม่สามารถวางได้ด้วยค่าปัจจุบัน:\n\n- " + "\n- ".join(problems))
    else:
        left, right = st.columns([2.1, 1], gap="large")
        with left:
            st.pyplot(
                draw_carton_stack_only(
                    cart_L_cm, cart_W_cm, cart_H_cm,
                    nx, ny, layers, gap=0.0, scale=0.95
                ),
                use_container_width=False
            )
        with right:
            total_boxes  = per_layer * layers
            stack_H      = layers * cart_H_cm
            footprint_L  = nx * cart_L_cm
            footprint_W  = ny * cart_W_cm
            area_pallet_m2 = (footprint_L * footprint_W) / 1e4
            cbm_per_pallet = (footprint_L * footprint_W * stack_H) / 1e6

            st.markdown("### 📄 สรุปการเรียงบนพาเลต")
            st.markdown("""
            <style>
              .sumcardB{background:#eef6ff;padding:12px 14px;border-radius:10px;
                        border:1px solid #cde2ff;box-shadow:0 4px 10px rgba(0,0,0,.04);font-size:15px}
              .sumhB{font-size:18px;font-weight:800;color:#113e7e;margin-bottom:8px}
              .bul{margin:0;padding-left:18px}
              .bul li{margin:6px 0}
              .subbul{margin:2px 0 6px 18px;padding-left:18px;list-style:circle}
            </style>
            """, unsafe_allow_html=True)

            html = f"""
            <div class='sumcardB'>
              <div class='sumhB'>สรุป</div>
              <ul class='bul'>
                <li><b>ขนาดกองกล่องรวม</b>: {footprint_L:.0f} × {footprint_W:.0f} × {stack_H:.1f} cm</li>
                <ul class='subbul'>
                  <li>ยาว (L): <b>{footprint_L:.0f} cm</b></li>
                  <li>กว้าง (W): <b>{footprint_W:.0f} cm</b></li>
                  <li>สูง (H): <b>{stack_H:.1f} cm</b></li>
                </ul>
                <li><b>วางต่อชั้น</b>: {per_layer} กล่อง (nx={nx}, ny={ny})</li>
                <li><b>จำนวนชั้น</b>: {layers} ชั้น</li>
                <li><b>รวมต่อพาเลต</b>: {total_boxes} กล่อง</li>
                <li><b>พื้นที่ฐานพาเลต (L×W)</b>: {area_pallet_m2:.4f} m²</li>
                <li><b>CBM ต่อพาเลต</b>: {cbm_per_pallet:.4f} m³</li>
              </ul>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
