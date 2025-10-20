# APP2.py — Carton Designer (ทรงเหลี่ยม) + Visualization + Pallet (Manual-only)
# Run: streamlit run APP2.py

import math
from dataclasses import dataclass
from typing import Tuple
import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------- Thai font ----------------
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Tahoma", "Sarabun", "TH Sarabun New", "Noto Sans Thai",
    "Arial Unicode MS", "Segoe UI", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

# ---------------- Palette ----------------
PALETTE = {
    "frame":  "#d9a362",                 # น้ำตาลกรอบนอก
    "inside": (0.42, 0.67, 0.86, 0.95),  # ฟ้าด้านใน
    "grid":   "#ffffff",                 # เส้นแบ่งสีขาว
    "edge":   "#4a4a4a",                 # ขอบด้านใน
    "pallet": "#9c6a3a",                 # สีพาเลต
}

# ---------------- Unit helper ----------------
class Unit:
    factor_to_mm = {"mm": 1.0, "cm": 10.0, "in": 25.4}
    @staticmethod
    def to_mm(x, unit): return x * Unit.factor_to_mm[unit]
    @staticmethod
    def from_mm(x, unit): return x / Unit.factor_to_mm[unit]
    @staticmethod
    def to_cm(x, unit):  # -> cm
        return x * (Unit.factor_to_mm[unit] / 10.0)

# ---------------- Models & utils ----------------
@dataclass
class BasePlan:
    pattern: str
    per_layer: int
    nx: int               # columns along L
    ny: int               # rows along W
    spacing_x: float      # step along W
    spacing_y: float      # step along L
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

def surface_area_box(W, L, H): 
    return 2 * (W * L + W * H + L * H)

def _permutations_with_height_labels(w: float, l: float, h: float):
    return [
        (w,l,h,"h"), (w,h,l,"l"),
        (l,w,h,"h"), (l,h,w,"w"),
        (h,w,l,"l"), (h,l,w,"w"),
    ]

def rect_design_min_carton(qty, w, l, h, locked_axis=None, force_layers=None,
                           disallow_front_up=False, keep_top_up=False):
    perms = _permutations_with_height_labels(w,l,h)
    if keep_top_up:        perms = [p for p in perms if p[3] == "h"]
    if disallow_front_up:  perms = [p for p in perms if p[3] != "l"]
    if locked_axis:        perms = [p for p in perms if p[3] == locked_axis]
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
                total = nx*ny*layers
                if total < qty: 
                    continue
                SA = surface_area_box(W, L, H)
                res = PackResult((iw, il, ih), layers, nx*ny, total, plan,
                                 {"SA": SA, "height_from": hfrom})
                if (best is None) or (SA < best.meta["SA"]) or (SA == best.meta["SA"] and total < best.total):
                    best = res
            if force_layers:
                break
    return best

# ---------------- Drawing (Top / Side / 3D carton) ----------------
def make_top_view(W, L, plan, unit, show_fill=False, show_index=False,
                  scale=0.55, title_text=None, title_size=12):
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
                           facecolor=PALETTE["inside"], edgecolor=PALETTE["edge"],
                           linewidth=1.5))

    bw, bl = plan.spacing_x, plan.spacing_y
    idx = 1
    for iy in range(plan.ny):
        for ix in range(plan.nx):
            x, y = ix*bl, iy*bw
            ax.add_patch(Rectangle((x, y), bl, bw, 
                                   fill=False, edgecolor=PALETTE["grid"], linewidth=1.6))
            if show_index:
                ax.text(x + bl/2, y + bw/2, str(idx), ha="center", va="center", fontsize=7, color="#333")
            idx += 1

    ax.set_xlim(-frame_pad, L + frame_pad)
    ax.set_ylim(-frame_pad, W + frame_pad)
    ax.set_aspect("equal")

    if title_text is None:
        title_text = f"มุมบน: {W:.2f}×{L:.2f} {unit}"
    if title_text != "":
        ax.set_title(title_text, fontsize=title_size, fontweight="bold")

    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    return fig

def make_side_view(W, L, H, layers, plan, ih, scale=0.55, title_text=None, title_size=11):
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

    if title_text is None:
        title_text = f"มุมข้าง: {W:.2f}×{H:.2f}"
    if title_text != "":
        ax.set_title(title_text, fontsize=title_size, fontweight="bold")

    ax.set_xlim(-frame_pad, W + frame_pad)
    ax.set_ylim(-frame_pad, H + frame_pad)
    ax.set_aspect("auto")

    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    return fig

def make_3d_stack(W, L, plan, layers, ih, scale=0.8):
    """3D ซ้อนชั้นในกล่อง"""
    fig = plt.figure(figsize=(4.2*scale, 3.1*scale))
    ax = fig.add_subplot(111, projection="3d")
    z_unit = ih

    for z in range(layers):
        for iy in range(plan.ny):
            for ix in range(plan.nx):
                ax.bar3d(ix*plan.spacing_y, iy*plan.spacing_x, z*z_unit,
                         plan.spacing_y,    plan.spacing_x,    z_unit,
                         shade=True, alpha=.43)

    face = [(0,0,0), (L,0,0), (L,0,layers*z_unit), (0,0,layers*z_unit)]
    poly = Poly3DCollection([face], facecolors=(0.1,0.6,1.0,0.08),
                            edgecolors="dodgerblue", linewidths=1.0)
    ax.add_collection3d(poly)
    ax.text(L/2, -0.06*W, layers*z_unit*0.95, "FRONT", ha="center", color="dodgerblue", fontsize=7)

    ax.set_xlabel("L"); ax.set_ylabel("W"); ax.set_zlabel("H")
    ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, layers*z_unit)
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout(pad=0.05)
    return fig

def render_wlh_diagram_oriented(w, l, h, up="h", figsize=(2.6,2.0)):
    up = (up or "h").lower()
    if up == "h":    x_len,y_len,z_len = w,l,h
    elif up == "w":  x_len,y_len,z_len = l,h,w
    else:            x_len,y_len,z_len = w,h,l

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    verts = [
        [(0,0,0),(x_len,0,0),(x_len,y_len,0),(0,y_len,0)],
        [(0,0,z_len),(x_len,0,z_len),(x_len,y_len,z_len),(0,y_len,z_len)],
        [(0,0,0),(x_len,0,0),(x_len,0,z_len),(0,0,z_len)],
        [(0,y_len,0),(x_len,y_len,0),(x_len,y_len,z_len),(0,y_len,z_len)],
        [(0,0,0),(0,y_len,0),(0,y_len,z_len),(0,0,z_len)],
        [(x_len,0,0),(x_len,y_len,0),(x_len,y_len,z_len),(x_len,0,z_len)],
    ]
    box = Poly3DCollection(verts, facecolors=(0.75,0.88,1.0,0.45),
                           edgecolors="navy", linewidths=0.6)
    ax.add_collection3d(box)
    front = Poly3DCollection([[(0,0,0),(x_len,0,0),(x_len,0,z_len),(0,0,z_len)]],
                             facecolors=(0.1,0.6,1.0,0.10),
                             edgecolors="dodgerblue", linewidths=0.9)
    ax.add_collection3d(front)
    ax.text(x_len*0.5, -0.06*y_len, z_len*0.9, "FRONT", color="dodgerblue",
            ha="center", fontsize=7)

    ax.set_box_aspect((x_len,y_len,z_len))
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()
    plt.subplots_adjust(0,0,1,1)
    fig.tight_layout(pad=0.02)
    return fig

# ---------------- Summary card ----------------
def render_summary_box(L,W,H,layers,per_layer,sa=None,tol=None,unit="cm",cartons_needed=None):
    # หน่วยสำหรับแสดงผล จะส่ง "cm" เข้ามาเสมอ
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
    </style>
    """, unsafe_allow_html=True)

    size_line = f"{L:.2f}×{W:.2f}×{H:.2f} {unit}"
    dims_html = (
        f"<div class='dim'>"
        f"<span>กว้าง (W): <b>{W:.2f} {unit}</b></span>"
        f"<span>ยาว (L): <b>{L:.2f} {unit}</b></span>"
        f"<span>สูง (H): <b>{H:.2f} {unit}</b></span>"
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
    if sa is not None:
        rows.append(("Surface Area", f"{sa:.2f} {unit}²"))   # แสดงเป็น cm²
    if tol is not None:
        rows.append(("Tolerance", f"{tol:.2f} {unit}"))

    html = "".join(f"<div class='sumk'>{k}</div><div class='sumv'>{v}</div>" for k, v in rows)
    st.markdown(f"<div class='sumcard'><div class='sumh'>กล่องแนะนำ</div><div class='sumrow'>{html}</div></div>",
                unsafe_allow_html=True)

# ---------------- Pallet 3D (boxes separated, centered & smaller) ----------------
def draw_pallet_boxes_3d(carton_L_cm, carton_W_cm, carton_H_cm,
                         nx, ny, layers,
                         pallet_L_cm=120.0, pallet_W_cm=100.0,
                         pallet_height_cm=12.0, gap=0.0, scale=0.55,
                         placement="center"):  # "center" | "front_left" | "back_left"
    """วาดพาเลต + กองกล่อง โดยเลือกตำแหน่งการวางบนพาเลตได้"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def faces(x, y, z, dx, dy, dz):
        return [
            [(x,y,z), (x+dx,y,z), (x+dx,y+dy,z), (x,y+dy,z)],
            [(x,y,z+dz), (x+dx,y,z+dz), (x+dx,y+dy,z+dz), (x,y+dy,z+dz)],
            [(x,y,z), (x+dx,y,z), (x+dx,y,z+dz), (x,y,z+dz)],
            [(x,y+dy,z), (x+dx,y+dy,z), (x+dx,y+dy,z+dz), (x,y+dy,z+dz)],
            [(x,y,z), (x,y+dy,z), (x,y+dy,z+dz), (x,y,z+dz)],
            [(x+dx,y,z), (x+dx,y+dy,z), (x+dx,y+dy,z+dz), (x+dx,y,z+dz)],
        ]

    # ขนาดรูป
    fig = plt.figure(figsize=(4.6*scale, 3.4*scale))
    ax  = fig.add_subplot(111, projection="3d")

    # พาเลต
    pal = Poly3DCollection(
        faces(0, 0, 0, pallet_L_cm, pallet_W_cm, pallet_height_cm),
        facecolors=PALETTE["pallet"], edgecolors="#5a3a1a",
        linewidths=1.0, alpha=0.85, zsort="average",
    )
    ax.add_collection3d(pal)

    # พื้นที่กองกล่อง
    used_L = nx*carton_L_cm + max(0, nx-1)*gap
    used_W = ny*carton_W_cm + max(0, ny-1)*gap

    # -------- ตำแหน่งการวาง ----------
    margin = 2.0
    if placement == "front_left":
        off_x = margin
        off_y = margin                      # ชิดด้านหน้า
    elif placement == "back_left":
        off_x = margin
        off_y = max(0.0, pallet_W_cm - used_W - margin)  # ชิดด้านหลัง
    else:  # "center" (ค่าเริ่มต้น)
        off_x = max(0.0, (pallet_L_cm - used_L)/2.0)
        off_y = max(0.0, (pallet_W_cm - used_W)/2.0)

    # กันการทับกับผิวพาเลตเล็กน้อย
    z_eps = 0.03
    box_color, box_edge = (0.30, 0.50, 0.75, 0.80), "#1a3a5a"

    # วางกล่อง
    for k in range(layers):
        z0 = pallet_height_cm + k*carton_H_cm + z_eps
        for j in range(ny):
            for i in range(nx):
                x = off_x + i*(carton_L_cm + gap)
                y = off_y + j*(carton_W_cm + gap)
                ax.add_collection3d(Poly3DCollection(
                    faces(x, y, z0, carton_L_cm, carton_W_cm, carton_H_cm),
                    facecolors=box_color, edgecolors=box_edge,
                    linewidths=0.6, alpha=0.85, zsort="average",
                ))

    used_H = pallet_height_cm + layers*carton_H_cm

    # กรอบ/มุมกล้อง
    ax.set_xlim(0, pallet_L_cm)
    ax.set_ylim(0, pallet_W_cm)
    ax.set_zlim(0, used_H + 4)
    ax.set_box_aspect((pallet_L_cm, pallet_W_cm, used_H))
    ax.view_init(elev=24, azim=-52)  # มุมมองสบายตา
    ax.set_axis_off()
    fig.tight_layout(pad=0.05)

    # ป้ายขนาด
    ax.text(pallet_L_cm*0.50, -6, used_H + 2, f"{int(round(pallet_L_cm))} CM",
            ha="center", fontsize=9, fontweight="bold")
    ax.text(-6, pallet_W_cm*0.50,  used_H + 2, f"{int(round(pallet_W_cm))} CM",
            va="center", rotation=90, fontsize=9, fontweight="bold")
    ax.text(pallet_L_cm + 3, pallet_W_cm*0.70, used_H*0.55,
            f"{int(round(used_H))} CM", rotation=90, va="center",
            fontsize=9, fontweight="bold")

    return fig

# ---------------- UI: 2 หน้าหลัก ----------------
st.set_page_config(page_title="Carton Designer", layout="wide")
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;
padding:1.0rem;border-radius:12px;text-align:center;margin-bottom:0.8rem}
.card{background:#fff;border:1px solid #ebedf0;border-radius:12px;padding:14px 16px;
box-shadow:0 5px 14px rgba(0,0,0,.04);margin-bottom:12px}
</style>
<div class="main-header"><h1>📦 ตัวช่วยออกแบบขนาดกล่อง & วางพาเลต</h1>
<p>Giftbox ➜ Carton + Visualization + Carton ➜ Pallet (Manual)</p></div>
""", unsafe_allow_html=True)

tab_carton, tab_pallet = st.tabs(["🎁 Giftbox ➜ Carton", "🧱 Carton ➜ Pallet"])

# ---------------- PAGE 1: Giftbox ➜ Carton ----------------
with tab_carton:
    # ซ้าย = ลักษณะสินค้า, ขวา = ตั้งค่าการวาง + ระยะเผื่อ
    col_left, col_right = st.columns([6.5,5.5], gap="large")

    # ===== Block 1: ลักษณะของสินค้า =====
    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧩 ลักษณะของสินค้า")

        # ลิสต์ลักษณะ (ยังไม่ผูกตั้งค่าอัตโนมัติ)
        product_profiles = [
            "— ไม่เลือก —",
            "แท่งยาว/ดินสอ (ลิป, มาสคาร่า, ดินสอเขียนคิ้ว)",
            "ของเหลว/ครีม (ลิปกลอส, ทินต์, อายไลเนอร์น้ำ)",
            "มีชิ้นส่วนยื่น/โบว์/ฝาพับ",
            "เปราะบาง/แตกง่าย",
            "ต้องหงายกราฟิก/โลโก้",
            "ชุดกิฟต์เซ็ตหลายชิ้น (มีช่องว่างภายใน)",
            "ทรงเตี้ย/แบน",
            "ทรงสูง/ผอมสูง",
            "สินค้าบางเบา (วางได้หลายชั้น)",
            "สินค้าหนัก (จำกัดจำนวนชั้น)",
        ]
        st.selectbox("ประเภท/ลักษณะสินค้า (ใช้กำหนด Preset ในอนาคต — ตอนนี้ยังไม่ปรับค่าอัตโนมัติ)",
                     product_profiles, index=0)

        # ตัวเลือกทิศทางแบบภาษาไทย
        lock_options = [
            ("ไม่ล็อก — ให้ระบบลองทุกแบบ", None, "ระบบจะลองวางทุกทิศ (H/W/L ขึ้น) แล้วเลือกแบบที่เหมาะที่สุด"),
            ("วางแบบปกติ — H ขึ้น",        "h",  "ด้านสูง (H) อยู่ด้านบน — ไม่พลิกแกน"),
            ("พลิกให้ด้านกว้างขึ้น — W ขึ้น", "w",  "หมุนให้ด้านกว้าง (W) เป็นแกนตั้ง/ด้านบน"),
            ("พลิกให้ด้านยาวขึ้น — L ขึ้น",  "l",  "หมุนให้ด้านยาว (L) เป็นแกนตั้ง/ด้านบน"),
        ]
        labels = [name for name, _, _ in lock_options]
        selection = st.radio(
            "🔒 เลือกทิศทางการวาง",
            labels, index=0, horizontal=False,
            help="กำหนดว่าให้แกนใดของกล่องเป็นด้าน 'บน' ตอนจัดวาง",
        )
        locked_axis = next(val for name, val, _ in lock_options if name == selection)
        desc = next(d for name, _, d in lock_options if name == selection)
        st.caption(f"คำอธิบาย: {desc}")

        diag_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== Block 2: ตั้งค่าการวาง Gift box + ระยะเผื่อ =====
    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📦 ตั้งค่าการวาง Gift box")

        # หน่วยที่ใช้ (ย้ายมาบนสุด)
        unit = st.selectbox("📏 หน่วยที่ใช้", ["mm","cm","in"], index=0)

        # ขนาด giftbox
        st.markdown("**ขนาด Giftbox**")
        c1, c2, c3 = st.columns(3)
        gb_w = c1.number_input("กว้าง (W)", min_value=0.1, value=120.0, step=1.0)
        gb_l = c2.number_input("ยาว (L)",   min_value=0.1, value=200.0, step=1.0)
        gb_h = c3.number_input("สูง (H)",   min_value=0.1, value=80.0,  step=1.0)

        st.divider()

        # จำนวนสินค้า
        st.markdown("**จำนวนสินค้า**")
        desired_qty = st.number_input("จำนวน giftbox ที่ต้องการ (รวม)", 1, 999999, 20, 1)

        st.divider()

        # จำนวนชั้น (โชว์ตลอด + เลือกให้คำนวณเองได้)
        st.markdown("**จำนวนชั้นที่ต้องการวาง**")
        cl1, cl2 = st.columns([1.1, 1])
        layers_input = cl1.number_input("จำนวนชั้น (ต่อ 1 กล่อง)", 1, 50, 1, 1)
        auto_layers  = cl2.checkbox("คิดจำนวนชั้นเอง", value=False,
                                    help="ติ๊กเพื่อให้ระบบคำนวณจำนวนชั้นให้เอง")
        layers_for_one = None if auto_layers else layers_input

        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Tolerance
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🛠️ ระยะเผื่อ (Tolerance)")
        base_tol = st.number_input(f"เผื่อรอบชิ้นงานต่อด้าน ({unit})", 0.0, 50.0, 0.0, 0.5)
        st.caption("Protrusion margin (เผื่อเฉพาะจุดยื่น) — ระบุเพิ่มทีละด้าน")
        g1, g2, g3 = st.columns(3)
        with g1:
            pm_top    = st.number_input(f"Top (+H) ({unit})",    0.0, 100.0, 0.0, 0.5)
            pm_bottom = st.number_input(f"Bottom (-H) ({unit})", 0.0, 100.0, 0.0, 0.5)
        with g2:
            pm_left   = st.number_input(f"Left (-W) ({unit})",   0.0, 100.0, 0.0, 0.5)
            pm_right  = st.number_input(f"Right (+W) ({unit})",  0.0, 100.0, 0.0, 0.5)
        with g3:
            pm_front  = st.number_input(f"Front (-L) ({unit})",  0.0, 100.0, 0.0, 0.5)
            pm_back   = st.number_input(f"Back (+L) ({unit})",   0.0, 100.0, 0.0, 0.5)
        try_bruteforce = st.checkbox("🔎 ลองค่า tolerance หลายค่า (0–0.5)", value=False,
                                     help="ให้ระบบลอง tolerance หลายค่า (0–0.5) เพื่อหากล่องที่เหมาะที่สุด")
        st.markdown("</div>", unsafe_allow_html=True)

    # วาดรูปช่วยจำด้วยขนาดจริง
    with col_left:
        diag_placeholder.pyplot(
            render_wlh_diagram_oriented(gb_w, gb_l, gb_h, up=locked_axis or "h"),
            use_container_width=False
        )

    # ---- Compute
    st.markdown("<br>", unsafe_allow_html=True)
    calc = st.button("🚀 คำนวณขนาดกล่องที่เหมาะสม", use_container_width=True)

    def apply_margins_and_tol_rect(w,l,h,tol,pm):
        W = w + (pm["left"] + pm["right"]) + 2*tol
        L = l + (pm["front"] + pm["back"]) + 2*tol
        H = h + (pm["bottom"] + pm["top"]) + 2*tol
        return W,L,H

    if calc:
        t_values = [round(x,2) for x in np.arange(0,0.51,0.05)] if try_bruteforce else [base_tol]
        best = None; best_tol = None
        for t in t_values:
            iw, il, ih = apply_margins_and_tol_rect(
                gb_w, gb_l, gb_h, t,
                {"top":pm_top,"bottom":pm_bottom,"left":pm_left,"right":pm_right,"front":pm_front,"back":pm_back}
            )
            res = rect_design_min_carton(
                desired_qty, iw, il, ih,
                locked_axis=locked_axis, force_layers=layers_for_one,
                disallow_front_up=False, keep_top_up=False
            )
            if res and (best is None or res.meta["SA"] < best.meta["SA"]):
                best = res; best_tol = t

        if not best:
            st.error("ไม่พบวิธีจัดวางที่เหมาะสม (ลองลดข้อจำกัดหรือเพิ่มค่าเผื่อ)")
        else:
            st.session_state.result = best
            st.session_state.tol = best_tol
            st.session_state.unit = unit
            st.session_state.qty = desired_qty
            st.success("✅ คำนวณสำเร็จ! — ดูสรุปด้านล่างหรือไปหน้า 'Carton ➜ Pallet' ต่อได้เลย")

    # ---- Show result (Summary in cm + Visualization)
    res = st.session_state.get("result")
    if res:
        tol  = st.session_state.get("tol", 0)
        unit = st.session_state.get("unit", "mm")
        qty  = st.session_state.get("qty", 1)

        iw,il,ih = res.orientation
        W = res.plan.ny * res.plan.spacing_x
        L = res.plan.nx * res.plan.spacing_y
        H = res.layers * ih
        per_carton = res.layers * res.per_layer
        need_cartons = math.ceil(qty / per_carton)

        # แสดงผลเป็น cm เสมอ
        W_cm = Unit.to_cm(W, unit); L_cm = Unit.to_cm(L, unit); H_cm = Unit.to_cm(H, unit)
        tol_cm = Unit.to_cm(tol, unit)
        SA_cm = surface_area_box(W_cm, L_cm, H_cm)  # cm²

        s1, s2 = st.columns([2,3])
        with s1:
            st.pyplot(make_top_view(W, L, res.plan, unit, True, False, 0.45, "", 10),
                      use_container_width=False)
        with s2:
            render_summary_box(L_cm, W_cm, H_cm, res.layers, res.per_layer, SA_cm, tol_cm, "cm", need_cartons)

        st.divider()
        st.header("🖼️ Visualization")

        tab2d, tab3d = st.tabs(["📐 มุมบน (2D)", "📦 ซ้อนชั้น (3D)"])
        with tab2d:
            left, right = st.columns([3,2])
            with right:
                fill = st.checkbox("🟩 แสดงสี", True)
                idx  = st.checkbox("🔢 หมายเลข", False)
                st.markdown("### สรุป")
                render_summary_box(L_cm, W_cm, H_cm, res.layers, res.per_layer, SA_cm, tol_cm, "cm", need_cartons)
            with left:
                st.pyplot(make_top_view(W,L,res.plan,unit, show_fill=fill, show_index=idx,
                                        scale=0.55, title_text=None, title_size=12),
                          use_container_width=False)

            st.markdown("---")
            st.caption("🧩 ภาพเล็ก 2 มุมมอง")
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**มุมบน**")
                st.pyplot(make_top_view(W,L,res.plan,unit, True, False, 0.45, "", 10),
                          use_container_width=False)
            with cB:
                st.markdown("**มุมข้าง**")
                st.pyplot(make_side_view(W,L,H,res.layers,res.plan,ih, 0.45, "", 10),
                          use_container_width=False)

        with tab3d:
            st.pyplot(make_3d_stack(W,L,res.plan,res.layers,ih, scale=0.7),
                      use_container_width=False)

# ---------------- PAGE 2: Carton ➜ Pallet (Single Block) ----------------
with tab_pallet:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧱 Carton ➜ Pallet")

    # 0) ดึงขนาดจากหน้าแรก (ถ้ามีผลคำนวณ)
    use_from_prev = st.checkbox("ดึงขนาดจากหน้า Giftbox ➜ Carton (ถ้ามีผลคำนวณ)", value=True)

    # ค่ากล่อง (cm)
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
        st.info(f"ดึงมาแล้ว: W={cart_W_cm:.2f} cm, L={cart_L_cm:.2f} cm, H={cart_H_cm:.2f} cm")
    else:
        c1,c2,c3 = st.columns(3)
        cart_W_cm = c1.number_input("กว้าง (W) ของกล่อง (cm)", 1.0, 200.0, 30.0, 0.5)
        cart_L_cm = c2.number_input("ยาว (L) ของกล่อง (cm)",   1.0, 200.0, 40.0, 0.5)
        cart_H_cm = c3.number_input("สูง (H) ของกล่อง (cm)",   1.0, 200.0, 20.0, 0.5)

    st.divider()

    # 1) ตั้งค่าพาเลต
    st.markdown("**ตั้งค่าพาเลต (Pallet)**")
    p1, p2, p3, p4 = st.columns(4)
    pal_W_cm = p1.number_input("กว้างพาเลต (cm)", 50.0, 200.0, 100.0, 1.0)
    pal_L_cm = p2.number_input("ยาวพาเลต (cm)",   60.0, 200.0, 120.0, 1.0)
    pal_Hmax = p3.number_input("สูงรวมไม่เกิน (cm)", 60.0, 300.0, 180.0, 1.0)
    pal_H    = p4.number_input("ความสูงพาเลต (cm)",  5.0,  40.0,  12.0, 0.5)

    st.divider()

    # 2) จำนวนกล่องต่อชั้น
    st.markdown("**จำนวนกล่องต่อชั้น**")
    n1, n2, n3 = st.columns([1.1,1.1,1])
    nx = n1.number_input("จำนวนคอลัมน์ตามแนวยาวพาเลต (nx)", 0, 100, 2, 1)
    ny = n2.number_input("จำนวนแถวตามแนวกว้างพาเลต (ny)", 0, 100, 2, 1)
    per_layer = nx * ny
    n3.metric("รวมต่อชั้น", per_layer)

    st.divider()

    # 3) จำนวนชั้น (layers)
    st.markdown("**จำนวนชั้น (layers)**")
    layers = st.number_input("วางกี่ชั้น", 0, 50, 5, 1)

    # ---------- ตรวจสอบเงื่อนไข ----------
    problems = []
    if nx * cart_L_cm > pal_L_cm:
        problems.append(f"กล่องตามแนวยาวล้นพาเลต: nx×L = {nx*cart_L_cm:.1f} > {pal_L_cm:.1f} cm")
    if ny * cart_W_cm > pal_W_cm:
        problems.append(f"กล่องตามแนวกว้างล้นพาเลต: ny×W = {ny*cart_W_cm:.1f} > {pal_W_cm:.1f} cm")
    total_height = pal_H + layers * cart_H_cm
    if total_height > pal_Hmax:
        problems.append(f"ความสูงรวมเกินกำหนด: {total_height:.1f} > {pal_Hmax:.1f} cm")

    if problems:
        st.error("ไม่สามารถวางได้ด้วยค่าปัจจุบัน:\n\n- " + "\n- ".join(problems))
    else:
        # 3D
        st.pyplot(
            draw_pallet_boxes_3d(cart_L_cm, cart_W_cm, cart_H_cm,
                                 nx, ny, layers,
                                 pal_L_cm, pal_W_cm, pal_H,
                                 gap=0.0, scale=.55, placement="center"),
            use_container_width=False
        )
        # สรุป
        total_boxes = per_layer * layers
        st.markdown("### 📄 สรุปการเรียงบนพาเลต")
        st.markdown(
            f"""
**ขนาดกล่อง (OD)** : {cart_L_cm:.0f}×{cart_W_cm:.0f}×{cart_H_cm:.0f} cm  
**จำนวนต่อชั้น** : {per_layer} กล่อง  (nx={nx}, ny={ny})  
**จำนวนชั้น** : {layers} ชั้น  
**ความสูงรวม** : {total_height:.1f} cm  *(พาเลต {pal_H:.0f} + กล่อง {layers}×{cart_H_cm:.0f})*  
**รวมต่อพาเลต** : **{total_boxes} กล่อง**
            """
        )

    st.markdown("</div>", unsafe_allow_html=True)
