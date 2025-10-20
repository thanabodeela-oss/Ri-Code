# APP2.py — Carton Designer (ทรงเหลี่ยม) + Visualization + Pallet (Carton-only)
# Run: streamlit run APP2.py

import math
from dataclasses import dataclass
from typing import Tuple, Optional
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
    "frame":  "#d9a362",
    "inside": (0.42, 0.67, 0.86, 0.95),
    "grid":   "#ffffff",
    "edge":   "#4a4a4a",
    "pallet": "#9c6a3a",
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

def rect_design_min_carton(qty, w, l, h, locked_axis: Optional[str]=None, force_layers: Optional[int]=None):
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

# ---------------- Drawing helpers ----------------
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
                           facecolor=PALETTE["inside"] if show_fill else (1,1,1,1),
                           edgecolor=PALETTE["edge"], linewidth=1.5))

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

# ---------------- Summary card (always in cm / cm²) ----------------
def render_summary_box_cm(L_cm,W_cm,H_cm,layers,per_layer,sa_cm2=None,tol_cm=None,cartons_needed=None):
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
    if tol_cm is not None:
        rows.append(("Tolerance", f"{tol_cm:.2f} cm"))

    html = "".join(f"<div class='sumk'>{k}</div><div class='sumv'>{v}</div>" for k, v in rows)
    st.markdown(f"<div class='sumcard'><div class='sumh'>กล่องแนะนำ</div><div class='sumrow'>{html}</div></div>",
                unsafe_allow_html=True)

# ---------------- 3D utils ----------------
def _faces(x, y, z, dx, dy, dz):
    return [
        [(x,y,z), (x+dx,y,z), (x+dx,y+dy,z), (x,y+dy,z)],
        [(x,y,z+dz), (x+dx,y,z+dz), (x+dx,y+dy,z+dz), (x,y+dy,z+dz)],
        [(x,y,z), (x+dx,y,z), (x+dx,y,z+dz), (x,y,z+dz)],
        [(x,y+dy,z), (x+dx,y+dy,z), (x+dx,y+dy,z+dz), (x,y+dy,z+dz)],
        [(x,y,z), (x,y+dy,z), (x,y+dy,z+dz), (x,y,z+dz)],
        [(x+dx,y,z), (x+dx,y+dy,z), (x+dx,y+dy,z+dz), (x+dx,y,z+dz)],
    ]

# ---------------- Carton-only 3D (no pallet) ----------------
def draw_carton_stack_only(carton_L_cm, carton_W_cm, carton_H_cm,
                           nx, ny, layers, gap=0.0, scale=0.90):
    """วาดเฉพาะกองกล่อง (orthographic) ไม่มีพาเลต"""
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

# ---------------- UI ----------------
st.set_page_config(page_title="Carton Designer", layout="wide")
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;
padding:1.0rem;border-radius:12px;text-align:center;margin-bottom:0.8rem}
.card{background:#fff;border:1px solid #ebedf0;border-radius:12px;padding:14px 16px;
box-shadow:0 5px 14px rgba(0,0,0,.04);margin-bottom:12px}
</style>
<div class="main-header"><h1>📦 ตัวช่วยออกแบบขนาดกล่อง & วางพาเลต</h1>
<p>Giftbox ➜ Carton + Visualization + Carton ➜ Pallet</p></div>
""", unsafe_allow_html=True)

tab_carton, tab_pallet = st.tabs(["🎁 Giftbox ➜ Carton", "🧱 Carton ➜ Pallet"])

# ---------------- PAGE 1: Giftbox ➜ Carton ----------------
with tab_carton:
    col_main, col_side = st.columns([8,4], gap="large")

    with col_main:
        # 1) ลักษณะของสินค้า (Cosmetics OEM list only, no locking)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧩 ลักษณะของสินค้า")
        product_type = st.selectbox(
            "เลือกหมวดสินค้า (ยังไม่ล็อกกฎใด ๆ ใช้เพื่ออ้างอิงเท่านั้น)",
            [
                # กล่องแข็ง/พาเลตเมคอัพ
                "ตลับแป้ง/บลัช/อายแชโดว์ (Compact/Palette)",
                "แท่งลิปสติก/ลิปรูจ/ลิปบาล์ม (Lip Stick/Rouge/Balm)",
                "ดินสอเขียนคิ้ว/อายไลเนอร์ (Pencil/Auto Pencil)",
                "มาสคาร่า/คอนซีลเลอร์แบบแท่ง (Tube with Wiper)",

                # ขวด/กระปุกสกินแคร์
                "กระปุกครีม (Jar – Glass/Plastic)",
                "ขวดดรอปเปอร์/เซรั่ม (Dropper Bottle)",
                "ขวดปั๊ม/เออร์เลส (Pump/Airless)",
                "ขวดสเปรย์/โทนเนอร์ (Spray/Toner Bottle)",
                "แอมพูล/ไวอัล (Ampoule/Vial)",

                # บรรจุภัณฑ์นิ่ม/ซอง
                "ซองครีม/ซองเจล (Sachet)",
                "แผ่นมาสก์ชีท (Sheet Mask)",
                "ซองตั้ง/สแตนด์อัพพาวช์ (Stand-up Pouch)",
                "บลิสเตอร์/การ์ดแพ็ก (Blister/Card)",

                # ชุดเซ็ต/ของแถม
                "กิฟต์เซ็ต/มัลติแพ็ก (Gift Set / Multipack)",
                "กล่องพร้อมไส้/อินเสิร์ต (Carton + Insert/Tray)",

                # อื่น ๆ
                "อื่น ๆ",
            ],
            index=0
        )
        st.caption("ทรงเหลี่ยม (Rectangular) — รุ่นนี้รองรับทรงเหลี่ยมเท่านั้น (ยังไม่ล็อกกฎการวาง) ")
        st.markdown("</div>", unsafe_allow_html=True)

        # 2) ตั้งค่าการวาง Gift box
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📦 ตั้งค่าการวาง Gift box")
        unit = st.selectbox("📏 หน่วยที่ใช้", ["mm","cm","in"], index=0)
        st.divider()

        c1,c2,c3 = st.columns(3)
        gb_w = c1.number_input("ขนาด Giftbox — กว้าง (W)", min_value=0.1, value=120.0, step=1.0)
        gb_l = c2.number_input("ขนาด Giftbox — ยาว (L)",   min_value=0.1, value=200.0, step=1.0)
        gb_h = c3.number_input("ขนาด Giftbox — สูง (H)",   min_value=0.1, value=80.0,  step=1.0)

        qty = st.number_input("จำนวน giftbox ที่ต้องการ (รวม)", 1, 999999, 20, 1)

        auto_layers = st.checkbox("คำนวณจำนวนชั้นอัตโนมัติ", value=True)
        layers_for_one = None
        if not auto_layers:
            layers_for_one = st.number_input("จำนวนชั้นที่ต้องการวาง (ต่อ 1 กล่อง)", 1, 50, 1, 1)

        st.markdown("### 🔁 เลือกทิศทางการวาง")
        lock_opt = st.radio(
            "ทิศทาง (แกนที่ตั้งอยู่ด้านบน)",
            ["ไม่ล็อก – ให้ระบบลองทุกแบบ", "วางแบบเดิม – H ขึ้น", "พลิกให้ด้านกว้างขึ้น – W ขึ้น", "พลิกให้ด้านยาวขึ้น – L ขึ้น"],
            horizontal=False, index=0, label_visibility="visible")
        lock_map = {
            "ไม่ล็อก – ให้ระบบลองทุกแบบ":"auto",
            "วางแบบเดิม – H ขึ้น":"h",
            "พลิกให้ด้านกว้างขึ้น – W ขึ้น":"w",
            "พลิกให้ด้านยาวขึ้น – L ขึ้น":"l",
        }
        locked_axis = lock_map[lock_opt]

        st.caption("รูปช่วยจำอัตราส่วน W/L/H")
        st.pyplot(render_wlh_diagram_oriented(gb_w, gb_l, gb_h,
                  up=(locked_axis if locked_axis!="auto" else "h")),
                  use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_side:
        # 3) ระยะเผื่อ
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🛠️ ระยะเผื่อ (Tolerance)")
        base_tol = st.number_input(f"เผื่อรอบชิ้นงานต่อด้าน ({unit})", 0.0, 50.0, 0.0, 0.5)
        st.caption("Protrusion margin (เผื่อเฉพาะจุดยื่น) — ระบุเพิ่มทีละด้าน")
        g1,g2,g3 = st.columns(3)
        with g1:
            pm_top    = st.number_input(f"Top (+H) ({unit})",    0.0, 100.0, 0.0, 0.5)
            pm_bottom = st.number_input(f"Bottom (-H) ({unit})", 0.0, 100.0, 0.0, 0.5)
        with g2:
            pm_left   = st.number_input(f"Left (-W) ({unit})",   0.0, 100.0, 0.0, 0.5)
            pm_right  = st.number_input(f"Right (+W) ({unit})",  0.0, 100.0, 0.0, 0.5)
        with g3:
            pm_front  = st.number_input(f"Front (-L) ({unit})",  0.0, 100.0, 0.0, 0.5)
            pm_back   = st.number_input(f"Back (+L) ({unit})",   0.0, 100.0, 0.0, 0.5)
        try_bruteforce = st.checkbox("🔎 ลอง tolerance หลายค่า (0–0.5)", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    calc = st.button("🚀 คำนวณขนาดกล่องที่เหมาะสม", use_container_width=True)

    # ---- Compute
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
            axis_forced = None if locked_axis=="auto" else locked_axis
            res = rect_design_min_carton(
                qty, iw, il, ih,
                locked_axis=axis_forced, force_layers=layers_for_one
            )
            if res and (best is None or res.meta["SA"] < best.meta["SA"]):
                best = res; best_tol = t

        if not best:
            st.error("ไม่พบวิธีจัดวางที่เหมาะสม (ลองลดข้อจำกัดหรือเพิ่มค่าเผื่อ)")
        else:
            st.session_state.result = best
            st.session_state.tol = best_tol
            st.session_state.unit = unit
            st.session_state.qty = qty
            st.success("✅ คำนวณสำเร็จ! — ไปหน้า 'Carton ➜ Pallet' เพื่อคำนวณต่อได้เลย")

    # ---- Show result
    res = st.session_state.get("result")
    if res:
        tol  = st.session_state.get("tol", 0.0)
        base_unit = st.session_state.get("unit", "mm")
        qty_saved  = st.session_state.get("qty", 1)

        iw,il,ih = res.orientation
        W = res.plan.ny * res.plan.spacing_x
        L = res.plan.nx * res.plan.spacing_y
        H = res.layers * ih

        # เป็น cm สำหรับแสดงผล
        W_cm = Unit.to_cm(W, base_unit)
        L_cm = Unit.to_cm(L, base_unit)
        H_cm = Unit.to_cm(H, base_unit)
        SA_cm2 = 2*(W_cm*L_cm + W_cm*H_cm + L_cm*H_cm)
        tol_cm = Unit.to_cm(tol, base_unit)

        per_carton = res.layers * res.per_layer
        need_cartons = math.ceil(qty_saved / per_carton)

        st.divider()
        st.header("🖼️ Visualization")

        tab2d, tab3d = st.tabs(["📐 มุมบน/ข้าง (2D)", "📦 ซ้อนชั้น (3D)"])
        with tab2d:
            left, right = st.columns([3,2])
            with right:
                fill = st.checkbox("🟩 แสดงสี", True)
                idx  = st.checkbox("🔢 หมายเลข", False)
                st.markdown("### สรุป (หน่วย cm)")
                render_summary_box_cm(L_cm, W_cm, H_cm, res.layers, res.per_layer, SA_cm2, tol_cm, need_cartons)
            with left:
                st.pyplot(make_top_view(W,L,res.plan,base_unit, show_fill=fill, show_index=idx,
                                        scale=0.55, title_text=None, title_size=12),
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

# ---------------- PAGE 2: Carton ➜ Pallet (Carton-only view) ----------------
with tab_pallet:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧱 วาง Carton บนพาเลต (โชว์เฉพาะกองกล่อง)")

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

    # พาเลตใช้สำหรับคำนวณตรวจสอบเงื่อนไขเท่านั้น (ภาพจะไม่วาดพาเลต)
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
            total_boxes = per_layer * layers
            footprint_L = nx*cart_L_cm
            footprint_W = ny*cart_W_cm

            st.markdown("### 📄 สรุปการเรียงบนพาเลต")
            st.markdown("""
            <style>
            .sumcardB{background:#eef6ff;padding:12px 14px;border-radius:10px;
                     border:1px solid #cde2ff;box-shadow:0 4px 10px rgba(0,0,0,.04);font-size:15px}
            .sumhB{font-size:18px;font-weight:800;color:#113e7e;margin-bottom:8px}
            .sumrowB{display:grid;grid-template-columns:auto 1fr;gap:6px 10px}
            .sumkB{color:#3c5f8f}.sumvB{font-weight:700}
            </style>
            """, unsafe_allow_html=True)

            def row(k,v): return f"<div class='sumkB'>{k}</div><div class='sumvB'>{v}</div>"

            html = "".join([
                row("ขนาดกล่อง (OD)", f"{cart_L_cm:.0f} × {cart_W_cm:.0f} × {cart_H_cm:.0f} cm"),
                row("วางต่อชั้น",     f"{per_layer} กล่อง (nx={nx}, ny={ny})"),
                row("จำนวนชั้น",      f"{layers} ชั้น"),
                row("รอยเท้าบนพาเลต", f"{footprint_L:.0f} × {footprint_W:.0f} cm"),
                row("ความสูงรวม",     f"{total_height:.1f} cm  (พาเลต {pal_H:.0f} + กล่อง {layers}×{cart_H_cm:.0f})"),
                row("รวมต่อพาเลต",    f"{total_boxes} กล่อง"),
            ])
            st.markdown(f"<div class='sumcardB'><div class='sumhB'>สรุป</div><div class='sumrowB'>{html}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
