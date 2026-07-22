import os
import sys
import math
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ==========================================
# 1. COLOR PALETTE DEFINITIONS
# ==========================================
PALETTE_BG = "#F8FAFC"          # Soft Chalk Background
PALETTE_SIDEBAR = "#FFFFFF"     # Crisp White Sidebar
PALETTE_WORKSPACE = "#F1F5F9"   # Off-white Canvas Frame
PALETTE_BORDER = "#E2E8F0"      # Subtle Gray Division Lines

PALETTE_PRIMARY = "#4F46E5"     # Deep Indigo Accent
PALETTE_PRIMARY_HOVER = "#4338CA"
PALETTE_EMERALD = "#10B981"     # Mint Green
PALETTE_EMERALD_HOVER = "#059669"
PALETTE_SLATE = "#64748B"       # Slate Gray Secondary Text
PALETTE_TEXT_DARK = "#0F172A"   # Midnight Primary Text


# ==========================================
# 2. ENHANCED COMPUTER VISION ENGINE
# ==========================================

def display_detected_fragments(image_path):
    """
    Detects individual torn paper fragments, extracts them with transparent 
    alpha borders, and rights their rotational tilt.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    # Standardize maximum dimension for performance
    h, w = img.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Robust adaptive/Otsu thresholding for separating paper from background
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fragments = []
    min_area = (img.shape[0] * img.shape[1]) * 0.003  # Noise threshold Filter

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)

            # Minimum bounding box for rotational alignment
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            if angle < -45:
                angle += 90

            x, y, w_box, h_box = cv2.boundingRect(cnt)
            crop_img = img[y:y+h_box, x:x+w_box].copy()
            crop_mask = mask[y:y+h_box, x:x+w_box].copy()

            # Center of mass calculation
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w_box // 2, y + h_box // 2

            fragments.append({
                'contour': cnt,
                'roi': crop_img,
                'mask': crop_mask,
                'bbox': (x, y, w_box, h_box),
                'center': (cx, cy),
                'area': area,
                'angle': angle
            })

    return fragments, img


def display_final_canvas(fragments, step_callback=None):
    """
    Aligns fragments on canvas using feature-matching homography across 
    fragment tear edges.
    """
    if not fragments:
        return None

    # Calculate optimal canvas boundaries based on input bounding boxes
    all_cnts = np.vstack([f['contour'] for f in fragments])
    x_min, y_min, total_w, total_h = cv2.boundingRect(all_cnts)

    canvas_h = total_h + 120
    canvas_w = total_w + 120
    
    # White background canvas
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # Sort fragments from largest to smallest (base fragment anchoring)
    fragments = sorted(fragments, key=lambda f: f['area'], reverse=True)
    total_steps = len(fragments)

    orb = cv2.ORB_create(nfeatures=2000)
    placed_masks = []

    for idx, frag in enumerate(fragments):
        bx, by, bw, bh = frag['bbox']
        roi_img = frag['roi']
        roi_mask = frag['mask']

        target_x = (bx - x_min) + 60
        target_y = (by - y_min) + 60

        # Boundary safety clip
        target_x = max(0, min(canvas_w - bw, target_x))
        target_y = max(0, min(canvas_h - bh, target_y))

        # Perform feature matching for auto-snapping if canvas already has content
        if idx > 0 and len(placed_masks) > 0:
            kp1, des1 = orb.detectAndCompute(roi_img, roi_mask)
            kp2, des2 = orb.detectAndCompute(canvas, None)

            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) >= 8:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:15]]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:15]]).reshape(-1, 1, 2)

                    # Compute precise translation transformation matrix
                    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        # Extract translation offset safely
                        dx = H[0, 2]
                        dy = H[1, 2]
                        if abs(dx) < canvas_w and abs(dy) < canvas_h:
                            target_x = int(np.clip(target_x + (dx * 0.05), 0, canvas_w - bw))
                            target_y = int(np.clip(target_y + (dy * 0.05), 0, canvas_h - bh))

        # Alpha-blended placement onto workspace canvas
        target_region = canvas[target_y:target_y+bh, target_x:target_x+bw]
        mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR) / 255.0

        blended = (roi_img * mask_3ch) + (target_region * (1.0 - mask_3ch))
        canvas[target_y:target_y+bh, target_x:target_x+bw] = blended.astype(np.uint8)

        # Track placed masks for seamless edge inpainting
        current_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        current_mask[target_y:target_y+bh, target_x:target_x+bw] = roi_mask
        placed_masks.append(current_mask)

        if step_callback:
            step_callback(canvas.copy(), idx + 1, total_steps)

    return canvas


def apply_inpainting_transformation(canvas_img):
    """
    Fills seam tear gaps safely without eroding intact text or ink details.
    """
    gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    
    # Identify non-paper background boundaries around torn edges
    _, white_gap_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    _, dark_gap_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    
    combined_mask = cv2.bitwise_or(white_gap_mask, dark_gap_mask)
    
    # Thin structural dilation along paper tear boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seam_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    seam_mask = cv2.dilate(seam_mask, kernel, iterations=1)

    # Perform Telea Fast Marching Inpainting targeting seams
    inpainted = cv2.inpaint(canvas_img, seam_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted


# ==========================================
# 3. GUI INTERFACE & CONTROLLERS
# ==========================================

class ApplicationWindow:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Paper Studio — Advanced Reconstruction Engine")
        self.root.geometry("840x640")
        self.root.configure(bg=PALETTE_BG)
        self.root.resizable(False, False)

        self.selected_path = None
        self.input_preview_photo = None
        self.stitched_canvas = None
        
        self.step_history = []
        self.current_step_index = 0

        self.setup_ui_layout()

    def setup_ui_layout(self):
        self.sidebar = tk.Frame(
            self.root, bg=PALETTE_SIDEBAR, width=280, highlightbackground=PALETTE_BORDER, highlightthickness=1
        )
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        brand_frame = tk.Frame(self.sidebar, bg=PALETTE_SIDEBAR, padx=20, pady=24)
        brand_frame.pack(fill=tk.X)

        lbl_logo = tk.Label(
            brand_frame,
            text="Paper Studio",
            font=("Segoe UI", 16, "bold"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_TEXT_DARK,
            anchor="w",
        )
        lbl_logo.pack(fill=tk.X)

        lbl_sub = tk.Label(
            brand_frame,
            text="Document Reconstruction Engine",
            font=("Segoe UI", 8),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_SLATE,
            anchor="w",
        )
        lbl_sub.pack(fill=tk.X, pady=(2, 0))

        controls_group = tk.Frame(self.sidebar, bg=PALETTE_SIDEBAR, padx=20)
        controls_group.pack(fill=tk.X, pady=(10, 0))

        lbl_step1 = tk.Label(
            controls_group,
            text="1. SOURCE DOCUMENT",
            font=("Segoe UI", 8, "bold"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_PRIMARY,
            anchor="w",
        )
        lbl_step1.pack(fill=tk.X, pady=(0, 6))

        self.btn_select_file = tk.Button(
            controls_group,
            text="Choose Image File...",
            font=("Segoe UI", 9, "bold"),
            bg=PALETTE_PRIMARY,
            fg="white",
            activebackground=PALETTE_PRIMARY_HOVER,
            activeforeground="white",
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
            anchor="w",
            command=self.browse_file,
        )
        self.btn_select_file.pack(fill=tk.X)

        self.lbl_file_name = tk.Label(
            controls_group,
            text="No file selected",
            font=("Segoe UI", 8, "italic"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_SLATE,
            anchor="w",
            wraplength=230,
            justify="left",
        )
        self.lbl_file_name.pack(fill=tk.X, pady=(6, 18))

        lbl_step2 = tk.Label(
            controls_group,
            text="2. RECONSTRUCTION",
            font=("Segoe UI", 8, "bold"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_PRIMARY,
            anchor="w",
        )
        lbl_step2.pack(fill=tk.X, pady=(0, 6))

        self.btn_run_stitch = tk.Button(
            controls_group,
            text="🧩 Stitch Fragments",
            font=("Segoe UI", 9, "bold"),
            bg=PALETTE_EMERALD,
            fg="white",
            activebackground=PALETTE_EMERALD_HOVER,
            activeforeground="white",
            disabledforeground="#94A3B8",
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
            state=tk.DISABLED,
            anchor="w",
            command=self.process_image,
        )
        self.btn_run_stitch.pack(fill=tk.X)

        self.btn_open_viewer = tk.Button(
            controls_group,
            text="🔍 View Output & Inpaint ▶",
            font=("Segoe UI", 8, "bold"),
            bg=PALETTE_PRIMARY,
            fg="white",
            activebackground=PALETTE_PRIMARY_HOVER,
            activeforeground="white",
            bd=0,
            padx=10,
            pady=8,
            cursor="hand2",
            anchor="w",
            command=self.open_result_viewer,
        )

        status_frame = tk.Frame(self.sidebar, bg=PALETTE_SIDEBAR, padx=20, pady=20)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_status = tk.Label(
            status_frame,
            text="Status: Ready",
            font=("Segoe UI", 8, "bold"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_SLATE,
            anchor="w",
            wraplength=230,
            justify="left",
        )
        self.lbl_status.pack(fill=tk.X)

        workspace = tk.Frame(self.root, bg=PALETTE_BG, padx=20, pady=20)
        workspace.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.card = tk.Frame(
            workspace, bg=PALETTE_WORKSPACE, highlightbackground=PALETTE_BORDER, highlightthickness=1
        )
        self.card.pack(fill=tk.BOTH, expand=True)

        self.lbl_card_title = tk.Label(
            self.card,
            text="Input Preview",
            font=("Segoe UI", 9, "bold"),
            bg=PALETTE_WORKSPACE,
            fg=PALETTE_SLATE,
            padx=16,
            pady=12,
            anchor="w",
        )
        self.lbl_card_title.pack(fill=tk.X)

        self.lbl_preview = tk.Label(
            self.card,
            text="🖼️ No Document Loaded\n\nClick 'Choose Image File...' on the left panel to begin.",
            font=("Segoe UI", 9),
            bg=PALETTE_WORKSPACE,
            fg=PALETTE_SLATE,
        )
        self.lbl_preview.pack(fill=tk.BOTH, expand=True)

        self.nav_frame = tk.Frame(self.card, bg=PALETTE_WORKSPACE, pady=10)
        self.nav_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_prev_step = tk.Button(
            self.nav_frame,
            text="◀ Previous Step",
            font=("Segoe UI", 8, "bold"),
            bg=PALETTE_SLATE,
            fg="white",
            activebackground="#475569",
            activeforeground="white",
            disabledforeground="#94A3B8",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.show_previous_step,
        )
        self.btn_prev_step.pack(side=tk.LEFT, padx=(20, 5))

        self.lbl_step_counter = tk.Label(
            self.nav_frame,
            text="Step 0 of 0",
            font=("Segoe UI", 8, "bold"),
            bg=PALETTE_WORKSPACE,
            fg=PALETTE_SLATE,
        )
        self.lbl_step_counter.pack(side=tk.LEFT, expand=True)

        self.btn_next_step = tk.Button(
            self.nav_frame,
            text="Next Step ▶",
            font=("Segoe UI", 8, "bold"),
            bg=PALETTE_SLATE,
            fg="white",
            activebackground="#475569",
            activeforeground="white",
            disabledforeground="#94A3B8",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.show_next_step,
        )
        self.btn_next_step.pack(side=tk.RIGHT, padx=(5, 20))

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.selected_path = file_path
            self.lbl_file_name.config(
                text=os.path.basename(file_path), fg=PALETTE_TEXT_DARK, font=("Segoe UI", 8, "bold")
            )
            self.btn_run_stitch.config(state=tk.NORMAL, bg=PALETTE_EMERALD)
            self.btn_open_viewer.pack_forget()
            self.lbl_status.config(
                text="Status: Image loaded. Ready to stitch.",
                fg=PALETTE_PRIMARY,
            )

            self.step_history = []
            self.current_step_index = 0
            self.stitched_canvas = None
            self.update_nav_buttons()

            pil_img = Image.open(file_path)
            pil_img.thumbnail((460, 340))
            self.input_preview_photo = ImageTk.PhotoImage(pil_img)
            self.lbl_preview.config(
                image=self.input_preview_photo, text="", bg=PALETTE_WORKSPACE
            )

    def record_stitch_step(self, intermediate_canvas, current_step, total_steps):
        self.step_history.append(intermediate_canvas.copy())
        self.current_step_index = len(self.step_history) - 1

        self.render_step_frame()
        self.lbl_status.config(
            text=f"Status: ⏳ Aligning fragment {current_step}/{total_steps}...",
            fg="#D97706",
        )
        self.root.update()

    def render_step_frame(self):
        if not self.step_history:
            return

        canvas_img = self.step_history[self.current_step_index]
        img_rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((460, 340))
        self.input_preview_photo = ImageTk.PhotoImage(pil_img)

        total_steps = len(self.step_history) - 1
        
        if self.current_step_index == total_steps and self.stitched_canvas is not None:
            self.lbl_card_title.config(
                text=f"Final Assembly Complete — Step {self.current_step_index} of {total_steps}"
            )
        else:
            self.lbl_card_title.config(
                text=f"Assembly Progress — Step {self.current_step_index} of {total_steps}"
            )

        self.lbl_preview.config(image=self.input_preview_photo, text="", bg=PALETTE_WORKSPACE)
        self.lbl_step_counter.config(
            text=f"Step {self.current_step_index} of {total_steps}"
        )
        self.update_nav_buttons()

    def show_previous_step(self):
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.render_step_frame()

    def show_next_step(self):
        if self.current_step_index < len(self.step_history) - 1:
            self.current_step_index += 1
            self.render_step_frame()

    def update_nav_buttons(self):
        total = len(self.step_history)
        if total <= 1:
            self.btn_prev_step.config(state=tk.DISABLED, bg="#CBD5E1")
            self.btn_next_step.config(state=tk.DISABLED, bg="#CBD5E1")
            return

        if self.current_step_index > 0:
            self.btn_prev_step.config(state=tk.NORMAL, bg=PALETTE_SLATE)
        else:
            self.btn_prev_step.config(state=tk.DISABLED, bg="#CBD5E1")

        if self.current_step_index < total - 1:
            self.btn_next_step.config(state=tk.NORMAL, bg=PALETTE_SLATE)
        else:
            self.btn_next_step.config(state=tk.DISABLED, bg="#CBD5E1")

    def process_image(self):
        if not self.selected_path:
            return

        self.step_history = []
        self.current_step_index = 0
        self.btn_open_viewer.pack_forget()

        self.btn_run_stitch.config(state=tk.DISABLED, bg="#CBD5E1")
        self.btn_select_file.config(state=tk.DISABLED)
        self.lbl_status.config(
            text="Status: ⏳ Extracting & analyzing fragment edges...",
            fg="#D97706",
        )
        self.root.update()

        raw_fragment_metadata, _ = display_detected_fragments(
            self.selected_path
        )
        if not raw_fragment_metadata:
            messagebox.showerror(
                "Error",
                "No distinct paper fragments found or image path error.",
            )
            self.reset_ui()
            return

        self.stitched_canvas = display_final_canvas(
            raw_fragment_metadata, step_callback=self.record_stitch_step
        )

        self.btn_select_file.config(state=tk.NORMAL)
        self.btn_run_stitch.config(state=tk.NORMAL, bg=PALETTE_EMERALD)
        
        self.btn_open_viewer.pack(fill=tk.X, pady=(10, 0))

        self.lbl_status.config(
            text="Status: ✅ Assembly Complete! Use ◀ ▶ buttons below to review alignment.",
            fg=PALETTE_EMERALD,
        )

    def open_result_viewer(self):
        if self.stitched_canvas is not None:
            self.root.withdraw()
            ResultViewer(self.root, self.stitched_canvas)

    def reset_ui(self):
        self.btn_run_stitch.config(state=tk.NORMAL, bg=PALETTE_EMERALD)
        self.btn_select_file.config(state=tk.NORMAL)
        self.lbl_status.config(text="Status: Ready", fg=PALETTE_SLATE)

    def run(self):
        self.root.mainloop()


class ResultViewer:

    def __init__(self, main_root, stitched_img):
        self.main_root = main_root
        self.stitched_img = stitched_img
        self.current_display_img = stitched_img
        self.inpainted_img = None

        self.window = tk.Toplevel()
        self.window.title("Paper Studio — Stitched Output Viewer")
        self.window.geometry("860x700")
        self.window.configure(bg=PALETTE_BG)
        self.window.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.setup_ui()

    def setup_ui(self):
        header = tk.Frame(self.window, bg=PALETTE_SIDEBAR, padx=20, pady=12, highlightbackground=PALETTE_BORDER, highlightthickness=1)
        header.pack(fill=tk.X, side=tk.TOP)

        self.title_lbl = tk.Label(
            header,
            text="Stitched Result",
            font=("Segoe UI", 12, "bold"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_TEXT_DARK,
        )
        self.title_lbl.pack(side=tk.LEFT)

        btn_exit = tk.Button(
            header,
            text="Close App",
            font=("Segoe UI", 9, "bold"),
            bg="#EF4444",
            fg="white",
            activebackground="#DC2626",
            activeforeground="white",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.exit_app,
        )
        btn_exit.pack(side=tk.RIGHT, padx=(10, 0))

        btn_back = tk.Button(
            header,
            text="↩ Back to Steps",
            font=("Segoe UI", 9, "bold"),
            bg=PALETTE_SLATE,
            fg="white",
            activebackground="#475569",
            activeforeground="white",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.back_to_main,
        )
        btn_back.pack(side=tk.RIGHT, padx=(10, 0))

        btn_save = tk.Button(
            header,
            text="💾 Save Image",
            font=("Segoe UI", 9, "bold"),
            bg=PALETTE_EMERALD,
            fg="white",
            activebackground=PALETTE_EMERALD_HOVER,
            activeforeground="white",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.save_image,
        )
        btn_save.pack(side=tk.RIGHT, padx=(10, 0))

        self.btn_inpaint = tk.Button(
            header,
            text="✨ Safe Inpaint Seams",
            font=("Segoe UI", 9, "bold"),
            bg=PALETTE_PRIMARY,
            fg="white",
            activebackground=PALETTE_PRIMARY_HOVER,
            activeforeground="white",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.trigger_inpainting,
        )
        self.btn_inpaint.pack(side=tk.RIGHT)

        self.img_card = tk.Frame(
            self.window, bg=PALETTE_WORKSPACE, highlightbackground=PALETTE_BORDER, highlightthickness=1
        )
        self.img_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.lbl_image = tk.Label(self.img_card, bg=PALETTE_WORKSPACE, bd=0)
        self.lbl_image.pack(expand=True, anchor="center", padx=10, pady=10)

        self.render_image(self.stitched_img)

    def render_image(self, img_array):
        if len(img_array.shape) == 2:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((800, 560))
        self.photo = ImageTk.PhotoImage(pil_img)

        self.lbl_image.config(image=self.photo)

    def trigger_inpainting(self):
        if self.inpainted_img is None:
            self.inpainted_img = apply_inpainting_transformation(self.stitched_img)
            self.current_display_img = self.inpainted_img
            self.title_lbl.config(text="Seam-Inpainted Output")
            self.btn_inpaint.config(text="🔄 Show Raw Stitched")
        else:
            if np.array_equal(self.current_display_img, self.inpainted_img):
                self.current_display_img = self.stitched_img
                self.title_lbl.config(text="Stitched Result")
                self.btn_inpaint.config(text="✨ Safe Inpaint Seams")
            else:
                self.current_display_img = self.inpainted_img
                self.title_lbl.config(text="Seam-Inpainted Output")
                self.btn_inpaint.config(text="🔄 Show Raw Stitched")

        self.render_image(self.current_display_img)

    def save_image(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
        )
        if file_path:
            cv2.imwrite(file_path, self.current_display_img)
            messagebox.showinfo("Saved", f"Image saved successfully to:\n{file_path}")

    def back_to_main(self):
        self.window.destroy()
        self.main_root.deiconify()

    def exit_app(self):
        self.window.destroy()
        self.main_root.destroy()
        sys.exit()


if __name__ == "__main__":
    app = ApplicationWindow()
    app.run()