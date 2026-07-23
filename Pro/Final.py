import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


def run_case1_pipeline(image_path, step_callback=None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
    pieces, masks, boundaries = [], [], []

    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cropped_piece = image[y : y + h, x : x + w]

        mask = np.zeros_like(closed)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        cropped_mask = mask[y : y + h, x : x + w]

        dilated = cv2.dilate(
            cropped_mask, np.ones((3, 3), np.uint8), iterations=1
        )
        boundary = cv2.absdiff(dilated, cropped_mask)
        isolated_piece = cv2.bitwise_and(
            cropped_piece, cropped_piece, mask=cropped_mask
        )

        pieces.append(isolated_piece)
        masks.append(cropped_mask)
        boundaries.append(boundary)

    num_pieces = len(pieces)

    def find_best_translation(i, j):
        bi, bj = boundaries[i], boundaries[j]
        mi, mj = masks[i], masks[j]
        hi, wi = bi.shape
        hj, wj = bj.shape

        padded_b = cv2.copyMakeBorder(
            bi, hj, hj, wj, wj, cv2.BORDER_CONSTANT, value=0
        )
        corr = cv2.matchTemplate(padded_b, bj, cv2.TM_CCORR)
        flat_indices = np.argsort(corr.ravel())[-150:][::-1]

        best_score, best_dx, best_dy = -1, 0, 0

        for idx in flat_indices:
            r, c = np.unravel_index(idx, corr.shape)
            dy, dx = r - hj, c - wj

            y_start_i, y_end_i = max(0, dy), min(hi, dy + hj)
            x_start_i, x_end_i = max(0, dx), min(wi, dx + wj)
            y_start_j, y_end_j = max(0, -dy), min(hj, hi - dy)
            x_start_j, x_end_j = max(0, -dx), min(wj, wi - dx)

            if y_end_i <= y_start_i or x_end_i <= x_start_i:
                continue

            overlap_mask_i = mi[y_start_i:y_end_i, x_start_i:x_end_i]
            overlap_mask_j = mj[y_start_j:y_end_j, x_start_j:x_end_j]

            overlap_area = np.sum((overlap_mask_i > 0) & (overlap_mask_j > 0))
            min_mask_area = min(np.sum(mi > 0), np.sum(mj > 0))

            if overlap_area > (0.015 * min_mask_area):
                continue

            contact_score = np.sum(
                (bi[y_start_i:y_end_i, x_start_i:x_end_i] > 0)
                & (bj[y_start_j:y_end_j, x_start_j:x_end_j] > 0)
            )

            if contact_score > best_score:
                best_score, best_dx, best_dy = contact_score, dx, dy

        return best_dx, best_dy, best_score

    match_graph = []
    for i in range(num_pieces):
        for j in range(i + 1, num_pieces):
            dx, dy, score = find_best_translation(i, j)
            if score > 5:
                match_graph.append(
                    {"i": i, "j": j, "dx": dx, "dy": dy, "score": score}
                )

    match_graph.sort(key=lambda x: x["score"], reverse=True)

    root_piece = np.argmax([np.sum(m > 0) for m in masks])
    positions = {root_piece: (0, 0)}
    placed = {root_piece}

    def render_current_canvas():
        min_x = min(pos[0] for pos in positions.values())
        min_y = min(pos[1] for pos in positions.values())

        norm_pos = {
            k: (pos[0] - min_x, pos[1] - min_y) for k, pos in positions.items()
        }
        max_w = max(px + pieces[k].shape[1] for k, (px, py) in norm_pos.items())
        max_h = max(py + pieces[k].shape[0] for k, (px, py) in norm_pos.items())

        curr_canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        for idx, (px, py) in norm_pos.items():
            img, mask = pieces[idx], masks[idx]
            h, w = img.shape[:2]
            curr_canvas[py : py + h, px : px + w][mask > 0] = img[mask > 0]

        return curr_canvas

    if step_callback:
        step_callback(render_current_canvas(), len(placed), num_pieces)

    while len(placed) < num_pieces:
        progress = False
        for match in match_graph:
            i, j, dx, dy = (
                match["i"],
                match["j"],
                match["dx"],
                match["dy"],
            )

            if i in placed and j not in placed:
                positions[j] = (positions[i][0] + dx, positions[i][1] + dy)
                placed.add(j)
                progress = True
            elif j in placed and i not in placed:
                positions[i] = (positions[j][0] - dx, positions[j][1] - dy)
                placed.add(i)
                progress = True

            if progress:
                if step_callback:
                    step_callback(
                        render_current_canvas(), len(placed), num_pieces
                    )
                break

        if not progress:
            unplaced = list(set(range(num_pieces)) - placed)
            if not unplaced:
                break
            positions[unplaced[0]] = (0, 0)
            placed.add(unplaced[0])
            if step_callback:
                step_callback(render_current_canvas(), len(placed), num_pieces)

    return render_current_canvas(), len(pieces)


def apply_inpainting_transformation(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    paper_mask = (gray > 15).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_paper = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel)
    gaps_mask = cv2.bitwise_and(closed_paper, cv2.bitwise_not(paper_mask))

    _, bright_seams = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    seams_mask = cv2.bitwise_and(
        cv2.bitwise_or(gaps_mask, bright_seams), closed_paper
    )

    inpainted = cv2.inpaint(
        canvas, seams_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )

    lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    return cv2.adaptiveThreshold(
        cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY),
        200,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        3,
        2,
    )


PALETTE_BG, PALETTE_SIDEBAR, PALETTE_WORKSPACE, PALETTE_BORDER = (
    "#F8FAFC",
    "#FFFFFF",
    "#F1F5F9",
    "#E2E8F0",
)
PALETTE_PRIMARY, PALETTE_PRIMARY_HOVER, PALETTE_EMERALD, PALETTE_SLATE, PALETTE_TEXT_DARK = (
    "#4F46E5",
    "#4338CA",
    "#10B981",
    "#64748B",
    "#0F172A",
)


class ApplicationWindow:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Paper Studio — Document Reconstruction Engine")
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
            self.root,
            bg=PALETTE_SIDEBAR,
            width=280,
            highlightbackground=PALETTE_BORDER,
            highlightthickness=1,
        )
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        brand = tk.Frame(self.sidebar, bg=PALETTE_SIDEBAR, padx=20, pady=24)
        brand.pack(fill=tk.X)
        tk.Label(
            brand,
            text="Paper Studio",
            font=("Segoe UI", 16, "bold"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_TEXT_DARK,
            anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            brand,
            text="Document Reconstruction Engine",
            font=("Segoe UI", 8),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_SLATE,
            anchor="w",
        ).pack(fill=tk.X, pady=(2, 0))

        controls = tk.Frame(self.sidebar, bg=PALETTE_SIDEBAR, padx=20)
        controls.pack(fill=tk.X, pady=(10, 0))

        tk.Label(
            controls,
            text="1. SOURCE DOCUMENT",
            font=("Segoe UI", 8, "bold"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_PRIMARY,
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 6))

        self.btn_select_file = tk.Button(
            controls,
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
            controls,
            text="No file selected",
            font=("Segoe UI", 8, "italic"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_SLATE,
            anchor="w",
            wraplength=230,
            justify="left",
        )
        self.lbl_file_name.pack(fill=tk.X, pady=(6, 18))

        tk.Label(
            controls,
            text="2. RECONSTRUCTION",
            font=("Segoe UI", 8, "bold"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_PRIMARY,
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 6))

        self.btn_run_stitch = tk.Button(
            controls,
            text="🧩 Stitch Fragments",
            font=("Segoe UI", 9, "bold"),
            bg=PALETTE_EMERALD,
            fg="white",
            activebackground="#059669",
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
            controls,
            text="🔍 View Final Output & Inpaint ▶",
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

        status_frame = tk.Frame(
            self.sidebar, bg=PALETTE_SIDEBAR, padx=20, pady=20
        )
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
            workspace,
            bg=PALETTE_WORKSPACE,
            highlightbackground=PALETTE_BORDER,
            highlightthickness=1,
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
            text=(
                "🖼️ No Document Loaded\n\nClick 'Choose Image File...' on the"
                " left panel to begin."
            ),
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
                text=os.path.basename(file_path),
                fg=PALETTE_TEXT_DARK,
                font=("Segoe UI", 8, "bold"),
            )
            self.btn_run_stitch.config(state=tk.NORMAL, bg=PALETTE_EMERALD)
            self.btn_open_viewer.pack_forget()
            self.lbl_status.config(
                text="Status: Image loaded. Ready to stitch.",
                fg=PALETTE_PRIMARY,
            )

            self.step_history, self.current_step_index, self.stitched_canvas = (
                [],
                0,
                None,
            )
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
            text=f"Status: ⏳ Stitching step {current_step}/{total_steps}...",
            fg="#D97706",
        )
        self.root.update()

    def render_step_frame(self):
        if not self.step_history:
            return

        img_rgb = cv2.cvtColor(
            self.step_history[self.current_step_index], cv2.COLOR_BGR2RGB
        )
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((460, 340))
        self.input_preview_photo = ImageTk.PhotoImage(pil_img)

        total_steps = len(self.step_history) - 1
        prefix = (
            "Final Assembly Complete"
            if (
                self.current_step_index == total_steps
                and self.stitched_canvas is not None
            )
            else "Assembly Progress"
        )
        self.lbl_card_title.config(
            text=f"{prefix} — Step {self.current_step_index} of {total_steps}"
        )

        self.lbl_preview.config(
            image=self.input_preview_photo, text="", bg=PALETTE_WORKSPACE
        )
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
        has_prev = total > 1 and self.current_step_index > 0
        has_next = total > 1 and self.current_step_index < total - 1

        self.btn_prev_step.config(
            state=tk.NORMAL if has_prev else tk.DISABLED,
            bg=PALETTE_SLATE if has_prev else "#CBD5E1",
        )
        self.btn_next_step.config(
            state=tk.NORMAL if has_next else tk.DISABLED,
            bg=PALETTE_SLATE if has_next else "#CBD5E1",
        )

    def process_image(self):
        if not self.selected_path:
            return

        self.step_history, self.current_step_index = [], 0
        self.btn_open_viewer.pack_forget()

        self.btn_run_stitch.config(state=tk.DISABLED, bg="#CBD5E1")
        self.btn_select_file.config(state=tk.DISABLED)
        self.lbl_status.config(
            text="Status: ⏳ Executing core logic...", fg="#D97706"
        )
        self.root.update()

        try:
            self.stitched_canvas, _ = run_case1_pipeline(
                self.selected_path, step_callback=self.record_stitch_step
            )
        except Exception as e:
            messagebox.showerror(
                "Error", f"An error occurred during processing:\n{e}"
            )
            self.reset_ui()
            return

        self.btn_select_file.config(state=tk.NORMAL)
        self.btn_run_stitch.config(state=tk.NORMAL, bg=PALETTE_EMERALD)
        self.btn_open_viewer.pack(fill=tk.X, pady=(10, 0))
        self.lbl_status.config(
            text=(
                "Status: ✅ Assembly Complete! Use ◀ ▶ buttons below to review"
                " all steps."
            ),
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
        self.main_root, self.stitched_img = main_root, stitched_img
        self.current_display_img, self.inpainted_img = stitched_img, None

        self.window = tk.Toplevel()
        self.window.title("Paper Studio — Stitched Output Viewer")
        self.window.geometry("860x700")
        self.window.configure(bg=PALETTE_BG)
        self.window.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.setup_ui()

    def setup_ui(self):
        header = tk.Frame(
            self.window,
            bg=PALETTE_SIDEBAR,
            padx=20,
            pady=12,
            highlightbackground=PALETTE_BORDER,
            highlightthickness=1,
        )
        header.pack(fill=tk.X, side=tk.TOP)

        self.title_lbl = tk.Label(
            header,
            text="Stitched Result",
            font=("Segoe UI", 12, "bold"),
            bg=PALETTE_SIDEBAR,
            fg=PALETTE_TEXT_DARK,
        )
        self.title_lbl.pack(side=tk.LEFT)

        buttons = [
            ("Close App", "#EF4444", "#DC2626", self.exit_app),
            ("↩ Back to Steps", PALETTE_SLATE, "#475569", self.back_to_main),
            ("💾 Save Image", PALETTE_EMERALD, "#059669", self.save_image),
        ]

        for text, bg, active_bg, cmd in buttons:
            tk.Button(
                header,
                text=text,
                font=("Segoe UI", 9, "bold"),
                bg=bg,
                fg="white",
                activebackground=active_bg,
                activeforeground="white",
                bd=0,
                padx=14,
                pady=6,
                cursor="hand2",
                command=cmd,
            ).pack(side=tk.RIGHT, padx=(10, 0))

        self.btn_inpaint = tk.Button(
            header,
            text="✨ Apply Inpainting",
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
            self.window,
            bg=PALETTE_WORKSPACE,
            highlightbackground=PALETTE_BORDER,
            highlightthickness=1,
        )
        self.img_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.lbl_image = tk.Label(self.img_card, bg=PALETTE_WORKSPACE, bd=0)
        self.lbl_image.pack(expand=True, anchor="center", padx=10, pady=10)

        self.render_image(self.stitched_img)

    def render_image(self, img_array):
        img_rgb = (
            cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            if len(img_array.shape) == 2
            else cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        )
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((800, 560))
        self.photo = ImageTk.PhotoImage(pil_img)
        self.lbl_image.config(image=self.photo)

    def trigger_inpainting(self):
        if self.inpainted_img is None:
            self.inpainted_img = apply_inpainting_transformation(
                self.stitched_img
            )

        is_inpainted = np.array_equal(
            self.current_display_img, self.inpainted_img
        )
        self.current_display_img = (
            self.stitched_img if is_inpainted else self.inpainted_img
        )

        self.title_lbl.config(
            text="Stitched Result"
            if is_inpainted
            else "Inpainted & Binarized Output"
        )
        self.btn_inpaint.config(
            text="✨ Apply Inpainting" if is_inpainted else "🔄 Show Raw Stitched"
        )
        self.render_image(self.current_display_img)

    def save_image(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            cv2.imwrite(file_path, self.current_display_img)
            messagebox.showinfo(
                "Saved", f"Image saved successfully to:\n{file_path}"
            )

    def back_to_main(self):
        self.window.destroy()
        self.main_root.deiconify()

    def exit_app(self):
        self.window.destroy()
        self.main_root.destroy()
        sys.exit()


if __name__ == "__main__":
    ApplicationWindow().run()