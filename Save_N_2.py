import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter.simpledialog  # Ajoute en haut
import time
import threading

# Constants for high-fidelity rendering
DPI = 150
BASE_SCALE = DPI / 72  # PDF points are 1/72 inch

class PDFImageDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Détection d'images dans PDF")
        self.geometry("1200x900")

        # Initial state
        self.doc = None
        self.page = None
        self.current_page = 0
        self.zoom = 1.0

        # UI widgets
        self.load_pdf_button = ctk.CTkButton(self, text="Charger PDF", command=self.load_pdf)
        self.load_pdf_button.pack(pady=5)

        self.load_img_button = ctk.CTkButton(self, text="Charger Image", command=self.load_image)
        self.load_img_button.pack(pady=5)

        self.progress = ctk.CTkProgressBar(self)
        self.progress.pack(pady=10)
        self.progress.set(0)
        self.progress_label = ctk.CTkLabel(self, text="Progression : 0%")
        self.progress_label.pack()

        self.canvas = ctk.CTkCanvas(self, bg="gray20")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>",   self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>",   self.on_zoom)
        self.canvas.bind("<Configure>", self._resize_canvas)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.scroll_y = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        self.rect = None
        self.start = (0, 0)
        self.end = (0, 0)

    def load_pdf(self):
        path = filedialog.askopenfilename(title="Sélectionner un PDF", filetypes=[("PDF Files", "*.pdf")])
        if not path:
            return
        self.doc = fitz.open(path)
        self.current_page = 0
        self.page = self.doc.load_page(self.current_page)
        self.render_page()

    def load_image(self):
        path = filedialog.askopenfilename(title="Sélectionner une image", filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            print("Erreur lors du chargement de l'image.")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_image = img_rgb
        self.doc = None  # Désactive le PDF
        self.page = None
        self.render_image(img_rgb)

    def render_image(self, img_rgb):
        # Redimensionne pour tenir dans la fenêtre si besoin
        h, w, _ = img_rgb.shape
        max_w, max_h = 1000, 1400
        scale = min(max_w / w, max_h / h, 1.0)
        img_disp = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(img_disp)
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.display_scale = scale
        self.zoom = 1.0

    def render_page(self):
        # Affichage à basse résolution pour éviter MemoryError
        self.display_scale = min(1.0, 800 / self.page.rect.width)  # max 800px de large
        mat = fitz.Matrix(BASE_SCALE * self.zoom * self.display_scale, BASE_SCALE * self.zoom * self.display_scale)
        pix = self.page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        self.tk_img = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def on_zoom(self, event):
        # Zoom in/out around cursor
        factor = 1.1 if event.delta > 0 else 1/1.1
        self.zoom *= factor
        self.render_page()

    def on_mouse_down(self, event):
        self.start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(*self.start, *self.start, outline="orange", width=2)

    def on_mouse_drag(self, event):
        cur = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.coords(self.rect, *self.start, *cur)

    def on_mouse_up(self, event):
        self.end = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

        if self.page is not None:
            # --- CAS PDF ---
            scale_factor = BASE_SCALE * self.zoom * self.display_scale
            x1, y1 = [coord / scale_factor for coord in self.start]
            x2, y2 = [coord / scale_factor for coord in self.end]
            clip_rect = fitz.Rect(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

            high_res_scale = 300 / 72  # ou 150/72 selon ta RAM
            sel_pix = self.page.get_pixmap(matrix=fitz.Matrix(high_res_scale, high_res_scale), clip=clip_rect)
            sel_np = np.frombuffer(sel_pix.samples, dtype=np.uint8).reshape(sel_pix.height, sel_pix.width, sel_pix.n)
            sel_rgb = cv2.cvtColor(sel_np, cv2.COLOR_BGR2RGB)

            # Recadrage du symbole
            cropped_sel_rgb = crop_to_symbol(sel_rgb)

            # Sauvegarde du template
            template_dir = "template"
            os.makedirs(template_dir, exist_ok=True)
            tpl_name = tkinter.simpledialog.askstring("Nom du template", "Nommer la sélection :")
            if tpl_name:
                tpl_path = os.path.join(template_dir, f"{tpl_name}.png")
                cv2.imwrite(tpl_path, cropped_sel_rgb)
                print(f"Template sauvegardé : {tpl_path}")

                # Extraction page complète à la même résolution
                full_pix = self.page.get_pixmap(matrix=fitz.Matrix(high_res_scale, high_res_scale), alpha=False)
                full_np = np.frombuffer(full_pix.samples, dtype=np.uint8).reshape(full_pix.height, full_pix.width, full_pix.n)
                full_rgb = cv2.cvtColor(full_np, cv2.COLOR_BGR2RGB)

                # --- Détection robuste comme pour l'image ---
                tpl_gray = cv2.cvtColor(cropped_sel_rgb, cv2.COLOR_RGB2GRAY)
                tpl_gray = cv2.equalizeHist(tpl_gray)
                img_gray = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2GRAY)
                img_gray = cv2.equalizeHist(img_gray)

                h_img, w_img = img_gray.shape[:2]
                h_tpl, w_tpl = tpl_gray.shape[:2]
                min_scale = max(0.2, 5 / min(h_tpl, w_tpl))
                max_scale = min(2.0, min(h_img / h_tpl, w_img / w_tpl))
                scales = np.linspace(min_scale, max_scale, 25)

                all_boxes = []
                for tpl in [tpl_gray, cv2.rotate(tpl_gray, cv2.ROTATE_180)]:
                    for scale in scales:
                        nh, nw = int(h_tpl * scale), int(w_tpl * scale)
                        if nh < 10 or nw < 10 or nh > h_img // 2 or nw > w_img // 2:
                            continue
                        resized_tpl = cv2.resize(tpl, (nw, nh), interpolation=cv2.INTER_AREA)
                        try:
                            res = cv2.matchTemplate(img_gray, resized_tpl, cv2.TM_CCOEFF_NORMED)
                        except cv2.error:
                            continue
                        loc = np.where(res >= 0.7)
                        for pt in zip(*loc[::-1]):
                            all_boxes.append((pt[0], pt[1], nw, nh))
                boxes = self.nms(all_boxes, overlapThresh=0.3)
                print(f"Occurrences détectées : {len(boxes)}")

                # Dessine les rectangles sur une copie de la page
                img_detect = full_rgb.copy()
                for idx, (x, y, w, h) in enumerate(boxes, 1):
                    cv2.rectangle(img_detect, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # Ajoute le numéro en haut à gauche du rectangle
                    cv2.putText(
                        img_detect, str(idx), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                    )

                # Affiche l'image annotée dans le canvas
                img_disp = cv2.cvtColor(img_detect, cv2.COLOR_RGB2BGR)
                h, w, _ = img_disp.shape
                max_w, max_h = 1000, 1400
                scale = min(max_w / w, max_h / h, 1.0)
                img_disp = cv2.resize(img_disp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                img_pil = Image.fromarray(img_disp)
                self.tk_img = ImageTk.PhotoImage(img_pil)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
                self.canvas.config(scrollregion=self.canvas.bbox("all"))
                cv2.imwrite("image_detected.png", cv2.cvtColor(img_detect, cv2.COLOR_RGB2BGR))

        elif hasattr(self, "current_image") and self.current_image is not None:
            # --- CAS IMAGE ---
            x1, y1 = [int(coord / self.display_scale) for coord in self.start]
            x2, y2 = [int(coord / self.display_scale) for coord in self.end]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)

            h, w, _ = self.current_image.shape
            xmin = max(0, min(xmin, w-1))
            xmax = max(0, min(xmax, w))
            ymin = max(0, min(ymin, h-1))
            ymax = max(0, min(ymax, h))

            if xmax - xmin < 5 or ymax - ymin < 5:
                print("Sélection trop petite ou hors image.")
                return

            sel_rgb = self.current_image[ymin:ymax, xmin:xmax]
            if sel_rgb.size == 0:
                print("Sélection vide.")
                return

            cropped_sel_rgb = crop_to_symbol(sel_rgb)
            template_dir = "template"
            os.makedirs(template_dir, exist_ok=True)
            tpl_name = tkinter.simpledialog.askstring("Nom du template", "Nommer la sélection :")
            if tpl_name:
                tpl_path = os.path.join(template_dir, f"{tpl_name}.png")
                cv2.imwrite(tpl_path, cropped_sel_rgb)
                print(f"Template sauvegardé : {tpl_path}")

                
               

                img_detect = self.current_image.copy()
                for idx, (x, y, w, h) in enumerate(positions, 1):
                    cv2.rectangle(img_detect, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # Ajoute le numéro en haut à gauche du rectangle
                    cv2.putText(
                        img_detect, str(idx), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                    )
                
                # Affiche l'image annotée dans le canvas
                img_disp = cv2.cvtColor(img_detect, cv2.COLOR_RGB2BGR)
                h, w, _ = img_disp.shape
                max_w, max_h = 1000, 1400
                scale = min(max_w / w, max_h / h, 1.0)
                img_disp = cv2.resize(img_disp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                img_pil = Image.fromarray(img_disp)
                self.tk_img = ImageTk.PhotoImage(img_pil)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
                self.canvas.config(scrollregion=self.canvas.bbox("all"))
                cv2.imwrite("image_detected.png", cv2.cvtColor(img_detect, cv2.COLOR_RGB2BGR))

            counts, positions = self.detect_all_templates(self.current_image, "template")
            # Pour afficher les détections de tous les templates :
            img_detect = self.current_image.copy()
            for tpl_name, boxes in positions.items():
                for idx, (x, y, w, h) in enumerate(boxes, 1):
                    cv2.rectangle(img_detect, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # Ajoute le numéro en haut à gauche du rectangle
                    cv2.putText(
                        img_detect, str(idx), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                    )
            # Affiche ou sauvegarde img_detect si besoin

        else:
            print("Aucun document ou image chargé.")

    

    def detect_all_templates(self, img_rgb, templates_dir, threshold=0.58):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        counts = {}
        positions = {}

        fnames = [f for f in os.listdir(templates_dir) if f.lower().endswith(".png")]
        total = len(fnames)
        start_time = time.time()

        for idx_file, fname in enumerate(fnames, 1):
            tpl_name = os.path.splitext(fname)[0]
            template = cv2.imread(os.path.join(templates_dir, fname), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            template_flipped = cv2.rotate(template, cv2.ROTATE_180)
            all_pts = []
            for tpl in [template, template_flipped]:
                h, w = tpl.shape[:2]
                if h >= 10 and w >= 10 and h <= img_gray.shape[0] and w <= img_gray.shape[1]:
                    res = cv2.matchTemplate(img_gray, tpl, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= threshold)
                    for pt in zip(*loc[::-1]):
                        all_pts.append((pt[0], pt[1], w, h))
            filtered_pts = self.nms(all_pts, overlapThresh=0.3)
            counts[tpl_name] = len(filtered_pts)
            positions[tpl_name] = filtered_pts

            # Mise à jour de la barre de progression
            progress = idx_file / total
            self.progress.set(progress)
            self.progress_label.configure(text=f"Progression : {int(progress*100)}%")
            self.update_idletasks()
            self.update()  # Ajoute cette ligne !

        elapsed = time.time() - start_time
        print(f"Temps total de détection : {elapsed:.2f} secondes")
        self.progress_label.configure(text=f"Terminé en {elapsed:.2f} sec")
        return counts, positions

    @staticmethod
    def nms(boxes, overlapThresh=0.3):
        if not boxes:
            return []
        arr = np.array(boxes)
        x1, y1 = arr[:,0], arr[:,1]
        x2, y2 = x1 + arr[:,2], y1 + arr[:,3]
        areas = arr[:,2] * arr[:,3]
        idxs = np.argsort(areas)[::-1]
        pick = []
        while len(idxs) > 0:
            i = idxs[0]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / areas[idxs[1:]]
            idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlapThresh)[0] + 1)))
        return [tuple(arr[i]) for i in pick]

    def annotate_and_save(self, boxes, out_path):
        # Draw red rectangles on PDF in vector coordinates
        page = self.page
        for (x_px, y_px, w_px, h_px) in boxes:
            rect = fitz.Rect(
                x_px / BASE_SCALE,
                y_px / BASE_SCALE,
                (x_px + w_px) / BASE_SCALE,
                (y_px + h_px) / BASE_SCALE,
            )
            page.draw_rect(rect, color=(1, 0, 0), width=2)
        # Supprime le fichier s'il existe déjà
        if os.path.exists(out_path):
            os.remove(out_path)
        self.doc.save(out_path)

    def _resize_canvas(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        # Pour Windows
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def start_detection(self):
        threading.Thread(
            target=self.detect_all_templates,
            args=(self.current_image, "template"),
            daemon=True
        ).start()

def crop_to_symbol(model_img):
    # Si l'image est déjà en niveaux de gris, ne pas reconvertir
    if len(model_img.shape) == 3 and model_img.shape[2] == 3:
        gray = cv2.cvtColor(model_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = model_img
    # Seuillage adaptatif pour isoler le symbole
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return model_img[y:y+h, x:x+w]
    return model_img

if __name__ == "__main__":
    app = PDFImageDetectorApp()
    app.mainloop()