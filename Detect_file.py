import os
import cv2
import fitz  # PyMuPDF
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, simpledialog
import threading
import platform
import subprocess

# Configuration de l'apparence
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Constantes
DPI = 200  # Pour le rendu haute qualité des PDF
BASE_SCALE = DPI / 72  # PDF points à pouces
TEMPLATE_DIR = "templates"
RESULTS_DIR = "results"

# Créer les répertoires s'ils n'existent pas
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def crop_to_symbol(image):
    """Recadre l'image pour ne conserver que la forme noire principale."""
    # Conversion en niveaux de gris si nécessaire
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Seuillage pour isoler la forme noire
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Trouver le plus grand contour
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # Ajouter une marge de 2 pixels
        margin = 2
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2 * margin)
        h = min(gray.shape[0] - y, h + 2 * margin)
        
        # Recadrer l'image originale
        if len(image.shape) == 3:
            cropped = image[y:y+h, x:x+w]
        else:
            cropped = gray[y:y+h, x:x+w]
        
        return cropped
    
    # Si pas de contour, retourner l'image originale
    return image


class PDFTemplateDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configuration de la fenêtre
        self.title("Détecteur de Templates PDF")
        self.geometry("1200x800")
        
        # Variables d'état
        self.doc = None
        self.page = None
        self.current_page = 0
        self.zoom = 1.0
        self.start = None
        self.end = None
        self.rect = None
        self.current_image = None
        self.display_scale = 1.0
        
        # Variables pour la gestion des détections
        self.detected_image = None
        self.detection_positions = {}
        self.current_scale = 1.0
        
        # Création de l'interface
        self.create_widgets()
        
    def create_widgets(self):
        """Crée tous les widgets de l'interface."""
        # Cadre supérieur pour les boutons
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(fill="x", padx=10, pady=5)
        
        # Bouton de chargement de PDF
        self.load_pdf_button = ctk.CTkButton(top_frame, text="Charger PDF", command=self.load_pdf)
        self.load_pdf_button.pack(side="left", padx=5, pady=5)
        
        # Bouton pour ouvrir le PDF dans le lecteur par défaut
        self.open_pdf_button = ctk.CTkButton(top_frame, text="Ouvrir PDF", 
                                           command=self.open_pdf_external,
                                           fg_color="#3a7ebf", hover_color="#2a5a8f")
        self.open_pdf_button.pack(side="left", padx=5, pady=5)
        
        # Contrôles de navigation PDF
        self.pdf_controls_frame = ctk.CTkFrame(top_frame)
        self.pdf_controls_frame.pack(side="left", padx=20)
        
        self.prev_page_button = ctk.CTkButton(self.pdf_controls_frame, text="< Page préc.", 
                                            command=self.previous_page, width=100)
        self.prev_page_button.pack(side="left", padx=5)
        
        self.page_label = ctk.CTkLabel(self.pdf_controls_frame, text="Page: 0/0")
        self.page_label.pack(side="left", padx=10)
        
        self.next_page_button = ctk.CTkButton(self.pdf_controls_frame, text="Page suiv. >", 
                                            command=self.next_page, width=100)
        self.next_page_button.pack(side="left", padx=5)
        
        # Bouton de détection
        self.detect_button = ctk.CTkButton(top_frame, text="Détecter Templates", 
                                          command=self.start_detection,
                                          fg_color="green", hover_color="darkgreen")
        self.detect_button.pack(side="left", padx=10)
        
        # Bouton de réinitialisation
        self.reset_button = ctk.CTkButton(top_frame, text="Réinitialiser", 
                                         command=self.reset_app,
                                         fg_color="firebrick", hover_color="darkred")
        self.reset_button.pack(side="right", padx=10)
        
        # Affichage des détections
        self.show_detections_var = ctk.BooleanVar(value=True)
        self.show_detections_checkbox = ctk.CTkCheckBox(
            top_frame, text="Afficher détections", 
            variable=self.show_detections_var,
            command=self.toggle_detections_display)
        self.show_detections_checkbox.pack(side="right", padx=10)
        
        # Cadre pour le seuil de similarité
        similarity_frame = ctk.CTkFrame(self)
        similarity_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        similarity_label = ctk.CTkLabel(similarity_frame, text="Seuil de similarité:")
        similarity_label.pack(side="left", padx=10, pady=5)
        
        # Valeur du seuil (0.0 à 1.0)
        self.threshold_var = ctk.DoubleVar(value=0.7)
        self.threshold_value_label = ctk.CTkLabel(similarity_frame, 
                                                text=f"{self.threshold_var.get():.2f}")
        self.threshold_value_label.pack(side="right", padx=10, pady=5)
        
        # Curseur
        self.threshold_slider = ctk.CTkSlider(similarity_frame, 
                                             from_=0.5, to=1.0, 
                                             number_of_steps=50,
                                             variable=self.threshold_var,
                                             command=self.update_threshold_label)
        self.threshold_slider.pack(side="left", fill="x", expand=True, padx=10, pady=5)
        
        # Barre de progression et étiquette d'état
        self.progress_frame = ctk.CTkFrame(self)
        self.progress_frame.pack(fill="x", padx=10, pady=5)
        
        self.progress = ctk.CTkProgressBar(self.progress_frame)
        self.progress.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.progress.set(0)
        
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Prêt")
        self.progress_label.pack(side="right", padx=10)
        
        # Cadre principal contenant le canvas
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Canvas pour l'affichage du PDF
        self.canvas_frame = ctk.CTkFrame(main_frame)
        self.canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.canvas = ctk.CTkCanvas(self.canvas_frame, bg="#2a2a2a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Scrollbars
        self.h_scrollbar = ctk.CTkScrollbar(self.canvas_frame, orientation="horizontal", command=self.canvas.xview)
        self.h_scrollbar.pack(side="bottom", fill="x")
        
        self.v_scrollbar = ctk.CTkScrollbar(self.canvas_frame, orientation="vertical", command=self.canvas.yview)
        self.v_scrollbar.pack(side="right", fill="y")
        
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_zoom)  # Windows
        self.canvas.bind("<Button-4>", self.on_zoom)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_zoom)    # Linux scroll down
        
        # Étiquette résultat en bas
        self.result_label = ctk.CTkLabel(self, text="")
        self.result_label.pack(pady=5)
        
        # Initialiser
        self.reset_app()
    
    def update_threshold_label(self, value):
        """Met à jour l'affichage du seuil de similarité."""
        self.threshold_value_label.configure(text=f"{float(value):.2f}")
        
    def open_pdf_external(self):
        """Ouvre le fichier PDF courant dans le lecteur PDF par défaut du système."""
        if not self.doc:
            self.progress_label.configure(text="Aucun PDF chargé")
            return
        
        try:
            # Récupérer le chemin du fichier PDF
            pdf_path = self.doc.name
            
            # Ouvrir avec le lecteur par défaut selon le système d'exploitation
            system = platform.system()
            if system == 'Windows':
                os.startfile(pdf_path)
            elif system == 'Darwin':  # macOS
                subprocess.call(('open', pdf_path))
            else:  # Linux et autres
                subprocess.call(('xdg-open', pdf_path))
                
            self.progress_label.configure(text=f"PDF ouvert dans le lecteur par défaut")
        except Exception as e:
            self.progress_label.configure(text=f"Erreur lors de l'ouverture du PDF: {e}")
    
    def reset_app(self):
        """Réinitialise l'application pour une nouvelle session."""
        # Réinitialiser les variables
        if self.doc:
            self.doc.close()
        self.doc = None
        self.page = None
        self.current_page = 0
        self.zoom = 1.0
        self.start = None
        self.end = None
        self.rect = None
        self.current_image = None
        self.detected_image = None
        self.detection_positions = {}
        
        # Nettoyer le canvas
        self.canvas.delete("all")
        
        # Réinitialiser les étiquettes
        self.page_label.configure(text="Page: 0/0")
        self.progress_label.configure(text="Prêt")
        self.progress.set(0)
        self.result_label.configure(text="")
        
    def load_pdf(self):
        """Charge un fichier PDF sélectionné par l'utilisateur."""
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return
            
        self.reset_app()  # Réinitialiser avant de charger un nouveau PDF
        
        try:
            self.doc = fitz.open(file_path)
            self.current_page = 0
            self.page_label.configure(text=f"Page: {self.current_page + 1}/{len(self.doc)}")
            self.render_page()
        except Exception as e:
            self.progress_label.configure(text=f"Erreur: {e}")
    
    def render_page(self):
        """Affiche la page courante du PDF."""
        if not self.doc or self.current_page >= len(self.doc):
            return
            
        self.page = self.doc[self.current_page]
        
        # Rendu à la résolution souhaitée
        pix = self.page.get_pixmap(matrix=fitz.Matrix(BASE_SCALE * self.zoom, BASE_SCALE * self.zoom))
        
        # Conversion en image pour tkinter
        img_data = pix.samples
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        self.tk_img = ImageTk.PhotoImage(image=img)
        
        # Afficher dans le canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def next_page(self):
        """Passe à la page suivante du PDF."""
        if self.doc and self.current_page < len(self.doc) - 1:
            self.current_page += 1
            self.page_label.configure(text=f"Page: {self.current_page + 1}/{len(self.doc)}")
            self.render_page()
    
    def previous_page(self):
        """Passe à la page précédente du PDF."""
        if self.doc and self.current_page > 0:
            self.current_page -= 1
            self.page_label.configure(text=f"Page: {self.current_page + 1}/{len(self.doc)}")
            self.render_page()
    
    def on_zoom(self, event):
        """Gère le zoom sur la molette de la souris."""
        # Déterminer la direction du zoom
        if event.delta > 0 or event.num == 4:  # Zoom in
            factor = 1.1
        else:  # Zoom out
            factor = 0.9
        
        self.zoom *= factor
        
        # Réafficher selon le contenu actif
        if hasattr(self, "detected_image") and self.detected_image is not None:
            self.display_image(self.detected_image, factor=self.zoom)
            if self.show_detections_var.get():
                self.display_image_with_detections()
        elif self.page:
            self.render_page()
    
    def on_mouse_down(self, event):
        """Initialise le dessin d'un rectangle de sélection."""
        self.start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if hasattr(self, "rect") and self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(*self.start, *self.start, outline="orange", width=2)
    
    def on_mouse_drag(self, event):
        """Met à jour le rectangle de sélection pendant le mouvement de la souris."""
        cur = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.coords(self.rect, *self.start, *cur)
    
    def on_mouse_up(self, event):
        """Finalise la sélection du template."""
        if not self.page:
            return
            
        self.end = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        
        # Convertir les coordonnées canvas en coordonnées PDF
        x1, y1 = self.start
        x2, y2 = self.end
        
        # Ajuster les coordonnées pour l'échelle actuelle
        scale = BASE_SCALE * self.zoom
        x1, y1 = int(x1 / scale), int(y1 / scale)
        x2, y2 = int(x2 / scale), int(y2 / scale)
        
        # Définir le rectangle de sélection
        clip_rect = fitz.Rect(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        
        # Haute résolution pour la détection
        high_res_scale = 300 / 72
        sel_pix = self.page.get_pixmap(matrix=fitz.Matrix(high_res_scale, high_res_scale), clip=clip_rect)
        sel_np = np.frombuffer(sel_pix.samples, dtype=np.uint8).reshape(sel_pix.height, sel_pix.width, sel_pix.n)
        sel_rgb = cv2.cvtColor(sel_np, cv2.COLOR_RGB2BGR)
        
        # Recadrage intelligent pour ne garder que le symbole
        cropped_sel_rgb = crop_to_symbol(sel_rgb)
        
        # NOUVEAU: Stocker la version originale et la version prétraitée
        original_template = cropped_sel_rgb.copy()
        preprocessed_template = self.preprocess_for_detection(cropped_sel_rgb)
        
        # Demander un nom pour le template et le sauvegarder
        tpl_name = simpledialog.askstring("Nom du template", "Nommer la sélection :")
        if tpl_name:
            tpl_path = os.path.join(TEMPLATE_DIR, f"{tpl_name}.png")
            cv2.imwrite(tpl_path, cropped_sel_rgb)
            
            # Sauvegarder aussi la version prétraitée
            preproc_path = os.path.join(TEMPLATE_DIR, f"{tpl_name}_preprocessed.png")
            cv2.imwrite(preproc_path, preprocessed_template)
            
            self.progress_label.configure(text=f"Template '{tpl_name}' sauvegardé")
            
            # Stocker les deux versions du template pour la détection
            self.template_image = cropped_sel_rgb
            self.preprocessed_template = preprocessed_template
            self.template_name = tpl_name
    
    def preprocess_for_detection(self, image):
        """Applique un prétraitement cohérent pour la détection.
        Cette fonction doit être utilisée à la fois pour le template et l'image source."""
        
        # Conversion en niveaux de gris
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Amélioration du contraste - CLAHE (plus efficace que l'égalisation simple)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Réduction du bruit
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Normalisation de l'éclairage
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        
        # Binarisation adaptative pour une meilleure robustesse aux variations d'éclairage
        binary = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
                                      
        return binary
    
    def start_detection(self):
        """Démarre la détection des templates dans le PDF."""
        if not hasattr(self, "template_image") or self.template_image is None:
            self.progress_label.configure(text="Veuillez d'abord sélectionner un template")
            return
            
        if not self.doc:
            self.progress_label.configure(text="Veuillez d'abord charger un PDF")
            return
        
        # Réinitialiser l'interface de progression
        self.progress.set(0)
        self.progress_label.configure(text="Démarrage de la détection...")
        self.update_idletasks()
        
        # Lancer la détection dans un thread séparé avec le seuil actuel
        threshold = self.threshold_var.get()
        threading.Thread(target=lambda: self._run_detection(threshold), daemon=True).start()
    
    def _run_detection(self, threshold):
        """Exécute la détection de template dans le document complet."""
        template = self.template_image
        template_name = self.template_name
        
        # Étape cruciale: s'assurer que le template est prétraité exactement comme l'image source le sera
        if not hasattr(self, "preprocessed_template"):
            self.preprocessed_template = self.preprocess_for_detection(template.copy())
        
        total_pages = len(self.doc)
        all_detections = {}
        
        # Pour stocker le résultat final
        result_image = None
        first_detection = True
        
        for page_num in range(total_pages):
            # Mise à jour de la progression...
            
            # Charger la page
            page = self.doc[page_num]
            
            # IMPORTANT: Utiliser EXACTEMENT la même résolution que lors de la sélection du template
            high_res_scale = 300 / 72  # Doit être identique à celle utilisée dans on_mouse_up
            pix = page.get_pixmap(matrix=fitz.Matrix(high_res_scale, high_res_scale))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Détection avec traitement cohérent
            result, detections = self.detect_template(img_bgr, template, threshold=threshold)
            
            if len(detections) > 0:
                all_detections[page_num] = detections
                
                # Si c'est la première détection, on garde cette page comme résultat
                if first_detection:
                    result_image = result
                    first_detection = False
        
        # Traiter les résultats
        if not all_detections:
            self.update_ui(lambda: self.progress_label.configure(text="Aucune détection trouvée"))
            return
        
        # Compter le total des détections
        total_detections = sum(len(dets) for dets in all_detections.values())
        
        # Stocker les détections pour l'affichage
        self.detection_positions[template_name] = all_detections.get(self.current_page, [])
        
        # Mettre à jour l'affichage
        if result_image is not None:
            self.detected_image = result_image
            
            # Afficher l'image avec les détections
            self.update_ui(lambda: (
                self.display_image(result_image),
                self.display_image_with_detections(),
                self.progress.set(1.0),
                self.progress_label.configure(text=f"Détection terminée: {total_detections} occurrences trouvées"),
                self.result_label.configure(text=f"Template '{template_name}' : {total_detections} occurrences dans le document")
            ))
    
    def detect_template(self, img, template, threshold=0.7):
        """Détecte les occurrences du template avec un prétraitement cohérent."""
        
        # IMPORTANT: Utiliser le template prétraité stocké
        preprocessed_tpl = self.preprocessed_template
        
        # Ne prétraiter l'image source qu'après le même recadrage que le template
        # pour garantir une comparaison équitable
        
        # Multi-échelle avec pas plus fin
        detections = []
        
        # Utiliser davantage d'échelles pour une meilleure détection
        scales = np.linspace(0.6, 1.4, 17)  # Plus d'échelles pour une meilleure couverture
        
        # Au lieu de prétraiter l'image entière, prétraitons des portions de l'image
        # à chaque position de détection potentielle
        h, w = preprocessed_tpl.shape[:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        
        for scale in scales:
            # Redimensionner le template ou l'image selon l'échelle
            if scale < 1.0:
                # Redimensionner le template
                w_scaled = int(w * scale)
                h_scaled = int(h * scale)
                if w_scaled < 8 or h_scaled < 8:  # Éviter les templates trop petits
                    continue
                resized_tpl = cv2.resize(preprocessed_tpl, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
                search_img = img_gray
            else:
                # Redimensionner l'image source (plus efficace pour les grandes échelles)
                scale_inv = 1.0/scale
                h_img, w_img = img_gray.shape[:2]
                w_img_scaled = int(w_img * scale_inv)
                h_img_scaled = int(h_img * scale_inv)
                search_img = cv2.resize(img_gray, (w_img_scaled, h_img_scaled), interpolation=cv2.INTER_AREA)
                resized_tpl = preprocessed_tpl
                w_scaled, h_scaled = w, h
            
            # Prétraiter l'image de recherche de la même manière que le template
            # mais seulement pour le matching, pas pour l'affichage
            search_img_processed = self.preprocess_for_detection(search_img)
            
            # Template matching
            try:
                res = cv2.matchTemplate(search_img_processed, resized_tpl, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= threshold)
                
                # Ajuster les coordonnées selon l'échelle
                adjust_scale = 1.0 if scale < 1.0 else scale
                
                for pt in zip(*loc[::-1]):
                    x, y = pt
                    if scale >= 1.0:
                        x, y = int(x * scale), int(y * scale)
                    confidence = res[pt[1], pt[0]]
                    detections.append([x, y, int(w_scaled * adjust_scale), int(h_scaled * adjust_scale), scale, confidence])
            except Exception as e:
                continue  # En cas d'erreur, continuer avec l'échelle suivante
        
        # Non-maximum suppression pour éliminer les doublons
        if detections:
            detections = self.non_max_suppression(detections, overlap_thresh=0.2)
        
        # Dessiner les rectangles sur une copie de l'image
        result = img.copy()
        
        # Tri des détections: d'abord par rangée puis par position horizontale
        # Une rangée est définie par des y similaires (tolérance de 20% de la hauteur du template)
        if detections:
            # Calculer la hauteur moyenne des détections
            avg_height = np.mean([h for _, _, _, h, _, _ in detections])
            row_tolerance = avg_height * 0.2;
            
            # Grouper par rangées
            rows = {}
            for det in detections:
                y = det[1]
                row_found = False
                for row_y in rows:
                    if abs(y - row_y) < row_tolerance:
                        rows[row_y].append(det)
                        row_found = True
                        break
                if not row_found:
                    rows[y] = [det]
            
            # Trier chaque rangée par position x
            sorted_rows = sorted(rows.items(), key=lambda r: r[0])
            sorted_detections = []
            for _, row_dets in sorted_rows:
                row_dets.sort(key=lambda d: d[0])  # Tri par x
                sorted_detections.extend(row_dets)
            
            # Dessiner les rectangles numérotés dans l'ordre
            for i, (x, y, w, h, _, conf) in enumerate(sorted_detections, 1):
                cv2.rectangle(result, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                # Ajouter le numéro et la confiance
                label = f"{i} ({conf:.2f})"
                cv2.putText(result, label, (int(x), int(y-5)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return result, [(x, y, w, h, s) for x, y, w, h, s, _ in sorted_detections]
        
        return result, []
    
    def non_max_suppression(self, boxes, overlap_thresh=0.3):
        """Applique la suppression non-maximum pour éliminer les détections redondantes."""
        if len(boxes) == 0:
            return []
        
        # Convertir en numpy array
        boxes = np.array(boxes, dtype="float")
        
        # Coordonnées des boîtes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        
        # Scores de confiance (pour privilégier les détections avec un score élevé)
        confidence = boxes[:, 5]
        
        # Calculer l'aire des boîtes
        area = (x2 - x1) * (y2 - y1)
        
        # Trier par confiance décroissante
        idxs = np.argsort(confidence)[::-1]
        
        # Liste pour stocker les indices des boîtes à conserver
        pick = []
        
        # Boucle tant qu'il reste des indices à vérifier
        while len(idxs) > 0:
            # Prendre l'indice de la boîte avec la confiance la plus élevée
            i = idxs[0]
            pick.append(i)
            
            # Trouver les intersections
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            # Calculer largeur et hauteur de l'intersection
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            # Calculer le ratio de chevauchement
            overlap = (w * h) / area[idxs[1:]]
            
            # Supprimer les indices avec chevauchement élevé
            idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
        
        # Retourner les boîtes sélectionnées
        return boxes[pick].astype("float").tolist()
    
    def display_image(self, image, factor=1.0):
        """Affiche une image dans le canvas avec le facteur de zoom spécifié."""
        h, w = image.shape[:2]
        
        # Calculer l'échelle de base pour ajuster à la fenêtre
        max_w, max_h = 1000, 800
        base_scale = min(max_w / w, max_h / h, 1.0)
        scale = base_scale * factor
        self.current_scale = scale
        
        # Redimensionner pour l'affichage
        img_disp = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # Conversion pour tkinter
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(image=img_pil)
        
        # Affichage dans le canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def display_image_with_detections(self):
        """Affiche l'image avec les détections dans le canvas."""
        if not hasattr(self, "detected_image") or self.detected_image is None:
            return
        
        # Si l'affichage des détections est désactivé
        if not self.show_detections_var.get():
            return
        
        # Trier les détections de gauche à droite et de haut en bas
        all_detections = []
        for template_name, detections in self.detection_positions.items():
            for i, (x, y, w, h, scale) in enumerate(detections):
                all_detections.append((x, y, w, h, template_name))
        
        # Tri plus intelligent - par rangées puis par colonnes
        if all_detections:
            # Calculer la hauteur moyenne
            avg_height = np.mean([h for _, _, _, h, _ in all_detections])
            row_tolerance = avg_height * 0.2
            
            # Grouper par rangées
            rows = {}
            for det in all_detections:
                x, y, w, h, tpl = det
                row_found = False
                for row_y in rows:
                    if abs(y - row_y) < row_tolerance:
                        rows[row_y].append(det)
                        row_found = True
                        break
                if not row_found:
                    rows[y] = [det]
            
            # Trier les rangées de haut en bas, puis les éléments de gauche à droite
            sorted_detections = []
            for _, row_dets in sorted(rows.items(), key=lambda r: r[0]):
                sorted_detections.extend(sorted(row_dets, key=lambda d: d[0]))
        else:
            sorted_detections = all_detections
        
        # Dessiner les rectangles et numéros
        for i, (x, y, w, h, _) in enumerate(sorted_detections, 1):
            # Calculer les coordonnées à l'échelle actuelle
            scaled_x = int(x * self.current_scale)
            scaled_y = int(y * self.current_scale)
            scaled_w = int(w * self.current_scale)
            scaled_h = int(h * self.current_scale)
            
            # Créer rectangle avec des tags pour manipulation future
            rect_id = self.canvas.create_rectangle(scaled_x, scaled_y, 
                                          scaled_x + scaled_w, scaled_y + scaled_h, 
                                          outline="red", width=2, tags="detection_rect")
            
            # Créer texte avec numéro, avec contour blanc pour meilleure visibilité
            # SANS le rectangle noir
            text_id = self.canvas.create_text(scaled_x + 15, scaled_y - 10, 
                                       text=str(i), fill="red", 
                                       font=("Arial", 12, "bold"), tags="detection_text")
    
    def toggle_detections_display(self):
        """Active ou désactive l'affichage des détections."""
        if hasattr(self, "detected_image") and self.detected_image is not None:
            if self.show_detections_var.get():
                self.display_image_with_detections()
            else:
                # Supprimer les rectangles et textes de détection
                self.canvas.delete("detection_rect", "detection_text")
    
    def update_ui(self, func):
        """Met à jour l'interface utilisateur de façon thread-safe."""
        if not self.winfo_exists():
            return
        self.after(0, func)


if __name__ == "__main__":
    app = PDFTemplateDetectorApp()
    app.mainloop()