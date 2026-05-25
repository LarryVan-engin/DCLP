import argparse
import os
import glob
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import module_utils

def normalize_text(text):
    if text is None:
        return ""
    # Remove any whitespace and hyphens for comparison
    return text.replace(" ", "").replace("-", "").replace(".", "").upper()

def calculate_cer(pred, target):
    """
    Calculate Character Error Rate using Levenshtein distance.
    """
    if not target:
        return 1.0 if pred else 0.0
    import Levenshtein
    return Levenshtein.distance(pred, target) / len(target)

class LabelingApp:
    def __init__(self, root, img_paths, gt_path):
        self.root = root
        self.root.title("License Plate Labeling Tool")
        
        self.img_paths = img_paths
        self.gt_path = gt_path
        
        # Load existing GT to skip already labeled
        self.labeled_data = {}
        if os.path.exists(self.gt_path):
            with open(self.gt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",", 1)
                    if len(parts) == 2:
                        self.labeled_data[parts[0]] = parts[1]
                        
        self.pending_paths = [p for p in self.img_paths if os.path.basename(p) not in self.labeled_data]
        
        if not self.pending_paths:
            messagebox.showinfo("Done", "All images have been labeled!")
            self.root.quit()
            return
            
        self.current_idx = 0
        self.total_pending = len(self.pending_paths)
        
        # UI Elements
        self.lbl_info = tk.Label(root, text="", font=("Arial", 12))
        self.lbl_info.pack(pady=5)
        
        self.canvas = tk.Canvas(root, width=400, height=300, bg="gray")
        self.canvas.pack(pady=10)
        
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(root, textvariable=self.entry_var, font=("Arial", 24), justify="center")
        self.entry.pack(pady=10)
        self.entry.bind("<Return>", self.save_and_next)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        self.btn_next = tk.Button(btn_frame, text="Next (Enter)", command=self.save_and_next, font=("Arial", 12))
        self.btn_next.grid(row=0, column=0, padx=10)
        
        self.btn_skip = tk.Button(btn_frame, text="Skip", command=self.skip, font=("Arial", 12))
        self.btn_skip.grid(row=0, column=1, padx=10)
        
        self.btn_exit = tk.Button(root, text="Save & Exit", command=self.root.quit, font=("Arial", 12))
        self.btn_exit.pack(pady=5)
        
        self.load_image()
        
    def load_image(self):
        if self.current_idx >= self.total_pending:
            messagebox.showinfo("Done", "Finished labeling all images.")
            self.root.quit()
            return
            
        img_path = self.pending_paths[self.current_idx]
        self.lbl_info.config(text=f"Image {self.current_idx + 1} / {self.total_pending}: {os.path.basename(img_path)}")
        self.entry_var.set("")
        
        try:
            with Image.open(img_path) as img:
                # Resize for viewing
                img.thumbnail((400, 300))
                self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(200, 150, image=self.tk_img, anchor="center")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            self.skip()
            
        self.entry.focus()
        
    def save_and_next(self, event=None):
        label = self.entry_var.get().strip().upper()
        if not label:
            messagebox.showwarning("Warning", "Label cannot be empty! Use 'Skip' if you want to skip.")
            return
            
        img_path = self.pending_paths[self.current_idx]
        filename = os.path.basename(img_path)
        
        with open(self.gt_path, "a", encoding="utf-8") as f:
            f.write(f"{filename},{label}\n")
            
        self.current_idx += 1
        self.load_image()
        
    def skip(self):
        img_path = self.pending_paths[self.current_idx]
        try:
            os.remove(img_path)
            print(f"Deleted skipped image: {img_path}")
        except Exception as e:
            print(f"Failed to delete {img_path}: {e}")
            
        self.current_idx += 1
        self.load_image()

def run_labeling(img_dir, gt_path):
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    if not img_paths:
        print(f"No images found in {img_dir}")
        return
        
    root = tk.Tk()
    app = LabelingApp(root, sorted(img_paths), gt_path)
    root.mainloop()

def run_evaluation(img_dir, gt_path):
    if not os.path.exists(gt_path):
        print(f"Ground truth file not found: {gt_path}")
        return
        
    # Load YOLO model
    try:
        from ultralytics import YOLO
        plate_model = YOLO(r"D:\VSCode\DCLP\main_code\runs\detect\model_detect_license_plate.pt")
        print("Loaded plate detection model successfully.")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return

    # Read GT
    gt_data = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                gt_data[parts[0]] = parts[1]
                
    if not gt_data:
        print("Ground truth file is empty.")
        return
        
    total = 0
    correct = 0
    total_cer = 0.0
    
    error_log_path = "error_analysis.txt"
    with open(error_log_path, "w", encoding="utf-8") as err_log:
        err_log.write("Filename | Ground Truth | OCR Prediction | Match?\n")
        err_log.write("-" * 60 + "\n")
        
        for filename, gt_text in gt_data.items():
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                print(f"Warning: Image {filename} not found in {img_dir}, skipping.")
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            H, W = img.shape[:2]
            
            # Detect plate with YOLO
            results = plate_model(img, verbose=False)
            pred_text = ""
            ok = False
            
            best_det = None
            for r in results:
                for box in r.boxes:
                    if best_det is None or box.conf[0] > best_det.conf[0]:
                        best_det = box
            
            if best_det is not None:
                x1, y1, x2, y2 = best_det.xyxy[0].cpu().numpy()
                pred_text, ok = module_utils.read_license_plate_vn(
                    img, int(x1), int(y1), int(x2), int(y2), pad_ratio=0.18
                )
            else:
                pred_text, ok = module_utils.read_license_plate_vn(
                    img, 0, 0, W, H, pad_ratio=0.0
                )
            
            if pred_text is None:
                pred_text = ""
            
            norm_gt = normalize_text(gt_text)
            norm_pred = normalize_text(pred_text)
            
            is_match = (norm_gt == norm_pred)
            
            try:
                import Levenshtein
                cer = calculate_cer(norm_pred, norm_gt)
            except ImportError:
                # Fallback if Levenshtein is not installed
                cer = 0.0 if is_match else 1.0
                
            if is_match:
                correct += 1
            else:
                err_log.write(f"{filename} | {gt_text} | {pred_text} | False\n")
                
            total_cer += cer
            total += 1
            
            print(f"Processed {filename}: GT='{gt_text}' -> Pred='{pred_text}' | Match: {is_match}")
            
    if total > 0:
        accuracy = (correct / total) * 100
        avg_cer = (total_cer / total) * 100
        
        print("\n" + "=" * 40)
        print("EVALUATION RESULTS")
        print("=" * 40)
        print(f"Total evaluated  : {total}")
        print(f"Exact Match      : {correct}/{total} ({accuracy:.2f}%)")
        print(f"Average CER      : {avg_cer:.2f}%")
        print(f"Error log saved to: {os.path.abspath(error_log_path)}")
        print("=" * 40)
    else:
        print("No valid images found for evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR Accuracy")
    parser.add_argument("--mode", type=str, choices=["label", "test"], required=True, 
                        help="'label' to run labeling GUI, 'test' to run evaluation")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing cropped plate images")
    parser.add_argument("--gt", type=str, default="ground_truth.txt", help="Path to ground truth file")
    
    args = parser.parse_args()
    
    if args.mode == "label":
        run_labeling(args.img_dir, args.gt)
    elif args.mode == "test":
        run_evaluation(args.img_dir, args.gt)
