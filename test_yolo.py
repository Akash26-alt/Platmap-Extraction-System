"""
test_yolo_on_pdf.py
-------------------
Tests a trained YOLO model on all pages of a PDF (or single image) and visualizes detections.

Usage:
    python test_yolo_on_pdf.py --model path/to/your/model.pt --pdf input.pdf
    python test_yolo_on_pdf.py --model path/to/your/model.pt --image page.jpg

Options:
    --conf     Confidence threshold (default 0.25)
    --save     Save annotated images to ./yolo_test_output/
    --no-show  Don't display images (useful for batch testing)
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from ultralytics import YOLO


def setup_output_dir(output_dir="yolo_test_output"):
    """Create output directory for saved images."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def draw_detections(image, detections, conf_threshold=0.25):
    """
    Draw bounding boxes and labels on image.
    Returns annotated image as numpy array.
    """
    img_copy = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        class_name = det["class_name"]
        class_id = det["class_id"]
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Label text
        label = f"{class_name} (id:{class_id}) {conf:.2f}"
        # Put label above box
        cv2.putText(img_copy, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img_copy


def run_yolo_on_image(model, image_path, conf_threshold=0.25, save_path=None, show=True):
    """Run YOLO on a single image file."""
    print(f"\n--- Testing on image: {image_path} ---")
    results = model(image_path, conf=conf_threshold)[0]
    
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_name = results.names[cls_id] if results.names else str(cls_id)
            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
    
    # Load image for drawing
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return detections
    
    annotated = draw_detections(img, detections, conf_threshold)
    
    print(f"Found {len(detections)} detection(s):")
    for d in detections:
        print(f"  - {d['class_name']} (id={d['class_id']}) conf={d['confidence']:.3f} bbox={d['bbox']}")
    
    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Saved annotated image to {save_path}")
    
    if show:
        cv2.imshow("YOLO Detections", annotated)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detections


def run_yolo_on_pdf(model, pdf_path, conf_threshold=0.25, output_dir=None, show=False):
    """Run YOLO on each page of a PDF."""
    print(f"\n=== Testing YOLO on PDF: {pdf_path} ===")
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")
    
    all_page_detections = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page as image (higher DPI for better detection)
        zoom = 150 / 72  # 150 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        # Convert to OpenCV format
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        print(f"\n--- Page {page_num+1} ---")
        results = model(img, conf=conf_threshold)[0]
        
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_name = results.names[cls_id] if results.names else str(cls_id)
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
        
        all_page_detections.append(detections)
        print(f"Detections: {len(detections)}")
        for d in detections:
            print(f"  - {d['class_name']} (id={d['class_id']}) conf={d['confidence']:.3f} bbox={d['bbox']}")
        
        # Save annotated page if output_dir provided
        if output_dir and detections:
            annotated = draw_detections(img, detections, conf_threshold)
            out_path = os.path.join(output_dir, f"page_{page_num+1}_annotated.png")
            cv2.imwrite(out_path, annotated)
            print(f"  Saved: {out_path}")
    
    doc.close()
    return all_page_detections


def main():
    parser = argparse.ArgumentParser(description="Test YOLO model on platmap images or PDF")
    parser.add_argument("--model", required=True, help="Path to YOLO model .pt file")
    parser.add_argument("--pdf", help="Path to PDF file")
    parser.add_argument("--image", help="Path to single image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    parser.add_argument("--save", action="store_true", help="Save annotated images to ./yolo_test_output/")
    parser.add_argument("--no-show", action="store_true", help="Don't display images (useful for batch)")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print("Model loaded. Class names:")
    if hasattr(model, "names") and model.names:
        for idx, name in model.names.items():
            print(f"  {idx}: '{name}'")
    else:
        print("  Could not retrieve class names from model.")
    
    output_dir = setup_output_dir() if args.save else None
    
    if args.pdf:
        run_yolo_on_pdf(model, args.pdf, args.conf, output_dir, show=not args.no_show)
    elif args.image:
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, os.path.basename(args.image).replace(".", "_annotated."))
        run_yolo_on_image(model, args.image, args.conf, save_path, show=not args.no_show)
    else:
        print("Error: Please provide either --pdf or --image")
        parser.print_help()


if __name__ == "__main__":
    main()