import os
import glob
from utils.image_utils import load_image, save_image, crop_image, draw_bbox
from detectors.face_detector import FaceDetector

def run_tests():
    test_dir = "/Users/grover/Documents/3d model/face_3d_reconstruction/test case images"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all jpg/jpeg files
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.jpeg"))
    image_paths = sorted(image_paths)
    
    total_images = len(image_paths)
    successful_detections = 0
    failed_detections = 0
    multiple_faces = 0
    
    detector = FaceDetector()
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # Load
        try:
            image_rgb = load_image(img_path)
        except Exception as e:
            print(f"Failed to load {img_name}: {e}")
            continue
            
        h, w, _ = image_rgb.shape
        print(f"Processing: {img_name}")
        print(f"Resolution: {w}x{h}")
        
        # Detect
        result = detector.detect(image_rgb, return_metrics=True)
        bbox, metrics = result
        
        if bbox is None:
            print(f"No face detected in image: {img_name}")
            failed_detections += 1
            print("-" * 40)
            continue
            
        successful_detections += 1
        
        num_faces = metrics.get('num_faces', 0)
        if num_faces > 1:
            multiple_faces += 1
            
        print(f"Faces detected: {num_faces}")
        print(f"Selected face: {metrics.get('selected_index')}")
        print(f"Bounding box: (x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})")
        print(f"Detection confidence: {metrics.get('confidence'):.2f}")
        
        # Crop & Save margin cropped face
        cropped_face = crop_image(image_rgb, bbox)
        cropped_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
        save_image(cropped_face, cropped_path)
        print(f"Saved cropped image: {cropped_path}")
        
        # Draw bbox & Save debug image
        debug_image = draw_bbox(image_rgb, bbox)
        debug_path = os.path.join(output_dir, f"{base_name}_bbox.jpg")
        save_image(debug_image, debug_path)
        
        print("-" * 40)

    detector.close()
    
    # Generate Summary Report
    report_path = "stage1_test_report.txt"
    with open(report_path, "w") as f:
        f.write("Stage 1 Test Summary\n\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Successful Face Detections: {successful_detections}\n")
        f.write(f"Failed Detections: {failed_detections}\n")
        f.write(f"Multiple Face Cases: {multiple_faces}\n")
        
    print(f"\nSaved test report to {report_path}")

if __name__ == "__main__":
    run_tests()
