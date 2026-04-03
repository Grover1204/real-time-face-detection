import os
import glob
from utils.image_utils import load_image, save_image, crop_image
from detectors.face_detector import FaceDetector
from alignment.face_alignment import FaceAligner

def run_stage2_tests():
    test_dir = "/Users/grover/Documents/3d model/face_3d_reconstruction/test case images"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all jpg/jpeg files
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.jpeg"))
    image_paths = sorted(image_paths)
    
    total_images = len(image_paths)
    successful_alignments = 0
    failed_alignments = 0
    total_rotation_angle = 0.0
    
    detector = FaceDetector()
    aligner = FaceAligner()
    
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
        print(f"Original Resolution: {w}x{h}")
        
        # Stage 1: Detect and Crop
        result = detector.detect(image_rgb, return_metrics=True)
        bbox, _ = result
        
        if bbox is None:
            print(f"No face detected in image: {img_name} (Failed Stage 1)")
            failed_alignments += 1
            print("-" * 40)
            continue
            
        cropped_face = crop_image(image_rgb, bbox)
        ch, cw, _ = cropped_face.shape
        print(f"Cropped resolution: {cw}x{ch}")
        
        cropped_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
        save_image(cropped_face, cropped_path)
        
        # Stage 2: Align
        aligned_face, debug_landmarks, metrics = aligner.align(cropped_face, return_metrics=True)
        
        if debug_landmarks is None:
            print(f"Landmarks not detected for alignment in {img_name}")
            failed_alignments += 1
            print("-" * 40)
            continue
            
        successful_alignments += 1
        
        # Metrics
        print(f"Left eye: {metrics.get('left_eye')}")
        print(f"Right eye: {metrics.get('right_eye')}")
        rot_angle = metrics.get('rotation_angle', 0.0)
        print(f"Rotation angle: {rot_angle:.2f} degrees")
        
        total_rotation_angle += abs(rot_angle)
        
        ah, aw, _ = aligned_face.shape
        print(f"Aligned image resolution: {aw}x{ah}")
        
        aligned_path = os.path.join(output_dir, f"{base_name}_aligned.jpg")
        save_image(aligned_face, aligned_path)
        print(f"Saved aligned image: {aligned_path}")
        
        landmarks_path = os.path.join(output_dir, f"{base_name}_landmarks.jpg")
        save_image(debug_landmarks, landmarks_path)
        
        print("-" * 40)

    detector.close()
    aligner.close()
    
    # Generate Summary Report
    avg_rotation_angle = total_rotation_angle / successful_alignments if successful_alignments > 0 else 0.0
    report_path = "stage2_test_report.txt"
    with open(report_path, "w") as f:
        f.write("Stage 2 Alignment Test Summary\n\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Successful Alignments: {successful_alignments}\n")
        f.write(f"Failed Alignments: {failed_alignments}\n")
        f.write(f"Average Rotation Angle: {avg_rotation_angle:.2f}°\n")
        
    print(f"\nSaved test report to {report_path}")

if __name__ == "__main__":
    run_stage2_tests()
