import argparse
import generate_frames
import cnn_face_detection
import visualize

def main():
    img_path = './training_data/'
    dlib_path = './mmod_human_face_detector.dat'
    detected_head_shot = 'img_head_shots/'
    upsample_val = 1
    
    generate_frames.create_frames()
    cnn_face_detection.detect_heads(img_path,dlib_path,upsample_val)
    visualize.detect_emotions(detected_head_shot)

if __name__ == "__main__":
    main()