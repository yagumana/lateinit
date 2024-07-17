import os
import cv2
from pdf2image import convert_from_path

import os
from PIL import Image


def convert_images_to_pdfs(image_dir, output_dir):
    # Get list of PNG files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    if not image_files:
        print("No PNG files found in the directory.")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for image_file in image_files:
        # Open the image file
        img_path = os.path.join(image_dir, image_file)
        img = Image.open(img_path)
        
        # Convert image to RGB (if necessary)
        if img.mode in ('RGBA', 'LA'):
            img = img.convert('RGB')
        
        # Save as PDF
        pdf_path = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}.pdf')
        img.save(pdf_path, 'PDF', resolution=100.0)
        print(f'Saved {pdf_path}')

# Example usage
# image_dir = 'path/to/your/image/directory'
# output_dir = 'path/to/your/output/directory'
# convert_images_to_pdfs(image_dir, output_dir)




def convert_pdf_to_images(pdf_dir, image_dir):
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
    if not pdf_files:
        print("No PDF files found in the directory.")
    for i, pdf_file in enumerate(pdf_files):
        images = convert_from_path(os.path.join(pdf_dir, pdf_file))
        for j, image in enumerate(images):
            image.save(os.path.join(image_dir, f'image_{i*10+j:05d}.png'), 'PNG')

def create_video_from_images(image_dir, output_video_path, fps=30):
    images = sorted([img for img in os.listdir(image_dir) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for i, image in enumerate(images):
        frame = cv2.imread(os.path.join(image_dir, image))
        video.write(frame)

        if i >= 750:
            # 追加のフレームを挿入して表示時間を延ばす
            video.write(frame)
        if i >= 800:
            video.write(frame)
        if i >= 970:
            video.write(frame)
            video.write(frame)
        if i==1000:
            for j in range(50):
                video.write(frame)
        

    # cv2.destroyAllWindows()
    video.release()

def main():
    pdf_2d_jacobian_dir = '/workspace/data/2d_jacobian_squared_ver3'
    # image_2d_jacobian_dir = '/workspace/data/2d_jacobian_squared_ver3/images'
    image_2d_jacobian_dir = '/workspace/data/2d_jacobian_squared_re'
    # output_2d_jacobian_video_path = '/workspace/data/2d_jacobian_squared_ver3/videos/output_video.mp4'
    output_2d_jacobian_video_path = '/workspace/data/2d_jacobian_squared_re/videos/output_video.mp4'

    pdf_2d_vfield_dir = '/workspace/data/2d_vfield_ver3'
    image_2d_vfield_dir = '/workspace/data/2d_vfield_ver3/images'
    output_2d_vfield_video_path = '/workspace/data/2d_vfield_ver3/videos/output_video.mp4'

    image_3d_vfield_dir = '/workspace/data/3d_vfield_16'
    output_3d_vfield_pdf_dir = '/workspace/data/3d_vfield_16/pdf'
    output_3d_vfield_video_path = '/workspace/data/3d_vfield_16/videos/3d_vfield_16.mp4'

    image_3d_jacobi_sq_dir = '/workspace/data/3d_jacobi_sq_128'
    output_3d_jacobi_sq_pdf_dir = '/workspace/data/3d_jacobi_sq_128/pdf'
    output_3d_jacobi_sq_video_path = '/workspace/data/3d_jacobi_sq_128/videos/3d_jacobi_sq_128.mp4'

    image_ellipse_vfield_dir = '/workspace/data/ellipse21_vfield/png'
    output_ellipse_video_path = '/workspace/data/ellipse21_vfield/videos/ellipse21_vfield.mp4'
    
    # 2d jacobian, pdf -> video
    # if not os.path.exists(image_2d_jacobian_dir):
    #     os.makedirs(image_2d_jacobian_dir)

    # print("Converting PDF files to images...")
    # convert_pdf_to_images(pdf_2d_jacobian_dir, image_2d_jacobian_dir)

    # print("Creating video from images...")
    # create_video_from_images(image_2d_jacobian_dir, output_2d_jacobian_video_path)
    # print(f"Video saved to {output_2d_jacobian_video_path}")


    # 2d score vector field, pdf -> video
    # if not os.path.exists(image_2d_vfield_dir):
    #     os.makedirs(image_2d_vfield_dir)
    
    # print("Converting PDF files to images...")
    # convert_pdf_to_images(pdf_2d_vfield_dir, image_2d_vfield_dir)

    # print("Creating video from images...")
    # create_video_from_images(image_2d_vfield_dir, output_2d_vfield_video_path)
    # print(f"Video saved to {output_2d_vfield_video_path}")


    #3d score vector field, png -> pdf
    # if not os.path.exists(output_3d_vfield_pdf_dir):
    #     os.makedirs(output_3d_vfield_pdf_dir)
    # convert_images_to_pdfs(image_3d_vfield_dir, output_3d_vfield_pdf_dir)

    # 3d score vector field, png -> video

    # if not os.path.exists(image_2d_vfield_dir):
    #     os.makedirs(image_2d_vfield_dir)
    
    # print("Converting PDF files to images...")
    # convert_pdf_to_images(pdf_2d_vfield_dir, image_2d_vfield_dir)

    # print("Creating video from images...")
    # create_video_from_images(image_3d_vfield_dir, output_3d_vfield_video_path)
    # print(f"Video saved to {output_3d_vfield_video_path}")


    # #3d jacobi_sq, png -> pdf
    # if not os.path.exists(output_3d_jacobi_sq_pdf_dir):
    #     os.makedirs(output_3d_jacobi_sq_pdf_dir)
    # convert_images_to_pdfs(image_3d_jacobi_sq_dir, output_3d_jacobi_sq_pdf_dir)

    # 3d jacobi_sq, png -> video

    # print("Creating video from images...")
    # create_video_from_images(image_3d_jacobi_sq_dir, output_3d_jacobi_sq_video_path)
    # print(f"Video saved to {output_3d_jacobi_sq_video_path}")

    # ellipse png -> video
    print("Creating video from images...")
    create_video_from_images(image_ellipse_vfield_dir, output_ellipse_video_path)
    print(f"Video saved to {output_ellipse_video_path}")


    
    


if __name__ == "__main__":
    main()
