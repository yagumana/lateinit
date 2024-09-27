import cv2
import os

# 保存された画像のフォルダ
image_folder = '/workspace/images/fashion_mnist'

# 出力する動画ファイルのパスと名前
video_file = '/workspace/images/fashion_mnist/output_video.mp4'

# フレームレート (1秒間に表示する画像の数)
fps = 2

# 保存された画像ファイルのリストを取得し、ソートして順序を揃える
images = [img for img in os.listdir(image_folder) if img.startswith("decoded_images_") and img.endswith(".png")]
images.sort()  # ファイル名をアルファベット順にソート（Late_timeの順に表示されるように）

# 最初の画像のファイルパスを使って、動画のフレームサイズを決定
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# 動画ファイルの設定 (FourCCコードを指定してビデオフォーマットを定義)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' はMP4フォーマット用のコーデック
video = cv2.VideoWriter(video_file, fourcc, fps, (width, height))

# 画像を順に動画に追加
for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    video.write(frame)  # 画像をフレームとして動画に追加

# 動画ファイルを閉じて保存
video.release()
# cv2.destroyAllWindows()

print(f"Video saved as {video_file}")
