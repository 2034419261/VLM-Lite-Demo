# 使用模型预测的 top1 作为目标
python -m src.demo.clip_gradcam --ckpt checkpoints/clip_like_small.pth --image_path samples/cat2.png

# 指定目标文本（最好是 CIFAR 类名）
python -m src.demo.clip_gradcam --ckpt checkpoints/clip_like_small.pth --image_path samples/cat2.png --text "cat"

# 或者用 class idx
python -m src.demo.clip_gradcam --ckpt checkpoints/clip_like_small.pth --image_path samples/cat2.png --class_idx 3
