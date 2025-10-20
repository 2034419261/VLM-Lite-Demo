# 文本检索（query 要尽量是 CIFAR 类名或相似）
python -m src.demo.clip_retrieval --ckpt checkpoints/clip_like_small.pth --query "cat" --topk 5

# 从图片检索文本（image -> top-k labels）
python -m src.demo.clip_retrieval --ckpt checkpoints/clip_like_small.pth --image_path path/to/your_image.jpg --topk 5
