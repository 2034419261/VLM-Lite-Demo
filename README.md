# VLM-Lite-Demo
````markdown
# 🧠 VLM-Lite: A Minimal Vision–Language Model Playground

> A lightweight, educational, and fully runnable Vision–Language Model (VLM) framework —  
> featuring a **CLIP-style contrastive learner**, integrated **BLIP image captioning**,  
> plus visualization and demo tools: **retrieval**, **GradCAM**, and a captioned-video demo.  
> Designed for research, teaching, and fast prototyping without heavy data or compute needs.

---

## 🌍 Overview

**VLM-Lite** is a compact, modular playground for learning and prototyping vision-language ideas.  
It demonstrates core concepts in modern VLMs (CLIP, BLIP, LISA) with emphasis on:

- clarity and modularity (clean `src/` structure),
- low resource requirements (uses TorchVision CIFAR-10 by default),
- immediate visual demos (captioned video, retrieval grids, GradCAM heatmaps).

Key built-in capabilities:
- 🔹 CLIP-style contrastive training on CIFAR-10 (`src/train.py`).
- 🔹 BLIP integration for image captioning and a captioned-video demo (`src/integrations/blip_caption.py`, `src/demo/create_caption_video.py`).
- 🔹 **CLIP Retrieval**: text→image and image→text retrieval demo (`src/demo/clip_retrieval.py`).
- 🔹 **CLIP GradCAM**: attention/importance heatmaps for CLIP-like models (`src/demo/clip_gradcam.py`).
- 🔹 Optional: BLIP VQA / Gradio UI integration for interactive demos (examples and hooks included).

This repo is ideal for:
- students & researchers experimenting with VLM concepts,
- devs wanting a reproducible, educational PyTorch example,
- presenters preparing interactive demos for talks or tutorials.

---

## 🧩 Features (summary)

| Module | Description | Key tech |
|---|---:|---|
| `src/models/clip_like.py` | Minimal CLIP-like encoder (image + simple text embedding) | ResNet (timm), projection heads |
| `src/data/cifar_text.py` | CIFAR-10 wrapper that produces image–text pairs (class names) | torchvision |
| `src/losses/contrastive.py` | InfoNCE-style symmetric contrastive loss | PyTorch |
| `src/integrations/blip_caption.py` | BLIP wrapper (Hugging Face) with local/offline fallback | transformers |
| `src/demo/create_caption_video.py` | Generate an mp4 with BLIP captions over frames | PIL, OpenCV |
| `src/demo/clip_retrieval.py` | Text→Image / Image→Text retrieval demo, saves retrieval grid | NumPy, PIL |
| `src/demo/clip_gradcam.py` | Grad-CAM style heatmaps for CLIP-like models | PyTorch, OpenCV |
| `src/train.py` | End-to-end CLIP-like training on CIFAR-10 | AdamW, timm |

---

## ⚙️ Installation

```bash
# Clone
git clone https://github.com/yourusername/VLM-Lite.git
cd VLM-Lite

# (Optional) virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
````

### Core Dependencies

* `torch`, `torchvision`, `timm`
* `transformers`, `huggingface_hub` (for BLIP), `Pillow`, `opencv-python`
* `numpy`, `tqdm`, `scikit-learn`, `matplotlib` (for visualization)

> Optional: `open_clip_torch` or `sentence-transformers` if you want more powerful text encoders or pretrained CLIP variants (see **Extending**).

---

## 🚀 Quick Start

### 1) Train a CLIP-like model on CIFAR-10

```bash
# recommended: run from repository root
python -m src.train
```

Short run (2 epochs) example output:

```
Epoch 0: loss=3.33, zero-shot acc=0.8806
Epoch 1: loss=2.90, zero-shot acc=0.9115
```

Trained checkpoint is saved to `checkpoints/` (e.g. `checkpoints/clip_like_small.pth`).

---

### 2) Generate BLIP caption video (demo)

* If you have internet/ HF access (auto-download):

```bash
python -m src.demo.create_caption_video
```

* If offline but you previously downloaded BLIP to `./models/blip-base`:

```bash
python -m src.demo.create_caption_video --local_model ./models/blip-base
```

* Offline fallback (use CIFAR labels as captions; no HF download):

```bash
python -m src.demo.create_caption_video --offline
```

Output:

```
Saved video to output/blip_demo.mp4
```

Open `output/blip_demo.mp4` — each frame shows a CIFAR image with a caption.

---

## 🔎 Retrieval Demo (text → image) — Usage

Text query example (prefer CIFAR class names for best result):

```bash
python -m src.demo.clip_retrieval --ckpt checkpoints/clip_like_small.pth --query "cat" --topk 5
```

Image→text example:

```bash
python -m src.demo.clip_retrieval --ckpt checkpoints/clip_like_small.pth --image_path samples/my_image.jpg --topk 5
```

Outputs:

* Grid image saved to `output/retrieval/retrieval_<query>.jpg`.
* Console prints top-k indices, labels and similarity scores.

Notes:

* Retrieval uses model's text encoder for class labels. For arbitrary free-text queries, swap in a more powerful text encoder (see **Extending**).

---

## 🔥 GradCAM Demo (visualize CLIP attention)

Generate Grad-CAM overlay for an input image:

```bash
python -m src.demo.clip_gradcam --ckpt checkpoints/clip_like_small.pth --image_path samples/cat.png --text "cat"
```

* If `--text` omitted, the script will automatically use model's predicted top-1 class as the GradCAM target.
* Output saved to `output/gradcam/<image>_gradcam.jpg` (or custom `--out`).

Usage patterns:

* Inspect where the model looks for evidence of a class.
* Compare GradCAMs for different target texts.

---

## 🧠 Analysis & Probing

* TSNE / embedding visualization: `src/analysis/visualize_embeddings.py`
  Example:

  ```bash
  python -m src.analysis.visualize_embeddings --ckpt checkpoints/clip_like_small.pth --out output/embeddings_tsne.png
  ```

* Linear probe evaluation (freeze encoder, train small classifier):

  ```bash
  python -m src.linear_probe --ckpt checkpoints/clip_like_small.pth --epochs 5
  ```

---

## ⚖️ BLIP / Offline / Proxy Notes

* BLIP (Hugging Face) will download weights on first run. If your machine cannot reach Hugging Face:

  * Download model on another machine and copy `./models/blip-base` to this repo root. Then run with `--local_model ./models/blip-base`.
  * Or use `--offline` to use CIFAR class labels as captions for demo purposes.

* If your environment needs a proxy:

```bash
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
python -m src.demo.create_caption_video
```

---

## 🧩 Project Structure

```
VLM-Lite/
├── src/
│   ├── data/                 # dataset wrappers (CIFAR text)
│   ├── models/               # clip_like.py
│   ├── losses/               # contrastive loss
│   ├── integrations/         # blip wrapper with offline fallback
│   ├── demo/                 # create_caption_video.py, clip_retrieval.py, clip_gradcam.py
│   ├── analysis/             # t-SNE, probes
│   ├── train.py
│   └── utils.py
├── output/                   # auto-generated visuals & videos
├── models/                   # optional: local HF models (e.g. blip-base/)
├── checkpoints/              # saved model checkpoints
├── requirements.txt
└── README.md
```

---

## 🔍 Extending the Project

You can extend VLM-Lite easily:

* **Better text encoder / free-text queries**: integrate `open_clip_torch` or `sentence-transformers` (swap `SimpleTextEncoder`).
* **Use pretrained CLIP**: replace local `clip_like` with an official pre-trained CLIP (OpenAI / open_clip) to improve retrieval across arbitrary sentences.
* **BLIP-2 / VQA / Instruction-following**: add `Salesforce/blip2-flan-t5` or InstructBLIP for visual reasoning and VQA; integrate a small LLM for multi-turn dialogue.
* **Interactive UI**: build a Gradio app `src/app/vlm_gradio.py` that shows caption, GradCAM, and retrieval outputs from an uploaded image.
* **LISA / SegPoint**: add reasoning-driven segmentation (LLM→segmentation) and 3D point-cloud modules (requires separate datasets & compute).

---

## 📦 Output Examples

| Task               |                    Output | Note                             |
| ------------------ | ------------------------: | -------------------------------- |
| CLIP training      | console logs & checkpoint | small quick-run by default       |
| BLIP caption video |    `output/blip_demo.mp4` | 16 CIFAR frames + captions       |
| Retrieval          |  `output/retrieval/*.jpg` | top-k grid per query             |
| GradCAM            |    `output/gradcam/*.jpg` | heatmap overlay for given target |

---

## 🧑‍💻 Credits & References

This project draws inspiration from:

* **CLIP (OpenAI)** — [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
* **open_clip** — [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
* **BLIP (Salesforce)** — [https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP)
* **LISA (reasoning-seg)** — [https://github.com/dvlab-research/LISA](https://github.com/dvlab-research/LISA)
* **SegPoint (3D)** — [https://github.com/SegPoint](https://github.com/SegPoint)
* **timm** — [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

---

## 📜 License

MIT License. Note: third-party models (BLIP, CLIP variants) are subject to their original licenses.

---

## ❤️ Acknowledgements

Created by **Yu Zhang** as a compact educational and prototyping toolkit for multimodal research and demonstrations.
If this repo helps you, please ⭐ the project and share your demos!

```

中文版：

````markdown
# 🧠 VLM-Lite：轻量级视觉语言模型实验平台  

> 一个面向学习、研究与演示的轻量级 Vision–Language Model (VLM) 框架。  
> 它集成了 **CLIP 式图文对比学习**、**BLIP 图像描述生成**、  
> 以及 **图文检索** 和 **GradCAM 可视化** 等多模态功能。  
> 无需大规模数据或计算资源，即可体验现代 VLM 的核心原理与应用。

---

## 🌍 项目概述  

**VLM-Lite** 是一个小巧但完整的多模态学习框架，  
用最简代码复现了现代视觉语言模型（如 **CLIP**、**BLIP**、**LISA**）的核心思想，  
强调 **模块化、可读性强、资源占用低**。  

主要特性：
- 🔹 CLIP 式图文对齐模型，支持在 CIFAR-10 上快速训练。  
- 🔹 集成 **BLIP** 图像字幕生成模块，可生成带字幕视频。  
- 🔹 **CLIP 图文检索 Demo**：实现 text→image 和 image→text 检索。  
- 🔹 **CLIP GradCAM 可视化**：展示模型关注的图像区域。  
- 🔹 所有示例均基于 PyTorch 与 TorchVision 内置数据集，无需额外预处理。  

非常适合：
- 🎓 学习多模态对齐与表征学习原理的学生 / 研究者；  
- ⚙️ 想快速构建可视化 Demo 的开发者；  
- 💡 用于课堂教学、科研演示与论文补充实验。  

---

## 🧩 功能概览  

| 模块 | 功能说明 | 主要技术 |
|------|-----------|-----------|
| `src/models/clip_like.py` | 简化版 CLIP 模型（图像编码 + 文本嵌入） | ResNet (timm), 投影层 |
| `src/data/cifar_text.py` | CIFAR-10 图文配对封装（自动生成文本标签） | torchvision |
| `src/losses/contrastive.py` | InfoNCE 对比损失 | PyTorch |
| `src/integrations/blip_caption.py` | BLIP 图像字幕生成（含离线/镜像模式） | transformers |
| `src/demo/create_caption_video.py` | 生成带字幕视频 | PIL, OpenCV |
| `src/demo/clip_retrieval.py` | 图文检索（Text→Image / Image→Text） | NumPy, PIL |
| `src/demo/clip_gradcam.py` | GradCAM 热力图可视化 | PyTorch, OpenCV |
| `src/train.py` | CLIP 式对比训练 | AdamW, timm |

---

## ⚙️ 安装与环境配置  

```bash
# 克隆仓库
git clone https://github.com/yourusername/VLM-Lite.git
cd VLM-Lite

# （可选）创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
````

### 主要依赖

* `torch`, `torchvision`, `timm`
* `transformers`, `huggingface_hub`, `Pillow`, `opencv-python`
* `numpy`, `tqdm`, `matplotlib`, `scikit-learn`

> 可选：若要使用更强的预训练文本编码器，可安装 `open_clip_torch` 或 `sentence-transformers`。

---

## 🚀 快速上手

### 1️⃣ 在 CIFAR-10 上训练一个 CLIP 式模型

```bash
python -m src.train
```

示例输出：

```
Epoch 0: loss=3.33, zero-shot acc=0.8806
Epoch 1: loss=2.90, zero-shot acc=0.9115
```

训练后的权重将保存在 `checkpoints/clip_like_small.pth`。

---

### 2️⃣ 运行 BLIP 图像字幕视频 Demo

* 若网络可访问 Hugging Face：

```bash
python -m src.demo.create_caption_video
```

* 若已在本地下载过 BLIP 模型（例如放在 `./models/blip-base`）：

```bash
python -m src.demo.create_caption_video --local_model ./models/blip-base
```

* 若完全离线（使用 CIFAR 类别名称代替字幕）：

```bash
python -m src.demo.create_caption_video --offline
```

输出：

```
Saved video to output/blip_demo.mp4
```

打开 `output/blip_demo.mp4`，即可看到每张 CIFAR 图片配有 BLIP 自动生成的字幕。

---

## 🔍 CLIP 图文检索 Demo

**文本检索图片（Text → Image）**

```bash
python -m src.demo.clip_retrieval --ckpt checkpoints/clip_like_small.pth --query "cat" --topk 5
```

**图片检索文字（Image → Text）**

```bash
python -m src.demo.clip_retrieval --ckpt checkpoints/clip_like_small.pth --image_path samples/my_image.jpg --topk 5
```

输出：

* 检索结果可视化图保存在 `output/retrieval/retrieval_<query>.jpg`；
* 终端打印 top-k 结果的标签与相似度分数。

> 提示：若使用自定义文本查询，请确保模型的文本编码器足够泛化（或替换为 BERT/CLIP 编码器）。

---

## 🔥 CLIP GradCAM 可视化

生成热力图叠加效果：

```bash
python -m src.demo.clip_gradcam --ckpt checkpoints/clip_like_small.pth --image_path samples/cat.png --text "cat"
```

* 若省略 `--text`，脚本会自动选择模型预测的 top-1 类别；
* 输出图片保存在 `output/gradcam/<image>_gradcam.jpg`。

用途：

* 可视化模型关注区域；
* 对比不同文本目标下的注意力差异。

---

## 🧠 分析与扩展实验

* **特征可视化 (t-SNE)**：

```bash
python -m src.analysis.visualize_embeddings --ckpt checkpoints/clip_like_small.pth --out output/embeddings_tsne.png
```

* **线性探针 (Linear Probe)**：

```bash
python -m src.linear_probe --ckpt checkpoints/clip_like_small.pth --epochs 5
```

---

## ⚖️ BLIP / 离线 / 代理说明

* 若无法访问 Hugging Face，可手动下载 `Salesforce/blip-image-captioning-base` 并放入 `./models/blip-base`：

  ```bash
  python -m src.demo.create_caption_video --local_model ./models/blip-base
  ```
* 完全离线模式：

  ```bash
  python -m src.demo.create_caption_video --offline
  ```
* 若需要代理：

  ```bash
  export HTTP_PROXY="http://127.0.0.1:7890"
  export HTTPS_PROXY="http://127.0.0.1:7890"
  ```

---

## 🧰 项目结构

```
VLM-Lite/
├── src/
│   ├── data/                 # CIFAR-10 数据集封装
│   ├── models/               # clip_like 模型
│   ├── losses/               # 对比损失函数
│   ├── integrations/         # BLIP 接口
│   ├── demo/                 # create_caption_video / clip_retrieval / clip_gradcam
│   ├── analysis/             # 可视化与探针脚本
│   ├── train.py              # 训练脚本
│   └── utils.py
├── checkpoints/              # 训练模型保存目录
├── models/                   # 本地下载的 BLIP 模型
├── output/                   # 视频、热图、检索结果等输出
├── requirements.txt
└── README.md
```

---

## 📦 输出示例

| 任务      | 输出                       | 说明                  |
| ------- | ------------------------ | ------------------- |
| CLIP 训练 | 控制台日志 / checkpoints      | 快速 2 轮训练即可运行        |
| BLIP 视频 | `output/blip_demo.mp4`   | 每帧为 CIFAR 图像 + 生成字幕 |
| 图文检索    | `output/retrieval/*.jpg` | 检索结果可视化             |
| GradCAM | `output/gradcam/*.jpg`   | CLIP 注意力热力图         |

---

## 🔍 扩展方向

VLM-Lite 可以轻松扩展：

| 方向      | 示例                          | 参考项目                      |
| ------- | --------------------------- | ------------------------- |
| 更强文本编码器 | 替换为 BERT、GPT-2 或 CLIP 官方文本塔 | Hugging Face Transformers |
| 更高层生成模型 | 集成 BLIP-2 / CoCa，支持视觉推理与问答  | Salesforce/BLIP-2         |
| 交互式应用   | 通过 Gradio 创建在线可视化界面         | Gradio                    |
| 推理型分割   | 接入 LISA 框架，实现文字驱动分割         | dvlab-research/LISA       |
| 三维推理    | 扩展为 SegPoint 点云理解           | SegPoint Project          |

---

## 🧑‍💻 致谢与引用

本项目参考并学习自以下优秀开源工作：

* **CLIP (OpenAI)** — [OpenAI/CLIP](https://github.com/openai/CLIP)
* **open_clip** — [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
* **BLIP** — [Salesforce/BLIP](https://github.com/salesforce/BLIP)
* **LISA** — [dvlab-research/LISA](https://github.com/dvlab-research/LISA)
* **SegPoint** — [SegPoint Project](https://github.com/SegPoint)
* **timm** — [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

---

## 📜 开源协议

本项目基于 **MIT License** 开源。
BLIP 等模型需遵循其原始开源协议（Salesforce / Hugging Face）。

---

## ❤️ 作者说明

本项目由 **Yu Zhang** 创建，
作为 CVPR 研究原型与教学示例，
旨在帮助科研与开发人员快速理解视觉语言模型的核心机制与多模态应用。

> 如果你觉得这个项目有帮助，请点个 ⭐ 支持一下吧！

```

