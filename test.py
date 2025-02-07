import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# 可视化工具函数
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# 核心移动函数
def move_player_seamless(image, mask, original_box, min_x=220, min_y=300):
    # 获取精确的球员区域
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # 提取球员图像（带透明度通道）
    player_roi = image[y_min:y_max+1, x_min:x_max+1].copy()
    alpha_channel = (mask[y_min:y_max+1, x_min:x_max+1] * 255).astype(np.uint8)
    player_roi = cv2.cvtColor(player_roi, cv2.COLOR_RGB2RGBA)
    player_roi[:, :, 3] = alpha_channel

    # 创建修复掩模
    inpaint_mask = (mask * 255).astype(np.uint8)
    
    # 修复原始位置
    repaired_bg = cv2.inpaint(image, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
    
    # 生成新位置（确保在指定区域内）
    h, w = image.shape[:2]
    obj_h, obj_w = y_max - y_min + 1, x_max - x_min + 1
    new_x = np.random.randint(min_x, w - obj_w - 1)
    new_y = np.random.randint(min_y, h - obj_h - 1)

    # 泊松融合
    center = (new_x + obj_w//2, new_y + obj_h//2)
    output = cv2.seamlessClone(
        player_roi[:, :, :3],  # 原始RGB图像
        repaired_bg,           # 修复后的背景
        alpha_channel,         # 透明度作为掩模
        center,
        cv2.MIXED_CLONE
    )

    return output, (new_x, new_y, new_x+obj_w, new_y+obj_h)

# 主程序
if __name__ == "__main__":
    # 加载图像
    image = cv2.cvtColor(cv2.imread('Dataset/Corner-kick/ck_image.jpg'), cv2.COLOR_BGR2RGB)
    
    # 初始化SAM模型
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # 同时使用点和框提示
    input_point = np.array([[415, 420]])  # 球员中心点
    input_label = np.array([1])
    input_box = np.array([380, 380, 440, 480])  # 大致边界框

    # 生成预测
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box[None, :],
        multimask_output=False,
    )

    # 获取最佳掩码
    best_mask = masks[0].astype(np.uint8)

    # 移动球员
    final_image, new_box = move_player_seamless(image, best_mask, input_box)

    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 原始检测
    plt.subplot(231)
    plt.title("Original Image")
    plt.imshow(image)
    show_box(input_box, plt.gca())
    show_points(input_point, input_label, plt.gca())
    
    # 分割掩码
    plt.subplot(232)
    plt.title("Segmentation Mask")
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_box(input_box, plt.gca())
    
    # 修复后的背景
    plt.subplot(233)
    plt.title("Repaired Background")
    repaired_bg = cv2.inpaint(image, best_mask*255, 5, cv2.INPAINT_NS)
    plt.imshow(repaired_bg)
    
    # 最终结果
    plt.subplot(212)
    plt.title("Final Result with Player Moved")
    plt.imshow(final_image)
    show_box(new_box, plt.gca())
    plt.tight_layout()
    plt.show()