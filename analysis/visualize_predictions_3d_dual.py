import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import random

def main():
    # 1. 定义相关的 3D NIfTI 路径
    raw_images_dir = "/mnt/synology/ruihao.li/nnUNet_raw/Dataset004_CTPET/imagesTr"
    raw_labels_dir = "/mnt/synology/ruihao.li/nnUNet_raw/Dataset004_CTPET/labelsTr"
    predict_labels_dir = "/mnt/synology/ruihao.li/nnUNet_results/Dataset004_CTPET/nnUNetTrainer_500epochs__nnUNetPlans__2d/fold_0/validation"
    
    # 自动创建专门存放可视化结果的新文件夹，保持根目录整洁
    output_dir = "/mnt/synology/ruihao.li/ctpet_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 自动从 validation 目录下获取所有验证集病例
    valid_cases = [f.replace(".nii.gz", "") for f in os.listdir(predict_labels_dir) if f.endswith(".nii.gz")]
    if len(valid_cases) == 0:
        print("找不到任何 3D NIfTI 预测结果。")
        return

    # 🌟 一次性批量生成的图片数量（默认随机抽 10 张）
    num_samples = 10  
    print(f"验证集共有 {len(valid_cases)} 个病例。开始随机抽取 {num_samples} 个进行一键批量可视化...\n")

    # 从验证集中随机不重复地抽样
    sampled_cases = random.sample(valid_cases, min(num_samples, len(valid_cases)))

    for idx, case_id in enumerate(sampled_cases):
        ct_path = os.path.join(raw_images_dir, f"{case_id}_0000.nii.gz")  # 通道 0: CT
        pet_path = os.path.join(raw_images_dir, f"{case_id}_0001.nii.gz") # 通道 1: PET
        lbl_path = os.path.join(raw_labels_dir, f"{case_id}.nii.gz")       # 真值标签
        pred_path = os.path.join(predict_labels_dir, f"{case_id}.nii.gz")  # 预测标签
        
        # 安全性拦截：确认所需文件齐全
        if not os.path.exists(ct_path) or not os.path.exists(pet_path) or not os.path.exists(lbl_path) or not os.path.exists(pred_path):
            print(f"[{idx+1}/{num_samples}] ❌ 病例 {case_id} 数据不齐全，跳过。")
            continue
            
        try:
            # 3. 使用 SimpleITK 读取数据
            ct_volume = sitk.ReadImage(ct_path)
            pet_volume = sitk.ReadImage(pet_path)
            lbl_volume = sitk.ReadImage(lbl_path)
            pred_volume = sitk.ReadImage(pred_path)
            
            ct_arr = sitk.GetArrayFromImage(ct_volume)
            pet_arr = sitk.GetArrayFromImage(pet_volume)
            lbl_arr = sitk.GetArrayFromImage(lbl_volume)
            pred_arr = sitk.GetArrayFromImage(pred_volume)
            
            # 4. 自动寻找包含病灶（值为 1）的 2D 切片层
            # 逻辑：只要医生标了（lbl_arr == 1）或者模型预测了（pred_arr == 1），就提取该层进行完美对比
            z_indices = np.where((pred_arr == 1) | (lbl_arr == 1))[0]
            
            if len(z_indices) > 0:
                z_idx = z_indices[len(z_indices) // 2]
                status_str = "🟢 [检测到病灶]"
            else:
                z_idx = ct_arr.shape[0] // 2
                status_str = "⚪️ [未检测到病灶]"
                
            ct_slice = ct_arr[z_idx, :, :]
            pet_slice = pet_arr[z_idx, :, :]
            lbl_slice = lbl_arr[z_idx, :, :]
            pred_slice = pred_arr[z_idx, :, :]
            
            # 5. 绘制 4 栏式学术对比图
            fig = plt.figure(figsize=(24, 6))
            
            # 第 1 栏：原始解剖 CT（灰度）
            plt.subplot(1, 4, 1)
            plt.imshow(ct_slice, cmap='gray')
            plt.title(f"Anatomical CT (Slice {z_idx})")
            plt.axis('off')
            
            # 第 2 栏：功能代谢 PET（SUV 热力图）
            plt.subplot(1, 4, 2)
            plt.imshow(pet_slice, cmap='hot')
            plt.title("Metabolic PET (SUV)")
            plt.axis('off')
            
            # 第 3 栏：医生画的真值 Ground Truth Label（标红）
            plt.subplot(1, 4, 3)
            plt.imshow(lbl_slice, cmap='jet', vmin=0, vmax=1)
            plt.title("Ground Truth (Label)")
            plt.axis('off')
            
            # 第 4 栏：U-Net 预测的分割掩膜（标红）
            plt.subplot(1, 4, 4)
            plt.imshow(pred_slice, cmap='jet', vmin=0, vmax=1)
            plt.title("U-Net Predicted")
            plt.axis('off')
            
            # 唯一性命名（病例ID_slice_层数.png）
            output_name = f"{case_id}_slice_{z_idx}.png"
            output_path = os.path.join(output_dir, output_name)
            
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)  # 保存后立刻强行关闭画布，释放内存
            
            print(f"[{idx+1}/{num_samples}] {status_str} => 成功生成并存入: {output_name}")
            
        except Exception as e:
            print(f"[{idx+1}/{num_samples}] ❌ 处理病例 {case_id} 发生异常，错误原因: {e}")
            
    print(f"\n🎉 批量可视化全部结束！所有对比图已安全保存至全新文件夹：")
    print(f"📂 {output_dir}")

if __name__ == "__main__":
    main()