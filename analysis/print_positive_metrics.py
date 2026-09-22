import os
import json
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool

def check_and_calculate_positive_dice(args):
    """
    检查患者是否为真实阳性病例。
    如果是阳性：计算其 3D 体积级 Dice。
    如果是阴性：检查模型是否也完美预测为全黑（完美避错）。
    """
    case_id, raw_labels_dir, predict_labels_dir = args
    lbl_path = os.path.join(raw_labels_dir, f"{case_id}.nii.gz")
    pred_path = os.path.join(predict_labels_dir, f"{case_id}.nii.gz")
    
    try:
        lbl_img = sitk.ReadImage(lbl_path)
        lbl_arr = sitk.GetArrayFromImage(lbl_img)
        
        total_lbl_area = np.sum(lbl_arr > 0)
        
        # 读取模型预测卷
        pred_img = sitk.ReadImage(pred_path)
        pred_arr = sitk.GetArrayFromImage(pred_img)
        
        # 二值化
        pred_arr = (pred_arr > 0).astype(np.uint8)
        lbl_arr = (lbl_arr > 0).astype(np.uint8)
        
        # 1. 真实阴性病例（健康人）
        if total_lbl_area == 0:
            total_pred_area = np.sum(pred_arr)
            if total_pred_area == 0:
                return case_id, None, "NEG_PERFECT"  # 模型完美预测全黑，完全正确
            else:
                return case_id, total_pred_area, "NEG_FALSE_POSITIVE"  # 模型误诊了（产生了假阳性体积）
                
        # 2. 真实阳性病例
        total_pred_area = np.sum(pred_arr)
        total_intersection = np.sum(lbl_arr & pred_arr)
        
        dice_3d = (2.0 * total_intersection) / (total_lbl_area + total_pred_area)
        return case_id, dice_3d, "POSITIVE"

    except Exception as e:
        return case_id, None, f"ERROR: {str(e)}"

def evaluate_positive_only(dataset_id, trainer_folder):
    raw_labels_dir = f"/mnt/synology/ruihao.li/nnUNet_raw/{dataset_id}/labelsTr"
    predict_labels_dir = f"/mnt/synology/ruihao.li/nnUNet_results/{dataset_id}/{trainer_folder}/fold_0/validation"
    splits_json_path = f"/mnt/synology/ruihao.li/nnUNet_preprocessed/{dataset_id}/splits_final.json"
    fold = 0
    num_workers = 16

    if not os.path.exists(predict_labels_dir) or not os.path.exists(splits_json_path):
        return None

    with open(splits_json_path, 'r') as f:
        splits = json.load(f)
    val_cases = splits[fold]['val']

    tasks = [(case_id, raw_labels_dir, predict_labels_dir) for case_id in val_cases]
    with Pool(num_workers) as pool:
        results = pool.map(check_and_calculate_positive_dice, tasks)

    pos_dices = []
    neg_perfect_count = 0
    neg_fp_count = 0
    neg_fp_volumes = []
    errors = 0

    for case_id, value, status in results:
        if status == "POSITIVE":
            if value is not None:
                pos_dices.append(value)
        elif status == "NEG_PERFECT":
            neg_perfect_count += 1
        elif status == "NEG_FALSE_POSITIVE":
            neg_fp_count += 1
            neg_fp_volumes.append(value)
        else:
            errors += 1

    total_neg = neg_perfect_count + neg_fp_count
    neg_perfect_rate = (neg_perfect_count / total_neg * 100) if total_neg > 0 else 0.0

    return {
        "total_cases": len(val_cases),
        "pos_cases": len(pos_dices),
        "neg_cases": total_neg,
        "neg_perfect": neg_perfect_count,
        "neg_fp": neg_fp_count,
        "neg_perfect_rate": neg_perfect_rate,
        "errors": errors,
        "mean_pos_dice": np.mean(pos_dices) if pos_dices else 0.0,
        "median_pos_dice": np.median(pos_dices) if pos_dices else 0.0,
        "std_pos_dice": np.std(pos_dices) if pos_dices else 0.0
    }

def print_report(title, report):
    if not report:
        print(f"\n{title} 数据未就绪或未找到。请检查实验文件夹路径是否匹配 3D 架构。")
        return
    print("\n" + "="*45)
    print(title)
    print("="*45)
    print(f"总评估病例数: {report['total_cases']} 个")
    print(f"真实阳性患者数 (GT 包含病灶): {report['pos_cases']} 个")
    print(f"真实阴性对照数 (GT 为全黑):   {report['neg_cases']} 个")
    print(f"计算失败病例数:               {report['errors']} 个")
    print("-"*45)
    print(f"【阳性组表现】")
    print(f"阳性平均 Dice (Mean Pos Dice): {report['mean_pos_dice'] * 100:.2f}%")
    print(f"阳性中位数 Dice (Median):      {report['median_pos_dice'] * 100:.2f}%")
    print(f"阳性标准差 (Std Dev):         {report['std_pos_dice'] * 100:.2f}%")
    print("-"*45)
    print(f"【阴性组避错表现】")
    print(f"避错患者数 (预测全黑):     {report['neg_perfect']} 个")
    print(f"误诊假阳性患者数 (漏判点):  {report['neg_fp']} 个")
    print(f"阴性控制率 (Specificity):  {report['neg_perfect_rate']:.2f}%")
    print("="*45)

def main():
    dataset_3 = "Dataset003_MyCT"
    dataset_4 = "Dataset004_CTPET"
    
    # 请确认你运行 3D 单双模态时使用的具体 nnunet 文件夹名称（此处以常见 3d_fullres 为例，如果不对请修改它）
    trainer_folder = "nnUNetTrainer__nnUNetPlans__3d_fullres"
    
    report_ct = evaluate_positive_only(dataset_3, trainer_folder)
    report_ctpet = evaluate_positive_only(dataset_4, trainer_folder)
    
    print_report("阳性患者评估报告 - 纯 CT (Dataset003)", report_ct)
    print_report("阳性患者评估报告 - CT+PET 双模态 (Dataset004)", report_ctpet)

if __name__ == "__main__":
    main()