import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    summary_path = "/mnt/synology/ruihao.li/nnUNet_results/Dataset004_CTPET/nnUNetTrainer_500epochs__nnUNetPlans__2d/fold_0/validation/summary.json"

    if not os.path.exists(summary_path):
        print(f"错误：找不到数据文件 {summary_path}")
        return

    # 1. 加载并解析数据
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    dices_only = []
    # 兼容处理：支持 metric_per_case 为列表或字典
    case_entries = data.get('metric_per_case', [])
    if isinstance(case_entries, dict):
        case_entries = case_entries.values()

    for case_entry in case_entries:
        metrics = case_entry.get('metrics', {})
        lesion_metrics = metrics.get('1', {})
        dice = lesion_metrics.get('Dice', float('nan'))
        if not np.isnan(dice):
            dices_only.append(dice * 100) # 转换为百分比 [0-100]

    if len(dices_only) == 0:
        print("未提取到有效的 Dice 数据。")
        return

    print(f"成功提取了 {len(dices_only)} 例有真实病灶患者的 Dice 分数进行画图。")

    # 计算全局统计量
    mean_val = np.mean(dices_only)
    median_val = np.median(dices_only)
    q25, q75 = np.percentile(dices_only, [25, 75])
    max_val, min_val = np.max(dices_only), np.min(dices_only)

    # 2. 开始绘图 (双子图并排)
    plt.style.use('seaborn-v0_8-whitegrid') # 经典精美白底网格
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # 颜色与样式配置
    box_color = '#AEC6CF'        # 柔和蓝
    violin_color = '#FFB3B3'     # 柔和红
    median_color = '#D62728'     # 亮红（中位数）
    mean_color = '#17BECF'       # 亮青（均值）
    quantile_color = '#1F77B4'   # 深蓝（分位数）

    # ------------------ A. 小提琴图 ------------------
    # 绘制小提琴图，同时要求展示均值(showmeans)、中位数(showmedians)，并传入 25% 和 75% 分位数
    parts = axes[0].violinplot(dices_only, showmeans=True, showmedians=True, quantiles=[0.25, 0.75])
    
    # 美化小提琴图外观
    for pc in parts['bodies']:
        pc.set_facecolor(violin_color)
        pc.set_edgecolor('#E377C2')
        pc.set_alpha(0.6)
    
    # 精细控制内部各个参数线条
    parts['cmeans'].set_colors(mean_color)
    parts['cmeans'].set_linewidth(2.5)
    parts['cmeans'].set_linestyle('--') # 均值用虚线区分
    
    parts['cmedians'].set_colors(median_color)
    parts['cmedians'].set_linewidth(3)  # 中位数加粗实线
    
    parts['cquantiles'].set_colors(quantile_color)
    parts['cquantiles'].set_linewidth(1.5)
    parts['cquantiles'].set_linestyle(':')  # 四分位数用点线
    
    # 外部轮廓和黑线调淡
    parts['cmins'].set_color('#7F7F7F')
    parts['cmaxes'].set_color('#7F7F7F')
    parts['cbars'].set_color('#7F7F7F')

    # 【核心改进：就地文字标注参数】
    # 在图中央直接用文字把中位数和均值写出来，避开线条冲突
    axes[0].text(1.15, median_val, f'Median: {median_val:.1f}%', color=median_color, weight='bold', va='center', fontsize=11)
    axes[0].text(1.15, mean_val, f'Mean: {mean_val:.1f}%', color=mean_color, weight='bold', va='center', fontsize=11)
    axes[0].text(1.12, q75, f'Q3: {q75:.1f}%', color=quantile_color, va='center', fontsize=9, alpha=0.8)
    axes[0].text(1.12, q25, f'Q1: {q25:.1f}%', color=quantile_color, va='center', fontsize=9, alpha=0.8)

    axes[0].set_title("Violin Plot + Density Distribution", fontsize=13, fontweight='bold', pad=15)
    axes[0].set_ylabel("Dice Score (%)", fontsize=12)
    axes[0].set_ylim(-5, 105)
    axes[0].set_xlim(0.7, 1.5)
    axes[0].set_xticks([])
    
    # 为小提琴图创建独立的专业图例
    violin_legend_elements = [
        Line2D([0], [0], color=median_color, lw=3, label=f'Median ({median_val:.1f}%)'),
        Line2D([0], [0], color=mean_color, lw=2.5, ls='--', label=f'Mean ({mean_val:.1f}%)'),
        Line2D([0], [0], color=quantile_color, lw=1.5, ls=':', label=f'IQR (Q1-Q3)')
    ]
    axes[0].legend(handles=violin_legend_elements, loc='lower left', frameon=True, facecolor='white', edgecolor='none')


    # ------------------ B. 箱线图 ------------------
    box = axes[1].boxplot(dices_only, patch_artist=True, widths=0.35, showfliers=True,
                           flierprops=dict(marker='o', markerfacecolor='#FF7F0E', markersize=5, markeredgecolor='none', alpha=0.7))
    
    # 美化箱线图
    for patch in box['boxes']:
        patch.set_facecolor(box_color)
        patch.set_edgecolor('#1F77B4')
        patch.set_alpha(0.7)
    for median in box['medians']:
        median.set_color(median_color)
        median.set_linewidth(3)
    for whisker in box['whiskers']:
        whisker.set_color('#7F7F7F')
        whisker.set_linewidth(1.5)
    for cap in box['caps']:
        cap.set_color('#7F7F7F')
        cap.set_linewidth(1.5)

    # 额外在箱线图里画一根均值线（箱线图原生没有均值横线，加上它对比更强烈）
    axes[1].axhline(mean_val, color=mean_color, linestyle='--', linewidth=2, xmin=0.3, xmax=0.7)

    # 【核心改进：为箱线图的重要拐点做精细文字侧边标注】
    axes[1].text(1.22, median_val, f'Median: {median_val:.1f}%', color=median_color, weight='bold', va='center', fontsize=11)
    axes[1].text(1.22, q75, f'Q3 (75%): {q75:.1f}%', color='black', va='center', fontsize=10)
    axes[1].text(1.22, q25, f'Q1 (25%): {q25:.1f}%', color='black', va='center', fontsize=10)
    axes[1].text(1.22, max_val if max_val < 100 else 100, f'Max: {max_val:.1f}%', color='#7F7F7F', va='center', fontsize=9)
    # 如果低分异常值太多，下界线（Whisker Cap）可以通过计算获得，此处直接标出整体极值点
    axes[1].text(1.22, min_val, f'Min: {min_val:.1f}%', color='orange', va='center', fontsize=9)

    axes[1].set_title("Box Plot + Quantiles & Outliers", fontsize=13, fontweight='bold', pad=15)
    axes[1].set_ylabel("Dice Score (%)", fontsize=12)
    axes[1].set_ylim(-5, 105)
    axes[1].set_xlim(0.7, 1.5)
    axes[1].set_xticks([])
    
    # 箱线图图例
    box_legend_elements = [
        Line2D([0], [0], color=median_color, lw=3, label='Median Line'),
        Line2D([0], [0], color=mean_color, lw=2, ls='--', label='Mean Line'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7F0E', markersize=6, label='Outliers (Low Dice)')
    ]
    axes[1].legend(handles=box_legend_elements, loc='lower left', frameon=True, facecolor='white', edgecolor='none')


    # ------------------ 全局大标题与网格美化 ------------------
    # 微调网格线，使其更淡，不喧宾夺主
    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.5, color='#CCCCCC')

    fig.suptitle(f"Dataset004 (CT+PET) Validation Dice Summary (N={len(dices_only)})\nOverall Mean: {mean_val:.2f}%  |  Overall Median: {median_val:.2f}%", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93]) # 留出顶部大标题空间
    output_path = "/mnt/synology/ruihao.li/dice_distribution_enhanced.png"
    plt.savefig(output_path, dpi=200) # 提升至 200 DPI，看文字更清晰
    print(f"\n🎉 终极参数增强版对比图已成功生成并保存至: {output_path}！")

if __name__ == "__main__":
    main()