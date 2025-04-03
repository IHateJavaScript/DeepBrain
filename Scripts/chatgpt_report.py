


import openai
import json
import argparse
import cv2
import numpy as np
from skimage.measure import label, regionprops


'''

python /mnt/project/DeepBrain/WebSite/DeepBrain/chatgpt_report.py --json /mnt/project/DeepBrain/WebSite/uploads/55ef5329-cae6-476a-8ac8-d1c6d80cbfc9/Result.json --mri_path /mnt/project/DeepBrain/Results/Segmentation/MatToJPG/image/12.mat_image.jpg --mask_path /mnt/project/DeepBrain/Results/Segmentation/MatToJPG/mask/12.mat_mask.jpg --job_id 55ef5329-cae6-476a-8ac8-d1c6d80cbfc9

'''

openai.api_key = "OOPS THATS MY API KEY"

def analyze_mri_and_mask(mri_path, mask_path):
    # 加载 MRI 图像和分割掩膜
    mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mri is None or mask is None:
        raise FileNotFoundError("无法加载 MRI 或分割文件，请检查路径。")

    # 确保掩膜为二值
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 计算肿瘤的像素面积和比例
    tumor_area_pixels = np.sum(binary_mask > 0)
    total_area_pixels = mri.shape[0] * mri.shape[1]
    tumor_area_ratio = tumor_area_pixels / total_area_pixels

    # 计算肿瘤的边界框和位置
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)
    tumor_info = []
    for region in regions:
        # 提取肿瘤的边界框和质心
        min_row, min_col, max_row, max_col = region.bbox
        centroid = region.centroid
        tumor_info.append({
            "bbox": (min_row, min_col, max_row, max_col),
            "centroid": centroid,
            "area": region.area
        })

    # 提取肿瘤信号强度
    tumor_mri_values = mri[binary_mask > 0]
    mean_signal = np.mean(tumor_mri_values)
    max_signal = np.max(tumor_mri_values)
    min_signal = np.min(tumor_mri_values)

    # 返回分析结果
    return {
        "tumor_area_pixels": tumor_area_pixels,
        "tumor_area_ratio": tumor_area_ratio,
        "tumor_bounding_boxes": tumor_info,
        "signal_statistics": {
            "mean_signal": mean_signal,
            "max_signal": max_signal,
            "min_signal": min_signal
        }
    }


def generate_brain_tumor_report(mri_findings):
    """
    :param mri_findings: 字符串，包含 MRI 关键信息或检查发现
    :return: ChatGPT 返回的脑肿瘤 MRI 报告
    """
    # 准备系统信息（可选，用于设定 ChatGPT 的角色或背景）
    system_message = {
        "role": "system",
        "content": (
            "你是一名具有丰富经验的放射科医生，专门从事神经影像诊断。"
            "你将根据提供的 MRI 影像信息，撰写详细的诊断报告。"
        )
    }

    # 准备用户消息（用户给 ChatGPT 的指令 / 输入信息）
    user_message = {
        "role": "user",
        "content": (
            f"以下是患者脑部 MRI 检查的关键信息，请你用英文撰写一份专业的诊断报告，"
            f"包括患者情况概述、影像学所见、诊断意见和建议等部分：\n\n{mri_findings}"
            f"请生成html格式的报告，开头和结尾需要包含 <div> 和 </div> 标签, 并且width=60%;居中。"
            f"报告的样式需要符合医学专业报告的格式。"
        )
    }

    # 进行 API 调用
    response = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[system_message, user_message],
        # 你可以调整下面两个参数，以控制内容的多样性和创造性
        temperature=0.7,
        max_tokens=1000
    )

    # 从 response 中获取生成的文本
    html_response = response["choices"][0]["message"]["content"]
    

    return html_response







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script using argparse")
    parser.add_argument("--json", type=str, help="Input image path")
    parser.add_argument("--mri_path", type=str, help="Input image path")
    parser.add_argument("--mask_path", type=str, help="Input image path")
    parser.add_argument("--job_id", type=str, help="Job ID")

    args = parser.parse_args()

    result_dict = json.load(open(args.json))

    mri_result_dict = analyze_mri_and_mask(args.mri_path, args.mask_path)

    # 假设我们有一些虚构的 MRI 结果描述
    mock_mri_findings = f"""
The class of the patient is {result_dict['class']}
The clinical type of the brain tumor is {result_dict['label']}
The MRI findings are as follows:
- Tumor area pixels: {mri_result_dict['tumor_area_pixels']}
- Tumor area ratio: {mri_result_dict['tumor_area_ratio']}
- Signal intensity statistics:
    - Mean signal: {mri_result_dict['signal_statistics']['mean_signal']}
    - Max signal: {mri_result_dict['signal_statistics']['max_signal']}
    - Min signal: {mri_result_dict['signal_statistics']['min_signal']}
- Tumor locations and bounding boxes: {mri_result_dict['tumor_bounding_boxes']}

    """

    # 调用函数生成报告
    html_response = generate_brain_tumor_report(mock_mri_findings)

    # 将报告写入文件
    with open(f"/mnt/DATA/home/cuisj1/projects/DeepBrain/WebSites/DeepBrain/uploads/{args.job_id}/clinical_report.html", "w") as file:
        file.write(html_response)
    














