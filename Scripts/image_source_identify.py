import openai
import json
import argparse

# 设置 OpenAI API Key（请替换为你的 API Key）
openai.api_key = "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"

def classify_medical_image(image_url):
    """
    识别医学影像的类别（CT/MRI/X-ray）及组织来源（脑部/肺部/心脏等）
    :param image_url: 医学图像的 URL
    :return: JSON 格式的分类结果
    """
    try:
        # 发送请求到 OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # 确保使用支持视觉的模型，如 gpt-4o
            messages=[
                {"role": "system", "content": "You are a medical imaging expert. Identify the type of medical image (CT, MRI, X-ray) and its tissue source (brain, lung, heart)."},
                {"role": "user", "content": [
                    {"type": "text", "text": """
                        You are an expert of medical image.
                        1. Identify the type of this medical image (Strictly select from: CT, 2D-MRI, X-ray and Others).
                        2. Identify its tissue source (Strictly select from: Brain, Lung). 
                        3. Provide the result in the json format with the following keys: 'image_source' and 'tissue_source'. 
                        4. Please output the results in json format directly and do not use markdown format.
                    """},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ],
            max_tokens=300
        )

        # 解析返回结果
        classification_json = response["choices"][0]["message"]["content"]
        # convert to json
        classification_json = json.loads(classification_json)

        # 返回 JSON 结果
        return classification_json
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script using argparse")
    parser.add_argument("--image_url", type=str, help="Input image path")
    parser.add_argument("--output_dir", type=str, help="Job ID")

    args = parser.parse_args()

    image_url = args.image_url
    output_dir = args.output_dir
    
    print(f"image_url: {image_url}")

    # 示例：调用函数，输入医学图像的URL
    result = classify_medical_image(image_url)

    # save the result to a json file
    with open(f"{output_dir}/predicted_image_source_result.json", "w") as f:
        json.dump(result, f, indent=4)
    
