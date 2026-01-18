"""
LoRA 权重合并脚本
将 LoRA 适配器合并到 Qwen3-8B 基座模型,生成完整的微调模型
"""
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import os
import torch

# 配置参数
BASE_MODEL_PATH = "./Qwen/Qwen3-8B"
LORA_ADAPTER_PATH = "./saves/qwen3_mind_news_recommend/checkpoint-8500"
OUTPUT_PATH = "./merged_models/qwen3_mind_news_recommend"


def merge_lora_weights():
    """合并 LoRA 权重到基座模型"""
    print("=" * 60)
    print("开始合并 LoRA 权重")
    print(f"基座模型: {BASE_MODEL_PATH}")
    print(f"LoRA 适配器: {LORA_ADAPTER_PATH}")
    print(f"输出路径: {OUTPUT_PATH}")
    print("=" * 60)

    try:
        # 1. 加载带 LoRA 的模型
        print("\n[1/4] 加载 LoRA 模型...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            LORA_ADAPTER_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("✓ LoRA 模型加载成功")

        # 2. 合并权重
        print("\n[2/4] 合并 LoRA 权重到基座模型...")
        model = model.merge_and_unload()
        print("✓ 权重合并完成")

        # 3. 保存合并后的模型
        print(f"\n[3/4] 保存合并后的模型到 {OUTPUT_PATH}...")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        model.save_pretrained(OUTPUT_PATH)
        print("✓ 模型保存成功")

        # 4. 保存 tokenizer
        print("\n[4/4] 保存 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(OUTPUT_PATH)
        print("✓ Tokenizer 保存成功")

        print("\n" + "=" * 60)
        print("✓ LoRA 权重合并完成!")
        print(f"合并后的模型路径: {os.path.abspath(OUTPUT_PATH)}")
        print("=" * 60)

        # 显示模型大小
        print("\n模型文件列表:")
        for root, dirs, files in os.walk(OUTPUT_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    size_mb = file_size / (1024 * 1024)
                    print(f"  {file} ({size_mb:.2f} MB)")

        print(f"\n后续步骤:")
        print(f"1. 使用 vLLM 加载合并后的模型:")
        print(f"   vllm serve {OUTPUT_PATH}")
        print(f"2. 或在评估脚本中使用:")
        print(f"   from vllm import LLM")
        print(f"   llm = LLM(model=\"{OUTPUT_PATH}\")")

        return OUTPUT_PATH

    except Exception as e:
        print(f"\n✗ 合并失败: {e}")
        print("\n请检查:")
        print("1. LoRA 适配器路径是否正确")
        print("2. 基座模型是否已下载")
        print("3. 显存是否充足 (建议 40GB+)")
        print("4. peft 库是否已安装: pip install peft")
        return None


if __name__ == "__main__":
    # 检查依赖
    try:
        import peft
        print(f"✓ PEFT 版本: {peft.__version__}")
    except ImportError:
        print("✗ 未安装 peft")
        print("请运行: pip install peft")
        exit(1)

    # 合并权重
    merged_model_path = merge_lora_weights()

    if merged_model_path:
        print(f"\n✓ 成功! 可以使用 vLLM 加载模型了")
