"""Qwen3
Qwen3-8B 模型下载脚本
从 ModelScope 下载模型文件到本地目录
"""
from modelscope import snapshot_download
import os

# 配置参数
MODEL_ID = "Qwen/Qwen3-8B"  # ModelScope 模型 ID
LOCAL_DIR = "./Qwen/Qwen3-8B"  # 本地保存路径


def download_model():
    """下载 Qwen3-8B 模型"""
    print("=" * 60)
    print(f"开始下载模型: {MODEL_ID}")
    print(f"保存路径: {os.path.abspath(LOCAL_DIR)}")
    print("=" * 60)

    try:
        # 创建保存目录
        os.makedirs(LOCAL_DIR, exist_ok=True)

        # 下载模型
        model_dir = snapshot_download(
            model_id=MODEL_ID,
            cache_dir=LOCAL_DIR,
            revision="master"  # 使用最新版本
        )

        print("\n" + "=" * 60)
        print("✓ 模型下载完成!")
        print(f"模型路径: {model_dir}")
        print("=" * 60)

        # 列出下载的文件
        print("\n下载的文件列表:")
        for root, dirs, files in os.walk(model_dir):
            level = root.replace(model_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    size_mb = file_size / (1024 * 1024)
                    print(f'{subindent}{file} ({size_mb:.2f} MB)')

        return model_dir

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n请检查:")
        print("1. modelscope 是否已安装: pip install modelscope")
        print("2. 网络连接是否正常")
        print("3. ModelScope 账号是否已登录 (如需要)")
        return None


if __name__ == "__main__":
    # 检查依赖
    try:
        import modelscope
        print(f"✓ ModelScope 版本: {modelscope.__version__}")
    except ImportError:
        print("✗ 未安装 modelscope")
        print("请运行: pip install modelscope")
        exit(1)

    # 下载模型
    model_path = download_model()

    if model_path:
        print(f"\n后续步骤:")
        print(f"1. 在 new_train.yaml 中设置:")
        print(f"   model_name_or_path: {model_path}")
        print(f"2. 在 news_inference.yaml 中设置:")
        print(f"   model_name_or_path: {model_path}")
