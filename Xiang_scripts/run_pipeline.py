import os
import time

def run_pipeline():
    """
    按顺序执行real_time_debug.py和combine_video.py
    """
    print("=== 开始执行视频处理流程 ===")
    
    # 执行real_time_debug.py
    os.system("python real_time_debug.py")
    
    # 短暂等待确保文件写入完成
    time.sleep(2)
    
    # 执行combine_video.py
    os.system("python combine_video.py")
    
    print("\n=== 所有处理完成 ===")

if __name__ == "__main__":
    # 确保当前工作目录正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 运行流程
    run_pipeline() 