#!/usr/bin/env python3
"""
run_ablation_quick.py — 一键运行消融实验

这个脚本:
1. 检查模型是否存在
2. 运行 eval_ablation.py
3. 自动生成诊断报告
"""

import os, sys, subprocess, argparse, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='快速消融实验')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    
    proj_root = Path(__file__).parent.parent
    os.chdir(proj_root)
    
    # 寻找模型
    if args.model is None:
        models_dir = proj_root / 'saved_models'
        candidates = sorted(models_dir.glob('visual_ppo_*.pth'), 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        if candidates:
            args.model = str(candidates[0])
            print(f"✓ 找到最新模型: {args.model}")
        else:
            print("❌ 找不到模型。请用 --model 指定")
            return 1
    
    if not os.path.exists(args.model):
        print(f"❌ 模型不存在: {args.model}")
        return 1
    
    # 运行
    cmd = [
        'python', 'scripts/eval_ablation.py',
        '--model', args.model,
        '--episodes', str(args.episodes),
    ]
    
    print(f"\n运行: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(proj_root))
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
