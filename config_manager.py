"""
config_manager.py
配置文件管理模块
负责加载配置，如果不存在则生成默认模板
"""

import json
import sys
from pathlib import Path

# ============================================================
# 默认配置模板
# ============================================================

DEFAULT_CONFIG = {
    "folders": {
        "before": "./before",
        "after": "./after",
        "project": "./project"
    },
    "pairing": {
        "mode": "exact",
        "extensions": [".png", ".jpg", ".jpeg", ".bmp"]
    },
    "training": {
        "sample_size": 80,
        "batch_size": 8,
        "epochs": 20,
        "learning_rate": 0.001
    },
    "classification": {
        "confidence_threshold": 0.85
    }
}

# ============================================================
# 配置加载器
# ============================================================

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path='config.json'):
        self.config_path = Path(config_path)
        self.config = None
    
    def load_or_create(self):
        """加载配置，如果不存在则创建默认模板"""
        if self.config_path.exists():
            # 加载现有配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"✅ 加载配置: {self.config_path}")
            return self.config
        else:
            # 配置文件不存在，创建默认模板
            self.create_default()
            print(f"❌ 配置文件不存在: {self.config_path}")
            print(f"✅ 已创建默认配置模板: {self.config_path}")
            print("\n请编辑 config.json 后重新运行！")
            sys.exit(1)
    
    def create_default(self):
        """创建默认配置模板"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        print(f"📄 已创建配置文件: {self.config_path}")
        print("\n默认配置内容:")
        print(json.dumps(DEFAULT_CONFIG, indent=4, ensure_ascii=False))
    
    def get(self, key, default=None):
        """获取配置项"""
        if self.config is None:
            self.load_or_create()
        
        # 支持点号分隔的嵌套访问，如 'folders.before'
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


# ============================================================
# 便捷函数
# ============================================================

def load_config(config_path='config.json'):
    """加载配置，不存在则报错并创建模板"""
    loader = ConfigLoader(config_path)
    return loader.load_or_create()


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    # 测试配置加载
    config = load_config('config.json')
    print("\n配置内容:")
    print(json.dumps(config, indent=4, ensure_ascii=False))