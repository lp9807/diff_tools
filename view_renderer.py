"""
view_renderer.py
三视图渲染器 - 完全可复用的视图模块

功能：
1. Split View：可拖拽分割线
2. Animation View：交替闪烁 before/after
3. Triple View：三图对比（Before | After | Diff）
4. 鼠标交互：拖拽分割线、滚轮缩放、平移
5. 视图状态管理：缩放、平移、分割位置、动画间隔

被以下模块使用：
    - part1_data_preparation.py
    - part4_review.py
"""

import cv2
import numpy as np
import time

# ============================================================
# 标记类型配置
# ============================================================

# view_renderer.py - 添加 to_review

ALL_CLASSES = [
    'to_review',                 # 新增：待复核样本（低置信度）
    'quality_improved',
    'quality_degradation',
    'render_failed',
    'uncertain_difference',
    'matched'
]

LABEL_DISPLAY = {
    'to_review': '📌 待复核',      # 新增
    'quality_improved': '✅ 质量提升',
    'quality_degradation': '⚠️ 质量退化',
    'render_failed': '❌ 渲染失败',
    'uncertain_difference': '❓ 不确定差异',
    'matched': '🔄 匹配'
}

LABEL_COLORS = {
    'to_review': '#e67e22',        # 橙色
    'quality_improved': '#2ecc71',
    'quality_degradation': '#f39c12',
    'render_failed': '#e74c3c',
    'uncertain_difference': '#f1c40f',
    'matched': '#3498db'
}


# ============================================================
# 视图状态管理
# ============================================================

class ViewState:
    """
    视图状态管理
    
    属性：
        display_mode: 'split', 'animation', 'triple'
        split_pos: 分割位置 (0-1)
        zoom: 缩放倍数
        pan_x: 水平平移
        pan_y: 垂直平移
        animation_interval: 动画切换间隔（毫秒）
        show_before: 动画模式是否显示before
        img_width: 当前图像宽度
        img_height: 当前图像高度
        is_dragging_split: 是否正在拖拽分割线
        is_panning: 是否正在平移
        last_mouse_x: 上次鼠标X位置
        last_mouse_y: 上次鼠标Y位置
    """
    
    def __init__(self):
        self.display_mode = 'split'
        self.split_pos = 0.5
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.animation_interval = 500
        self.show_before = True
        self.img_width = 0
        self.img_height = 0
        self.is_dragging_split = False
        self.is_panning = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
    
    def reset(self):
        """重置视图状态"""
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.split_pos = 0.5
        self.show_before = True
        print("🔄 重置视图")
    
    def toggle_mode(self):
        """切换显示模式"""
        modes = ['split', 'animation', 'triple']
        ci = modes.index(self.display_mode)
        self.display_mode = modes[(ci + 1) % len(modes)]
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        print(f"📺 切换模式: {self.display_mode}")
    
    def adjust_split(self, delta):
        """调整分割位置"""
        if self.display_mode == 'split':
            self.split_pos = max(0.05, min(0.95, self.split_pos + delta))
    
    def adjust_animation_interval(self, delta):
        """调整动画间隔"""
        if self.display_mode == 'animation':
            self.animation_interval = max(50, min(2000, self.animation_interval + delta))
            print(f"⏱️ 间隔: {self.animation_interval}ms")
    
    def toggle_animation_frame(self):
        """切换动画帧"""
        if self.display_mode == 'animation':
            self.show_before = not self.show_before


# ============================================================
# 鼠标交互处理器
# ============================================================

class MouseHandler:
    """鼠标交互处理器"""
    
    def __init__(self, view_state):
        self.view_state = view_state
    
    def callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        vs = self.view_state
        
        if event == cv2.EVENT_LBUTTONDOWN:
            split_x = int(vs.split_pos * vs.img_width)
            if abs(x - split_x) < 20:
                vs.is_dragging_split = True
                vs.last_mouse_x = x
            else:
                vs.is_panning = True
                vs.last_mouse_x = x
                vs.last_mouse_y = y
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if vs.is_dragging_split:
                new_pos = max(0.05, min(0.95, x / vs.img_width))
                vs.split_pos = new_pos
                vs.last_mouse_x = x
            elif vs.is_panning:
                dx = x - vs.last_mouse_x
                dy = y - vs.last_mouse_y
                vs.pan_x += dx
                vs.pan_y += dy
                vs.last_mouse_x = x
                vs.last_mouse_y = y
        
        elif event == cv2.EVENT_LBUTTONUP:
            vs.is_dragging_split = False
            vs.is_panning = False
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                vs.zoom = min(3.0, vs.zoom * 1.1)
            else:
                vs.zoom = max(0.3, vs.zoom / 1.1)


# ============================================================
# 视图渲染器
# ============================================================

class ViewRenderer:
    """三视图渲染器"""
    
    @staticmethod
    def hex_to_bgr(hex_color):
        """将十六进制颜色转换为BGR"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (b, g, r)
    
    @staticmethod
    def _resize_image(img, target_h=500):
        """调整图片大小"""
        h, w = img.shape[:2]
        target_w = int(w * target_h / h)
        return cv2.resize(img, (target_w, target_h)), target_w, target_h
    
    @staticmethod
    def create_split_view(before_img, after_img, split_pos=0.5):
        """
        创建分割视图 - 可拖拽分割线
        
        Args:
            before_img: before图片 (BGR)
            after_img: after图片 (BGR)
            split_pos: 分割位置 (0-1)
        
        Returns:
            combined: 渲染后的图像
        """
        h, w = before_img.shape[:2]
        target_h = 500
        target_w = int(w * target_h / h)
        
        before_resized = cv2.resize(before_img, (target_w, target_h))
        after_resized = cv2.resize(after_img, (target_w, target_h))
        
        split_x = int(split_pos * target_w)
        combined = before_resized.copy()
        combined[:, split_x:] = after_resized[:, split_x:]
        
        # 分割线
        cv2.line(combined, (split_x, 0), (split_x, target_h), (0, 255, 255), 4)
        cv2.circle(combined, (split_x, 30), 12, (0, 255, 255), -1)
        cv2.circle(combined, (split_x, target_h - 30), 12, (0, 255, 255), -1)
        cv2.circle(combined, (split_x, target_h // 2), 15, (0, 255, 255), -1)
        cv2.circle(combined, (split_x, target_h // 2), 17, (255, 255, 255), 2)
        
        # 标签
        cv2.rectangle(combined, (10, 10), (10 + 6*12 + 20, 45), (0, 0, 0), -1)
        cv2.putText(combined, "BEFORE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.rectangle(combined, (target_w - 5*12 - 30, 10), (target_w - 10, 45), (0, 0, 0), -1)
        cv2.putText(combined, "AFTER", (target_w - 5*12 - 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(combined, "↔ 拖拽分割线", (10, target_h - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return combined
    
    @staticmethod
    def create_animation_view(before_img, after_img, show_before, interval_ms=500):
        """
        创建动画视图 - 交替显示 before/after
        
        Args:
            before_img: before图片 (BGR)
            after_img: after图片 (BGR)
            show_before: True显示before，False显示after
            interval_ms: 切换间隔（毫秒）
        
        Returns:
            display: 渲染后的图像
        """
        h, w = before_img.shape[:2]
        target_h = 500
        target_w = int(w * target_h / h)
        
        before_resized = cv2.resize(before_img, (target_w, target_h))
        after_resized = cv2.resize(after_img, (target_w, target_h))
        
        if show_before:
            display = before_resized.copy()
            label = "BEFORE"
            label_color = (0, 255, 0)
        else:
            display = after_resized.copy()
            label = "AFTER"
            label_color = (255, 165, 0)
        
        cv2.rectangle(display, (10, 10), (10 + len(label)*12 + 20, 45), (0, 0, 0), -1)
        cv2.putText(display, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)
        
        cv2.rectangle(display, (target_w - 180, 10), (target_w - 10, 40), (0, 0, 0), -1)
        cv2.putText(display, "🔄", (target_w - 155, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(display, (10, target_h - 35), (10 + 220, target_h - 10), (0, 0, 0), -1)
        cv2.putText(display, f"间隔: {interval_ms}ms  (↑↓调整)", (20, target_h - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    @staticmethod
    def create_triple_view(before_img, after_img):
        """
        创建三视图：Before | After | Diff Heatmap
        
        Args:
            before_img: before图片 (BGR)
            after_img: after图片 (BGR)
        
        Returns:
            combined: 渲染后的图像
        """
        h, w = before_img.shape[:2]
        target_h = 500
        target_w = int(w * target_h / h)
        
        before_resized = cv2.resize(before_img, (target_w, target_h))
        after_resized = cv2.resize(after_img, (target_w, target_h))
        
        diff = cv2.absdiff(before_img, after_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        diff_resized = cv2.resize(diff_colored, (target_w, target_h))
        
        gap = 5
        combined = np.hstack([
            before_resized,
            np.full((target_h, gap, 3), 255, dtype=np.uint8),
            after_resized,
            np.full((target_h, gap, 3), 255, dtype=np.uint8),
            diff_resized
        ])
        
        labels = [
            ("BEFORE", (10, 30), (0, 255, 0)),
            ("AFTER", (target_w + gap + 10, 30), (0, 255, 0)),
            ("DIFF", (2*target_w + 2*gap + 10, 30), (0, 255, 255))
        ]
        for text, pos, color in labels:
            cv2.rectangle(combined, (pos[0]-5, pos[1]-25), 
                         (pos[0] + len(text)*10 + 5, pos[1]+5), (0, 0, 0), -1)
            cv2.putText(combined, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return combined
    
    @staticmethod
    def create_main_view(before_img, after_img, mode='split', split_pos=0.5,
                         show_before=True, interval_ms=500):
        """
        根据模式创建主视图
        
        Args:
            before_img: before图片 (BGR)
            after_img: after图片 (BGR)
            mode: 'split', 'animation', 'triple'
            split_pos: 分割位置
            show_before: 动画模式显示before
            interval_ms: 动画间隔
        
        Returns:
            view: 渲染后的主视图
            mode_hint: 模式提示文字
        """
        if mode == 'split':
            view = ViewRenderer.create_split_view(before_img, after_img, split_pos)
            mode_hint = "SPLIT VIEW - 拖拽分割线 | ← → 微调"
        elif mode == 'animation':
            view = ViewRenderer.create_animation_view(before_img, after_img, show_before, interval_ms)
            mode_hint = f"ANIMATION VIEW - 间隔: {interval_ms}ms | ↑↓ 调整"
        else:  # triple
            view = ViewRenderer.create_triple_view(before_img, after_img)
            mode_hint = "TRIPLE VIEW - Before | After | Diff"
        
        return view, mode_hint
    
    @staticmethod
    def apply_zoom_pan(image, zoom=1.0, pan_x=0, pan_y=0):
        """
        对图像应用缩放和平移
        
        Args:
            image: 输入图像
            zoom: 缩放倍数
            pan_x: 水平平移
            pan_y: 垂直平移
        
        Returns:
            image: 处理后的图像
        """
        h, w = image.shape[:2]
        
        if zoom == 1.0 and pan_x == 0 and pan_y == 0:
            return image
        
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        
        if zoom != 1.0:
            image = cv2.resize(image, (new_w, new_h))
        
        if pan_x != 0 or pan_y != 0:
            start_x = max(0, min(new_w - w, pan_x))
            start_y = max(0, min(new_h - h, pan_y))
            image = image[start_y:start_y+h, start_x:start_x+w]
            
            if image.shape[0] < h or image.shape[1] < w:
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                y_offset = max(0, (h - image.shape[0]) // 2)
                x_offset = max(0, (w - image.shape[1]) // 2)
                padded[y_offset:y_offset+image.shape[0], 
                       x_offset:x_offset+image.shape[1]] = image
                image = padded
        
        return image
    
    @staticmethod
    def create_info_panel(sample, idx, total_samples, 
                          show_shortcuts=True,
                          custom_labels=None):
        """
        创建信息面板
        
        Args:
            sample: 样本信息字典
            idx: 当前索引
            total_samples: 总样本数
            show_shortcuts: 是否显示快捷键
            custom_labels: 自定义标签列表 (用于 part1_data_preparation)
        
        Returns:
            panel: 信息面板图像
        """
        panel_w = 450
        panel_h = 500
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8) + 40
        
        y_offset = 20
        line_height = 30
        
        # 标题
        cv2.putText(panel, f"审核 #{idx+1}/{total_samples}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += line_height + 10
        
        # 样本名
        name = sample.get('name', 'unknown')
        cv2.putText(panel, f"样本: {name}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += line_height
        
        # 当前预测
        pred_label = sample.get('pred_label', 'unknown')
        pred_color = ViewRenderer.hex_to_bgr(LABEL_COLORS.get(pred_label, '#ffffff'))
        pred_text = f"当前: {LABEL_DISPLAY.get(pred_label, pred_label)}"
        
        if sample.get('auto_matched', False):
            pred_text += " (自动匹配)"
        elif sample.get('confidence', 0) < 0.6:
            pred_text += f" (置信度: {sample['confidence']:.2f} ⚠️)"
        else:
            pred_text += f" (置信度: {sample['confidence']:.2f})"
        
        cv2.putText(panel, pred_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.55, pred_color, 1)
        y_offset += line_height + 5
        
        # SSIM
        ssim_score = sample.get('ssim_score', 0)
        if ssim_score > 0:
            cv2.putText(panel, f"SSIM: {ssim_score:.6f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height
        
        # 分隔线
        cv2.line(panel, (10, y_offset), (panel_w - 10, y_offset), (100, 100, 100), 1)
        y_offset += line_height
        
        # 快捷键
        if custom_labels is not None:
            # 自定义标签（用于 part1_data_preparation）
            cv2.putText(panel, "快捷键:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += line_height
            for text in custom_labels:
                cv2.putText(panel, f"  {text}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_offset += line_height
        
        elif show_shortcuts:
            # 标准快捷键（用于 part4_review）
            cv2.putText(panel, "选择新分类:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += line_height
            
            for i, cls in enumerate(ALL_CLASSES, 1):
                display = LABEL_DISPLAY.get(cls, cls)
                color = ViewRenderer.hex_to_bgr(LABEL_COLORS.get(cls, '#ffffff'))
                cv2.putText(panel, f"  {i} → {display}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += line_height
            
            y_offset += 5
            cv2.putText(panel, "  v → 切换视图模式", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_offset += line_height
            cv2.putText(panel, "  ↑↓ → 调整动画间隔", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_offset += line_height
            cv2.putText(panel, "  r → 重置视图  s → 跳过  q → 退出", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        return panel
    
    @staticmethod
    def create_full_view(before_img, after_img, sample, idx, total_samples,
                         view_state, show_shortcuts=True, custom_labels=None):
        """
        创建完整的审核视图（包含主视图 + 信息面板）
        
        Args:
            before_img: before图片 (BGR)
            after_img: after图片 (BGR)
            sample: 样本信息
            idx: 当前索引
            total_samples: 总样本数
            view_state: ViewState 对象
            show_shortcuts: 是否显示快捷键
            custom_labels: 自定义快捷键标签
        
        Returns:
            final: 完整的渲染视图
        """
        # 创建主视图
        main_view, mode_hint = ViewRenderer.create_main_view(
            before_img, after_img,
            mode=view_state.display_mode,
            split_pos=view_state.split_pos,
            show_before=view_state.show_before,
            interval_ms=view_state.animation_interval
        )
        
        # 应用缩放和平移
        main_view = ViewRenderer.apply_zoom_pan(
            main_view,
            zoom=view_state.zoom,
            pan_x=view_state.pan_x,
            pan_y=view_state.pan_y
        )
        
        # 创建信息面板
        panel = ViewRenderer.create_info_panel(
            sample, idx, total_samples,
            show_shortcuts=show_shortcuts,
            custom_labels=custom_labels
        )
        
        # 调整面板高度与主视图一致
        panel_h, panel_w = panel.shape[:2]
        main_h, main_w = main_view.shape[:2]
        
        if panel_h != main_h:
            panel = cv2.resize(panel, (panel_w, main_h))
        
        # 拼接
        gap = 5
        final = np.hstack([
            main_view,
            np.full((main_h, gap, 3), 255, dtype=np.uint8),
            panel
        ])
        
        # 更新视图状态中的图像尺寸
        view_state.img_width = final.shape[1]
        view_state.img_height = final.shape[0]
        
        return final


# ============================================================
# 键盘事件处理器
# ============================================================

class KeyboardHandler:
    """键盘事件处理器"""
    
    def __init__(self, view_state):
        self.view_state = view_state
    
    def handle_key(self, key):
        """
        处理键盘事件
        
        Args:
            key: OpenCV 按键值
        
        Returns:
            str: 触发的动作 ('toggle_mode', 'reset', 'quit', None)
        """
        vs = self.view_state
        
        if key == ord('v'):
            vs.toggle_mode()
            return 'toggle_mode'
        
        elif key == ord('r'):
            vs.reset()
            return 'reset'
        
        elif key == ord('q'):
            return 'quit'
        
        elif key == 81:  # 左箭头
            vs.adjust_split(-0.02)
            return 'adjust_split'
        
        elif key == 83:  # 右箭头
            vs.adjust_split(0.02)
            return 'adjust_split'
        
        elif key == 82:  # 上箭头
            vs.adjust_animation_interval(50)
            return 'adjust_interval'
        
        elif key == 84:  # 下箭头
            vs.adjust_animation_interval(-50)
            return 'adjust_interval'
        
        return None


# ============================================================
# 便捷函数
# ============================================================

def setup_viewer():
    """
    创建并返回视图组件（ViewState, MouseHandler, KeyboardHandler）
    
    Returns:
        tuple: (view_state, mouse_handler, keyboard_handler)
    """
    view_state = ViewState()
    mouse_handler = MouseHandler(view_state)
    keyboard_handler = KeyboardHandler(view_state)
    return view_state, mouse_handler, keyboard_handler