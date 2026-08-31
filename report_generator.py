"""
report_generator.py
独立的报告生成模块

功能：
1. 从多个数据源（review/annotated/final_output）收集样本分类信息
2. 按优先级确定每个样本的最终分类
3. 生成可视化HTML报告
4. 支持按分类分组显示
5. 悬停显示 before/after/diff 缩略图
"""

import json
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import shutil
import webbrowser

# ============================================================
# 导入配置
# ============================================================

from model_config import ALL_CLASSES, LABEL_DISPLAY, LABEL_COLORS

# ============================================================
# 报告生成器
# ============================================================

class ReportGenerator:
    """
    报告生成器 - 从多个数据源聚合分类信息
    
    分类优先级（从高到低）：
        1. 审核修改 (07_review/review_history.json)
        2. 人工标注 (02_annotated/)
        3. 模型预测 (06_final_output/)
    """
    
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        self.output_dir = self.project_dir / "06_final_output"
        self.annotated_dir = self.project_dir / "02_annotated"
        self.review_dir = self.project_dir / "07_review"
        self.report_dir = self.review_dir
        
        # 存储所有样本信息
        self.samples = {}  # name -> sample_info
        
        # 创建报告目录
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_samples(self):
        """
        从所有数据源收集样本信息
        按优先级确定最终分类
        """
        print("\n📂 收集样本分类信息...")
        print(f"  优先级: 审核修改 > 人工标注 > 模型预测")
        
        # ============================================================
        # 1. 从 06_final_output 读取模型预测结果（最低优先级）
        # ============================================================
        if self.output_dir.exists():
            for cls in ALL_CLASSES:
                cls_dir = self.output_dir / cls
                if not cls_dir.exists():
                    continue
                
                for pair_dir in cls_dir.iterdir():
                    if not pair_dir.is_dir():
                        continue
                    
                    name = pair_dir.name
                    if name not in self.samples:
                        self.samples[name] = {
                            'name': name,
                            'pred_label': cls,
                            'confidence': 0.0,
                            'ssim_score': 0.0,
                            'auto_matched': False,
                            'has_prediction': True
                        }
                    
                    # 读取预测信息
                    info_file = pair_dir / 'prediction_info.json'
                    if info_file.exists():
                        try:
                            with open(info_file, 'r') as f:
                                info = json.load(f)
                            self.samples[name]['confidence'] = info.get('confidence', 0.0)
                            self.samples[name]['ssim_score'] = info.get('ssim_score', 0.0)
                            self.samples[name]['auto_matched'] = info.get('auto_matched', False)
                            if info.get('low_confidence', False):
                                self.samples[name]['is_low_confidence'] = True
                        except:
                            pass
                    
                    # 记录源路径
                    self.samples[name]['pred_source'] = str(pair_dir)
        
        print(f"  📊 模型预测: {len(self.samples)} 个样本")
        
        # ============================================================
        # 2. 从 02_annotated 读取人工标注（中优先级）
        # ============================================================
        if self.annotated_dir.exists():
            annotated_count = 0
            for cls in ALL_CLASSES:
                cls_dir = self.annotated_dir / cls
                if not cls_dir.exists():
                    continue
                
                for pair_dir in cls_dir.iterdir():
                    if not pair_dir.is_dir():
                        continue
                    
                    name = pair_dir.name
                    annotated_count += 1
                    
                    if name not in self.samples:
                        self.samples[name] = {
                            'name': name,
                            'has_prediction': False
                        }
                    
                    # 人工标注覆盖预测结果
                    self.samples[name]['annotated_label'] = cls
                    self.samples[name]['annotated_source'] = str(pair_dir)
                    self.samples[name]['has_annotation'] = True
            
            print(f"  📝 人工标注: {annotated_count} 个样本")
        
        # ============================================================
        # 3. 从 07_review 读取审核修改（最高优先级）
        # ============================================================
        review_history_file = self.review_dir / 'review_history.json'
        if review_history_file.exists():
            try:
                with open(review_history_file, 'r') as f:
                    review_history = json.load(f)
                
                for entry in review_history:
                    name = entry.get('name')
                    if not name:
                        continue
                    
                    new_label = entry.get('new_label')
                    if not new_label:
                        continue
                    
                    if name not in self.samples:
                        self.samples[name] = {
                            'name': name,
                            'has_prediction': False
                        }
                    
                    # 审核修改覆盖所有
                    self.samples[name]['reviewed_label'] = new_label
                    self.samples[name]['old_label'] = entry.get('old_label')
                    self.samples[name]['reviewed_at'] = entry.get('timestamp')
                    self.samples[name]['has_review'] = True
                
                print(f"  🔄 审核修改: {len(review_history)} 个样本")
            except Exception as e:
                print(f"  ⚠️ 加载审核历史失败: {e}")
        
        # ============================================================
        # 4. 确定每个样本的最终分类（按优先级）
        # ============================================================
        for name, info in self.samples.items():
            # 优先级1: 审核修改
            if info.get('has_review', False):
                info['final_label'] = info['reviewed_label']
                info['source'] = 'review'
            
            # 优先级2: 人工标注
            elif info.get('has_annotation', False):
                info['final_label'] = info['annotated_label']
                info['source'] = 'annotated'
            
            # 优先级3: 模型预测
            elif info.get('has_prediction', False):
                info['final_label'] = info['pred_label']
                info['source'] = 'prediction'
            
            else:
                info['final_label'] = 'unknown'
                info['source'] = 'unknown'
        
        print(f"\n✅ 共收集 {len(self.samples)} 个样本")
        
        # 统计最终分类分布
        class_counts = defaultdict(int)
        source_counts = defaultdict(int)
        for info in self.samples.values():
            class_counts[info.get('final_label', 'unknown')] += 1
            source_counts[info.get('source', 'unknown')] += 1
        
        print("\n📊 最终分类分布:")
        for cls, count in class_counts.items():
            display = LABEL_DISPLAY.get(cls, cls)
            print(f"  - {display}: {count}")
        
        print("\n📊 数据来源:")
        for src, count in source_counts.items():
            src_display = {'review': '审核修改', 'annotated': '人工标注', 'prediction': '模型预测'}.get(src, src)
            print(f"  - {src_display}: {count}")
        
        return self.samples
    
    def generate_html_report(self):
        """生成HTML报告"""
        if not self.samples:
            print("❌ 没有样本数据，请先运行 collect_samples()")
            return
        
        print("\n📄 生成可视化HTML报告...")
        
        # 按最终分类分组
        grouped = defaultdict(list)
        for info in self.samples.values():
            final_label = info.get('final_label', 'unknown')
            grouped[final_label].append(info)
        
        # 创建报告资源目录
        report_files_dir = self.report_dir / 'report_files'
        report_files_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        total = len(self.samples)
        class_counts = {cls: len(items) for cls, items in grouped.items()}
        
        # 构建HTML
        html = self._build_html(grouped, class_counts, total, report_files_dir)
        
        html_path = self.report_dir / 'review_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ HTML报告已生成: {html_path}")
        print(f"📁 图片资源: {report_files_dir}")
        
        try:
            webbrowser.open(str(html_path.absolute()))
            print("🌐 已在浏览器中打开报告")
        except:
            print(f"请手动打开: {html_path}")
    
    def _build_html(self, grouped, class_counts, total, report_files_dir):
        """构建HTML内容"""
        
        # 生成缩略图文件
        self._generate_thumbnails(grouped, report_files_dir)
        
        # 生成每个类别的HTML
        class_sections = []
        display_order = ['to_review', 'render_failed', 'quality_degradation', 
                         'uncertain_difference', 'quality_improved', 'matched', 'unknown']
        
        for cls in display_order:
            if cls not in grouped:
                continue
            
            items = grouped[cls]
            display_name = LABEL_DISPLAY.get(cls, cls)
            color = LABEL_COLORS.get(cls, '#3498db')
            
            rows = []
            for info in items:
                name = info['name']
                source = info.get('source', 'unknown')
                confidence = info.get('confidence', 0)
                ssim_score = info.get('ssim_score', 0)
                auto_matched = info.get('auto_matched', False)
                
                # 构建缩略图
                thumb_html = self._build_thumb_html(name, report_files_dir)
                
                # 数据来源标签
                source_display = {
                    'review': '🔄 审核修改',
                    'annotated': '📝 人工标注',
                    'prediction': '🤖 模型预测'
                }.get(source, '❓ 未知')
                
                # 置信度显示
                conf_display = f"{confidence:.3f}" if confidence > 0 else '-'
                if auto_matched:
                    conf_display += " (自动匹配)"
                elif confidence < 0.6 and confidence > 0:
                    conf_display += " ⚠️"
                
                # ============================================================
                # 使用 format() 而不是 f-string
                # ============================================================
                row_html = '''
                <tr>
                    <td class="sample-name">{name}</td>
                    <td class="thumb-cell">{thumb_html}</td>
                    <td>{source_display}</td>
                    <td>{conf_display}</td>
                    <td>{ssim_display}</td>
                </tr>
                '''.format(
                    name=name,
                    thumb_html=thumb_html,
                    source_display=source_display,
                    conf_display=conf_display,
                    ssim_display=f"{ssim_score:.6f}" if ssim_score > 0 else '-'
                )
                rows.append(row_html)
            
            # 默认展开 render_failed 和 to_review
            is_default_expanded = cls in ['render_failed', 'to_review']
            icon_symbol = '▼' if is_default_expanded else '▶'
            display_style = 'block' if is_default_expanded else 'none'
            
            # ============================================================
            # 使用 format() 构建类别部分
            # ============================================================
            section_html = '''
            <div class="class-section">
                <div class="class-header" onclick="toggleClass('{cls}')" 
                     style="background: {color}22; border-left: 4px solid {color};">
                    <span class="class-name">{display_name}</span>
                    <span class="class-count">{count} 个样本</span>
                    <span class="source-info">来源: {sources}</span>
                    <span class="toggle-icon" id="icon-{cls}">{icon_symbol}</span>
                </div>
                <div class="class-body" id="body-{cls}" style="display: {display_style};">
                    <table class="details-table">
                        <thead>
                            <tr>
                                <th>样本名</th>
                                <th>图片预览 (悬停查看)</th>
                                <th>数据来源</th>
                                <th>置信度</th>
                                <th>SSIM</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
            '''.format(
                cls=cls,
                color=color,
                display_name=display_name,
                count=len(items),
                sources=self._get_class_sources(items),
                icon_symbol=icon_symbol,
                display_style=display_style,
                rows=''.join(rows)
            )
            class_sections.append(section_html)
        
        # 统计卡片
        stat_cards = self._build_stat_cards(class_counts, total)
        
        # ============================================================
        # 完整 HTML - 使用 format()
        # ============================================================
        html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>分类结果审核报告</title>
    <style>
        * {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; box-sizing: border-box; }}
        body {{ background: #f0f2f5; margin: 0; padding: 20px; color: #2c3e50; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 20px; }}
        .subtitle {{ color: #7f8c8d; margin-bottom: 25px; font-size: 14px; }}
        
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 20px 0 30px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 12px 16px; border-radius: 8px; text-align: center; border-left: 4px solid #3498db; }}
        .stat-card .number {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .stat-card .label {{ font-size: 13px; color: #7f8c8d; }}
        .stat-card.green {{ border-color: #2ecc71; }}
        .stat-card.orange {{ border-color: #f39c12; }}
        .stat-card.red {{ border-color: #e74c3c; }}
        .stat-card.blue {{ border-color: #3498db; }}
        .stat-card.purple {{ border-color: #9b59b6; }}
        
        .class-section {{ margin: 15px 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }}
        .class-header {{ padding: 12px 20px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; user-select: none; transition: background 0.2s; }}
        .class-header:hover {{ filter: brightness(0.95); }}
        .class-name {{ font-size: 16px; font-weight: 600; }}
        .class-count {{ font-size: 14px; color: #7f8c8d; margin-left: 15px; }}
        .source-info {{ font-size: 12px; color: #95a5a6; margin-left: auto; margin-right: 15px; }}
        .toggle-icon {{ font-size: 16px; font-weight: bold; color: #7f8c8d; }}
        .class-body {{ padding: 0; overflow-x: auto; }}
        
        .details-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .details-table th {{ background: #34495e; color: white; padding: 10px 12px; text-align: left; position: sticky; top: 0; z-index: 10; }}
        .details-table td {{ padding: 8px 12px; border-bottom: 1px solid #ecf0f1; vertical-align: middle; }}
        .details-table tr:hover {{ background: #f8f9fa; }}
        
        .thumb-cell {{ position: relative; width: 120px; min-width: 120px; }}
        .thumb-trigger {{ color: #3498db; cursor: pointer; font-size: 13px; padding: 4px 10px; border: 1px dashed #3498db; border-radius: 4px; display: inline-block; background: #f0f7ff; transition: all 0.2s; }}
        .thumb-trigger:hover {{ background: #3498db; color: white; }}
        
        .thumb-popup {{ display: none; position: fixed; z-index: 1000; background: white; border-radius: 10px; box-shadow: 0 8px 30px rgba(0,0,0,0.3); padding: 12px; max-width: 700px; min-width: 400px; pointer-events: none; border: 2px solid #e0e0e0; }}
        .thumb-popup.show {{ display: block; }}
        .thumb-popup .thumb-row {{ display: flex; gap: 8px; align-items: center; justify-content: center; }}
        .thumb-popup .thumb-img {{ max-width: 200px; max-height: 150px; border-radius: 6px; border: 1px solid #ddd; object-fit: contain; }}
        .thumb-popup .thumb-label {{ text-align: center; font-size: 11px; color: #7f8c8d; margin-top: 2px; }}
        .thumb-popup .thumb-arrow {{ font-size: 20px; color: #95a5a6; padding: 0 4px; }}
        .thumb-popup .sample-name-popup {{ text-align: center; font-weight: 600; font-size: 12px; color: #2c3e50; margin-bottom: 6px; background: #f0f2f5; padding: 4px 10px; border-radius: 4px; }}
        
        .footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #ecf0f1; color: #95a5a6; font-size: 13px; text-align: center; }}
        
        .legend {{ margin: 15px 0 20px 0; font-size: 13px; display: flex; gap: 20px; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-dot {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; }}
        
        @media (max-width: 768px) {{
            .stats {{ grid-template-columns: repeat(2, 1fr); }}
            .details-table {{ font-size: 12px; }}
            .thumb-popup {{ max-width: 300px; min-width: 200px; }}
            .thumb-popup .thumb-img {{ max-width: 100px; max-height: 80px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 分类结果审核报告</h1>
        <div class="subtitle">
            生成时间: {timestamp} &nbsp;|&nbsp; 
            总计: {total} 个样本 &nbsp;|&nbsp;
            分类优先级: 审核修改 > 人工标注 > 模型预测
        </div>
        
        <div class="stats">{stat_cards}</div>
        
        <div class="legend">
            <span class="legend-item"><span class="legend-dot" style="background:#2ecc71;"></span> 已确认</span>
            <span class="legend-item"><span class="legend-dot" style="background:#f39c12;"></span> 已修改</span>
            <span class="legend-item"><span class="legend-dot" style="background:#bdc3c7;"></span> 待审核</span>
            <span style="color:#7f8c8d;">💡 点击类别标题展开/收起</span>
            <span style="color:#7f8c8d;">🖱️ 悬停 "查看图片" 预览 before/after/diff</span>
            <span style="color:#e74c3c;">🔴 render_failed 和 to_review 默认展开</span>
        </div>
        
        {class_sections}
        
        <div class="footer">
            报告由 Report Generator 自动生成 &nbsp;|&nbsp; 
            数据来源: 06_final_output (预测) + 02_annotated (标注) + 07_review (审核)
        </div>
    </div>
    
    <div class="thumb-popup" id="thumbPopup">
        <div class="sample-name-popup" id="popupName">样本名</div>
        <div class="thumb-row">
            <div><div class="thumb-label">BEFORE</div><img class="thumb-img" id="popupBefore" src="" alt="before" /></div>
            <span class="thumb-arrow">→</span>
            <div><div class="thumb-label">AFTER</div><img class="thumb-img" id="popupAfter" src="" alt="after" /></div>
            <span class="thumb-arrow">→</span>
            <div><div class="thumb-label">DIFF</div><img class="thumb-img" id="popupDiff" src="" alt="diff" /></div>
        </div>
    </div>
    
    <script>
        function toggleClass(cls) {{
            var body = document.getElementById('body-' + cls);
            var icon = document.getElementById('icon-' + cls);
            if (body.style.display === 'none') {{
                body.style.display = 'block';
                icon.textContent = '▼';
            }} else {{
                body.style.display = 'none';
                icon.textContent = '▶';
            }}
        }}
        
        var popup = document.getElementById('thumbPopup');
        var popupBefore = document.getElementById('popupBefore');
        var popupAfter = document.getElementById('popupAfter');
        var popupDiff = document.getElementById('popupDiff');
        var popupName = document.getElementById('popupName');
        var hideTimeout = null;
        
        function showThumbPopup(event, name, beforeSrc, afterSrc, diffSrc) {{
            if (hideTimeout) {{ clearTimeout(hideTimeout); hideTimeout = null; }}
            popupName.textContent = name;
            popupBefore.src = beforeSrc || '';
            popupAfter.src = afterSrc || '';
            popupDiff.src = diffSrc || '';
            
            var rect = event.target.getBoundingClientRect();
            var left = rect.left + rect.width / 2 - 200;
            var top = rect.bottom + 10;
            if (left < 10) left = 10;
            if (left + 400 > window.innerWidth - 10) left = window.innerWidth - 410;
            if (top + 200 > window.innerHeight - 10) {{ top = rect.top - 220; }}
            
            popup.style.left = left + 'px';
            popup.style.top = top + 'px';
            popup.classList.add('show');
        }}
        
        function hideThumbPopup() {{
            hideTimeout = setTimeout(function() {{
                popup.classList.remove('show');
                hideTimeout = null;
            }}, 200);
        }}
        
        popup.addEventListener('mouseenter', function() {{
            if (hideTimeout) {{ clearTimeout(hideTimeout); hideTimeout = null; }}
        }});
        popup.addEventListener('mouseleave', function() {{ hideThumbPopup(); }});
    </script>
</body>
</html>
        '''.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total=total,
            stat_cards=stat_cards,
            class_sections=''.join(class_sections)
        )
        
        return html
    
    def _get_class_sources(self, items):
        """获取类别中样本的数据来源分布"""
        sources = defaultdict(int)
        for info in items:
            sources[info.get('source', 'unknown')] += 1
        parts = []
        for src, count in sources.items():
            src_display = {'review': '审核', 'annotated': '标注', 'prediction': '预测'}.get(src, src)
            parts.append(f"{src_display}:{count}")
        return ' | '.join(parts)
    
    def _build_stat_cards(self, class_counts, total):
        """构建统计卡片"""
        cards = []
        # 总样本
        cards.append('''
        <div class="stat-card blue">
            <div class="number">{total}</div>
            <div class="label">📊 总样本</div>
        </div>
        '''.format(total=total))
        
        # 各类别
        for cls in ['render_failed', 'to_review', 'quality_degradation', 
                    'uncertain_difference', 'quality_improved', 'matched']:
            count = class_counts.get(cls, 0)
            if count == 0:
                continue
            color = LABEL_COLORS.get(cls, '#3498db')
            display = LABEL_DISPLAY.get(cls, cls)
            cards.append('''
            <div class="stat-card" style="border-color: {color};">
                <div class="number">{count}</div>
                <div class="label">{display}</div>
            </div>
            '''.format(color=color, count=count, display=display))
        
        return ''.join(cards)
    
    def _generate_thumbnails(self, grouped, report_files_dir):
        """生成所有缩略图"""
        total_items = sum(len(items) for items in grouped.values())
        print(f"  生成 {total_items} 个样本的缩略图...")
        
        for cls, items in grouped.items():
            for info in items:
                name = info['name']
                
                # 查找源路径（按优先级）
                source_dir = None
                
                # 从 final_output 找
                for pred_cls in ALL_CLASSES:
                    path = self.output_dir / pred_cls / name
                    if path.exists():
                        source_dir = path
                        break
                
                # 从 annotated 找
                if not source_dir:
                    for ann_cls in ALL_CLASSES:
                        path = self.annotated_dir / ann_cls / name
                        if path.exists():
                            source_dir = path
                            break
                
                if source_dir:
                    before_path = source_dir / 'before.png'
                    after_path = source_dir / 'after.png'
                    
                    # 复制 before
                    if before_path.exists():
                        dest = report_files_dir / f"{name}_before.png"
                        shutil.copy2(before_path, dest)
                    
                    # 复制 after
                    if after_path.exists():
                        dest = report_files_dir / f"{name}_after.png"
                        shutil.copy2(after_path, dest)
                    
                    # 生成 diff
                    if before_path.exists() and after_path.exists():
                        diff_data = self._compute_diff_image(str(before_path), str(after_path))
                        if diff_data:
                            diff_dest = report_files_dir / f"{name}_diff.png"
                            with open(diff_dest, 'wb') as f:
                                f.write(diff_data)
    
    def _compute_diff_image(self, before_path, after_path):
        """计算差异图"""
        try:
            before = cv2.imread(before_path)
            after = cv2.imread(after_path)
            if before is None or after is None:
                return None
            if before.shape != after.shape:
                h = min(before.shape[0], after.shape[0])
                w = min(before.shape[1], after.shape[1])
                before = cv2.resize(before, (w, h))
                after = cv2.resize(after, (w, h))
            diff = cv2.absdiff(before, after)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            _, buffer = cv2.imencode('.png', diff_colored)
            return buffer.tobytes()
        except Exception as e:
            return None
    
    def _build_thumb_html(self, name, report_files_dir):
        """构建缩略图悬停HTML"""
        before_src = f"report_files/{name}_before.png"
        after_src = f"report_files/{name}_after.png"
        diff_src = f"report_files/{name}_diff.png"
        
        # 检查文件是否存在
        if not (report_files_dir / f"{name}_before.png").exists():
            before_src = ""
        if not (report_files_dir / f"{name}_after.png").exists():
            after_src = ""
        if not (report_files_dir / f"{name}_diff.png").exists():
            diff_src = ""
        
        name_escaped = name.replace("'", "\\'")
        
        return '''
        <span class="thumb-trigger" 
              onmouseenter="showThumbPopup(event, '{name}', '{before}', '{after}', '{diff}')"
              onmouseleave="hideThumbPopup()">
            🖼️ 查看图片
        </span>
        '''.format(name=name_escaped, before=before_src, after=after_src, diff=diff_src)


# ============================================================
# 便捷函数
# ============================================================

def generate_report(project_dir):
    """生成报告的便捷函数"""
    generator = ReportGenerator(project_dir)
    generator.collect_samples()
    generator.generate_html_report()
    return generator


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        project_dir = "./project"
    generate_report(project_dir)