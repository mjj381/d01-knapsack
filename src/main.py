import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os

# 常量定义
MAX_LINE_LENGTH = 120  # 每行最大字符数
FUNCTION_MAX_LINES = 80  # 函数最大行数

class DataHandler:
    """
    数据处理类：负责数据集读取、排序、结果保存
    """
    def read_dataset(self, file_path: str) -> pd.DataFrame:
        """
        读取D{0-1}KP数据集
        :param file_path: 数据集文件路径（支持txt/excel）
        :return: 解析后的DataFrame，列：item_set_id, w1, v1, w2, v2, w3, v3
        """
        try:
            if file_path.endswith('.txt'):
                # 假设TXT格式：每行是一个项集，格式为 项集ID 物品1重量 物品1价值 物品2重量 物品2价值 物品3重量 物品3价值
                df = pd.read_csv(
                    file_path,
                    sep=r'\s+',  # 原始字符串，避免转义警告
                    names=['item_set_id', 'w1', 'v1', 'w2', 'v2', 'w3', 'v3'],
                    dtype={'item_set_id': int, 'w1': int, 'v1': int, 'w2': int, 'v2': int, 'w3': int, 'v3': int}
                )
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, dtype={'item_set_id': int, 'w1': int, 'v1': int, 'w2': int, 'v2': int, 'w3': int, 'v3': int})
            else:
                raise ValueError("仅支持TXT/Excel格式的数据集")
            
            # 数据校验：检查第三项的价值是否为前两项之和，重量是否小于前两项之和
            df['v3_check'] = df['v1'] + df['v2']
            df['w3_check'] = df['w1'] + df['w2']
            invalid = df[(df['v3'] != df['v3_check']) | (df['w3'] >= df['w3_check'])]
            if not invalid.empty:
                messagebox.showwarning("数据校验警告", f"以下项集不符合D{0-1}KP规则：{invalid['item_set_id'].tolist()}")
            
            return df[['item_set_id', 'w1', 'v1', 'w2', 'v2', 'w3', 'v3']]
        except Exception as e:
            messagebox.showerror("读取失败", f"数据集读取错误：{str(e)}")
            return pd.DataFrame()

    def sort_by_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按项集第三项的价值/重量比非递增排序
        :param df: 原始数据集
        :return: 排序后的数据集
        """
        if df.empty:
            return df
        # 计算第三项的价值重量比（处理除数为0的情况）
        df['ratio_3'] = df['v3'] / df['w3'].replace(0, np.inf)
        df_sorted = df.sort_values(by='ratio_3', ascending=False).reset_index(drop=True)
        return df_sorted

    def save_result(self, result: dict, save_path: str, file_format: str = 'txt') -> None:
        """
        保存求解结果
        :param result: 结果字典，包含max_value、selected_items、solve_time、capacity等
        :param save_path: 保存路径
        :param file_format: 保存格式（txt/excel）
        """
        try:
            if file_format == 'txt':
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write("=== D{0-1}KP最优解结果 ===\n")
                    f.write(f"背包容量：{result['capacity']}\n")
                    f.write(f"最大价值：{result['max_value']}\n")
                    f.write(f"求解耗时：{result['solve_time']:.6f}秒\n")
                    f.write("选中的项集及物品：\n")
                    for item in result['selected_items']:
                        f.write(f"项集ID：{item['item_set_id']}，选中物品：{item['selected_item']}，重量：{item['weight']}，价值：{item['value']}\n")
            elif file_format == 'excel':
                # 拆分结果为基础信息和选中物品
                basic_info = pd.DataFrame({
                    '背包容量': [result['capacity']],
                    '最大价值': [result['max_value']],
                    '求解耗时(秒)': [result['solve_time']]
                })
                selected_items = pd.DataFrame(result['selected_items'])
                
                # 写入Excel的不同sheet
                with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                    basic_info.to_excel(writer, sheet_name='基础结果', index=False)
                    selected_items.to_excel(writer, sheet_name='选中物品', index=False)
            messagebox.showinfo("保存成功", f"结果已保存至：{save_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"结果保存错误：{str(e)}")

class KnapsackSolver:
    """
    背包求解类：实现动态规划/贪心算法求解D{0-1}KP
    """
    def __init__(self, capacity: int, dataset: pd.DataFrame):
        self.capacity = int(capacity)  # 强制转为整数，避免浮点类型
        self.dataset = dataset.copy()  # 复制数据集，避免修改原数据
        self.n = len(dataset)          # 项集数量

    def dynamic_programming(self) -> tuple[float, list, float]:
        """
        动态规划求解D{0-1}KP（空间优化：一维数组）
        核心修复：所有数值强制转为int，避免numpy.float64类型导致的索引错误
        :return: (最大价值, 选中物品列表, 求解耗时)
        """
        start_time = time.time()
        
        if self.n == 0 or self.capacity <= 0:
            return 0.0, [], time.time() - start_time
        
        # 初始化dp数组：dp[j]表示容量j下的最大价值（空间优化：一维数组）
        dp = [0] * (self.capacity + 1)
        # 记录选择路径：path[j] = [项集索引, 选中物品编号, 重量, 价值]
        path = [[None for _ in range(4)] for _ in range(self.capacity + 1)]
        
        # 遍历每个项集
        for i in range(self.n):
            item_set = self.dataset.iloc[i]
            # 关键修复：强制转换为int，避免numpy.float64类型
            items = [
                (int(item_set['w1']), int(item_set['v1'])),
                (int(item_set['w2']), int(item_set['v2'])),
                (int(item_set['w3']), int(item_set['v3']))
            ]
            # 逆序遍历容量，避免重复选择
            for j in range(self.capacity, -1, -1):
                # 遍历当前项集的3个物品，选择最优解
                for k in range(3):
                    w, v = items[k]
                    # 确保j和w都是int，j-w不会出现浮点类型
                    if w <= j and dp[j - w] + v > dp[j]:
                        dp[j] = dp[j - w] + v
                        path[j] = [i, k+1, w, v]  # k+1：物品编号1/2/3
        
        # 回溯路径，找到选中的物品
        current_cap = self.capacity
        selected_items = []
        while current_cap > 0 and path[current_cap][0] is not None:
            i, selected_item, w, v = path[current_cap]
            selected_items.append({
                'item_set_id': int(self.dataset.iloc[i]['item_set_id']),
                'selected_item': selected_item,
                'weight': int(w),
                'value': int(v)
            })
            current_cap -= int(w)  # 强制int，避免浮点误差
        
        # 反转列表，恢复选择顺序
        selected_items.reverse()
        solve_time = time.time() - start_time
        
        return float(dp[self.capacity]), selected_items, solve_time

class Visualization:
    """
    可视化类：绘制散点图、容量曲线等
    """
    def plot_scatter(self, df: pd.DataFrame, selected_items: list = None):
        """
        绘制重量-价值散点图
        :param df: 数据集
        :param selected_items: 选中的物品列表（可选，用于标注）
        """
        if df.empty:
            messagebox.showwarning("无数据", "请先读取有效的数据集！")
            return
        
        # 设置中文字体（避免乱码）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 兼容不同系统
        plt.rcParams['axes.unicode_minus'] = False                      # 解决负号显示问题
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 提取所有物品的重量和价值
        weights = []
        values = []
        labels = []
        for idx, row in df.iterrows():
            # 物品1
            weights.append(int(row['w1']))
            values.append(int(row['v1']))
            labels.append(f"项集{row['item_set_id']}-物品1")
            # 物品2
            weights.append(int(row['w2']))
            values.append(int(row['v2']))
            labels.append(f"项集{row['item_set_id']}-物品2")
            # 物品3
            weights.append(int(row['w3']))
            values.append(int(row['v3']))
            labels.append(f"项集{row['item_set_id']}-物品3")
        
        # 绘制所有物品的散点图
        ax.scatter(weights, values, c='#1f77b4', alpha=0.6, label='所有物品', s=60)
        
        # 标注选中的物品
        if selected_items and len(selected_items) > 0:
            sel_weights = [int(item['weight']) for item in selected_items]
            sel_values = [int(item['value']) for item in selected_items]
            sel_labels = [f"项集{item['item_set_id']}-物品{item['selected_item']}" for item in selected_items]
            ax.scatter(sel_weights, sel_values, c='#d62728', s=100, marker='*', label='选中物品')
            # 添加文本标注
            for w, v, l in zip(sel_weights, sel_values, sel_labels):
                ax.annotate(l, (w, v), xytext=(5, 5), textcoords='offset points', fontsize=9, color='#d62728')
        
        # 设置图表属性
        ax.set_xlabel('重量', fontsize=12, fontweight='bold')
        ax.set_ylabel('价值', fontsize=12, fontweight='bold')
        ax.set_title('D{0-1}KP物品重量-价值散点图', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 显示图表
        plt.tight_layout()
        plt.show()

class MainUI:
    """
    主界面类：创建GUI界面，整合所有功能
    """
    def __init__(self):
        # 初始化核心组件
        self.data_handler = DataHandler()
        self.solver = None
        self.visualization = Visualization()
        self.current_df = pd.DataFrame()  # 当前加载的数据集
        self.current_result = {}         # 当前求解结果
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("D{0-1}KP动态规划求解系统")
        self.root.geometry("850x650")
        self.root.resizable(True, True)  # 允许窗口缩放
        
        # 创建界面组件
        self.create_widgets()
        
        # 启动主循环
        self.root.mainloop()

    def create_widgets(self):
        """创建UI组件（修复Menu组件错误，优化布局）"""
        # 1. 菜单栏（核心修复：使用tk.Menu而非ttk.Menu）
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开数据集", command=self.on_select_file)
        file_menu.add_command(label="保存结果", command=self.on_save_result)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 功能菜单
        func_menu = tk.Menu(menubar, tearoff=0)
        func_menu.add_command(label="绘制散点图", command=self.on_plot_scatter)
        func_menu.add_command(label="按价值重量比排序", command=self.on_sort_data)
        func_menu.add_command(label="求解最优解", command=self.on_solve)
        menubar.add_cascade(label="功能", menu=func_menu)
        
        # 2. 左侧控制面板
        control_frame = ttk.LabelFrame(self.root, text="参数设置", padding=(10, 5))
        control_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        # 背包容量输入
        ttk.Label(control_frame, text="背包容量：").grid(row=0, column=0, padx=5, pady=8, sticky='w')
        self.capacity_entry = ttk.Entry(control_frame, width=12)
        self.capacity_entry.grid(row=0, column=1, padx=5, pady=8)
        self.capacity_entry.insert(0, "100")  # 默认值
        
        # 功能按钮
        btn_width = 18
        ttk.Button(control_frame, text="读取数据集", command=self.on_select_file, width=btn_width).grid(row=1, column=0, columnspan=2, padx=5, pady=8)
        ttk.Button(control_frame, text="绘制散点图", command=self.on_plot_scatter, width=btn_width).grid(row=2, column=0, columnspan=2, padx=5, pady=8)
        ttk.Button(control_frame, text="数据排序", command=self.on_sort_data, width=btn_width).grid(row=3, column=0, columnspan=2, padx=5, pady=8)
        ttk.Button(control_frame, text="求解最优解", command=self.on_solve, width=btn_width).grid(row=4, column=0, columnspan=2, padx=5, pady=8)
        ttk.Button(control_frame, text="保存结果", command=self.on_save_result, width=btn_width).grid(row=5, column=0, columnspan=2, padx=5, pady=8)
        
        # 3. 右侧结果展示区
        result_frame = ttk.LabelFrame(self.root, text="结果展示", padding=(10, 5))
        result_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # 滚动文本框（优化显示）
        scrollbar = ttk.Scrollbar(result_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.result_text = tk.Text(result_frame, wrap='word', yscrollcommand=scrollbar.set, font=('Consolas', 10), bg='#f8f9fa')
        self.result_text.pack(fill='both', expand=True, padx=5, pady=5)
        scrollbar.config(command=self.result_text.yview)
        
        # 初始提示文本
        self.result_text.insert('end', "===== D{0-1}KP动态规划求解系统 =====\n\n操作步骤：\n1. 点击【读取数据集】选择TXT/Excel文件\n2. 输入背包容量（正整数）\n3. 点击【求解最优解】计算结果\n4. 可绘制散点图/排序数据/保存结果\n")
        self.result_text.config(state='disabled')  # 初始只读

    def update_result_text(self, content: str):
        """更新结果展示区文本（安全更新，避免编辑）"""
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, content)
        self.result_text.config(state='disabled')

    def on_select_file(self):
        """选择并读取数据集"""
        file_path = filedialog.askopenfilename(
            title="选择D{0-1}KP数据集",
            filetypes=[("数据文件", "*.txt *.xlsx *.xls"), ("所有文件", "*.*")],
            initialdir=os.path.join(os.getcwd(), "datasets")  # 默认打开datasets文件夹
        )
        if file_path:
            self.current_df = self.data_handler.read_dataset(file_path)
            if not self.current_df.empty:
                self.update_result_text(f"✅ 成功读取数据集：{os.path.basename(file_path)}\n📊 项集数量：{len(self.current_df)}\n\n数据集前5行：\n{self.current_df.head().to_string()}")
                self.current_result = {}  # 清空之前的求解结果

    def on_plot_scatter(self):
        """绘制散点图"""
        if self.current_df.empty:
            messagebox.showwarning("无数据", "请先读取有效的数据集！")
            return
        # 绘制散点图（若有求解结果，标注选中物品）
        self.visualization.plot_scatter(self.current_df, self.current_result.get('selected_items'))

    def on_sort_data(self):
        """按第三项价值重量比排序"""
        if self.current_df.empty:
            messagebox.showwarning("无数据", "请先读取有效的数据集！")
            return
        self.current_df = self.data_handler.sort_by_ratio(self.current_df)
        # 显示排序后的前5行数据
        self.update_result_text("✅ 数据集已按项集第三项的价值/重量比降序排序\n\n排序后前5行数据：\n" + self.current_df.head(10).to_string())

    def on_solve(self):
        """求解最优解"""
        if self.current_df.empty:
            messagebox.showwarning("无数据", "请先读取有效的数据集！")
            return
        
        # 获取并校验背包容量
        try:
            capacity_input = self.capacity_entry.get().strip()
            if not capacity_input:
                raise ValueError("容量不能为空")
            capacity = int(capacity_input)
            if capacity <= 0:
                raise ValueError("容量必须为正整数")
        except ValueError as e:
            messagebox.showerror("参数错误", f"背包容量输入无效：{str(e)}\n请输入正整数！")
            return
        
        # 执行动态规划求解
        self.solver = KnapsackSolver(capacity, self.current_df)
        max_value, selected_items, solve_time = self.solver.dynamic_programming()
        
        # 保存结果
        self.current_result = {
            'capacity': capacity,
            'max_value': max_value,
            'selected_items': selected_items,
            'solve_time': solve_time
        }
        
        # 显示求解结果
        result_text = f"=== D{0-1}KP最优解求解结果 ===\n"
        result_text += f"🎒 背包容量：{capacity}\n"
        result_text += f"💰 最大价值：{max_value:.0f}\n"
        result_text += f"⏱️ 求解耗时：{solve_time:.6f}秒\n"
        result_text += f"📌 选中项集数量：{len(selected_items)}\n\n"
        
        if selected_items:
            result_text += "📋 选中物品详情：\n"
            total_weight = 0
            for idx, item in enumerate(selected_items, 1):
                result_text += f"{idx}. 项集ID：{item['item_set_id']} | 物品：{item['selected_item']} | 重量：{item['weight']} | 价值：{item['value']}\n"
                total_weight += item['weight']
            result_text += f"\n📊 选中物品总重量：{total_weight}（≤ 背包容量{capacity}）"
        else:
            result_text += "⚠️ 无可用物品可选中（容量过小或无有效物品）"
        
        self.update_result_text(result_text)

    def on_save_result(self):
        """保存求解结果"""
        if not self.current_result:
            messagebox.showwarning("无结果", "请先求解最优解！")
            return
        
        # 选择保存路径和格式
        save_path = filedialog.asksaveasfilename(
            title="保存求解结果",
            defaultextension=".txt",
            filetypes=[("TXT文本文件", "*.txt"), ("Excel文件", "*.xlsx")],
            initialdir=os.path.join(os.getcwd(), "results")  # 默认保存到results文件夹
        )
        if save_path:
            # 创建results文件夹（如果不存在）
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 判断保存格式
            file_format = 'txt' if save_path.endswith('.txt') else 'excel'
            self.data_handler.save_result(self.current_result, save_path, file_format)

# 程序入口
if __name__ == "__main__":
    # 检查依赖库
    required_libs = {
        'pandas': '数据处理',
        'matplotlib': '绘图可视化',
        'openpyxl': 'Excel文件读写',
        'numpy': '数值计算'
    }
    missing_libs = []
    
    for lib, desc in required_libs.items():
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(f"{lib}（{desc}）")
    
    if missing_libs:
        print(f"❌ 缺少以下依赖库：{', '.join(missing_libs)}")
        print("📦 请执行以下命令安装：")
        print("pip install pandas matplotlib openpyxl numpy")
    else:
        # 创建必要的文件夹（如果不存在）
        for folder in ['datasets', 'results']:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"📁 创建文件夹：{folder}")
        
        # 启动程序
        print("🚀 启动D{0-1}KP动态规划求解系统...")
        MainUI()