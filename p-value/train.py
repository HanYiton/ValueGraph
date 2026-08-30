from model import BotRGCN, BotGCN, BotGAT
from model_512 import BotRGCN_512, BotGCN_512, BotGAT_512
from Dataset import Twibot22
import torch
from torch import nn
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
import json
import os

device = 'cuda:0'
embedding_size,dropout,lr,weight_decay=32,0.1,1e-2,5e-2


def train(model, optimizer, loss_fn, des_tensor, tweets_tensor, num_prop, category_prop, 
          edge_index, edge_type, labels, train_idx, val_idx, epoch):
    """训练函数"""
    model.train()
    output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
    loss_train = loss_fn(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train, loss_train

def test(model, loss_fn, des_tensor, tweets_tensor, num_prop, category_prop,
         edge_index, edge_type, labels, test_idx):
    """测试函数"""
    model.eval()
    output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
    loss_test = loss_fn(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    output_pred = output.max(1)[1][test_idx].to('cpu').detach().numpy()
    output_proba = torch.softmax(output[test_idx], dim=1)[:, 1].to('cpu').detach().numpy()
    label = labels[test_idx].to('cpu').detach().numpy()
    f1 = f1_score(label, output_pred)
    #mcc=matthews_corrcoef(label, output_pred)
    precision = precision_score(label, output_pred)
    recall = recall_score(label, output_pred)
    fpr, tpr, thresholds = roc_curve(label, output_proba, pos_label=1)
    Auc = auc(fpr, tpr)
    
    results = {
        'test_loss': loss_test.item(),
        'test_accuracy': acc_test.item(),
        'precision': precision.item() if isinstance(precision, np.ndarray) else precision,
        'recall': recall.item() if isinstance(recall, np.ndarray) else recall,
        'f1_score': f1.item() if isinstance(f1, np.ndarray) else f1,
        'auc': Auc.item() if isinstance(Auc, np.ndarray) else Auc,
    }
    
    print("Test set results:",
            "test_loss= {:.4f}".format(results['test_loss']),
            "test_accuracy= {:.4f}".format(results['test_accuracy']),
            "precision= {:.4f}".format(results['precision']),
            "recall= {:.4f}".format(results['recall']),
            "f1_score= {:.4f}".format(results['f1_score']),
            #"mcc= {:.4f}".format(mcc.item()),
            "auc= {:.4f}".format(results['auc']),
            )
    return results

def load_data(root_path):
    """加载数据"""
    # 如果路径是相对路径，基于脚本所在目录解析
    if not os.path.isabs(root_path):
        # 获取脚本文件所在的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.join(script_dir, root_path)
    
    # 转换为绝对路径并规范化
    root_path = os.path.abspath(os.path.normpath(root_path))
    if not root_path.endswith(os.sep):
        root_path = root_path + os.sep
    
    # 检查必要的文件是否存在（Dataset类使用 root + 'label.pt' 的方式拼接路径）
    label_path = root_path + 'label.pt'
    if not os.path.exists(label_path):
        # 也尝试用 os.path.join 的方式检查
        label_path_alt = os.path.join(root_path, 'label.pt')
        if not os.path.exists(label_path_alt):
            print(f"错误：在 {root_path} 中找不到 label.pt 文件")
            print(f"请确保数据已经预处理")
            print(f"尝试查找的文件路径1: {label_path}")
            print(f"尝试查找的文件路径2: {label_path_alt}")
            raise FileNotFoundError(f"缺少必要文件: label.pt 不在 {root_path} 目录中")
        else:
            label_path = label_path_alt
    
    print(f"  数据路径: {root_path}")
    print(f"  找到 label.pt 文件: {label_path}")
    
    # 如果数据已经预处理，使用 process=False 直接加载
    # 注意：Dataset 类在 process=False 时不会创建 df_data_labeled，所以 label.pt 必须存在
    # Dataset类的load_labels()方法使用 self.root + 'label.pt' 来检查路径，所以确保root以/结尾
    try:
        dataset = Twibot22(root=root_path, device=device, process=False, save=False)
        des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx = dataset.dataloader()
        print(f"  ✓ 数据加载成功")
        return des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx
    except AttributeError as e:
        if 'df_data_labeled' in str(e):
            print(f"错误：数据集加载失败，Dataset类尝试访问 df_data_labeled 属性")
            print(f"这通常意味着 label.pt 文件在Dataset类内部无法找到")
            print(f"Dataset使用的root路径: {root_path}")
            print(f"提示：请确保 {root_path} 目录中包含所有预处理文件（label.pt, des_tensor.pt等）")
            raise FileNotFoundError(f"数据加载失败，label.pt文件路径问题: {e}")
        else:
            raise
    
def calculate_single_metric_pvalue(values1, values2, metric_name, source1_name, source2_name):
    """
    计算单个指标的统计显著性（P值）
    
    参数:
        values1: 数据源1的指标值数组
        values2: 数据源2的指标值数组
        metric_name: 指标名称
        source1_name: 数据源1的名称
        source2_name: 数据源2的名称
    
    返回:
        包含该指标所有统计信息的字典
    """
    # 确保长度一致
    min_len = min(len(values1), len(values2))
    values1 = np.array(values1[:min_len])
    values2 = np.array(values2[:min_len])
    
    if len(values1) < 2 or len(values2) < 2:
        print(f"警告：{metric_name} 至少需要2次运行才能计算统计显著性")
        return None
    
    # 计算统计量
    mean1 = float(np.mean(values1))
    mean2 = float(np.mean(values2))
    std1 = float(np.std(values1))
    std2 = float(np.std(values2))
    mean_diff = mean1 - mean2
    
    # 配对t检验
    pvalue_t = np.nan
    t_stat = np.nan
    try:
        t_stat, pvalue_t = ttest_rel(values1, values2)
        pvalue_t = float(pvalue_t)
        t_stat = float(t_stat)
    except Exception as e:
        print(f"  警告：{metric_name} 配对t检验失败: {e}")
    
    # Wilcoxon符号秩检验（非参数）
    pvalue_w = np.nan
    w_stat = np.nan
    try:
        w_stat, pvalue_w = wilcoxon(values1, values2, alternative='two-sided')
        pvalue_w = float(pvalue_w)
        w_stat = float(w_stat)
    except Exception as e:
        print(f"  警告：{metric_name} Wilcoxon检验失败: {e}")
    
    result = {
        'metric': metric_name,
        'source1_name': source1_name,
        'source2_name': source2_name,
        'source1_mean': mean1,
        'source2_mean': mean2,
        'source1_std': std1,
        'source2_std': std2,
        'mean_diff': mean_diff,
        't_statistic': t_stat,
        't_test_pvalue': pvalue_t,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': pvalue_w,
        'n_runs': len(values1),
        'source1_values': values1.tolist(),
        'source2_values': values2.tolist()
    }
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"指标: {metric_name}")
    print(f"{'='*60}")
    print(f"  {source1_name}: {mean1:.4f} ± {std1:.4f}")
    print(f"  {source2_name}: {mean2:.4f} ± {std2:.4f}")
    print(f"  均值差: {mean_diff:.4f}")
    print(f"  配对t检验:")
    print(f"    t统计量: {t_stat:.6f}")
    print(f"    P值: {pvalue_t:.6f}")
    print(f"  Wilcoxon符号秩检验:")
    print(f"    W统计量: {w_stat:.6f}")
    print(f"    P值: {pvalue_w:.6f}")
    print(f"  运行次数: {len(values1)}")
    
    return result


def calculate_pvalues_two_sources(results_source1, results_source2, source1_name, source2_name):
    """
    计算两个数据源之间的统计显著性（p值）
    results_source1: 数据源1的多次运行结果列表
    results_source2: 数据源2的多次运行结果列表
    source1_name: 数据源1的名称
    source2_name: 数据源2的名称
    
    返回: 统计结果字典，包含p值
    """
    if len(results_source1) < 2 or len(results_source2) < 2:
        print("警告：每个数据源至少需要2次运行才能计算统计显著性")
        return None
    
    if len(results_source1) != len(results_source2):
        print(f"警告：两个数据源的运行次数不一致 ({len(results_source1)} vs {len(results_source2)})")
        print("将使用较少的运行次数进行比较")
        min_runs = min(len(results_source1), len(results_source2))
        results_source1 = results_source1[:min_runs]
        results_source2 = results_source2[:min_runs]
    
    # 提取各指标
    metrics = ['test_accuracy', 'f1_score', 'precision', 'recall', 'auc']
    
    # 计算每个数据源的统计量
    stats_source1 = {}
    stats_source2 = {}
    all_metrics_source1 = {}
    all_metrics_source2 = {}
    
    for metric in metrics:
        values1 = [r[metric] for r in results_source1 if metric in r]
        values2 = [r[metric] for r in results_source2 if metric in r]
        
        if len(values1) > 0:
            all_metrics_source1[metric] = np.array(values1)
            stats_source1[metric] = {
                'mean': float(np.mean(values1)),
                'std': float(np.std(values1)),
                'values': values1
            }
        
        if len(values2) > 0:
            all_metrics_source2[metric] = np.array(values2)
            stats_source2[metric] = {
                'mean': float(np.mean(values2)),
                'std': float(np.std(values2)),
                'values': values2
            }
    
    # 进行配对检验 - 分别计算每个指标的P值
    pvalues = {}
    metric_results = {}  # 存储每个指标的详细结果
    
    print(f"\n{'='*80}")
    print(f"计算两个数据源的统计显著性比较: {source1_name} vs {source2_name}")
    print(f"{'='*80}")
    
    for metric in metrics:
        if metric in all_metrics_source1 and metric in all_metrics_source2:
            values1 = all_metrics_source1[metric]
            values2 = all_metrics_source2[metric]
            
            # 使用单独的函数计算单个指标的P值
            metric_result = calculate_single_metric_pvalue(
                values1, values2, metric, source1_name, source2_name
            )
            
            if metric_result:
                metric_results[metric] = metric_result
                
                # 保存到pvalues字典中（保持向后兼容）
                pvalues[f'{metric}_t_test'] = metric_result['t_test_pvalue']
                pvalues[f'{metric}_wilcoxon'] = metric_result['wilcoxon_pvalue']
                pvalues[f'{metric}_mean_diff'] = metric_result['mean_diff']
                pvalues[f'{metric}_{source1_name}_mean'] = metric_result['source1_mean']
                pvalues[f'{metric}_{source2_name}_mean'] = metric_result['source2_mean']
    
    return {
        'source1_name': source1_name,
        'source2_name': source2_name,
        'source1_statistics': stats_source1,
        'source2_statistics': stats_source2,
        'pvalues': pvalues,
        'metric_results': metric_results,  # 新增：每个指标的详细结果
        'n_runs': len(results_source1)
    }


def run_single_experiment(des_tensor, tweets_tensor, num_prop, category_prop, 
                          edge_index, edge_type, labels, train_idx, val_idx, test_idx,
                          run_id, random_seed=None, epochs=200, source_name="", model_class=BotRGCN):
    """运行单次实验"""
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    
    print(f"\n{'='*60}")
    print(f"数据源: {source_name} | 运行 #{run_id + 1} (random_seed={random_seed})")
    print(f"{'='*60}")
    
    # 初始化模型和优化器
    model = model_class(cat_prop_size=3, embedding_dimension=embedding_size).to(device)
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    # 训练
    for epoch in range(epochs):
        train(model, optimizer, loss_fn, des_tensor, tweets_tensor, num_prop, 
              category_prop, edge_index, edge_type, labels, train_idx, val_idx, epoch)
    
    # 测试
    results = test(model, loss_fn, des_tensor, tweets_tensor, num_prop, 
                   category_prop, edge_index, edge_type, labels, test_idx)
    return results


def save_statistics_results(stats_results, output_dir='./results'):
    """保存两个数据源比较的统计结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON格式的详细结果
    with open(os.path.join(output_dir, 'two_source_comparison_results.json'), 'w', encoding='utf-8') as f:
        json.dump(stats_results, f, indent=2, ensure_ascii=False)
    
    # 保存CSV格式的P值汇总结果
    if stats_results and 'pvalues' in stats_results:
        metrics = ['test_accuracy', 'f1_score', 'precision', 'recall', 'auc']
        summary_data = []
        
        # 优先使用metric_results中的详细数据
        if 'metric_results' in stats_results and stats_results['metric_results']:
            for metric in metrics:
                if metric in stats_results['metric_results']:
                    metric_result = stats_results['metric_results'][metric]
                    row = {
                        '指标': metric,
                        f'{stats_results["source1_name"]}_均值': f"{metric_result['source1_mean']:.6f}",
                        f'{stats_results["source2_name"]}_均值': f"{metric_result['source2_mean']:.6f}",
                        f'{stats_results["source1_name"]}_标准差': f"{metric_result['source1_std']:.6f}",
                        f'{stats_results["source2_name"]}_标准差': f"{metric_result['source2_std']:.6f}",
                        '均值差': f"{metric_result['mean_diff']:.6f}",
                        '配对t检验_t统计量': f"{metric_result['t_statistic']:.6f}" if not np.isnan(metric_result['t_statistic']) else 'N/A',
                        '配对t检验_P值': f"{metric_result['t_test_pvalue']:.6f}" if not np.isnan(metric_result['t_test_pvalue']) else 'N/A',
                        'Wilcoxon检验_W统计量': f"{metric_result['wilcoxon_statistic']:.6f}" if not np.isnan(metric_result['wilcoxon_statistic']) else 'N/A',
                        'Wilcoxon检验_P值': f"{metric_result['wilcoxon_pvalue']:.6f}" if not np.isnan(metric_result['wilcoxon_pvalue']) else 'N/A',
                        '运行次数': metric_result['n_runs']
                    }
                    summary_data.append(row)
        else:
            # 向后兼容：如果没有metric_results，使用pvalues
            for metric in metrics:
                if f'{metric}_mean_diff' in stats_results['pvalues']:
                    row = {
                        '指标': metric,
                        f'{stats_results["source1_name"]}_均值': stats_results['pvalues'].get(f'{metric}_{stats_results["source1_name"]}_mean', 'N/A'),
                        f'{stats_results["source2_name"]}_均值': stats_results['pvalues'].get(f'{metric}_{stats_results["source2_name"]}_mean', 'N/A'),
                        '均值差': stats_results['pvalues'].get(f'{metric}_mean_diff', 'N/A'),
                        '配对t检验_P值': f"{stats_results['pvalues'].get(f'{metric}_t_test', np.nan):.6f}" if not np.isnan(stats_results['pvalues'].get(f'{metric}_t_test', np.nan)) else 'N/A',
                        'Wilcoxon检验_P值': f"{stats_results['pvalues'].get(f'{metric}_wilcoxon', np.nan):.6f}" if not np.isnan(stats_results['pvalues'].get(f'{metric}_wilcoxon', np.nan)) else 'N/A',
                        '运行次数': stats_results['n_runs']
                    }
                    summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(output_dir, 'botrgcn_two_source_comparison_pvalues.csv'), index=False, encoding='utf-8-sig')
            print(f"\n✓ 统计结果已保存到 {output_dir}/")
            print(f"  - botrgcn_two_source_comparison_results.json (详细结果)")
            print(f"  - botrgcn_two_source_comparison_pvalues.csv (所有指标的P值汇总)")


# ============ 主程序 ============
# 配置参数
NUM_RUNS = 10  # 每个数据源的运行次数，用于计算统计显著性
EPOCHS = 200  # 训练轮数

# ============ 配置两个数据源 ============
# 数据源1配置
DATA_SOURCE_1_ROOT = './processed_data/'  # 修改为第一个数据源的路径
DATA_SOURCE_1_NAME = '数据源1'

# 数据源2配置  
DATA_SOURCE_2_ROOT = './processed_data-2/'  # 修改为第二个数据源的路径
DATA_SOURCE_2_NAME = '数据源2'

# 是否比较两个数据源
COMPARE_TWO_SOURCES = True  # 设置为True比较两个数据源，False则只运行单个数据源

if COMPARE_TWO_SOURCES:
    print("="*80)
    print("开始两个数据源的比较实验")
    print("="*80)
    
    # 加载两个数据源
    print("\n加载数据源1...")
    des_tensor1, tweets_tensor1, num_prop1, category_prop1, edge_index1, edge_type1, labels1, train_idx1, val_idx1, test_idx1 = load_data(DATA_SOURCE_1_ROOT)
    
    print("\n加载数据源2...")
    des_tensor2, tweets_tensor2, num_prop2, category_prop2, edge_index2, edge_type2, labels2, train_idx2, val_idx2, test_idx2 = load_data(DATA_SOURCE_2_ROOT)
    
    # 对每个数据源运行多次实验
    print(f"\n{'='*80}")
    print(f"对数据源1 ({DATA_SOURCE_1_NAME}) 进行 {NUM_RUNS} 次运行...")
    print(f"{'='*80}")
    results_source1 = []
    
    for run_id in range(NUM_RUNS):
        random_seed = 42 + run_id  # 每次使用不同的随机种子
        results = run_single_experiment(
            des_tensor1, tweets_tensor1, num_prop1, category_prop1,
            edge_index1, edge_type1, labels1, train_idx1, val_idx1, test_idx1,
            run_id, random_seed, EPOCHS, DATA_SOURCE_1_NAME
        )
        results_source1.append(results)
    
    print(f"\n{'='*80}")
    print(f"对数据源2 ({DATA_SOURCE_2_NAME}) 进行 {NUM_RUNS} 次运行...")
    print(f"{'='*80}")
    results_source2 = []
    
    for run_id in range(NUM_RUNS):
        random_seed = 42 + run_id  # 使用相同的随机种子序列以进行配对比较
        results = run_single_experiment(
            des_tensor2, tweets_tensor2, num_prop2, category_prop2,
            edge_index2, edge_type2, labels2, train_idx2, val_idx2, test_idx2,
            run_id, random_seed, EPOCHS, DATA_SOURCE_2_NAME, model_class=BotRGCN_512
        )
        results_source2.append(results)
    
    # 计算两个数据源之间的统计显著性（P值）
    print(f"\n{'='*80}")
    print("计算两个数据源的统计显著性比较...")
    print(f"{'='*80}")
    stats_results = calculate_pvalues_two_sources(
        results_source1, results_source2, 
        DATA_SOURCE_1_NAME, DATA_SOURCE_2_NAME
    )
    
    if stats_results:
        # 保存结果
        save_statistics_results(stats_results)
        
        # 打印总结
        print(f"\n{'='*80}")
        print("实验完成！")
        print(f"{'='*80}")
        print(f"\n两个数据源比较的P值已计算并保存。")
        print(f"数据源1: {DATA_SOURCE_1_NAME}")
        print(f"数据源2: {DATA_SOURCE_2_NAME}")
        print(f"运行次数: {NUM_RUNS}")

else:
    # 单数据源模式（原始行为）
    print("="*80)
    print("单数据源运行模式")
    print("="*80)
    
    root = './processed_data/'
    des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx = load_data(root)
    
    if NUM_RUNS > 1:
        # 多次运行模式
        print(f"\n进行 {NUM_RUNS} 次运行...")
        all_results = []
        
        for run_id in range(NUM_RUNS):
            random_seed = 42 + run_id
            results = run_single_experiment(
                des_tensor, tweets_tensor, num_prop, category_prop,
                edge_index, edge_type, labels, train_idx, val_idx, test_idx,
                run_id, random_seed, EPOCHS, "单数据源"
            )
            all_results.append(results)
        
        # 打印汇总
        print("\n" + "="*60)
        print("所有运行结果汇总:")
        print("="*60)
        metrics = ['test_accuracy', 'f1_score', 'precision', 'recall', 'auc']
        for metric in metrics:
            values = [r[metric] for r in all_results if metric in r]
            if values:
                print(f"{metric}: 均值={np.mean(values):.4f}, 标准差={np.std(values):.4f}")
    else:
        # 单次运行模式
        model = BotRGCN(cat_prop_size=3, embedding_dimension=embedding_size).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(EPOCHS):
            train(model, optimizer, loss_fn, des_tensor, tweets_tensor, num_prop,
                  category_prop, edge_index, edge_type, labels, train_idx, val_idx, epoch)
        
        test(model, loss_fn, des_tensor, tweets_tensor, num_prop,
             category_prop, edge_index, edge_type, labels, test_idx)