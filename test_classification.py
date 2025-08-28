"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=41, type=int, choices=[10, 41],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    return parser.parse_args()


# 为ModelNet40定义颜色映射（基于预测正确/错误）
def get_prediction_color(true_label, pred_label):
    """根据预测是否正确返回颜色，正确为绿色，错误为红色"""
    if true_label == pred_label:
        return [0, 255, 0]  # 绿色 - 正确
    else:
        return [255, 0, 0]  # 红色 - 错误


def get_gt_color(true_label):
    """真实标签的颜色（统一用蓝色）"""
    return [0, 0, 255]  # 蓝色


def test(model, loader, num_class=40, vote_num=1, visual_dir=None, logger=None):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    # ModelNet40的类别名称
    class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 
                   'bottle', 'bowl', 'car', 'chair', 'cone', 
                   'cup', 'curtain', 'desk', 'door', 'dresser', 
                   'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 
                   'laptop', 'mantel', 'monitor', 'night_stand', 'person', 
                   'piano', 'plant', 'radio', 'range_hood', 'sink', 
                   'sofa', 'stairs', 'stool', 'table', 'tent', 
                   'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    # 样本计数器
    sample_count = 0

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        # --- 新增代码：为每个样本保存单独的可视化文件 ---
        if args.visual and visual_dir is not None:
            # 将Tensor转换为numpy数组
            points_np = points.transpose(2, 1).cpu().numpy()  # 转换回 (B, N, C) 格式
            target_np = target.cpu().numpy()
            pred_np = pred_choice.cpu().numpy()
            
            # 遍历当前批次中的每个样本
            for i in range(points_np.shape[0]):
                # 获取单个点云、其真实标签和预测标签
                single_pc = points_np[i, ...]  # (N, C)
                true_label = int(target_np[i])
                pred_label = int(pred_np[i])
                
                # 获取类别名称
                true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
                pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
                
                # 获取颜色
                pred_color = get_prediction_color(true_label, pred_label)
                gt_color = get_gt_color(true_label)
                
                # 为每个样本创建单独的文件
                file_prefix = f"sample_{sample_count:04d}_true_{true_name}_pred_{pred_name}"
                pred_filename = os.path.join(visual_dir, file_prefix + '_pred.obj')
                gt_filename = os.path.join(visual_dir, file_prefix + '_gt.obj')
                
                # 写入预测结果文件
                with open(pred_filename, 'w') as pred_fout:
                    for point_idx in range(single_pc.shape[0]):
                        # 只取前3个坐标（x, y, z），忽略法线信息
                        x, y, z = single_pc[point_idx, 0], single_pc[point_idx, 1], single_pc[point_idx, 2]
                        pred_fout.write('v %f %f %f %d %d %d\n' % (
                            x, y, z, pred_color[0], pred_color[1], pred_color[2]))
                
                # 写入真实标签文件
                with open(gt_filename, 'w') as gt_fout:
                    for point_idx in range(single_pc.shape[0]):
                        x, y, z = single_pc[point_idx, 0], single_pc[point_idx, 1], single_pc[point_idx, 2]
                        gt_fout.write('v %f %f %f %d %d %d\n' % (
                            x, y, z, gt_color[0], gt_color[1], gt_color[2]))
                
                # 记录日志
                if logger and sample_count % 100 == 0:  # 每100个样本记录一次
                    status = "CORRECT" if true_label == pred_label else "WRONG"
                    logger.info(f'Sample {sample_count}: True={true_name}, Pred={pred_name} [{status}]')
                
                sample_count += 1
        # --- 新增代码结束 ---

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir
    
    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    def log_string(str):
        logger.info(str)
        print(str)

    log_string('PARAMETER ...')
    log_string(str(args))

    '''VISUAL DIR'''
    visual_dir = os.path.join(experiment_dir, 'visual')
    if args.visual:
        os.makedirs(visual_dir, exist_ok=True)
        log_string(f'Visualization directory: {visual_dir}')

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    # 获取测试数据集大小
    test_size = len(test_dataset)
    log_string(f'The size of test data is: {test_size}')

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, 
                                      vote_num=args.num_votes, 
                                      num_class=num_class,
                                      visual_dir=visual_dir if args.visual else None,
                                      logger=logger if args.visual else None)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        
        if args.visual:
            log_string(f'Visualization files saved to: {visual_dir}')
            log_string(f'Generated {test_size} pairs of .obj files (pred/gt for each sample)')


if __name__ == '__main__':
    args = parse_args()
    main(args)

##比原版多了可视化代码