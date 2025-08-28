#!/usr/bin/env python3
"""
RealSense D435i + PointNet2 实时点云分类器 - 带地面分割和点云可视化
"""

import argparse
import time
import numpy as np
import cv2
import torch
import open3d as o3d
import pyrealsense2 as rs
import threading
from queue import Queue
import sys

# 添加项目路径
sys.path.append('/home/rong/Pointnet_Pointnet2_pytorch')

from models.pointnet2_cls_ssg import get_model

# ModelNet10 类别名称
class_names = [
    'bed', 'bookshelf', 'car', 'chair', 'cup',
    'desk', 'person', 'sofa', 'table', 'tennis_ball'
]

class RealSenseCamera:
    """RealSense相机封装类，提供更稳定的连接"""
    
    def __init__(self):
        self.pipeline = None
        self.align = None
        self.pc = None
        self.running = False
        self.frame_queue = Queue(maxsize=1)
        
    def initialize(self, max_retries=5):
        """初始化相机，支持重试机制"""
        for attempt in range(max_retries):
            try:
                print(f"尝试初始化相机 (尝试 {attempt + 1}/{max_retries})...")
                
                # 检查设备
                ctx = rs.context()
                if len(ctx.query_devices()) == 0:
                    print("未检测到RealSense设备")
                    time.sleep(1)
                    continue
                
                # 创建新的pipeline
                if self.pipeline:
                    try:
                        self.pipeline.stop()
                    except:
                        pass
                
                self.pipeline = rs.pipeline()
                config = rs.config()
                
                # 配置流 - 使用较低分辨率提高稳定性
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                # 启动pipeline
                pipeline_profile = self.pipeline.start(config)
                
                # 创建对齐对象
                align_to = rs.stream.color
                self.align = rs.align(align_to)
                
                # 创建点云对象
                self.pc = rs.pointcloud()
                
                # 等待几帧让相机稳定
                for _ in range(10):
                    self.pipeline.wait_for_frames()
                
                print("相机初始化成功!")
                self.running = True
                return True
                
            except Exception as e:
                print(f"相机初始化失败 (尝试 {attempt + 1}): {e}")
                if self.pipeline:
                    try:
                        self.pipeline.stop()
                    except:
                        pass
                time.sleep(0.5)
        
        return False
    
    def start_capture(self):
        """启动相机捕获线程"""
        def capture_thread():
            while self.running:
                try:
                    # 设置较短超时时间
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    aligned_frames = self.align.process(frames)
                    
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    
                    if depth_frame and color_frame:
                        # 清空队列并放入新帧
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except:
                                pass
                        self.frame_queue.put((depth_frame, color_frame))
                        
                except RuntimeError as e:
                    if "Frame didn't arrive" in str(e):
                        print("帧超时，尝试重新初始化...")
                        if not self.initialize():
                            break
                    else:
                        print(f"捕获错误: {e}")
                        time.sleep(0.1)
                except Exception as e:
                    print(f"捕获线程错误: {e}")
                    time.sleep(0.1)
        
        self.capture_thread = threading.Thread(target=capture_thread, daemon=True)
        self.capture_thread.start()
    
    def get_frames(self):
        """获取最新的帧"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None, None
    
    def stop(self):
        """停止相机"""
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass

def segment_ground_plane(points, distance_threshold=0.015, ransac_n=3, num_iterations=1000):
    """
    使用RANSAC算法分割地面平面
    """
    if len(points) < 10:
        return points, np.array([])
    
    try:
        # 转换为Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 使用RANSAC分割平面
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        # 提取非地面点（物体）
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        object_points = np.asarray(outlier_cloud.points)
        
        return object_points, np.asarray(inliers)
        
    except Exception as e:
        print(f"地面分割错误: {e}")
        return points, np.array([])

def remove_outliers(points, nb_neighbors=20, std_ratio=2.0):
    """
    移除离群点
    """
    if len(points) < 10:
        return points
    
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 统计离群点移除
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        
        filtered_pcd = pcd.select_by_index(ind)
        return np.asarray(filtered_pcd.points)
        
    except Exception as e:
        print(f"离群点移除错误: {e}")
        return points

def calculate_pointcloud_bounds(points):
    """
    计算点云在X、Y、Z三个方向上的坐标范围
    
    参数:
        points: numpy数组，形状为(N, 3)，包含点云的XYZ坐标
        
    返回:
        bounds_dict: 包含各轴最小值和最大值的字典
    """
    if len(points) == 0:
        return {
            'x_min': 0, 'x_max': 0,
            'y_min': 0, 'y_max': 0,
            'z_min': 0, 'z_max': 0
        }
    
    try:
        # 计算各轴的最小值和最大值
        x_min, y_min, z_min = np.min(points, axis=0)
        x_max, y_max, z_max = np.max(points, axis=0)
        
        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max
        }
    except Exception as e:
        print(f"计算点云范围错误: {e}")
        return {
            'x_min': 0, 'x_max': 0,
            'y_min': 0, 'y_max': 0,
            'z_min': 0, 'z_max': 0
        }

def preprocess_point_cloud(points, num_points=1024):
    """优化版点云预处理"""
    if len(points) < 50:  # 点数太少直接返回
        return np.zeros((num_points, 3), dtype=np.float32)
    
    try:
        # 移除无效点
        mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
        points = points[mask]
        
        if len(points) < 50:
            return np.zeros((num_points, 3), dtype=np.float32)
        
        # 使用numpy直接处理，避免Open3D开销
        if len(points) > num_points:
            # 随机下采样
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            # 重复采样
            repeat_times = (num_points + len(points) - 1) // len(points)
            points = np.tile(points, (repeat_times, 1))[:num_points]
        
        # 归一化
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        
        if max_dist > 1e-6:
            points /= max_dist
            
        return points.astype(np.float32)
        
    except Exception as e:
        print(f"预处理错误: {e}")
        return np.zeros((num_points, 3), dtype=np.float32)

def render_point_cloud(points, width=640, height=480):
    """
    将3D点云渲染为2D图像
    """
    if len(points) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    try:
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 设置视角参数（俯视角度）
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        vis.add_geometry(pcd)
        
        # 设置相机视角（俯视图）
        view_control = vis.get_view_control()
        view_control.set_front([0, 0, -1])  # 看向负Z轴
        view_control.set_up([0, -1, 0])     # Y轴朝下
        view_control.set_zoom(0.8)
        
        # 获取渲染图像
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        
        # 转换为numpy数组并调整格式
        image_np = np.asarray(image)
        image_np = (image_np * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        return image_np
        
    except Exception as e:
        print(f"点云渲染错误: {e}")
        return np.zeros((height, width, 3), dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(description='RealSense PointNet实时分类')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='训练好的模型路径')
    parser.add_argument('--num_points', type=int, default=1024,
                       help='点云采样数量')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                       help='使用CUDA加速')
    parser.add_argument('--ground_threshold', type=float, default=0.015,
                       help='地面分割距离阈值')
    parser.add_argument('--remove_outliers', action='store_true', default=True,
                       help='移除离群点')
    args = parser.parse_args()

    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载预训练模型
    print("加载模型...")
    try:
        # 修改num_class为10
        model = get_model(num_class=10, normal_channel=False)
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        
        # 检查模型权重是否匹配
        model_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                          if k in model_state_dict and model_state_dict[k].shape == v.shape}
        
        # 加载匹配的权重
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)
        
        # 对于不匹配的层（如分类层），保持随机初始化或使用其他策略
        missing_keys = [k for k in checkpoint['model_state_dict'].keys() if k not in pretrained_dict]
        if missing_keys:
            print(f"注意: 以下权重不匹配，将使用随机初始化: {missing_keys}")
        
        model.to(device)
        model.eval()
        print("模型加载成功!")
        
        # 预热GPU
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, args.num_points).to(device)
            model(dummy_input)
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 初始化RealSense相机
    print("初始化RealSense相机...")
    camera = RealSenseCamera()
    if not camera.initialize(max_retries=10):
        print("无法初始化相机，退出程序")
        return
    
    camera.start_capture()
    time.sleep(1)  # 给相机一些时间开始捕获

    # 3. 创建显示窗口
    cv2.namedWindow('RealSense PointNet Classification', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RealSense PointNet Classification', 1280, 480)
    
    print("开始实时分类...按ESC退出")
    
    # 性能统计
    frame_count = 0
    start_time = time.time()
    fps = 0
    processing_times = []
    
    try:
        while True:
            frame_start_time = time.time()
            
            # 获取最新帧
            depth_frame, color_frame = camera.get_frames()
            if depth_frame is None or color_frame is None:
                time.sleep(0.01)  # 短暂休眠避免CPU占用过高
                continue
            
            # 转换彩色图像
            color_image = np.asanyarray(color_frame.get_data())
            
            # 生成点云
            try:
                points = camera.pc.calculate(depth_frame)
                v = points.get_vertices()
                verts = np.asarray(v).view(np.float32).reshape(-1, 3)
            except Exception as e:
                print(f"点云生成错误: {e}")
                continue
            
            # 移除无效点
            valid_mask = ~np.isnan(verts).any(axis=1) & ~np.isinf(verts).any(axis=1)
            verts = verts[valid_mask]
            
            if len(verts) < 100:
                continue
            
            # 地面分割 - 只保留物体点云
            object_points, ground_indices = segment_ground_plane(
                verts, 
                distance_threshold=args.ground_threshold
            )
            
            # 可选：移除离群点
            if args.remove_outliers and len(object_points) > 50:
                object_points = remove_outliers(object_points)
            
            # 计算物体点云的坐标范围
            bounds = calculate_pointcloud_bounds(object_points)
            
            # 显示分割结果信息
            ground_percentage = len(ground_indices) / len(verts) * 100 if len(verts) > 0 else 0
            object_percentage = len(object_points) / len(verts) * 100 if len(verts) > 0 else 0
            
            # 渲染点云为图像
            pointcloud_image = render_point_cloud(object_points, width=640, height=480)
            
            # 在点云图像上添加信息
            cv2.putText(pointcloud_image, "Segmented Point Cloud", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(pointcloud_image, f"Points: {len(object_points)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 在点云图像上显示坐标范围
            y_offset = 90
            cv2.putText(pointcloud_image, f"X: {bounds['x_min']:.2f} - {bounds['x_max']:.2f} m", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            cv2.putText(pointcloud_image, f"Y: {bounds['y_min']:.2f} - {bounds['y_max']:.2f} m", 
                       (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(pointcloud_image, f"Z: {bounds['z_min']:.2f} - {bounds['z_max']:.2f} m", 
                       (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            # 预处理点云
            processed_cloud = preprocess_point_cloud(object_points, args.num_points)
            
            # 转换为模型输入格式并移动到GPU
            with torch.no_grad():
                points_tensor = torch.from_numpy(processed_cloud.T).unsqueeze(0).float().to(device)
                
                # 使用CUDA事件计时
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                pred, _ = model(points_tensor)
                end_event.record()
                
                # 等待CUDA操作完成
                torch.cuda.synchronize()
                
                gpu_time = start_event.elapsed_time(end_event)
                
                pred_choice = pred.data.max(1)[1]
                class_index = pred_choice.item()
                
                # 添加类别索引检查
                if class_index >= len(class_names):
                    class_name = f"Class_{class_index}"
                    print(f"警告: 预测类别索引 {class_index} 超出已知类别范围")
                else:
                    class_name = class_names[class_index]
                
                confidence = torch.nn.functional.softmax(pred, dim=1)[0, class_index].item()
            
            # 计算FPS
            frame_count += 1
            processing_time = time.time() - frame_start_time
            processing_times.append(processing_time)
            
            if len(processing_times) > 30:
                processing_times.pop(0)
            
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = current_time
            
            avg_process_time = np.mean(processing_times) if processing_times else 0
            
            # 在RGB图像上显示结果
            text = f"{class_name} ({confidence:.2f})"
            cv2.putText(color_image, text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(color_image, f"FPS: {fps:.1f}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, f"Ground: {ground_percentage:.1f}%", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(color_image, f"Object: {object_percentage:.1f}%", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(color_image, f"Points: {len(object_points)}", (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 在RGB图像上显示坐标范围
            cv2.putText(color_image, f"X: {bounds['x_min']:.2f}-{bounds['x_max']:.2f}m", 
                       (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            cv2.putText(color_image, f"Y: {bounds['y_min']:.2f}-{bounds['y_max']:.2f}m", 
                       (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            cv2.putText(color_image, f"Z: {bounds['z_min']:.2f}-{bounds['z_max']:.2f}m", 
                       (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
            
            cv2.putText(color_image, "ESC to exit", (20, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 水平拼接RGB图像和点云图像
            combined_image = np.hstack((color_image, pointcloud_image))
            
            cv2.imshow('RealSense PointNet Classification', combined_image)
            
            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break
                
    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        # 清理资源
        camera.stop()
        cv2.destroyAllWindows()
        print("程序结束")

if __name__ == "__main__":
    main()

#python realsense_pointnet2.py --model_path /home/rong/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_cls_ssg/checkpoints/best_model.pth --num_points 1024