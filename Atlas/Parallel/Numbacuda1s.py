import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib  # 处理nii文件

# =========================
# GPU Kernels (numba.cuda)
# =========================

from numba import cuda, float32, uint8
import math
import numpy as np

@cuda.jit(fastmath=True)
def depth_reduce_kernel_u8(masks_u8, original_center, area_center, depths):
    # 一个 block 处理一个 mask；block 内 256 线程做并行归约
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x

    n_masks = masks_u8.shape[0]
    n_voxels = masks_u8.shape[1]
    if bid >= n_masks:
        return

    sm = cuda.shared.array(shape=512, dtype=float32)  # 256*2
    sm_area = sm
    sm_dot  = sm[256:]

    partial_area = 0.0
    partial_dot  = 0.0

    for j in range(tid, n_voxels, bdim):
        v = float32(masks_u8[bid, j])       # u8 -> f32 (0/1)
        c = original_center[j]              # f32
        partial_area += v
        partial_dot  += v * c

    sm_area[tid] = partial_area
    sm_dot[tid]  = partial_dot
    cuda.syncthreads()

    offset = bdim // 2
    while offset > 0:
        if tid < offset:
            sm_area[tid] += sm_area[tid + offset]
            sm_dot[tid]  += sm_dot[tid + offset]
        cuda.syncthreads()
        offset >>= 1  # 每次减半，直到 1 合并所有thread的值


    if tid == 0:
        area_mask = sm_area[0]
        dot_c     = sm_dot[0]
        if area_mask > 0.0 and area_center > 0.0:
            s1 = dot_c / area_mask
            s2 = dot_c / area_center
            depths[bid] = s1 if s1 < s2 else s2
        else:
            depths[bid] = 0.0
def compute_inclusion_scores_single_stream(masks_data, batch_size=128):
    """单流单缓冲实现（用于对比测试）"""
    print("\n=== 单流单缓冲GPU计算 ===")
    
    t_total_start = time.time()
    
    # 预处理
    num_samples = masks_data.shape[0]
    masks_flat = masks_data.reshape(num_samples, -1)
    original_center = masks_flat.mean(axis=0, dtype=np.float32)
    area_center = np.float32(original_center.sum(dtype=np.float32))
    V = masks_flat.shape[1]
    
    # 单流（默认流）
    d_center = cuda.to_device(original_center)
    
    # 单缓冲
    B = int(batch_size)
    d_masks = cuda.device_array((B, V), dtype=np.float32)
    d_out = cuda.device_array(B, dtype=np.float32)
    h_out = cuda.pinned_array(B, dtype=np.float32)
    
    # 结果容器
    all_depths = np.empty(num_samples, dtype=np.float32)
    
    # 计时
    h2d_time = 0.0
    compute_time = 0.0
    d2h_time = 0.0
    
    # 主循环
    start = 0
    while start < num_samples:
        end = min(start + B, num_samples)
        this_bs = end - start
        
        # 1. H2D传输（同步）
        t1 = time.time()
        d_masks[:this_bs, :] = cuda.to_device(masks_flat[start:end, :])
        h2d_time += (time.time() - t1)
        
        # 2. 计算（同步）
        t2 = time.time()
        depth_reduce_kernel_u8[this_bs, 256](d_masks[:this_bs, :], d_center, area_center, d_out[:this_bs])
        cuda.synchronize()
        compute_time += (time.time() - t2)
        
        # 3. D2H传输（同步）
        t3 = time.time()
        h_out[:this_bs] = d_out[:this_bs].copy_to_host()
        d2h_time += (time.time() - t3)
        
        # 4. 结果拷贝
        all_depths[start:end] = h_out[:this_bs]
        
        start = end
    
    t_total = time.time() - t_total_start
    
    single_timing = {
        'total_time': t_total,
        'h2d_time': h2d_time,
        'compute_time': compute_time,
        'd2h_time': d2h_time,
        'transfer_time': h2d_time + d2h_time
    }
    
    print("=== 单流单缓冲性能统计 ===")
    print(f" 总计算时间: {t_total:.3f}秒")
    print(f"   H2D传输时间: {h2d_time:.3f}秒 ({h2d_time/t_total*100:.1f}%)")
    print(f"   GPU计算时间: {compute_time:.3f}秒 ({compute_time/t_total*100:.1f}%)")
    print(f"   D2H传输时间: {d2h_time:.3f}秒 ({d2h_time/t_total*100:.1f}%)")
    print(f"   传输总时间: {h2d_time + d2h_time:.3f}秒")
    
    return all_depths, single_timing

def compute_inclusion_scores(masks_data, use_gpu=True, batch_size=128, test_effectiveness=False):
    # ===== CPU 回退 =====
    if (not use_gpu) or (not cuda.is_available()):
        print("检测到未启用CUDA或无可用GPU，自动退回CPU计算。")
        # ——你的原CPU实现保留——
        num_samples = masks_data.shape[0]
        masks_flattened = masks_data.reshape(num_samples, -1)
        original_center = np.mean(masks_flattened, axis=0, dtype=np.float32)
        area_masks = np.sum(masks_flattened, axis=1, dtype=np.float32)
        area_center = np.sum(original_center, dtype=np.float32)
        inv_center = 1.0 - original_center
        s1 = 1.0 - np.sum(inv_center * masks_flattened, axis=1, dtype=np.float32) / np.where(area_masks>0, area_masks, 1.0)
        inv_masks = 1.0 - masks_flattened
        s2 = 1.0 - np.sum(inv_masks * original_center, axis=1, dtype=np.float32) / (area_center if area_center>0 else 1.0)
        s1 = np.nan_to_num(s1); s2 = np.nan_to_num(s2)
        depths = np.minimum(s1, s2)
        depths[area_masks == 0] = 0
        if area_center == 0: depths[:] = 0
        return depths.astype(np.float32, copy=False)

    # ===== GPU 路径（双缓冲 + 两流） =====
    print("\n=== 使用GPU加速 (双缓冲 + 传输/计算重叠) ===")
    # 启动总计时（包含预热）
    t_total0 = time.time()

    # 预处理：展平 + u8 压缩 + 计算中心
    t0 = time.time()
    num_samples = masks_data.shape[0]
    masks_flat = masks_data.reshape(num_samples, -1)              # (N, V) float32 assumed
    original_center = masks_flat.mean(axis=0, dtype=np.float32)   # (V,)
    area_center = np.float32(original_center.sum(dtype=np.float32))
    V = masks_flat.shape[1]
    preprocessing_time = time.time() - t0

    # 两条流
    t_setup0 = time.time()
    stream_h2d = cuda.stream()
    stream_k   = cuda.stream()
    # 复用的H2D事件（双缓冲）
    ev_h2d = [cuda.event(timing=False), cuda.event(timing=False)]
    setup_streams_time = time.time() - t_setup0

    # 把中心一次性上卡
    t0 = time.time()
    d_center = cuda.to_device(original_center, stream=stream_h2d)
    stream_h2d.synchronize()
    h2d_time = time.time() - t0  # 仅供参考，不计入后续累计

    # 预热：触发JIT编译与流/事件初始化，避免计入主统计
    t_pw0 = time.time()
    d_masks_pw = cuda.to_device(np.zeros((1, V), dtype=np.float32), stream=stream_h2d)
    d_out_pw = cuda.device_array(1, dtype=np.float32)
    ev_h2d[0].record(stream_h2d)
    ev_h2d[0].wait(stream_k)
    depth_reduce_kernel_u8[1, 256, stream_k](d_masks_pw[:1, :], d_center, area_center, d_out_pw[:1])
    d_out_pw[:1].copy_to_host(cuda.pinned_array(1, dtype=np.float32), stream=stream_k)
    stream_k.synchronize()
    stream_h2d.synchronize()
    prewarm_time = time.time() - t_pw0

    # 结果容器
    all_depths = np.empty(num_samples, dtype=np.float32)

    # 配置
    B = int(batch_size)
    threadsperblock = 256
    # 双份设备缓冲 + 双份 pinned host 缓冲（输入/输出）计时
    t_alloc0 = time.time()
    d_masks = [cuda.device_array((B, V), dtype=np.float32),
               cuda.device_array((B, V), dtype=np.float32)]
    d_out   = [cuda.device_array(B, dtype=np.float32),
               cuda.device_array(B, dtype=np.float32)]
    h_in  = [cuda.pinned_array((B, V), dtype=np.float32),
             cuda.pinned_array((B, V), dtype=np.float32)]
    h_out = [cuda.pinned_array(B, dtype=np.float32),
             cuda.pinned_array(B, dtype=np.float32)]
    alloc_buffers_time = time.time() - t_alloc0

    # 复用的D2H完成事件（双缓冲）
    done_evt   = [None, None]
    done_slice = [None, None]   # (start_idx, length) 记录结果范围

    # 预热后子计时清零（总计时不重置，以包含预热）
    gpu_compute_time = 0.0
    gpu_h2d_time = 0.0
    gpu_d2h_time = 0.0
    h2h_copy_time = 0.0  # Host->Host 拷贝 all_depths <- h_out 的时间

    # 主循环：交替使用缓冲 0 / 1
    # 主循环：交替使用缓冲 0 / 1
    start = 0
    batch_id = 0
    while start < num_samples:
        end = min(start + B, num_samples)
        this_bs = end - start
        buf = batch_id & 1  # 0/1

        # 如该缓冲上一次有未写回结果：等待事件 -> 写回 all_depths
        if done_evt[buf] is not None:
            t_d2h_wait = time.time()
            done_evt[buf].synchronize()
            gpu_d2h_time += (time.time() - t_d2h_wait)
            s0, ln = done_slice[buf]
            t_h2h = time.time()
            all_depths[s0:s0+ln] = h_out[buf][:ln]
            h2h_copy_time += (time.time() - t_h2h)
            done_evt[buf] = None
            done_slice[buf] = None

        # 1) 填充 pinned host 输入缓冲
        t1 = time.time()
        np.copyto(h_in[buf][:this_bs, :], masks_flat[start:end, :])

        # 2) 异步 H2D -> 记录事件（注意：事件在流上 record，后续 event.wait(另一个流)）
        d_masks[buf][:this_bs, :].copy_to_device(h_in[buf][:this_bs, :], stream=stream_h2d)
        ev_h2d[buf].record(stream_h2d)
        gpu_h2d_time += (time.time() - t1)

        # 3) 计算流等待 H2D 完成 → 启动 kernel（grid=this_bs）
        ev_h2d[buf].wait(stream_k)  # <<< 关键：用 event.wait(stream)
        blockspergrid = this_bs
        t2 = time.time()
        depth_reduce_kernel_u8[blockspergrid, threadsperblock, stream_k](
            d_masks[buf][:this_bs, :], d_center, area_center, d_out[buf][:this_bs]
        )

        # 4) 异步 D2H 回到 pinned host 输出缓冲，并在计算流上记录“完成事件”
        d_out[buf][:this_bs].copy_to_host(h_out[buf][:this_bs], stream=stream_k)
        if done_evt[buf] is None:
            done_evt[buf] = cuda.event(timing=False)
        done_evt[buf].record(stream_k)
        done_slice[buf] = (start, this_bs)

        # 统计
        gpu_compute_time += (time.time() - t2)

        # 前进到下一批
        start   = end
        batch_id += 1


    # 循环结束后，冲洗两块缓冲区的剩余结果
    for buf in (0, 1):
        if done_evt[buf] is not None:
            t_d2h_wait = time.time()
            done_evt[buf].synchronize()
            gpu_d2h_time += (time.time() - t_d2h_wait)
            s0, ln = done_slice[buf]
            t_h2h = time.time()
            all_depths[s0:s0+ln] = h_out[buf][:ln]
            h2h_copy_time += (time.time() - t_h2h)
            done_evt[buf] = None
            done_slice[buf] = None

    t_total = time.time() - t_total0
    other_time = t_total - preprocessing_time - prewarm_time - gpu_compute_time - gpu_h2d_time - gpu_d2h_time

    # 进一步拆解 “其他开销”
    other_setup_alloc_copy = setup_streams_time + alloc_buffers_time + h2h_copy_time
    misc_time = other_time - other_setup_alloc_copy
    if misc_time < 0:
        misc_time = 0.0

    print("\n=== GPU计算性能统计（双缓冲） ===")
    print(f" 总计算时间: {t_total:.3f}秒")
    print(f"   数据预处理: {preprocessing_time:.3f}秒 ({preprocessing_time/t_total*100:.1f}%)")
    print(f"   GPU计算时间(含D2H排队): {gpu_compute_time:.3f}秒 ({gpu_compute_time/t_total*100:.1f}%)")
    print(f"   H2D传输时间(累计): {gpu_h2d_time:.3f}秒 ({gpu_h2d_time/t_total*100:.1f}%)")
    print(f"   预热时间: {prewarm_time:.3f}秒 ({prewarm_time/t_total*100:.1f}%)")
    print(f"   D2H等待时间(累计): {gpu_d2h_time:.3f}秒 ({gpu_d2h_time/t_total*100:.1f}%)")
    print(f"   其他开销: {other_time:.3f}秒")
    print("     ├─ 流/事件创建: {:.3f}秒".format(setup_streams_time))
    print("     ├─ 设备/固定内存分配: {:.3f}秒".format(alloc_buffers_time))
    print("     ├─ Host→Host结果拷贝: {:.3f}秒".format(h2h_copy_time))
    print("     └─ 其余杂项: {:.3f}秒".format(misc_time))
    
    # 如果需要测试有效性，返回时间统计
    if test_effectiveness:
        double_timing = {
            'total_time': t_total,
            'h2d_time': gpu_h2d_time,
            'compute_time': gpu_compute_time,
            'd2h_time': gpu_d2h_time,
            'transfer_time': gpu_h2d_time + gpu_d2h_time,
            'preprocessing_time': preprocessing_time,
            'prewarm_time': prewarm_time,
            'other_time': other_time
        }
        return all_depths, double_timing
    
    return all_depths


# =========================
# 数据加载/分析/可视化
# =========================

def load_all_masks(mask_dir):
    """
    加载目录中的所有mask文件
    返回: masks_data, mask_names, affine, header
    """
    print(f"\n=== 开始加载Mask文件 ===")
    start_time = time.time()

    nii_files = glob.glob(os.path.join(mask_dir, "*.nii"))
    if not nii_files:
        print(f"在目录 {mask_dir} 中未找到nii文件")
        return None, None, None, None

    print(f"找到 {len(nii_files)} 个nii文件")

    first_img = nib.load(nii_files[0])
    first_data = first_img.get_fdata()
    mask_shape = first_data.shape
    affine = first_img.affine
    header = first_img.header

    print(f"Mask文件形状: {mask_shape}")
    print(f"数据类型: {first_data.dtype}")

    masks_data = np.zeros((len(nii_files),) + mask_shape, dtype=np.float32)
    mask_names = []

    successful_loads = 0
    for i, nii_file in enumerate(nii_files):
        try:
            print(f"正在加载 ({i+1}/{len(nii_files)}): {os.path.basename(nii_file)}")
            img = nib.load(nii_file)
            data = img.get_fdata()
            if data.shape != mask_shape:
                print(f"  警告: 形状不匹配 {data.shape} vs {mask_shape}，跳过")
                continue
            data_f32 = data.astype(np.float32, copy=False)
            masks_data[successful_loads] = data_f32
            mask_names.append(os.path.basename(nii_file))
            successful_loads += 1
            voxel_count = np.count_nonzero(data_f32)
            print(f"  激活体素数: {voxel_count:,}")
        except Exception as e:
            print(f"  加载失败: {str(e)}")
            continue

    if successful_loads < len(nii_files):
        masks_data = masks_data[:successful_loads]
        print(f"实际成功加载: {successful_loads} 个文件")

    end_time = time.time()
    print(f"Mask文件加载完成，耗时: {end_time - start_time:.2f}秒")
    print(f"最终数据形状: {masks_data.shape}")

    return masks_data, mask_names, affine, header


def evaluate_double_buffering_effectiveness(single_timing, double_timing, masks_data):
    """评估双流双缓冲的有效性"""
    
    print(f"\n{'='*60}")
    print("双流双缓冲有效性评估")
    print(f"{'='*60}")
    
    # 基本性能对比
    speedup_total = single_timing['total_time'] / double_timing['total_time']
    speedup_transfer = single_timing['transfer_time'] / double_timing['transfer_time']
    
    print(f"\n=== 性能对比分析 ===")
    print(f"{'指标':<15} {'单流单缓冲':<12} {'双流双缓冲':<12} {'加速比':<8} {'改善%':<8}")
    print("-" * 65)
    
    metrics = [
        ('总时间', 'total_time'),
        ('H2D传输', 'h2d_time'),
        ('计算时间', 'compute_time'),
        ('D2H等待', 'd2h_time'),
        ('传输总计', 'transfer_time')
    ]
    
    for name, key in metrics:
        single_val = single_timing[key]
        double_val = double_timing[key]
        speedup = single_val / double_val if double_val > 0 else float('inf')
        improvement = (single_val - double_val) / single_val * 100 if single_val > 0 else 0
        
        print(f"{name:<15} {single_val:<12.3f} {double_val:<12.3f} {speedup:<8.2f} {improvement:<8.1f}%")
    
    # 重叠效率分析
    print(f"\n=== 重叠效率分析 ===")
    
    # 理论串行时间 vs 实际并行时间
    theoretical_serial = double_timing['h2d_time'] + double_timing['compute_time'] + double_timing['d2h_time']
    actual_parallel = double_timing['total_time'] - double_timing['preprocessing_time'] - double_timing['prewarm_time']
    
    # 理论最佳时间（最慢的操作）
    theoretical_best = max(double_timing['h2d_time'], double_timing['compute_time'], double_timing['d2h_time'])
    
    overlap_efficiency = theoretical_best / actual_parallel * 100 if actual_parallel > 0 else 0
    
    print(f"理论串行时间: {theoretical_serial:.3f}秒")
    print(f"实际并行时间: {actual_parallel:.3f}秒")
    print(f"理论最佳时间: {theoretical_best:.3f}秒")
    print(f"重叠效率: {overlap_efficiency:.1f}%")
    
    # 瓶颈识别
    bottleneck_time = max(double_timing['h2d_time'], double_timing['compute_time'], double_timing['d2h_time'])
    if bottleneck_time == double_timing['h2d_time']:
        bottleneck = "H2D传输"
    elif bottleneck_time == double_timing['compute_time']:
        bottleneck = "GPU计算"
    else:
        bottleneck = "D2H传输"
    
    print(f"当前瓶颈: {bottleneck} ({bottleneck_time:.3f}秒)")
    
    # 带宽利用率分析
    print(f"\n=== 带宽利用率分析 ===")
    data_size_gb = masks_data.nbytes / 1024**3
    
    # PCIe带宽估算 (双向传输)
    pcie_bandwidth_single = data_size_gb * 2 / single_timing['transfer_time']
    pcie_bandwidth_double = data_size_gb * 2 / double_timing['transfer_time']
    
    print(f"数据总量: {data_size_gb:.2f} GB")
    print(f"单流PCIe带宽: {pcie_bandwidth_single:.2f} GB/s")
    print(f"双流PCIe带宽: {pcie_bandwidth_double:.2f} GB/s")
    print(f"带宽改善: {(pcie_bandwidth_double/pcie_bandwidth_single-1)*100:.1f}%")
    
    # 有效性判断
    print(f"\n=== 双流双缓冲有效性判断 ===")
    
    criteria = {
        "总体加速比 > 1.2": speedup_total > 1.2,
        "传输时间减少 > 10%": (single_timing['transfer_time'] - double_timing['transfer_time']) / single_timing['transfer_time'] > 0.1,
        "重叠效率 > 60%": overlap_efficiency > 60,
        "带宽利用率提升 > 15%": (pcie_bandwidth_double - pcie_bandwidth_single) / pcie_bandwidth_single > 0.15
    }
    
    passed_count = 0
    for criterion, passed in criteria.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {criterion}: {status}")
        if passed:
            passed_count += 1
    
    effectiveness_score = passed_count / len(criteria) * 100
    
    if effectiveness_score >= 75:
        effectiveness = "✓ 非常有效"
        color_code = "绿色"
    elif effectiveness_score >= 50:
        effectiveness = "⚠ 部分有效"
        color_code = "黄色"
    else:
        effectiveness = "✗ 效果不佳"
        color_code = "红色"
    
    print(f"\n=== 总体评估 ===")
    print(f"有效性得分: {effectiveness_score:.1f}% ({passed_count}/{len(criteria)})")
    print(f"评估结果: {effectiveness}")
    
    # 优化建议
    print(f"\n=== 优化建议 ===")
    recommendations = []
    
    if speedup_total < 1.2:
        recommendations.append("总体加速比较低，检查数据大小和批次配置")
    
    if overlap_efficiency < 60:
        recommendations.append("重叠效率不理想，考虑调整H2D/计算/D2H的时间平衡")
    
    if bottleneck == "H2D传输":
        recommendations.append("H2D传输是瓶颈，建议：增大批次大小或优化数据格式")
    elif bottleneck == "GPU计算":
        recommendations.append("GPU计算是瓶颈，建议：优化kernel或增加并行度")
    elif bottleneck == "D2H传输":
        recommendations.append("D2H传输是瓶颈，建议：减少输出数据量或优化内存布局")
    
    if pcie_bandwidth_single < 5.0:  # 假设PCIe 3.0 x16理论带宽约16GB/s
        recommendations.append("PCIe带宽利用率较低，检查数据传输效率")
    
    if not recommendations:
        recommendations.append("当前实现表现良好，可进行微调优化")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return {
        'speedup_total': speedup_total,
        'overlap_efficiency': overlap_efficiency,
        'bottleneck': bottleneck,
        'effectiveness_score': effectiveness_score,
        'recommendations': recommendations
    }

def analyze_depth_ranking(inclusion_scores, mask_names):
    print(f"\n=== 开始深度排序分析 ===")
    sorted_indices = np.argsort(inclusion_scores)[::-1]
    sorted_scores = inclusion_scores[sorted_indices]
    sorted_mask_names = [mask_names[i] for i in sorted_indices]

    print(f"排序结果:")
    print(f"深度最高的5个mask:")
    for i in range(min(5, len(sorted_mask_names))):
        print(f"  {i+1}. {sorted_mask_names[i]} (深度: {sorted_scores[i]:.4f})")

    print(f"\n深度最低的5个mask:")
    for i in range(max(0, len(sorted_mask_names)-5), len(sorted_mask_names)):
        rank = len(sorted_mask_names) - i
        print(f"  倒数{rank}. {sorted_mask_names[i]} (深度: {sorted_scores[i]:.4f})")

    return sorted_indices, sorted_mask_names, sorted_scores


def visualize_depth_analysis(inclusion_scores, mask_names, sorted_indices, output_dir=None, effectiveness_results=None):
    print(f"\n=== 开始生成深度分析可视化 ===")

    # 如果有有效性评估结果，创建更大的图形来容纳额外的subplot
    if effectiveness_results:
        fig = plt.figure(figsize=(24, 16))
        total_subplots = (3, 4)  # 3行4列
    else:
        fig = plt.figure(figsize=(20, 12))
        total_subplots = (2, 4)  # 2行4列

    # 1. 深度分布直方图
    plt.subplot(total_subplots[0], total_subplots[1], 1)
    plt.hist(inclusion_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Inclusion Score Depth')
    plt.ylabel('Frequency')
    plt.title('Distribution of Inclusion Score Depths')
    plt.grid(True, alpha=0.3)

    # 2. 深度排序图
    plt.subplot(total_subplots[0], total_subplots[1], 2)
    sorted_scores = inclusion_scores[sorted_indices]
    plt.plot(range(len(sorted_scores)), sorted_scores, 'o-', alpha=0.7)
    plt.xlabel('Rank (sorted by depth)')
    plt.ylabel('Inclusion Score Depth')
    plt.title('Sorted Inclusion Score Depths')
    plt.grid(True, alpha=0.3)

    # 3. 累计深度分布
    plt.subplot(total_subplots[0], total_subplots[1], 3)
    cumulative_percent = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    plt.plot(sorted_scores, cumulative_percent, 'o-', alpha=0.7)
    plt.xlabel('Inclusion Score Depth')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Depth Distribution')
    plt.grid(True, alpha=0.3)

    # 4. 前后50%对比
    plt.subplot(total_subplots[0], total_subplots[1], 4)
    mid_point = len(sorted_scores) // 2
    top_50_scores = sorted_scores[:mid_point]
    bottom_50_scores = sorted_scores[mid_point:]
    plt.hist([top_50_scores, bottom_50_scores], bins=15, alpha=0.7, label=['Top 50%', 'Bottom 50%'])
    plt.xlabel('Inclusion Score Depth')
    plt.ylabel('Frequency')
    plt.title('Top 50% vs Bottom 50% Depth Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. 箱线图
    plt.subplot(total_subplots[0], total_subplots[1], 5)
    plt.boxplot(inclusion_scores, vert=True)
    plt.ylabel('Inclusion Score Depth')
    plt.title('Inclusion Score Depth Box Plot')
    plt.grid(True, alpha=0.3)

    # 6. 统计文本
    plt.subplot(total_subplots[0], total_subplots[1], 6)
    stats_text = f"""Inclusion Score Depth Statistics:

Count: {len(inclusion_scores)}
Mean: {np.mean(inclusion_scores):.4f}
Std: {np.std(inclusion_scores):.4f}
Min: {np.min(inclusion_scores):.4f}
Max: {np.max(inclusion_scores):.4f}
Median: {np.median(inclusion_scores):.4f}

Percentiles:
25%: {np.percentile(inclusion_scores, 25):.4f}
50%: {np.percentile(inclusion_scores, 50):.4f}
75%: {np.percentile(inclusion_scores, 75):.4f}
90%: {np.percentile(inclusion_scores, 90):.4f}
95%: {np.percentile(inclusion_scores, 95):.4f}"""
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    plt.axis('off')
    plt.title('Statistical Summary')

    # 7. Top10
    plt.subplot(total_subplots[0], total_subplots[1], 7)
    top_10_indices = sorted_indices[:10]
    top_10_scores = inclusion_scores[top_10_indices]
    top_10_names = [mask_names[i][:15] for i in top_10_indices]
    plt.barh(range(len(top_10_names)), top_10_scores, alpha=0.7)
    plt.yticks(range(len(top_10_names)), top_10_names)
    plt.xlabel('Inclusion Score Depth')
    plt.title('Top 10 Highest Depth Masks')
    plt.grid(True, alpha=0.3)

    # 8. 原顺序散点
    plt.subplot(total_subplots[0], total_subplots[1], 8)
    plt.plot(range(len(inclusion_scores)), inclusion_scores, 'o', alpha=0.6, markersize=4)
    plt.xlabel('Mask Index (original order)')
    plt.ylabel('Inclusion Score Depth')
    plt.title('Depth Scores by Original Order')
    plt.grid(True, alpha=0.3)

    # 如果有有效性评估结果，添加额外的可视化
    if effectiveness_results:
        # 9. 性能对比柱状图
        plt.subplot(total_subplots[0], total_subplots[1], 9)
        single_timing = effectiveness_results.get('single_timing', {})
        double_timing = effectiveness_results.get('double_timing', {})
        
        if single_timing and double_timing:
            categories = ['Total Time', 'H2D Transfer', 'Compute', 'D2H Transfer']
            single_times = [single_timing.get('total_time', 0), single_timing.get('h2d_time', 0), 
                           single_timing.get('compute_time', 0), single_timing.get('d2h_time', 0)]
            double_times = [double_timing.get('total_time', 0), double_timing.get('h2d_time', 0), 
                           double_timing.get('compute_time', 0), double_timing.get('d2h_time', 0)]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, single_times, width, label='Single Stream', alpha=0.8)
            plt.bar(x + width/2, double_times, width, label='Double Stream', alpha=0.8)
            plt.xlabel('Operation Type')
            plt.ylabel('Time (seconds)')
            plt.title('Performance Comparison')
            plt.xticks(x, categories, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 10. 加速比图
        plt.subplot(total_subplots[0], total_subplots[1], 10)
        if single_timing and double_timing:
            speedups = []
            for cat in ['total_time', 'h2d_time', 'compute_time', 'd2h_time']:
                single_val = single_timing.get(cat, 0)
                double_val = double_timing.get(cat, 0)
                speedup = single_val / double_val if double_val > 0 else 0
                speedups.append(speedup)
            
            colors = ['green' if s > 1.2 else 'orange' if s > 1.0 else 'red' for s in speedups]
            plt.bar(['Total', 'H2D', 'Compute', 'D2H'], speedups, color=colors, alpha=0.7)
            plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            plt.axhline(y=1.2, color='green', linestyle='--', alpha=0.5, label='Good (>1.2x)')
            plt.ylabel('Speedup Ratio')
            plt.title('Speedup Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 11. 有效性评估雷达图（简化为饼图）
        plt.subplot(total_subplots[0], total_subplots[1], 11)
        eval_results = effectiveness_results.get('evaluation', {})
        effectiveness_score = eval_results.get('effectiveness_score', 0)
        
        # 创建饼图显示有效性得分
        sizes = [effectiveness_score, 100 - effectiveness_score]
        colors = ['green' if effectiveness_score >= 75 else 'orange' if effectiveness_score >= 50 else 'red', 'lightgray']
        labels = [f'Effective ({effectiveness_score:.1f}%)', f'Ineffective ({100-effectiveness_score:.1f}%)']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Double Buffering Effectiveness')

        # 12. 优化建议文本
        plt.subplot(total_subplots[0], total_subplots[1], 12)
        recommendations = eval_results.get('recommendations', [])
        rec_text = "Optimization Recommendations:\n\n"
        for i, rec in enumerate(recommendations[:5], 1):  # 只显示前5条建议
            rec_text += f"{i}. {rec}\n"
        
        if len(recommendations) > 5:
            rec_text += f"... and {len(recommendations)-5} more"
        
        plt.text(0.1, 0.9, rec_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        plt.axis('off')
        plt.title('Optimization Suggestions')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, "depth_analysis_visualization.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"深度分析可视化结果已保存: {fig_path}")

    plt.show()


# =========================
# 主流程
# =========================

def main():
    print("=== 包含分数深度分析工具（numba.cuda 版本） ===")
    total_start_time = time.time()

    preprocessing_time = 0.0
    gpu_compute_time = 0.0
    data_transfer_time = 0.0
    other_overhead_time = 0.0

    # 路径自行修改
    mask_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\Allseg\haimaprocessed_binary"
    output_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\Allseg\haimaprocessed_binary\DepthAnalysis"

    print(f"Mask目录: {mask_dir}")
    print(f"输出目录: {output_dir}")

    if not os.path.exists(mask_dir):
        print(f"错误: 目录不存在: {mask_dir}")
        return

    # 1. 加载
    t0 = time.time()
    masks_data, mask_names, affine, header = load_all_masks(mask_dir)
    # 确保传入 compute_inclusion_scores 的数据已为 float32
    masks_data = masks_data.astype(np.float32, copy=False)
    preprocessing_time += time.time() - t0

    if masks_data is None:
        print("没有成功加载任何mask文件")
        return
    def pick_batch_size(V, bytes_per_voxel=1, safety=0.85, max_batch=1024):
        free_mem, total_mem = cuda.current_context().get_memory_info()
        avail = int(free_mem * safety)
        # 需要双缓冲两份输入 + 一份输出(忽略) + center (≈ V*4)
        overhead = V * 4
        per_batch = V * bytes_per_voxel
        if avail <= overhead + per_batch:
            return 32  # 退而求其次
        max_by_mem = (avail - overhead) // (2 * per_batch)
        B = int(max(1, min(max_by_mem, max_batch)))
        # 对齐到 32 的倍数
        return max(64, (B // 64) * 64)

    # 在进入 GPU 分支前：
    V = np.prod(masks_data.shape[1:], dtype=np.int64)
    #B = pick_batch_size(V, bytes_per_voxel=4)   # 使用 float32，每体素 4 字节
    B = 32   # 固定批次大小为128
    # 然后用 B 作为 batch_size
    print(f"B: {B}")
    # 2. 计算（>50 用批处理）
    use_batch_processing = masks_data.shape[0] > 50
    want_gpu = True  # 想用GPU就设 True
    
    # 判断是否进行双流双缓冲有效性测试
    test_double_buffering = masks_data.shape[0] >= 100  # 样本数量足够时才测试
    
    effectiveness_results = None
    
    if test_double_buffering and want_gpu:
        print(f"\n{'='*60}")
        print("双流双缓冲有效性测试")
        print(f"{'='*60}")
        print(f"检测到样本数量({masks_data.shape[0]})足够，开始进行双流双缓冲有效性测试...")
        
        # 先运行单流单缓冲
        print("\n第1步：运行单流单缓冲基准测试")
        single_results, single_timing = compute_inclusion_scores_single_stream(masks_data, batch_size=B)
        
        # 再运行双流双缓冲
        print("\n第2步：运行双流双缓冲测试")
        t0 = time.time()
        double_results, double_timing = compute_inclusion_scores(
            masks_data,
            use_gpu=want_gpu,
            batch_size=B,
            test_effectiveness=True
        )
        
        # 验证结果一致性
        print("\n第3步：验证结果一致性")
        diff = np.abs(single_results - double_results)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"最大差异: {max_diff:.6f}")
        print(f"平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            print("✓ 结果一致性验证通过")
            
            # 进行有效性评估
            print("\n第4步：进行有效性评估")
            evaluation = evaluate_double_buffering_effectiveness(single_timing, double_timing, masks_data)
            
            effectiveness_results = {
                'single_timing': single_timing,
                'double_timing': double_timing,
                'evaluation': evaluation
            }
            
            inclusion_scores = double_results  # 使用双流双缓冲的结果
        else:
            print("✗ 结果一致性验证失败，使用单流单缓冲结果")
            inclusion_scores = single_results
    else:
        # 常规计算
        t0 = time.time()
        inclusion_scores = compute_inclusion_scores(
            masks_data,
            use_gpu=want_gpu,
            batch_size=B
        )
        gpu_compute_time = time.time() - t0  # 粗略记录（包含函数内部预处理/传输）

    # 3. 排序分析
    t0 = time.time()
    sorted_indices, sorted_mask_names, sorted_scores = analyze_depth_ranking(inclusion_scores, mask_names)
    other_overhead_time += time.time() - t0

    # 4. 可视化
    t0 = time.time()
    # 遵循“不另行要求不保存图像”的规则，这里仅显示，不保存
    visualize_depth_analysis(inclusion_scores, mask_names, sorted_indices, output_dir=None, effectiveness_results=effectiveness_results)
    other_overhead_time += time.time() - t0

    # 5. 保存结果
    print(f"\n=== 保存分析结果 ===")
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "inclusion_scores.npy"), inclusion_scores)
    np.save(os.path.join(output_dir, "sorted_indices.npy"), sorted_indices)
    data_transfer_time = time.time() - t0
    print(f"包含分数深度分析结果已保存到: {output_dir}")

    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    gpu_utilization = (gpu_compute_time / total_runtime) * 100 if total_runtime > 0 else 0.0

    print(f"\n=== 性能统计（总） ===")
    print(f" 总计算时间: {total_runtime:.3f}秒")
    print(f"   数据预处理: {preprocessing_time:.3f}秒 ({preprocessing_time/total_runtime*100:.1f}%)")
    print(f"   GPU计算时间(含函数内部): {gpu_compute_time:.3f}秒 ({gpu_compute_time/total_runtime*100:.1f}%)")
    print(f"   数据传输时间(保存): {data_transfer_time:.3f}秒 ({data_transfer_time/total_runtime*100:.1f}%)")
    print(f"   其他开销: {other_overhead_time:.3f}秒")
    print(f" GPU利用率(粗略): {gpu_utilization:.1f}%")

    print(f"\n=== 分析总结 ===")
    print(f"处理了 {len(mask_names)} 个mask文件")
    print(f"深度范围: {np.min(inclusion_scores):.4f} - {np.max(inclusion_scores):.4f}")
    print(f"平均深度: {np.mean(inclusion_scores):.4f}")
    
    # 如果进行了双流双缓冲测试，显示总结
    if effectiveness_results:
        eval_results = effectiveness_results['evaluation']
        print(f"\n=== 双流双缓冲测试总结 ===")
        print(f"总体加速比: {eval_results['speedup_total']:.2f}x")
        print(f"重叠效率: {eval_results['overlap_efficiency']:.1f}%")
        print(f"性能瓶颈: {eval_results['bottleneck']}")
        print(f"有效性得分: {eval_results['effectiveness_score']:.1f}%")
        
        if eval_results['effectiveness_score'] >= 75:
            print("✓ 双流双缓冲非常有效！")
        elif eval_results['effectiveness_score'] >= 50:
            print("⚠ 双流双缓冲部分有效，有优化空间")
        else:
            print("✗ 双流双缓冲效果不理想，需要优化")


if __name__ == "__main__":
    main()
