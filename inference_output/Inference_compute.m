clear; clc; close all;

%% ==========================================
% 1. 路径与参数配置 (请根据实际情况修改)
% ==========================================
pred_dir = './db_test/predictions'; 
meta_dir = './TH'; 

% 定义三种帧在 16 帧序列中的具体位置 (MATLAB 索引从 1 开始)
idx_180 =[1, 2, 3, 4, 13, 14, 15, 16]; % 共 8 帧
idx_mix = [5, 12];                      % 共 2 帧
idx_60  =[6, 7, 8, 9, 10, 11];         % 共 6 帧

frame_titles = {'180MHz (8 frames/seq)', 'Mixed (2 frames/seq)', '60MHz (6 frames/seq)'};

%% ==========================================
% 2. 初始化累加器 (1:180MHz, 2:Mixed, 3:60MHz)
% ==========================================
sum_psnr_in_db   = zeros(1, 3);
sum_ssim_in_db   = zeros(1, 3);
sum_psnr_pred_db = zeros(1, 3);
sum_ssim_pred_db = zeros(1, 3);

sum_psnr_in_lin   = zeros(1, 3);
sum_ssim_in_lin   = zeros(1, 3);
sum_psnr_pred_lin = zeros(1, 3);
sum_ssim_pred_lin = zeros(1, 3);

valid_count = 0; 

%% ==========================================
% 3. 遍历所有文件，执行 16 帧全量统计
% ==========================================
pred_files = dir(fullfile(pred_dir, '*.mat'));

fprintf('🚀 启动全序列 16 帧全量指标统计，共探测到 %d 个序列文件...\n', length(pred_files));
fprintf(repmat('-', 1, 70) + "\n");

for file_idx = 1:length(pred_files)
    filename_ext = pred_files(file_idx).name;
    pred_filepath = fullfile(pred_dir, filename_ext);
    [~, filename, ~] = fileparts(pred_filepath);
    
    % --- 提取底图名字以匹配元数据 ---
    parts = strsplit(filename, '_DB_seq_');
    core_name = parts{1}; 
    
    core_name = append('SAR_Dataset_', core_name);
    meta_filepath = fullfile(meta_dir, [core_name, '.mat']);
    if ~exist(meta_filepath, 'file')
        warning('⚠️ 找不到对应的阈值文件: %s，跳过该序列。', meta_filepath);
        continue;
    end

    % --- 加载元数据 ---
    meta_data = load(meta_filepath);
    vmax_db = meta_data.V_MAX_GT; 
    vmin_db = meta_data.V_MIN_GT;

    % --- 加载预测数据并执行物理域逆向还原 ---
    data = load(pred_filepath);
    seq_pred_Linear_raw = sar_inverse_normalize_modality_advanced(data.seq_pred_DB, vmax_db, vmin_db);
    
    % 预分配当前文件的临时存储数组
    cur_psnr_in_db = zeros(1, 16);   cur_ssim_in_db = zeros(1, 16);
    cur_psnr_pred_db = zeros(1, 16); cur_ssim_pred_db = zeros(1, 16);
    cur_psnr_in_lin = zeros(1, 16);  cur_ssim_in_lin = zeros(1, 16);
    cur_psnr_pred_lin = zeros(1, 16);cur_ssim_pred_lin = zeros(1, 16);

    % ==========================================
    % 对 16 帧逐一算分
    % ==========================================
    for f = 1:16
        in_db   = data.seq_input_DB(:, :, f);
        pred_db = data.seq_pred_DB(:, :, f);
        gt_db   = data.seq_GT_DB(:, :, f);
        
        in_lin   = data.seq_input_Linear(:, :, f)/255;
        gt_lin   = data.seq_GT_Linear(:, :, f)/255;
        % 线性域预测结果应用你在推理脚本里的归一化逻辑
        pred_lin = minmaxnormalize_image(seq_pred_Linear_raw(:, :, f), meta_data.V_MAX_GT_L, meta_data.V_MIN_GT_L);
        
        % 计算 DB 域指标 (图像在 [0,1] 空间)
        cur_psnr_in_db(f)   = psnr(in_db, gt_db, 1.0);
        cur_ssim_in_db(f)   = ssim(in_db, gt_db);
        cur_psnr_pred_db(f) = psnr(pred_db, gt_db, 1.0);
        cur_ssim_pred_db(f) = ssim(pred_db, gt_db);
        
        % 计算 Linear 域指标 (图像在 [0,1] 空间)
        cur_psnr_in_lin(f)   = psnr(in_lin, gt_lin, 1.0);
        cur_ssim_in_lin(f)   = ssim(in_lin, gt_lin);
        cur_psnr_pred_lin(f) = psnr(pred_lin, gt_lin, 1.0);
        cur_ssim_pred_lin(f) = ssim(pred_lin, gt_lin);
    end
    
    % ==========================================
    % 将 16 帧的成绩归类装入 3 个累加器箱子
    % ==========================================
    % [箱子 1: 180MHz]
    sum_psnr_in_db(1)   = sum_psnr_in_db(1)   + sum(cur_psnr_in_db(idx_180));
    sum_ssim_in_db(1)   = sum_ssim_in_db(1)   + sum(cur_ssim_in_db(idx_180));
    sum_psnr_pred_db(1) = sum_psnr_pred_db(1) + sum(cur_psnr_pred_db(idx_180));
    sum_ssim_pred_db(1) = sum_ssim_pred_db(1) + sum(cur_ssim_pred_db(idx_180));
    
    sum_psnr_in_lin(1)   = sum_psnr_in_lin(1)   + sum(cur_psnr_in_lin(idx_180));
    sum_ssim_in_lin(1)   = sum_ssim_in_lin(1)   + sum(cur_ssim_in_lin(idx_180));
    sum_psnr_pred_lin(1) = sum_psnr_pred_lin(1) + sum(cur_psnr_pred_lin(idx_180));
    sum_ssim_pred_lin(1) = sum_ssim_pred_lin(1) + sum(cur_ssim_pred_lin(idx_180));

    % [箱子 2: Mixed]
    sum_psnr_in_db(2)   = sum_psnr_in_db(2)   + sum(cur_psnr_in_db(idx_mix));
    sum_ssim_in_db(2)   = sum_ssim_in_db(2)   + sum(cur_ssim_in_db(idx_mix));
    sum_psnr_pred_db(2) = sum_psnr_pred_db(2) + sum(cur_psnr_pred_db(idx_mix));
    sum_ssim_pred_db(2) = sum_ssim_pred_db(2) + sum(cur_ssim_pred_db(idx_mix));
    
    sum_psnr_in_lin(2)   = sum_psnr_in_lin(2)   + sum(cur_psnr_in_lin(idx_mix));
    sum_ssim_in_lin(2)   = sum_ssim_in_lin(2)   + sum(cur_ssim_in_lin(idx_mix));
    sum_psnr_pred_lin(2) = sum_psnr_pred_lin(2) + sum(cur_psnr_pred_lin(idx_mix));
    sum_ssim_pred_lin(2) = sum_ssim_pred_lin(2) + sum(cur_ssim_pred_lin(idx_mix));

    % [箱子 3: 60MHz]
    sum_psnr_in_db(3)   = sum_psnr_in_db(3)   + sum(cur_psnr_in_db(idx_60));
    sum_ssim_in_db(3)   = sum_ssim_in_db(3)   + sum(cur_ssim_in_db(idx_60));
    sum_psnr_pred_db(3) = sum_psnr_pred_db(3) + sum(cur_psnr_pred_db(idx_60));
    sum_ssim_pred_db(3) = sum_ssim_pred_db(3) + sum(cur_ssim_pred_db(idx_60));
    
    sum_psnr_in_lin(3)   = sum_psnr_in_lin(3)   + sum(cur_psnr_in_lin(idx_60));
    sum_ssim_in_lin(3)   = sum_ssim_in_lin(3)   + sum(cur_ssim_in_lin(idx_60));
    sum_psnr_pred_lin(3) = sum_psnr_pred_lin(3) + sum(cur_psnr_pred_lin(idx_60));
    sum_ssim_pred_lin(3) = sum_ssim_pred_lin(3) + sum(cur_ssim_pred_lin(idx_60));

    valid_count = valid_count + 1;
    fprintf('  ✔ 成功处理: [%d/%d] %s\n', valid_count, length(pred_files), filename);
end

if valid_count == 0
    error('❌ 没有成功处理任何文件，请检查文件夹或元数据是否匹配！');
end

%% ==========================================
% 4. 计算真正的全量平均值 (总和 / 总帧数)
% ==========================================
% 计算各类别包含的总帧数
total_frames_180 = valid_count * length(idx_180);
total_frames_mix = valid_count * length(idx_mix);
total_frames_60  = valid_count * length(idx_60);
divisor =[total_frames_180, total_frames_mix, total_frames_60];

avg_psnr_in_db   = sum_psnr_in_db ./ divisor;
avg_ssim_in_db   = sum_ssim_in_db ./ divisor;
avg_psnr_pred_db = sum_psnr_pred_db ./ divisor;
avg_ssim_pred_db = sum_ssim_pred_db ./ divisor;

avg_psnr_in_lin   = sum_psnr_in_lin ./ divisor;
avg_ssim_in_lin   = sum_ssim_in_lin ./ divisor;
avg_psnr_pred_lin = sum_psnr_pred_lin ./ divisor;
avg_ssim_pred_lin = sum_ssim_pred_lin ./ divisor;

%% ==========================================
% 5. 打印漂亮的学术表格
% ==========================================
fprintf('\n');
fprintf(repmat('=', 1, 75) + "\n");
fprintf('🏆 全量自动化评测总成绩 (共处理 %d 个序列，总计 %d 帧图像)\n', valid_count, valid_count * 16);
fprintf(repmat('=', 1, 75) + "\n\n");

% ------------------ 输出 DB 域表格 ------------------
fprintf('【1】 DB 域 (对数域) 评测结果：\n');
fprintf('--------------------------------------------------------------------------\n');
fprintf(' %-18s | %-16s | %-16s | %-16s\n', '指标 \ 帧类型', frame_titles{1}, frame_titles{2}, frame_titles{3});
fprintf('--------------------------------------------------------------------------\n');
fprintf(' %-18s | %10.2f dB     | %10.2f dB     | %10.2f dB\n', 'Input PSNR', avg_psnr_in_db(1), avg_psnr_in_db(2), avg_psnr_in_db(3));
fprintf(' %-18s | %10.2f dB     | %10.2f dB     | %10.2f dB\n', 'Pred  PSNR (U-Net)', avg_psnr_pred_db(1), avg_psnr_pred_db(2), avg_psnr_pred_db(3));
fprintf('--------------------------------------------------------------------------\n');
fprintf(' %-18s | %10.4f        | %10.4f        | %10.4f\n', 'Input SSIM', avg_ssim_in_db(1), avg_ssim_in_db(2), avg_ssim_in_db(3));
fprintf(' %-18s | %10.4f        | %10.4f        | %10.4f\n', 'Pred  SSIM (U-Net)', avg_ssim_pred_db(1), avg_ssim_pred_db(2), avg_ssim_pred_db(3));
fprintf('--------------------------------------------------------------------------\n\n');

% ------------------ 输出 Linear 域表格 ------------------
fprintf('【2】 Linear 域 (线性幅度域) 评测结果：\n');
fprintf('--------------------------------------------------------------------------\n');
fprintf(' %-18s | %-16s | %-16s | %-16s\n', '指标 \ 帧类型', frame_titles{1}, frame_titles{2}, frame_titles{3});
fprintf('--------------------------------------------------------------------------\n');
fprintf(' %-18s | %10.2f dB     | %10.2f dB     | %10.2f dB\n', 'Input PSNR', avg_psnr_in_lin(1), avg_psnr_in_lin(2), avg_psnr_in_lin(3));
fprintf(' %-18s | %10.2f dB     | %10.2f dB     | %10.2f dB\n', 'Pred  PSNR (U-Net)', avg_psnr_pred_lin(1), avg_psnr_pred_lin(2), avg_psnr_pred_lin(3));
fprintf('--------------------------------------------------------------------------\n');
fprintf(' %-18s | %10.4f        | %10.4f        | %10.4f\n', 'Input SSIM', avg_ssim_in_lin(1), avg_ssim_in_lin(2), avg_ssim_in_lin(3));
fprintf(' %-18s | %10.4f        | %10.4f        | %10.4f\n', 'Pred  SSIM (U-Net)', avg_ssim_pred_lin(1), avg_ssim_pred_lin(2), avg_ssim_pred_lin(3));
fprintf(repmat('=', 1, 75) + "\n");

%% ==========================================
% 附：逆向物理还原函数 (Advanced版)
% ==========================================
function img_linear = sar_inverse_normalize_modality_advanced(img_norm, v_max, v_min)
    img_db = img_norm .* (v_max - v_min) + v_min;
    img_linear = 10.^(img_db / 20) - 1e-5;
    img_linear(img_linear < 0) = 0;
end