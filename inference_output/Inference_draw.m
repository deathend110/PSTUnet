clear; clc; close all;

%% ==========================================
% 1. 路径与参数配置
% ==========================================
% 指定存放 _dual_pred.mat 文件的目标文件夹
pred_dir = './db_test/predictions'; 

% 存放 Bangkok_1.mat 等元数据阈值文件的文件夹
meta_dir = './TH'; 

% 你要抽取的帧索引 (MATLAB 索引从 1 开始)
% 180MHz(第2帧), 混合(第5帧), 60MHz(第8帧)
frame_indices =[2, 5, 9]; 
frame_titles = {'Frame 2 (180MHz)', 'Frame 5 (Mixed)', 'Frame 9 (60MHz)'};

%% ==========================================
% 2. 获取所有文件并开始批量循环
% ==========================================
pred_files = dir(fullfile(pred_dir, '*.mat'));
if isempty(pred_files)
    error('❌ 在文件夹 %s 中找不到任何 *_dual_pred.mat 文件！', pred_dir);
end

fprintf('🚀 启动批量自动化双域分析，共探测到 %d 个序列文件...\n', length(pred_files));
fprintf(repmat('=', 1, 50) + "\n");

for file_idx = 1:length(pred_files)
    
    % 当前处理的文件路径与文件名
    filename_ext = pred_files(file_idx).name;
    pred_filepath = fullfile(pred_dir, filename_ext);
    [~, filename, ~] = fileparts(pred_filepath);
    
    fprintf('[%d/%d] 正在分析序列: %s\n', file_idx, length(pred_files), filename);

    % --- 提取底图名字以匹配元数据 ---
    % 例如 SAR_Dataset_Bangkok_1_DB_seq_000008_dual_pred -> Bangkok_1
    parts = strsplit(filename, '_DB_seq_');
    core_name = parts{1}; 
    
    core_name = append('SAR_Dataset_', core_name);
    meta_filepath = fullfile(meta_dir, [core_name, '.mat']);
    if ~exist(meta_filepath, 'file')
        warning('⚠️ 找不到对应的阈值文件: %s，跳过该序列。', meta_filepath);
        continue;
    end

    meta_data = load(meta_filepath);
    vmax_db = meta_data.V_MAX_GT; 
    vmin_db = meta_data.V_MIN_GT;

    %% ==========================================
    % 3. 加载预测数据并执行物理域逆向还原
    % ==========================================
    data = load(pred_filepath);

    % 将网络的预测结果 (DB 域 [0,1]) 完美逆向还原回 Linear 域
    seq_pred_Linear = sar_inverse_normalize_modality_advanced(data.seq_pred_DB, vmax_db, vmin_db);
    % t = seq_pred_Linear(:,:,1);
    % imagesc(t); colormap(gray); axis off image;

    %% ==========================================
    % 4. 创建两个独立的 3x3 高清画布 (后台静默绘制)
    % ==========================================
    % 注意：'Visible', 'off' 会让画板在后台生成
    fig_db = figure('Color', 'w');
    sgtitle(sprintf('DB-Domain Restoration: %s', filename), 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');

    fig_lin = figure('Color', 'w');
    sgtitle(sprintf('Linear-Domain Restoration: %s', filename), 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');

    for i = 1:3
        f_idx = frame_indices(i);
        f_name = frame_titles{i};
        
        % ---------- 提取当前帧的数据 ----------
        in_db   = data.seq_input_DB(:, :, f_idx);
        pred_db = data.seq_pred_DB(:, :, f_idx);
        gt_db   = data.seq_GT_DB(:, :, f_idx);
        
        in_lin   = data.seq_input_Linear(:, :, f_idx)/255;
        pred_lin_raw = seq_pred_Linear(:, :, f_idx);
        
        % 严格保留你的自定义归一化逻辑
        pred_lin = minmaxnormalize_image(pred_lin_raw, meta_data.V_MAX_GT_L, meta_data.V_MIN_GT_L);
        gt_lin   = data.seq_GT_Linear(:, :, f_idx)/255;
        % imagesc(pred_lin); colormap(gray); axis off image;
        % ---------- 计算 DB 域指标 ----------
        psnr_in_db   = psnr(in_db, gt_db, 1.0);
        ssim_in_db   = ssim(in_db, gt_db);
        psnr_pred_db = psnr(pred_db, gt_db, 1.0);
        ssim_pred_db = ssim(pred_db, gt_db);
        
        % ---------- 计算 Linear 域指标 ----------
        psnr_in_lin   = psnr(in_lin, gt_lin);
        ssim_in_lin   = ssim(in_lin, gt_lin);
        psnr_pred_lin = psnr(pred_lin, gt_lin);
        ssim_pred_lin = ssim(pred_lin, gt_lin);
        
        % ---------- 确定 Linear 域的可视化显示上限 ----------
        disp_max_lin = prctile(gt_lin(:), 99.5);

        %% ---------- 开始绘制 (每图 3 行 3 列) ----------
        row_offset = (i - 1) * 3;
        
        % ================= 画布 1: DB 域 =================
        set(0, 'CurrentFigure', fig_db); % 激活 DB 画布
        
        subplot(3, 3, row_offset + 1);
        imagesc(in_db,[0, 1]); colormap(gray); axis off image;
        title(sprintf('%s Input\nPSNR: %.2fdB | SSIM: %.4f', f_name, psnr_in_db, ssim_in_db), 'FontSize', 12);
        
        subplot(3, 3, row_offset + 2);
        imagesc(pred_db, [0, 1]); colormap(gray); axis off image;
        title(sprintf('Prediction\nPSNR: %.2fdB | SSIM: %.4f', psnr_pred_db, ssim_pred_db), 'FontSize', 12, 'Color',[0.85 0.32 0.09]);
        
        subplot(3, 3, row_offset + 3);
        imagesc(gt_db, [0, 1]); colormap(gray); axis off image;
        title('Ground Truth', 'FontSize', 12);
        
        % ================= 画布 2: Linear 域 =================
        set(0, 'CurrentFigure', fig_lin); % 激活 Linear 画布
        
        subplot(3, 3, row_offset + 1);
        imagesc(in_lin,[0, disp_max_lin]); colormap(gray); axis off image;
        title(sprintf('%s Input\nPSNR: %.2fdB | SSIM: %.4f', f_name, psnr_in_lin, ssim_in_lin), 'FontSize', 12);
        
        subplot(3, 3, row_offset + 2);
        imagesc(pred_lin,[0, disp_max_lin]); colormap(gray); axis off image;
        title(sprintf('Prediction\nPSNR: %.2fdB | SSIM: %.4f', psnr_pred_lin, ssim_pred_lin), 'FontSize', 12, 'Color',[0 0.44 0.74]);
        
        subplot(3, 3, row_offset + 3);
        imagesc(gt_lin, [0, disp_max_lin]); colormap(gray); axis off image;
        title('Ground Truth', 'FontSize', 12);
    end

    %% ==========================================
    % 5. 导出两张独立的高清分析图
    % ==========================================
    % 动态生成保存文件名 (使用完整的序列名，防止同底图互相覆盖)
    % save_prefix   = strrep(filename, '_dual_pred', ''); % 比如 SAR_Dataset_Bangkok_1_DB_seq_000008
    % save_db_path  = sprintf('./img/Analysis_DB_%s.png', save_prefix);
    % save_lin_path = sprintf('./img/Analysis_Linear_%s.png', save_prefix);

    % 导出 300 DPI 的无损边缘高清图
    % exportgraphics(fig_db, save_db_path, 'Resolution', 300);
    % exportgraphics(fig_lin, save_lin_path, 'Resolution', 300);
    
    % 【极其重要】: 释放内存，关闭后台隐藏的画板
    close(fig_db);
    close(fig_lin);
    
    % fprintf('  👉 成功保存: %s\n', save_db_path);
    % fprintf('  👉 成功保存: %s\n', save_lin_path);
end

fprintf(repmat('=', 1, 50) + "\n");
fprintf('🎉 所有序列自动化评测与出图已全部完成！\n');

%% ==========================================
% 附：逆向物理还原函数 (Advanced版)
% ==========================================
function img_linear = sar_inverse_normalize_modality_advanced(img_norm, v_max, v_min)
    % 1. 从[0, 1] 逆向映射回 dB 域
    img_db = img_norm .* (v_max - v_min) + v_min;
    
    % 2. 从 dB 域还原到线性幅度域 (Linear Amplitude)
    img_linear = 10.^(img_db / 20);
    
    % 3. 截断保护：防止极微小的负数
    img_linear(img_linear < 0) = 0;
end