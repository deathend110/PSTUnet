clear; clc; close all;

%% ==========================================
% 1. 路径与参数配置
% ==========================================
pred_dir   = './db_test/predictions';
meta_dir   = './TH';

DB_dir     = 'G:\VSCODE-G\PST_Dataset\DB\testdata\';
Linear_dir = 'G:\VSCODE-G\PST_Dataset\Linear\testdata\';

% 你要抽取的帧索引（MATLAB 索引从 1 开始）
% frame_indices = [4, 5, 6, 7, 8, 9];
frame_indices = [8, 9, 10, 11, 12, 13];

% 三类帧号定义
idx_180 = [1, 2, 3, 4, 13, 14, 15, 16]; % 共 8 帧
idx_mix = [5, 12];                      % 共 2 帧
idx_60  = [6, 7, 8, 9, 10, 11];         % 共 6 帧

% 可选：手动标题。
% 若为空，则自动生成：Frame x (180MHz/Mixed/60MHz)
frame_titles = {};

%% ==========================================
% 2. 参数检查与自动标题生成
% ==========================================
num_frames = numel(frame_indices);

if numel(unique(frame_indices)) ~= num_frames
    error('frame_indices 中存在重复帧号，请检查！');
end

if isempty(frame_titles)
    frame_titles = cell(1, num_frames);
    for k = 1:num_frames
        f = frame_indices(k);

        if ismember(f, idx_180)
            band_name = '180MHz';
        elseif ismember(f, idx_mix)
            band_name = 'Mixed';
        elseif ismember(f, idx_60)
            band_name = '60MHz';
        else
            band_name = 'Unknown';
        end

        frame_titles{k} = sprintf('F-%d (%s)', f, band_name);
    end
end

if numel(frame_titles) ~= num_frames
    error('frame_titles 的数量必须与 frame_indices 一致！');
end

%% ==========================================
% 3. 获取所有文件并开始批量循环
% ==========================================
pred_files = dir(fullfile(pred_dir, '*.mat'));
if isempty(pred_files)
    error('❌ 在文件夹 %s 中找不到任何 .mat 文件！', pred_dir);
end

fprintf('🚀 启动批量自动化双域分析，共探测到 %d 个序列文件...\n', length(pred_files));
fprintf('%s\n', repmat('=', 1, 60));

for file_idx = 1:length(pred_files)

    % 当前处理文件
    filename_ext  = pred_files(file_idx).name;
    pred_filepath = fullfile(pred_dir, filename_ext);
    DB_filepath   = fullfile(DB_dir, filename_ext);
    filename_Linear = replace(filename_ext, "_DB_", "_L_");
    Linear_filepath = fullfile(Linear_dir, filename_Linear);

    [~, filename, ~] = fileparts(pred_filepath);

    fprintf('[%d/%d] 正在分析序列: %s\n', file_idx, length(pred_files), filename);

    % ---------- 检查文件存在 ----------
    if ~exist(pred_filepath, 'file')
        warning('⚠️ 预测文件不存在，跳过：%s', pred_filepath);
        continue;
    end
    if ~exist(DB_filepath, 'file')
        warning('⚠️ DB 文件不存在，跳过：%s', DB_filepath);
        continue;
    end
    if ~exist(Linear_filepath, 'file')
        warning('⚠️ Linear 文件不存在，跳过：%s', Linear_filepath);
        continue;
    end

    % ---------- 提取底图名字以匹配元数据 ----------
    parts = strsplit(filename, '_DB_seq_');
    core_name = parts{1};
    core_name = append('SAR_Dataset_', core_name);

    meta_filepath = fullfile(meta_dir, [core_name, '.mat']);
    if ~exist(meta_filepath, 'file')
        warning('⚠️ 找不到对应的阈值文件: %s，跳过该序列。', meta_filepath);
        continue;
    end

    %% ==========================================
    % 4. 加载元数据与数据
    % ==========================================
    meta_data = load(meta_filepath);
    vmax_db = meta_data.V_MAX_GT;
    vmin_db = meta_data.V_MIN_GT;

    pred_data = load(pred_filepath);
    seq_pred_DB      = double(pred_data.seq_pred_DB);

    seq_GT_DB        = double(h5read(DB_filepath, '/seq_GT')) / 255;
    seq_input_DB     = double(h5read(DB_filepath, '/seq_input')) / 255;

    seq_GT_Linear    = double(h5read(Linear_filepath, '/seq_GT_L')) / 255;
    seq_input_Linear = double(h5read(Linear_filepath, '/seq_input_L')) / 255;

    seq_pred_Linear_raw = sar_inverse_normalize_modality_advanced(seq_pred_DB, vmax_db, vmin_db);

    total_frames = size(seq_pred_DB, 3);
    if any(frame_indices < 1) || any(frame_indices > total_frames)
        warning('⚠️ 文件 %s 的总帧数为 %d，但 frame_indices 中存在越界索引，跳过该文件。', ...
            filename_ext, total_frames);
        continue;
    end

    %% ==========================================
    % 5. 创建两个独立画布
    %    每张图：3 行 n 列
    % ==========================================
    fig_db = figure('Color', 'w');
    sgtitle(sprintf('DB-Domain Restoration: %s', filename), ...
        'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');

    fig_lin = figure('Color', 'w');
    sgtitle(sprintf('Linear-Domain Restoration: %s', filename), ...
        'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');

    for i = 1:num_frames
        f_idx  = frame_indices(i);
        f_name = frame_titles{i};

        % ---------- 提取当前帧 ----------
        in_db   = seq_input_DB(:, :, f_idx);
        pred_db = seq_pred_DB(:, :, f_idx);
        gt_db   = seq_GT_DB(:, :, f_idx);

        in_lin       = seq_input_Linear(:, :, f_idx);
        pred_lin_raw = seq_pred_Linear_raw(:, :, f_idx);
        gt_lin       = seq_GT_Linear(:, :, f_idx);

        pred_lin = minmaxnormalize_image(pred_lin_raw, meta_data.V_MAX_GT_L, meta_data.V_MIN_GT_L);

        % ---------- 计算 DB 域指标 ----------
        psnr_in_db   = psnr(in_db, gt_db, 1.0);
        ssim_in_db   = ssim(in_db, gt_db);
        psnr_pred_db = psnr(pred_db, gt_db, 1.0);
        ssim_pred_db = ssim(pred_db, gt_db);

        % ---------- 计算 Linear 域指标 ----------
        psnr_in_lin   = psnr(in_lin, gt_lin, 1.0);
        ssim_in_lin   = ssim(in_lin, gt_lin);
        psnr_pred_lin = psnr(pred_lin, gt_lin, 1.0);
        ssim_pred_lin = ssim(pred_lin, gt_lin);

        % ---------- 确定 Linear 域显示上限 ----------
        disp_max_lin = prctile(gt_lin(:), 99.5);

        %% ================= DB 域 =================
        set(0, 'CurrentFigure', fig_db);

        subplot(3, num_frames, i);
        imagesc(in_db, [0, 1]); colormap(gray); axis off image;
        title(sprintf('%s Input\nPSNR: %.2fdB | SSIM: %.4f', ...
            f_name, psnr_in_db, ssim_in_db), 'FontSize', 12);

        subplot(3, num_frames, num_frames + i);
        imagesc(pred_db, [0, 1]); colormap(gray); axis off image;
        title(sprintf('Prediction\nPSNR: %.2fdB | SSIM: %.4f', ...
            psnr_pred_db, ssim_pred_db), ...
            'FontSize', 12, 'Color', [0.85 0.32 0.09]);

        subplot(3, num_frames, 2 * num_frames + i);
        imagesc(gt_db, [0, 1]); colormap(gray); axis off image;
        title('Ground Truth', 'FontSize', 12);

        %% ================= Linear 域 =================
        set(0, 'CurrentFigure', fig_lin);

        subplot(3, num_frames, i);
        imagesc(in_lin, [0, disp_max_lin]); colormap(gray); axis off image;
        title(sprintf('%s Input\nPSNR: %.2fdB | SSIM: %.4f', ...
            f_name, psnr_in_lin, ssim_in_lin), 'FontSize', 12);

        subplot(3, num_frames, num_frames + i);
        imagesc(pred_lin, [0, disp_max_lin]); colormap(gray); axis off image;
        title(sprintf('Prediction\nPSNR: %.2fdB | SSIM: %.4f', ...
            psnr_pred_lin, ssim_pred_lin), ...
            'FontSize', 12, 'Color', [0 0.44 0.74]);

        subplot(3, num_frames, 2 * num_frames + i);
        imagesc(gt_lin, [0, disp_max_lin]); colormap(gray); axis off image;
        title('Ground Truth', 'FontSize', 12);
    end

    % 如需导出可取消注释
    % save_prefix   = strrep(filename, '_dual_pred', '');
    % save_db_path  = sprintf('./img/Analysis_DB_%s.png', save_prefix);
    % save_lin_path = sprintf('./img/Analysis_Linear_%s.png', save_prefix);
    % exportgraphics(fig_db, save_db_path, 'Resolution', 300);
    % exportgraphics(fig_lin, save_lin_path, 'Resolution', 300);

    close(fig_db);
    close(fig_lin);
end

fprintf('%s\n', repmat('=', 1, 60));
fprintf('🎉 所有序列自动化评测与出图已全部完成！\n');

%% ==========================================
% 附：逆向物理还原函数 (Advanced版)
% ==========================================
function img_linear = sar_inverse_normalize_modality_advanced(img_norm, v_max, v_min)
    img_db = img_norm .* (v_max - v_min) + v_min;
    img_linear = 10.^(img_db / 20);
    img_linear(img_linear < 0) = 0;
end