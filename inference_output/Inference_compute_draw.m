clear; clc; close all;

%% ==========================================
% 1. Paths and configuration
% ==========================================
script_dir = fileparts(mfilename('fullpath'));

pred_dir = fullfile(script_dir, 'db_test', 'predictions');
meta_dir = fullfile(script_dir, 'TH');

dataset_root = 'G:\MATLAB-G\SAR Full PSF\Sequence_Dataset_AzimuthMix_q3_rt_only';
DB_dir = fullfile(dataset_root, 'DB', 'testdata');
Linear_dir = fullfile(dataset_root, 'Linear', 'testdata');

group_names = {'High', 'Mixed', 'Low'};
group_ids = [2, 3, 1];
group_colors = [
    0.90, 0.95, 1.00;  % high
    0.95, 0.95, 0.95;  % mixed
    1.00, 0.95, 0.90;  % low
];

%% ==========================================
% 2. Initialize accumulators
% ==========================================
pred_files = dir(fullfile(pred_dir, '*.mat'));
if isempty(pred_files)
    error('No prediction .mat files found under: %s', pred_dir);
end

sum_psnr_in_db = zeros(1, numel(group_ids));
sum_ssim_in_db = zeros(1, numel(group_ids));
sum_psnr_pred_db = zeros(1, numel(group_ids));
sum_ssim_pred_db = zeros(1, numel(group_ids));

sum_psnr_in_lin = zeros(1, numel(group_ids));
sum_ssim_in_lin = zeros(1, numel(group_ids));
sum_psnr_pred_lin = zeros(1, numel(group_ids));
sum_ssim_pred_lin = zeros(1, numel(group_ids));

group_counts = zeros(1, numel(group_ids));

sum_psnr_in_db_by_frame = [];
sum_ssim_in_db_by_frame = [];
sum_psnr_pred_db_by_frame = [];
sum_ssim_pred_db_by_frame = [];

sum_psnr_in_lin_by_frame = [];
sum_ssim_in_lin_by_frame = [];
sum_psnr_pred_lin_by_frame = [];
sum_ssim_pred_lin_by_frame = [];

reference_frame_mode_id = [];
valid_count = 0;

fprintf('Start sequence-wide dual-domain analysis. Found %d prediction files.\n', length(pred_files));
fprintf('%s\n', repmat('=', 1, 72));

%% ==========================================
% 3. Iterate all predicted sequences
% ==========================================
for file_idx = 1:length(pred_files)
    filename_ext = pred_files(file_idx).name;
    pred_filepath = fullfile(pred_dir, filename_ext);
    DB_filepath = fullfile(DB_dir, filename_ext);
    filename_linear = replace(filename_ext, '_DB_', '_L_');
    Linear_filepath = fullfile(Linear_dir, filename_linear);
    [~, filename, ~] = fileparts(pred_filepath);

    fprintf('[%d/%d] Processing sequence: %s\n', file_idx, length(pred_files), filename);

    if ~exist(DB_filepath, 'file')
        warning('Missing DB file, skip: %s', DB_filepath);
        continue;
    end
    if ~exist(Linear_filepath, 'file')
        warning('Missing Linear file, skip: %s', Linear_filepath);
        continue;
    end

    parts = strsplit(filename, '_DB_seq_');
    core_name = append('SAR_Dataset_', parts{1});
    meta_filepath = fullfile(meta_dir, [core_name, '.mat']);
    if ~exist(meta_filepath, 'file')
        warning('Missing normalization metadata, skip: %s', meta_filepath);
        continue;
    end

    meta_data = load(meta_filepath);
    pred_data = load(pred_filepath);

    seq_pred_DB = double(pred_data.seq_pred_DB);
    
    seq_GT_DB = double(h5read(DB_filepath, '/seq_GT')) / 255;

    seq_input_DB = double(h5read(DB_filepath, '/seq_input')) / 255;
    
    seq_GT_Linear = double(h5read(Linear_filepath, '/seq_GT_L')) / 255;
    
    seq_input_Linear = double(h5read(Linear_filepath, '/seq_input_L')) / 255;

    [seq_pred_DB, was_transposed, psnr_direct, psnr_transposed] = ...
        align_prediction_orientation(seq_pred_DB, seq_GT_DB);
    if was_transposed && file_idx == 1
        fprintf('Prediction orientation auto-corrected: direct %.4f dB, transposed %.4f dB\n', ...
            psnr_direct, psnr_transposed);
    end
    
    frame_mode_id = double(h5read(DB_filepath, '/frame_mode_id'));
    frame_mode_id = reshape(frame_mode_id, 1, []);

    seq_pred_Linear_raw = sar_inverse_normalize_modality_advanced( ...
        seq_pred_DB, meta_data.V_MAX_GT, meta_data.V_MIN_GT);

    total_frames = size(seq_pred_DB, 3);
    if isempty(sum_psnr_in_db_by_frame)
        sum_psnr_in_db_by_frame = zeros(1, total_frames);
        sum_ssim_in_db_by_frame = zeros(1, total_frames);
        sum_psnr_pred_db_by_frame = zeros(1, total_frames);
        sum_ssim_pred_db_by_frame = zeros(1, total_frames);
        sum_psnr_in_lin_by_frame = zeros(1, total_frames);
        sum_ssim_in_lin_by_frame = zeros(1, total_frames);
        sum_psnr_pred_lin_by_frame = zeros(1, total_frames);
        sum_ssim_pred_lin_by_frame = zeros(1, total_frames);
    elseif numel(sum_psnr_in_db_by_frame) ~= total_frames
        warning('Frame count mismatch in %s, skip this file.', filename_ext);
        continue;
    end

    if numel(frame_mode_id) ~= total_frames
        warning('frame_mode_id length mismatch in %s, skip this file.', filename_ext);
        continue;
    end

    if isempty(reference_frame_mode_id)
        reference_frame_mode_id = frame_mode_id;
    elseif ~isequal(reference_frame_mode_id, frame_mode_id)
        warning('Frame mode layout differs in %s. Plots use the first valid sequence as reference.', filename_ext);
    end

    cur_psnr_in_db = zeros(1, total_frames);
    cur_ssim_in_db = zeros(1, total_frames);
    cur_psnr_pred_db = zeros(1, total_frames);
    cur_ssim_pred_db = zeros(1, total_frames);

    cur_psnr_in_lin = zeros(1, total_frames);
    cur_ssim_in_lin = zeros(1, total_frames);
    cur_psnr_pred_lin = zeros(1, total_frames);
    cur_ssim_pred_lin = zeros(1, total_frames);

    for f = 1:total_frames
        in_db = seq_input_DB(:, :, f);
        pred_db = seq_pred_DB(:, :, f);
        gt_db = seq_GT_DB(:, :, f);

        in_lin = seq_input_Linear(:, :, f);
        gt_lin = seq_GT_Linear(:, :, f);
        pred_lin = minmaxnormalize_image( ...
            seq_pred_Linear_raw(:, :, f), meta_data.V_MAX_GT_L, meta_data.V_MIN_GT_L);

        cur_psnr_in_db(f) = psnr(in_db, gt_db, 1.0);
        cur_ssim_in_db(f) = ssim(in_db, gt_db);
        cur_psnr_pred_db(f) = psnr(pred_db, gt_db, 1.0);
        cur_ssim_pred_db(f) = ssim(pred_db, gt_db);

        cur_psnr_in_lin(f) = psnr(in_lin, gt_lin, 1.0);
        cur_ssim_in_lin(f) = ssim(in_lin, gt_lin);
        cur_psnr_pred_lin(f) = psnr(pred_lin, gt_lin, 1.0);
        cur_ssim_pred_lin(f) = ssim(pred_lin, gt_lin);
        subplot(121);imagesc(pred_lin); colormap(gca, 'gray'); axis image off;
        subplot(122);imagesc(gt_lin); colormap(gca, 'gray'); axis image off;
        
        group_idx = map_frame_mode_to_group(frame_mode_id(f), group_ids);
        if group_idx > 0
            sum_psnr_in_db(group_idx) = sum_psnr_in_db(group_idx) + cur_psnr_in_db(f);
            sum_ssim_in_db(group_idx) = sum_ssim_in_db(group_idx) + cur_ssim_in_db(f);
            sum_psnr_pred_db(group_idx) = sum_psnr_pred_db(group_idx) + cur_psnr_pred_db(f);
            sum_ssim_pred_db(group_idx) = sum_ssim_pred_db(group_idx) + cur_ssim_pred_db(f);

            sum_psnr_in_lin(group_idx) = sum_psnr_in_lin(group_idx) + cur_psnr_in_lin(f);
            sum_ssim_in_lin(group_idx) = sum_ssim_in_lin(group_idx) + cur_ssim_in_lin(f);
            sum_psnr_pred_lin(group_idx) = sum_psnr_pred_lin(group_idx) + cur_psnr_pred_lin(f);
            sum_ssim_pred_lin(group_idx) = sum_ssim_pred_lin(group_idx) + cur_ssim_pred_lin(f);

            group_counts(group_idx) = group_counts(group_idx) + 1;
        end
    end

    sum_psnr_in_db_by_frame = sum_psnr_in_db_by_frame + cur_psnr_in_db;
    sum_ssim_in_db_by_frame = sum_ssim_in_db_by_frame + cur_ssim_in_db;
    sum_psnr_pred_db_by_frame = sum_psnr_pred_db_by_frame + cur_psnr_pred_db;
    sum_ssim_pred_db_by_frame = sum_ssim_pred_db_by_frame + cur_ssim_pred_db;

    sum_psnr_in_lin_by_frame = sum_psnr_in_lin_by_frame + cur_psnr_in_lin;
    sum_ssim_in_lin_by_frame = sum_ssim_in_lin_by_frame + cur_ssim_in_lin;
    sum_psnr_pred_lin_by_frame = sum_psnr_pred_lin_by_frame + cur_psnr_pred_lin;
    sum_ssim_pred_lin_by_frame = sum_ssim_pred_lin_by_frame + cur_ssim_pred_lin;

    valid_count = valid_count + 1;
end

if valid_count == 0
    error('No valid sequence processed. Check predictions, dataset path, and TH metadata.');
end

%% ==========================================
% 4. Average metrics
% ==========================================
avg_psnr_in_db = safe_divide(sum_psnr_in_db, group_counts);
avg_ssim_in_db = safe_divide(sum_ssim_in_db, group_counts);
avg_psnr_pred_db = safe_divide(sum_psnr_pred_db, group_counts);
avg_ssim_pred_db = safe_divide(sum_ssim_pred_db, group_counts);

avg_psnr_in_lin = safe_divide(sum_psnr_in_lin, group_counts);
avg_ssim_in_lin = safe_divide(sum_ssim_in_lin, group_counts);
avg_psnr_pred_lin = safe_divide(sum_psnr_pred_lin, group_counts);
avg_ssim_pred_lin = safe_divide(sum_ssim_pred_lin, group_counts);

avg_psnr_in_db_by_frame = sum_psnr_in_db_by_frame / valid_count;
avg_ssim_in_db_by_frame = sum_ssim_in_db_by_frame / valid_count;
avg_psnr_pred_db_by_frame = sum_psnr_pred_db_by_frame / valid_count;
avg_ssim_pred_db_by_frame = sum_ssim_pred_db_by_frame / valid_count;

avg_psnr_in_lin_by_frame = sum_psnr_in_lin_by_frame / valid_count;
avg_ssim_in_lin_by_frame = sum_ssim_in_lin_by_frame / valid_count;
avg_psnr_pred_lin_by_frame = sum_psnr_pred_lin_by_frame / valid_count;
avg_ssim_pred_lin_by_frame = sum_ssim_pred_lin_by_frame / valid_count;

%% ==========================================
% 5. Print tables
% ==========================================
total_frames_all = valid_count * numel(reference_frame_mode_id);

fprintf('\n');
fprintf('%s\n', repmat('=', 1, 84));
fprintf('Sequence evaluation summary (%d sequences, %d total frames)\n', valid_count, total_frames_all);
fprintf('%s\n\n', repmat('=', 1, 84));

fprintf('DB domain metrics\n');
fprintf('------------------------------------------------------------------------------------\n');
fprintf(' %-18s | %-12s | %-12s | %-12s\n', 'Metric \\ Group', group_names{1}, group_names{2}, group_names{3});
fprintf('------------------------------------------------------------------------------------\n');
fprintf(' %-18s | %10.2f dB | %10.2f dB | %10.2f dB\n', 'Input PSNR', avg_psnr_in_db(1), avg_psnr_in_db(2), avg_psnr_in_db(3));
fprintf(' %-18s | %10.2f dB | %10.2f dB | %10.2f dB\n', 'Pred  PSNR', avg_psnr_pred_db(1), avg_psnr_pred_db(2), avg_psnr_pred_db(3));
fprintf('------------------------------------------------------------------------------------\n');
fprintf(' %-18s | %10.4f    | %10.4f    | %10.4f\n', 'Input SSIM', avg_ssim_in_db(1), avg_ssim_in_db(2), avg_ssim_in_db(3));
fprintf(' %-18s | %10.4f    | %10.4f    | %10.4f\n', 'Pred  SSIM', avg_ssim_pred_db(1), avg_ssim_pred_db(2), avg_ssim_pred_db(3));
fprintf(' %-18s | %10d    | %10d    | %10d\n', 'Frame Count', group_counts(1), group_counts(2), group_counts(3));
fprintf('------------------------------------------------------------------------------------\n\n');

fprintf('Linear domain metrics\n');
fprintf('------------------------------------------------------------------------------------\n');
fprintf(' %-18s | %-12s | %-12s | %-12s\n', 'Metric \\ Group', group_names{1}, group_names{2}, group_names{3});
fprintf('------------------------------------------------------------------------------------\n');
fprintf(' %-18s | %10.2f dB | %10.2f dB | %10.2f dB\n', 'Input PSNR', avg_psnr_in_lin(1), avg_psnr_in_lin(2), avg_psnr_in_lin(3));
fprintf(' %-18s | %10.2f dB | %10.2f dB | %10.2f dB\n', 'Pred  PSNR', avg_psnr_pred_lin(1), avg_psnr_pred_lin(2), avg_psnr_pred_lin(3));
fprintf('------------------------------------------------------------------------------------\n');
fprintf(' %-18s | %10.4f    | %10.4f    | %10.4f\n', 'Input SSIM', avg_ssim_in_lin(1), avg_ssim_in_lin(2), avg_ssim_in_lin(3));
fprintf(' %-18s | %10.4f    | %10.4f    | %10.4f\n', 'Pred  SSIM', avg_ssim_pred_lin(1), avg_ssim_pred_lin(2), avg_ssim_pred_lin(3));
fprintf(' %-18s | %10d    | %10d    | %10d\n', 'Frame Count', group_counts(1), group_counts(2), group_counts(3));
fprintf('%s\n', repmat('=', 1, 84));

%% ==========================================
% 6. Per-frame trend plots
% ==========================================
x = 1:numel(reference_frame_mode_id);

fig_psnr = figure('Color', 'w', 'Position', [100, 100, 1200, 440]);
ax1 = axes(fig_psnr);
hold(ax1, 'on');

yl1_tmp = [
    avg_psnr_in_db_by_frame, avg_psnr_pred_db_by_frame, ...
    avg_psnr_in_lin_by_frame, avg_psnr_pred_lin_by_frame
];
ymin1 = min(yl1_tmp) - 0.5;
ymax1 = max(yl1_tmp) + 0.5;

draw_mode_background(ax1, reference_frame_mode_id, ymin1, ymax1, group_ids, group_colors);

plot(x, avg_psnr_in_db_by_frame, '-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Input DB PSNR');
plot(x, avg_psnr_pred_db_by_frame, '-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Pred DB PSNR');
plot(x, avg_psnr_in_lin_by_frame, '-s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Input Linear PSNR');
plot(x, avg_psnr_pred_lin_by_frame, '-s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Pred Linear PSNR');

xlim([1 numel(x)]);
ylim([ymin1 ymax1]);
xticks(1:numel(x));
xlabel('Frame Index');
ylabel('PSNR / dB');
title(sprintf('Average PSNR over Frame Index (1~%d) across %d Sequences', numel(x), valid_count));
grid on;
legend('Location', 'best');
annotate_mode_segments(ax1, reference_frame_mode_id, ymax1, ymin1, group_ids, group_names);
hold(ax1, 'off');

fig_ssim = figure('Color', 'w', 'Position', [120, 160, 1200, 440]);
ax2 = axes(fig_ssim);
hold(ax2, 'on');

yl2_tmp = [
    avg_ssim_in_db_by_frame, avg_ssim_pred_db_by_frame, ...
    avg_ssim_in_lin_by_frame, avg_ssim_pred_lin_by_frame
];
ymin2 = min(yl2_tmp) - 0.02;
ymax2 = max(yl2_tmp) + 0.02;

draw_mode_background(ax2, reference_frame_mode_id, ymin2, ymax2, group_ids, group_colors);

plot(x, avg_ssim_in_db_by_frame, '-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Input DB SSIM');
plot(x, avg_ssim_pred_db_by_frame, '-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Pred DB SSIM');
plot(x, avg_ssim_in_lin_by_frame, '-s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Input Linear SSIM');
plot(x, avg_ssim_pred_lin_by_frame, '-s', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Pred Linear SSIM');

xlim([1 numel(x)]);
ylim([ymin2 ymax2]);
xticks(1:numel(x));
xlabel('Frame Index');
ylabel('SSIM');
title(sprintf('Average SSIM over Frame Index (1~%d) across %d Sequences', numel(x), valid_count));
grid on;
legend('Location', 'best');
annotate_mode_segments(ax2, reference_frame_mode_id, ymax2, ymin2, group_ids, group_names);
hold(ax2, 'off');

%% ==========================================
% 7. Auxiliary functions
% ==========================================
function out = safe_divide(numerator, denominator)
    out = nan(size(numerator));
    valid = denominator > 0;
    out(valid) = numerator(valid) ./ denominator(valid);
end

function group_idx = map_frame_mode_to_group(frame_mode_id, group_ids)
    group_idx = find(group_ids == frame_mode_id, 1);
    if isempty(group_idx)
        group_idx = 0;
    end
end

function draw_mode_background(ax, frame_mode_id, ymin, ymax, group_ids, group_colors)
    segments = find_mode_segments(frame_mode_id);
    axes(ax);
    for s = 1:size(segments, 1)
        seg_start = segments(s, 1);
        seg_end = segments(s, 2);
        seg_mode = segments(s, 3);
        color_idx = find(group_ids == seg_mode, 1);
        if isempty(color_idx)
            continue;
        end
        patch([seg_start - 0.5, seg_end + 0.5, seg_end + 0.5, seg_start - 0.5], ...
              [ymin, ymin, ymax, ymax], ...
              group_colors(color_idx, :), ...
              'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    end
end

function annotate_mode_segments(ax, frame_mode_id, ymax, ymin, group_ids, group_names)
    segments = find_mode_segments(frame_mode_id);
    label_y = ymax - 0.08 * (ymax - ymin);
    axes(ax);
    for s = 1:size(segments, 1)
        seg_start = segments(s, 1);
        seg_end = segments(s, 2);
        seg_mode = segments(s, 3);
        label_idx = find(group_ids == seg_mode, 1);
        if isempty(label_idx)
            continue;
        end
        text((seg_start + seg_end) / 2, label_y, group_names{label_idx}, ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
end

function segments = find_mode_segments(frame_mode_id)
    segments = zeros(0, 3);
    if isempty(frame_mode_id)
        return;
    end

    start_idx = 1;
    current_mode = frame_mode_id(1);
    for idx = 2:numel(frame_mode_id)
        if frame_mode_id(idx) ~= current_mode
            segments(end + 1, :) = [start_idx, idx - 1, current_mode]; %#ok<AGROW>
            start_idx = idx;
            current_mode = frame_mode_id(idx);
        end
    end
    segments(end + 1, :) = [start_idx, numel(frame_mode_id), current_mode];
end

function [seq_pred_DB, was_transposed, psnr_direct, psnr_transposed] = ...
    align_prediction_orientation(seq_pred_DB, seq_GT_DB)
    total_frames = size(seq_pred_DB, 3);
    probe_frames = unique(round(linspace(1, total_frames, min(total_frames, 3))));

    psnr_direct_vals = zeros(1, numel(probe_frames));
    psnr_transposed_vals = zeros(1, numel(probe_frames));
    for idx = 1:numel(probe_frames)
        f = probe_frames(idx);
        pred_frame = seq_pred_DB(:, :, f);
        gt_frame = seq_GT_DB(:, :, f);
        psnr_direct_vals(idx) = psnr(pred_frame, gt_frame, 1.0);
        psnr_transposed_vals(idx) = psnr(pred_frame.', gt_frame, 1.0);
    end

    psnr_direct = mean(psnr_direct_vals);
    psnr_transposed = mean(psnr_transposed_vals);
    was_transposed = psnr_transposed > psnr_direct + 0.1;
    if was_transposed
        seq_pred_DB = permute(seq_pred_DB, [2, 1, 3]);
    end
end

function img_linear = sar_inverse_normalize_modality_advanced(img_norm, v_max, v_min)
    img_db = img_norm .* (v_max - v_min) + v_min;
    img_linear = 10.^(img_db / 20) - 1e-5;
    img_linear(img_linear < 0) = 0;
end
