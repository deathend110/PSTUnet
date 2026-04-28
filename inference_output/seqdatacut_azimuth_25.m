clear; clc; close all;

%% =========================================================
%  主脚本：生成方位向 mixed 采样率序列数据集（去高斯噪声净化版）
%
%  核心逻辑：
%  1) 全局 512 有效轴定义 H/L/H/L/H = 640/896/512/896/640
%  2) 每条序列先切 1200x4272 的 60MHz 大块
%  3) 序列级生成 master 随机相位阈值场（无高斯噪声）
%  4) 每帧取 1200x1200 patch，GT / low / high / mixed 统一通过成像接口
%  5) 输出 600x600，再裁成 512x512，堆成 25 帧序列
%  6) HDF5(.mat) 保存 DB / Linear 两套
%% =========================================================

%% ==================== 配置区 ====================
DIR_LIST = ["SAR_Dataset_Bangkok_1", "SAR_Dataset_city1_histeq", ...
            "SAR_Dataset_city2_histeq", "SAR_Dataset_SAR_figure", ...
            "SAR_Dataset_filed", "SAR_Dataset_port", "SAR_Dataset_suburb"];

% 序列参数
N_SEQ          = 25;
STEP           = 128;
SEQ_STEP       = 4 * STEP;       % 序列之间滑动步长，可调整
SIG_H          = 1200;           % 每帧输入信号高度
SIG_W          = 1200;           % 每帧输入信号宽度
IMG_VALID      = 600;            % 成像后有效区域大小
PATCH_SIZE     = 512;            % 最终入序列 patch
VALID_MARGIN   = (SIG_W - PATCH_SIZE) / 2;   % 344
LOGIC_LEN_512  = PATCH_SIZE + (N_SEQ - 1) * STEP;   % 3584
INPUT_LEN_1200 = SIG_W      + (N_SEQ - 1) * STEP;   % 4272

% RT / mixed 参数
q             = 3;       % 方位向上采样倍率
As            = 0.6;     % RT 阈值系数
TRANS_WIDTH   = 64;      % mixed 边界平滑宽度
EDGE_BUFFER   = 50;      % mixed 能量对齐 buffer

% 数据集命名
BASE_TAG = "Sequence_Dataset_AzimuthMix_q3_rt_only";

rng(42);

%% ==================== 参数加载 ====================
S60 = load("FS60_params.mat");

%% ==================== 全局 512 有效轴模式 ====================
% H(640) - L(896) - H(512) - L(896) - H(640)
global_mode_mask_512 = build_global_mode_mask_512(LOGIC_LEN_512);

%% ==================== 输出目录 ====================
BASE_DIR = BASE_TAG;
if ~exist(BASE_DIR, 'dir'), mkdir(BASE_DIR); end
disp(BASE_DIR);

TRAIN_DB_DIR = fullfile(BASE_DIR, 'DB', 'traindata');
TEST_DB_DIR  = fullfile(BASE_DIR, 'DB', 'testdata');
TRAIN_L_DIR  = fullfile(BASE_DIR, 'Linear', 'traindata');
TEST_L_DIR   = fullfile(BASE_DIR, 'Linear', 'testdata');

if ~exist(TRAIN_DB_DIR, 'dir'), mkdir(TRAIN_DB_DIR); end
if ~exist(TEST_DB_DIR,  'dir'), mkdir(TEST_DB_DIR);  end
if ~exist(TRAIN_L_DIR,  'dir'), mkdir(TRAIN_L_DIR);  end
if ~exist(TEST_L_DIR,   'dir'), mkdir(TEST_L_DIR);   end

global_train_idx = 1;
global_test_idx  = 1;

%% ==================== 主循环 ====================
for d = 1:length(DIR_LIST)
    tic;
    ROOT_DIR = DIR_LIST(d);
    core_name = strrep(ROOT_DIR, 'SAR_Dataset_', '');

    % 搜索并排序轨迹文件
    filePattern = fullfile(ROOT_DIR, 'rstart*.mat');
    filestructs = dir(filePattern);
    [~, sort_idx] = sort({filestructs.name});
    filestructs = filestructs(sort_idx);

    num_files = length(filestructs);
    if num_files < 2
        warning('底图 [%s] 下文件不足 2 个，跳过。', ROOT_DIR);
        continue;
    end

    % 90% / 10% 划分 test
    num_test = round(num_files * 0.10);
    if num_test == 0 && num_files >= 2
        num_test = 1;
    end

    train_files = filestructs(1 : end - num_test);
    test_files  = filestructs(end - num_test + 1 : end);

    % 载入该底图归一化参数文件
    THpath = fullfile(ROOT_DIR, ROOT_DIR + ".mat");
    load(THpath);
    norm_ctx = collect_norm_context_from_workspace();

    fprintf('\n[%s] 共 %d 条轨迹 | 训练集: %d 条 (%.1f%%) | 测试集: %d 条\n', ...
        ROOT_DIR, num_files, length(train_files), ...
        (length(train_files) / num_files) * 100, length(test_files));

    split_sets  = {train_files, test_files};
    split_names = {'Train', 'Test'};

    for split_id = 1:2
        current_files = split_sets{split_id};
        current_mode  = split_names{split_id};

        for i = 1:length(current_files)
            filepath = fullfile(ROOT_DIR, current_files(i).name);
            fprintf('  正在处理 [%s]: %s\n', current_mode, current_files(i).name);

            tempData = load(filepath);
            varNames = fieldnames(tempData);
            raw_data = tempData.(varNames{1});    % 原始高采样率回波

            % 不再加高斯噪声，直接构造 60MHz clean / input 母体
            channel_60_clean = raw_data(1:3:end, :);
            channel_60_input = channel_60_clean;

            [h60, width60] = size(channel_60_clean);
            if h60 < SIG_H
                warning('60MHz 数据高度不足 %d，跳过: %s', SIG_H, filepath);
                continue;
            end

            % 能切出 1200x4272 大块的起点
            max_start = width60 - INPUT_LEN_1200 + 1;
            if max_start < 1
                continue;
            end

            block_starts = 1 : SEQ_STEP : max_start;
            if block_starts(end) < max_start
                block_starts(end + 1) = max_start;
            end

            for block_start = block_starts
                % ========== 切当前序列的大块：1200 x 4272 ==========
                seq60_clean = extract_sequence_block_60mhz(channel_60_clean, block_start, SIG_H, INPUT_LEN_1200);
                seq60_input = extract_sequence_block_60mhz(channel_60_input, block_start, SIG_H, INPUT_LEN_1200);

                % ========== 序列级 master 阈值场（无高斯噪声） ==========
                [U_master_seq, sigma_seq, A_rt] = build_master_threshold_seq(seq60_input, q, As);

                % ========== 序列容器 ==========
                seq_GT      = zeros(PATCH_SIZE, PATCH_SIZE, N_SEQ, 'single');
                seq_GT_L    = zeros(PATCH_SIZE, PATCH_SIZE, N_SEQ, 'single');
                seq_input   = zeros(PATCH_SIZE, PATCH_SIZE, N_SEQ, 'single');
                seq_input_L = zeros(PATCH_SIZE, PATCH_SIZE, N_SEQ, 'single');

                frame_mode_id     = zeros(1, N_SEQ, 'uint8');
                mode_mask_512_all = zeros(N_SEQ, PATCH_SIZE, 'uint8');

                for k = 1:N_SEQ
                    % ---------- 当前帧 patch ----------
                    [signal60_clean_patch, signal60_input_patch, U_master_patch] = ...
                        get_frame_patches(seq60_clean, seq60_input, U_master_seq, k, STEP, SIG_W, q);

                    % ---------- 当前帧模式掩码 ----------
                    [mode_mask_512, mode_mask_1200, frame_mode] = ...
                        get_frame_mode_masks(global_mode_mask_512, k, STEP, PATCH_SIZE, SIG_W, VALID_MARGIN);

                    % ---------- GT 成像 ----------
                    [img_gt_db_600, img_gt_l_600] = imaging_interface( ...
                        signal60_clean_patch, [], [], "gt", ...
                        S60, q, TRANS_WIDTH, EDGE_BUFFER, IMG_VALID, norm_ctx);

                    % ---------- input 成像 ----------
                    [img_in_db_600, img_in_l_600] = imaging_interface( ...
                        signal60_input_patch, U_master_patch, mode_mask_1200, frame_mode, ...
                        S60, q, TRANS_WIDTH, EDGE_BUFFER, IMG_VALID, norm_ctx);

                    % ---------- 600 -> 512 ----------
                    seq_GT(:,:,k)      = crop_center(img_gt_db_600, PATCH_SIZE);
                    seq_GT_L(:,:,k)    = crop_center(img_gt_l_600,  PATCH_SIZE);
                    seq_input(:,:,k)   = crop_center(img_in_db_600, PATCH_SIZE);
                    seq_input_L(:,:,k) = crop_center(img_in_l_600,  PATCH_SIZE);

                    frame_mode_id(k)       = encode_frame_mode(frame_mode);
                    mode_mask_512_all(k,:) = uint8(mode_mask_512);
                end

                % ========== 命名 ==========
                if split_id == 1
                    save_name_db = sprintf('%s_DB_seq_%06d.mat', core_name, global_train_idx);
                    save_path_db = fullfile(TRAIN_DB_DIR, save_name_db);

                    save_name_l = sprintf('%s_L_seq_%06d.mat', core_name, global_train_idx);
                    save_path_l = fullfile(TRAIN_L_DIR, save_name_l);
                else
                    save_name_db = sprintf('%s_DB_seq_%06d.mat', core_name, global_test_idx);
                    save_path_db = fullfile(TEST_DB_DIR, save_name_db);

                    save_name_l = sprintf('%s_L_seq_%06d.mat', core_name, global_test_idx);
                    save_path_l = fullfile(TEST_L_DIR, save_name_l);
                end

                % ========== 转 uint8 保存 ==========
                seq_GT_u8      = to_uint8_image(seq_GT);
                seq_GT_L_u8    = to_uint8_image(seq_GT_L);
                seq_input_u8   = to_uint8_image(seq_input);
                seq_input_L_u8 = to_uint8_image(seq_input_L);

                save_sequence_hdf5( ...
                    save_path_db, save_path_l, ...
                    seq_GT_u8, seq_input_u8, ...
                    seq_GT_L_u8, seq_input_L_u8, ...
                    frame_mode_id, mode_mask_512_all, sigma_seq, A_rt);

                if split_id == 1
                    global_train_idx = global_train_idx + 1;
                else
                    global_test_idx = global_test_idx + 1;
                end
            end
        end
    end

    t = toc;
    fprintf("✅ [%s] 处理完毕！耗时: %.2fs\n", ROOT_DIR, t);
end

fprintf('\n🎉 全部完成。\n');

%% =========================================================
%% ==================== 局部函数区 =========================
%% =========================================================

function global_mode_mask_512 = build_global_mode_mask_512(logic_len)
    assert(logic_len == 3584, '当前固定逻辑长度应为 3584');
    global_mode_mask_512 = false(1, logic_len);  % false=low, true=high

    % H(640) - L(896) - H(512) - L(896) - H(640)
    global_mode_mask_512(1:640)       = true;
    global_mode_mask_512(1537:2048)   = true;
    global_mode_mask_512(2945:3584)   = true;
end

function norm_ctx = collect_norm_context_from_workspace()
    cand = {
        'V_MAX_GT','V_MIN_GT','V_MAX_GT_L','V_MIN_GT_L', ...
        'V_MAX_60','V_MIN_60','V_MAX_60_L','V_MIN_60_L', ...
        'V_MAX_180','V_MIN_180','V_MAX_180_L','V_MIN_180_L', ...
        'V_MAX_RT','V_MIN_RT','V_MAX_RT_L','V_MIN_RT_L'
    };

    norm_ctx = struct();
    for i = 1:numel(cand)
        if evalin('caller', sprintf("exist('%s','var')", cand{i}))
            norm_ctx.(cand{i}) = evalin('caller', cand{i});
        end
    end
end

function seq_block = extract_sequence_block_60mhz(channel_60, block_start, sig_h, block_len)
    row_end = sig_h;
    col_end = block_start + block_len - 1;

    if size(channel_60, 1) < row_end
        error('extract_sequence_block_60mhz: 行数不足');
    end
    if size(channel_60, 2) < col_end
        error('extract_sequence_block_60mhz: 列数不足');
    end

    seq_block = channel_60(1:row_end, block_start:col_end);
end

function [U_master_seq, sigma_seq, A_rt] = build_master_threshold_seq(seq60_input, q, As)
    seq_up = azimuth_upsample_fft(seq60_input, q);   % [1200 x (3*4272)]

    sigma_seq = sqrt(2 / pi) * mean(abs(seq_up(:)));
    A_rt = As * sigma_seq;

    phi_seq = 2 * pi * rand(1, size(seq_up, 2));     % 序列级唯一随机相位
    U_master_seq = A_rt * exp(1i * phi_seq);         % [1 x (3*4272)]
end

function [signal60_clean_patch, signal60_input_patch, U_master_patch] = ...
    get_frame_patches(seq60_clean, seq60_input, U_master_seq, k, step, sig_w, q)

    input_s = 1 + (k - 1) * step;
    input_e = input_s + sig_w - 1;

    signal60_clean_patch = seq60_clean(:, input_s:input_e);
    signal60_input_patch = seq60_input(:, input_s:input_e);

    fine_s = q * (input_s - 1) + 1;
    fine_e = fine_s + q * sig_w - 1;
    U_master_patch = U_master_seq(fine_s:fine_e);   % [1 x 3600]
end

function [mode_mask_512, mode_mask_1200, frame_mode] = ...
    get_frame_mode_masks(global_mode_mask_512, k, step, patch_size, sig_w, valid_margin)

    valid_s = 1 + (k - 1) * step;
    valid_e = valid_s + patch_size - 1;

    mode_mask_512 = global_mode_mask_512(valid_s:valid_e);

    mode_mask_1200 = [ ...
        repmat(mode_mask_512(1),   1, valid_margin), ...
        mode_mask_512, ...
        repmat(mode_mask_512(end), 1, valid_margin) ...
    ];

    assert(numel(mode_mask_1200) == sig_w, 'mode_mask_1200 长度必须为 1200');

    if all(mode_mask_512)
        frame_mode = "high";
    elseif all(~mode_mask_512)
        frame_mode = "low";
    else
        frame_mode = "mixed";
    end
end

function [img_db_600, img_l_600] = imaging_interface( ...
    signal60_patch, U_master_patch, mode_mask_1200, frame_mode, ...
    S60, q, trans_width, edge_buffer, img_valid, norm_ctx)

    switch string(frame_mode)
        case "gt"
            RC = Range_Compress( ...
                signal60_patch, ...
                S60.fc, S60.tnrn, S60.gama, S60.R0, S60.C, S60.Fs, S60.Tp);
            norm_key = "gt";

        case "low"
            RC = build_low_rc(signal60_patch, U_master_patch, S60, q);
            norm_key = "low";

        case "high"
            RC = build_high_rc(signal60_patch, U_master_patch, S60, q);
            norm_key = "high";

        case "mixed"
            [RC, ~] = build_mixed_rc_from_master( ...
                signal60_patch, U_master_patch, mode_mask_1200, ...
                S60, q, trans_width, edge_buffer);
            norm_key = "mixed";

        otherwise
            error('未知 frame_mode: %s', string(frame_mode));
    end

    RCMC_out = RCMC(RC, S60.lambda, S60.fnrn, S60.fnan, S60.R0, S60.C, S60.v);
    IMG = SAR_Imaging(RCMC_out, S60.lambda, S60.Fs, S60.R0, S60.C, ...
                      S60.v, S60.tnan, S60.Ta, S60.prf);

    roi600 = extract_valid_image(abs(IMG), S60, img_valid);
    [img_db_600, img_l_600] = normalize_roi_pair(roi600, norm_key, norm_ctx);
end

function RC_low = build_low_rc(signal60_patch, U_master_patch, S60, q)
    [nrn, ~] = size(signal60_patch);

    u_low_vec = U_master_patch(1:q:end);   % 已验证 1:3:end 对齐原 coarse grid
    U_low = repmat(u_low_vec, nrn, 1);

    sig_low_1bit = quantize_1bit_with_U(signal60_patch, U_low);

    RC_low = Range_Compress( ...
        sig_low_1bit, ...
        S60.fc, S60.tnrn, S60.gama, S60.R0, S60.C, S60.Fs, S60.Tp);
end

function RC_high = build_high_rc(signal60_patch, U_master_patch, S60, q)
    [nrn, nan0] = size(signal60_patch);

    signal_up = azimuth_upsample_fft(signal60_patch, q);
    U_high = repmat(U_master_patch, nrn, 1);

    sig_high_1bit = quantize_1bit_with_U(signal_up, U_high);

    RC_high_up = Range_Compress( ...
        sig_high_1bit, ...
        S60.fc, S60.tnrn, S60.gama, S60.R0, S60.C, S60.Fs, S60.Tp);

    RC_high = crop_azimuth_doppler_to_width(RC_high_up, nan0);
end

function [RC_mix, dbg] = build_mixed_rc_from_master( ...
    signal60_patch, ...
    U_master_patch, ...
    mode_mask, ...
    S60, ...
    q, ...
    trans_width, ...
    edge_buffer)

    [nrn, nan0] = size(signal60_patch);
    assert(nan0 == 1200, 'signal60_patch 宽度必须是 1200');
    assert(numel(U_master_patch) == q * nan0, 'U_master_patch 长度必须是 q*1200');
    assert(numel(mode_mask) == nan0, 'mode_mask 长度必须是 1200');

    mode_mask = reshape(logical(mode_mask), 1, []);
    dbg = struct();

    u_low_vec = U_master_patch(1:q:end);
    U_low = repmat(u_low_vec, nrn, 1);

    sig_low_1bit = quantize_1bit_with_U(signal60_patch, U_low);
    RC_low = Range_Compress( ...
        sig_low_1bit, ...
        S60.fc, S60.tnrn, S60.gama, S60.R0, S60.C, S60.Fs, S60.Tp);

    signal_up = azimuth_upsample_fft(signal60_patch, q);
    U_high = repmat(U_master_patch, nrn, 1);

    sig_high_1bit = quantize_1bit_with_U(signal_up, U_high);
    RC_high_up = Range_Compress( ...
        sig_high_1bit, ...
        S60.fc, S60.tnrn, S60.gama, S60.R0, S60.C, S60.Fs, S60.Tp);

    RC_high = crop_azimuth_doppler_to_width(RC_high_up, nan0);

    if all(~mode_mask)
        RC_mix = RC_low;
        dbg.scale_factor = 1;
        dbg.w_high = zeros(1, nan0);
        dbg.boundaries = [];
        return;
    elseif all(mode_mask)
        RC_mix = RC_high;
        dbg.scale_factor = 1;
        dbg.w_high = ones(1, nan0);
        dbg.boundaries = [];
        return;
    end

    boundaries = find(diff(double(mode_mask)) ~= 0);
    scale_list = [];

    for b = boundaries
        left_is_high  = mode_mask(b);
        right_is_high = mode_mask(b + 1);

        idx_left  = max(1, b - edge_buffer + 1) : b;
        idx_right = (b + 1) : min(nan0, b + edge_buffer);

        if left_is_high && ~right_is_high
            p_high = mean(abs(RC_high(:, idx_left)).^2, 'all');
            p_low  = mean(abs(RC_low(:,  idx_right)).^2, 'all');
        elseif ~left_is_high && right_is_high
            p_high = mean(abs(RC_high(:, idx_right)).^2, 'all');
            p_low  = mean(abs(RC_low(:,  idx_left)).^2, 'all');
        else
            continue;
        end

        if p_high > 0 && p_low > 0
            scale_list(end + 1) = sqrt(p_low / p_high); %#ok<AGROW>
        end
    end

    if isempty(scale_list)
        scale_factor = 1;
    else
        scale_factor = median(scale_list);
    end

    RC_high = RC_high * scale_factor;

    w_high = build_soft_weight_from_mask(mode_mask, trans_width);
    W = repmat(w_high, nrn, 1);

    RC_mix = W .* RC_high + (1 - W) .* RC_low;

    dbg.scale_factor = scale_factor;
    dbg.w_high = w_high;
    dbg.boundaries = boundaries;
end

function S1 = quantize_1bit_with_U(S, U)
    re = ones(size(S), 'like', real(S));
    im = ones(size(S), 'like', real(S));

    re(real(S) + real(U) < 0) = -1;
    im(imag(S) + imag(U) < 0) = -1;

    S1 = complex(re, im);
end

function w_high = build_soft_weight_from_mask(mode_mask, trans_width)
    nan0 = numel(mode_mask);
    w_high = double(mode_mask);

    half_w = trans_width / 2;
    assert(mod(trans_width, 2) == 0, 'trans_width 建议取偶数，例如 64');

    boundaries = find(diff(double(mode_mask)) ~= 0);

    for b = boundaries
        l = max(1, b - half_w + 1);
        r = min(nan0, b + half_w);

        len = r - l + 1;
        if len <= 1
            continue;
        end

        if mode_mask(b) == 1 && mode_mask(b + 1) == 0
            ramp = linspace(1, 0, len);
        else
            ramp = linspace(0, 1, len);
        end

        w_high(l:r) = ramp;
    end
end

function S_up = azimuth_upsample_fft(S, q)
    [Nr, Na] = size(S);
    Na_up = q * Na;

    Sf = fftshift(fft(S, [], 2), 2);

    pad_total = Na_up - Na;
    pad_left  = floor(pad_total / 2);
    pad_right = pad_total - pad_left;

    Sf_up = [zeros(Nr, pad_left, 'like', Sf), ...
             Sf, ...
             zeros(Nr, pad_right, 'like', Sf)];

    S_up = ifft(ifftshift(Sf_up, 2), [], 2) * q;
end

function X_crop = crop_azimuth_doppler_to_width(X, target_width)
    [~, Na_up] = size(X);
    if target_width > Na_up
        error('target_width cannot be larger than current width.');
    end

    Xf = fftshift(fft(X, [], 2), 2);

    c = floor(Na_up / 2) + 1;
    h = floor(target_width / 2);

    if mod(target_width, 2) == 0
        idx = (c - h):(c + h - 1);
    else
        idx = (c - h):(c + h);
    end

    Xf_crop = Xf(:, idx);
    X_crop = ifft(ifftshift(Xf_crop, 2), [], 2);
end

function roi = extract_valid_image(img_abs, meta, valid_size)
    roi0 = extract_roi(img_abs, meta);
    roi = crop_center(roi0, valid_size);
end

function roi = extract_roi(img, meta)
    rngIdx = 1:size(img, 1);
    azIdx  = 1:size(img, 2);

    if isfield(meta, 'nrn') && isfield(meta, 'R_total')
        r0 = floor(meta.nrn / 2 - meta.R_total / 2) + 1;
        r1 = r0 + meta.R_total - 1;
        rngIdx = max(1, r0) : min(size(img, 1), r1);
    end

    if isfield(meta, 'nan') && isfield(meta, 'A_num')
        a0 = floor(meta.nan / 2 - meta.A_num / 2) + 1;
        a1 = a0 + meta.A_num - 1;
        azIdx = max(1, a0) : min(size(img, 2), a1);
    end

    roi = img(rngIdx, azIdx);
end

function patch = crop_center(img, patch_size)
    [h, w] = size(img);
    assert(patch_size <= h && patch_size <= w, 'crop_center: patch_size 超出图像范围');

    r0 = floor((h - patch_size) / 2) + 1;
    c0 = floor((w - patch_size) / 2) + 1;

    patch = img(r0:r0 + patch_size - 1, c0:c0 + patch_size - 1);
end

function [img_db, img_l] = normalize_roi_pair(roi_mag, mode_key, norm_ctx)
    [vmax_db, vmin_db, vmax_l, vmin_l] = pick_norm_pair(norm_ctx, mode_key);

    img_db = normalize_one(roi_mag, vmax_db, vmin_db, "db");
    img_l  = normalize_one(roi_mag, vmax_l,  vmin_l,  "linear");
end

function [vmax_db, vmin_db, vmax_l, vmin_l] = pick_norm_pair(norm_ctx, mode_key)
    vmax_db = []; vmin_db = []; vmax_l = []; vmin_l = [];

    switch string(mode_key)
        case "gt"
            [vmax_db, vmin_db] = read_pair(norm_ctx, 'V_MAX_GT', 'V_MIN_GT');
            [vmax_l,  vmin_l]  = read_pair(norm_ctx, 'V_MAX_GT_L', 'V_MIN_GT_L');

        case "low"
            [vmax_db, vmin_db] = read_pair(norm_ctx, 'V_MAX_60', 'V_MIN_60');
            [vmax_l,  vmin_l]  = read_pair(norm_ctx, 'V_MAX_60_L', 'V_MIN_60_L');

        case {"high", "mixed"}
            if isfield(norm_ctx, 'V_MAX_RT') && isfield(norm_ctx, 'V_MIN_RT')
                [vmax_db, vmin_db] = read_pair(norm_ctx, 'V_MAX_RT', 'V_MIN_RT');
                [vmax_l,  vmin_l]  = read_pair(norm_ctx, 'V_MAX_RT_L', 'V_MIN_RT_L');
            elseif isfield(norm_ctx, 'V_MAX_180') && isfield(norm_ctx, 'V_MIN_180')
                [vmax_db, vmin_db] = read_pair(norm_ctx, 'V_MAX_180', 'V_MIN_180');
                [vmax_l,  vmin_l]  = read_pair(norm_ctx, 'V_MAX_180_L', 'V_MIN_180_L');
            else
                [vmax_db, vmin_db] = read_pair(norm_ctx, 'V_MAX_60', 'V_MIN_60');
                [vmax_l,  vmin_l]  = read_pair(norm_ctx, 'V_MAX_60_L', 'V_MIN_60_L');
            end
    end
end

function [vmax, vmin] = read_pair(s, name_max, name_min)
    if isfield(s, name_max) && isfield(s, name_min)
        vmax = s.(name_max);
        vmin = s.(name_min);
    else
        vmax = [];
        vmin = [];
    end
end

function out = normalize_one(img, vmax, vmin, mode_kind)
    if ~isempty(vmax) && ~isempty(vmin)
        if mode_kind == "db" && exist('sar_normalize_modality', 'file')
            out = sar_normalize_modality(img, vmax, vmin);
        elseif mode_kind == "linear" && exist('minmaxnormalize_image', 'file')
            out = minmaxnormalize_image(img, vmax, vmin);
        else
            out = (img - vmin) ./ (vmax - vmin + eps);
        end
    else
        out = img ./ (max(img(:)) + eps);
    end

    out = max(0, min(1, out));
end

function out = to_uint8_image(x)
    x = max(0, min(1, x));
    out = uint8(round(x * 255));
end

function id = encode_frame_mode(frame_mode)
    switch string(frame_mode)
        case "low"
            id = uint8(1);
        case "high"
            id = uint8(2);
        case "mixed"
            id = uint8(3);
        case "gt"
            id = uint8(4);
        otherwise
            id = uint8(0);
    end
end

function save_sequence_hdf5( ...
    save_path_db, save_path_l, ...
    seq_GT_u8, seq_input_u8, ...
    seq_GT_L_u8, seq_input_L_u8, ...
    frame_mode_id, mode_mask_512_all, sigma_seq, A_rt)

    chunk_sz = [128, 128, size(seq_GT_u8, 3)];
    max_retries = 3;

    success_db = false;
    for retry_idx = 1:max_retries
        try
            if exist(save_path_db, 'file'), delete(save_path_db); end

            h5create(save_path_db, '/seq_GT',    size(seq_GT_u8),    ...
                'Datatype', 'uint8', 'ChunkSize', chunk_sz, 'Deflate', 5);
            h5create(save_path_db, '/seq_input', size(seq_input_u8), ...
                'Datatype', 'uint8', 'ChunkSize', chunk_sz, 'Deflate', 5);

            h5create(save_path_db, '/frame_mode_id', size(frame_mode_id), 'Datatype', 'uint8');
            h5create(save_path_db, '/mode_mask_512_all', size(mode_mask_512_all), 'Datatype', 'uint8');
            h5create(save_path_db, '/sigma_seq', 1, 'Datatype', 'single');
            h5create(save_path_db, '/A_rt',      1, 'Datatype', 'single');

            h5write(save_path_db, '/seq_GT',    seq_GT_u8);
            h5write(save_path_db, '/seq_input', seq_input_u8);
            h5write(save_path_db, '/frame_mode_id', frame_mode_id);
            h5write(save_path_db, '/mode_mask_512_all', mode_mask_512_all);
            h5write(save_path_db, '/sigma_seq', single(sigma_seq));
            h5write(save_path_db, '/A_rt',      single(A_rt));

            success_db = true;
            break;
        catch ME
            fprintf('  ⚠️ DB 写入失败 (第 %d 次): %s\n', retry_idx, ME.message);
            pause(2);
        end
    end

    if ~success_db
        error('DB 文件写入失败: %s', save_path_db);
    end

    success_l = false;
    for retry_idx = 1:max_retries
        try
            if exist(save_path_l, 'file'), delete(save_path_l); end

            h5create(save_path_l, '/seq_GT_L',    size(seq_GT_L_u8),    ...
                'Datatype', 'uint8', 'ChunkSize', chunk_sz, 'Deflate', 5);
            h5create(save_path_l, '/seq_input_L', size(seq_input_L_u8), ...
                'Datatype', 'uint8', 'ChunkSize', chunk_sz, 'Deflate', 5);

            h5create(save_path_l, '/frame_mode_id', size(frame_mode_id), 'Datatype', 'uint8');
            h5create(save_path_l, '/mode_mask_512_all', size(mode_mask_512_all), 'Datatype', 'uint8');
            h5create(save_path_l, '/sigma_seq', 1, 'Datatype', 'single');
            h5create(save_path_l, '/A_rt',      1, 'Datatype', 'single');

            h5write(save_path_l, '/seq_GT_L',    seq_GT_L_u8);
            h5write(save_path_l, '/seq_input_L', seq_input_L_u8);
            h5write(save_path_l, '/frame_mode_id', frame_mode_id);
            h5write(save_path_l, '/mode_mask_512_all', mode_mask_512_all);
            h5write(save_path_l, '/sigma_seq', single(sigma_seq));
            h5write(save_path_l, '/A_rt',      single(A_rt));

            success_l = true;
            break;
        catch ME
            fprintf('  ⚠️ Linear 写入失败 (第 %d 次): %s\n', retry_idx, ME.message);
            pause(2);
        end
    end

    if ~success_l
        error('Linear 文件写入失败: %s', save_path_l);
    end
end