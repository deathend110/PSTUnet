

%% SAR 序列相邻帧物理重叠区域 “零误差” 严格验证脚本
clear; clc; close all;

% ==================== 1. 参数设置 ====================
STEP = 128;         % 帧间滑动步长 (方位向)
PATCH_SIZE = 512;   % 图像切片大小
max_val = 65535.0;  % 根据保存时的量化极值设定 (uint16=65535)

% ==================== 2. 加载测试序列 ====================
% 请替换为你的真实文件路径 (这里强烈建议测试 GT 图像)
test_file = "G:\MATLAB-G\SAR Full PSF\Sequence_Dataset_AzimuthMix_q3_rt_only\DB\traindata\suburb_DB_seq_002175.mat";

if exist(test_file, 'file')
    fprintf('⏳ 正在读取文件: %s\n', test_file);
    try
        seq_data = h5read(test_file, '/seq_GT_L');
    catch
        seq_data = h5read(test_file, '/seq_GT');
    end
    seq_data = single(seq_data) / max_val;
else
    warning('❌ 未找到测试文件！生成虚拟张量用于代码演示...');
    seq_data = rand(PATCH_SIZE, PATCH_SIZE, 16, 'single');
end

% ==================== 3. 提取并裁剪相邻帧物理重叠区 ====================
k = 1; % 测试第 1 帧与第 2 帧的交界
frame_A = seq_data(:, :, k);     % 前一帧 (左侧块)
frame_B = seq_data(:, :, k+1);   % 后一帧 (右侧块)

% 截取绝对相同的物理区域
overlap_A = frame_A(:, STEP + 1 : end);  % 截掉前一帧独立的部分 (左侧128)
overlap_B = frame_B(:, 1 : end - STEP);  % 截掉后一帧独立的部分 (右侧128)

% ==================== 4. 计算减法误差与定量指标 ====================
% 1) 直接做带符号的物理减法
diff_map = overlap_A - overlap_B;

% 2) 计算统计学误差
mae_val  = mean(abs(diff_map(:)));  % 平均绝对误差 (MAE)
rmse_val = sqrt(mean(diff_map(:).^2)); % 均方根误差 (RMSE)
psnr_val = psnr(overlap_A, overlap_B);
ssim_val = ssim(overlap_A, overlap_B);

fprintf('\n====== 物理重叠区域 “零点” 评估 ======\n');
fprintf('🔥 平均绝对误差 (MAE)  : %.6f (越接近0越好)\n', mae_val);
fprintf('🔥 均方根误差   (RMSE) : %.6f (越接近0越好)\n', rmse_val);
fprintf('📈 区域 PSNR : %.2f dB\n', psnr_val);
fprintf('📈 区域 SSIM : %.4f\n', ssim_val);

% ==================== 5. 极致可视化 (1x5 布局) ====================
figure('Name', 'Zero-Difference Analysis', 'Position',[50, 200, 1800, 350]);

% 子图 1: 前一帧及裁剪框
subplot(1, 5, 1);
imagesc(frame_A); colormap(gca, 'gray'); axis image off;
title(sprintf('Frame %d (Full)', k), 'FontWeight', 'bold');
hold on; rectangle('Position',[STEP+1, 1, PATCH_SIZE-STEP-1, PATCH_SIZE-1], 'EdgeColor', 'g', 'LineWidth', 2);

% 子图 2: 后一帧及裁剪框
subplot(1, 5, 2);
imagesc(frame_B); colormap(gca, 'gray'); axis image off;
title(sprintf('Frame %d (Full)', k+1), 'FontWeight', 'bold');
hold on; rectangle('Position',[1, 1, PATCH_SIZE-STEP-1, PATCH_SIZE-1], 'EdgeColor', 'g', 'LineWidth', 2);

% 子图 3: 重叠区完美拼接对比 (Left:A | Right:B)
subplot(1, 5, 3);
half_width = floor(size(overlap_A, 2) / 2);
mix_img = overlap_A;
mix_img(:, half_width+1:end) = overlap_B(:, half_width+1:end);
imagesc(mix_img); colormap(gca, 'gray'); axis image off;
hold on; xline(half_width, 'r--', 'LineWidth', 2);
title('Seamless Split: A (Left) | B (Right)', 'FontWeight', 'bold');

% 子图 4: 零点发散冷暖误差图 (重点！)
subplot(1, 5, 4);
imagesc(diff_map); 
% 制作一个 蓝-白-红 的冷暖色表
bwr_cmap = interp1([1, 2, 3],[0 0 1; 1 1 1; 1 0 0], linspace(1, 3, 256));
colormap(gca, bwr_cmap);
% 强制色彩轴以 0 为绝对中心！(假设最大误差不会超过 0.1，超出会截断泛红/泛蓝)
max_err_disp = max(abs(diff_map(:)));
if max_err_disp == 0, max_err_disp = 1e-6; end % 防止纯0报错
clim([-max_err_disp, max_err_disp]); 
colorbar; axis image off;
title(sprintf('Signed Difference (A - B)\nMAE: %.4f', mae_val), 'FontWeight', 'bold');

% 子图 5: 误差分布直方图 (统计学证明)
subplot(1, 5, 5);
histogram(diff_map(:), 100, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none');
xline(0, 'r-', 'LineWidth', 2); % 画出绝对零点红线
grid on;
title('Error Histogram', 'FontWeight', 'bold');
xlabel('Difference Value (A - B)');
ylabel('Pixel Count');