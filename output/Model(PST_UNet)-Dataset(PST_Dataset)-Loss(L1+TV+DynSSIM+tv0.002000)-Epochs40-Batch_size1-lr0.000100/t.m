clc;
clear;
close all;

%% 文件路径
file_path = 'G:\VSCODE-G\PST_Dataset\Linear\testdata\Bangkok_1_L_seq_000017.mat';

%% 查看文件结构（可选）
disp('===== HDF5 文件结构 =====');
h5disp(file_path);

%% 读取数据
frame_type = h5read(file_path, '/frame_type');
% seq_GT     = h5read(file_path, '/seq_GT');
% seq_input  = h5read(file_path, '/seq_input');
seq_GT     = h5read(file_path, '/seq_GT_L');
seq_input  = h5read(file_path, '/seq_input_L');

%% 显示基本信息
disp('===== 数据信息 =====');
fprintf('frame_type size: %s, class: %s\n', mat2str(size(frame_type)), class(frame_type));
fprintf('seq_GT     size: %s, class: %s\n', mat2str(size(seq_GT)), class(seq_GT));
fprintf('seq_input  size: %s, class: %s\n', mat2str(size(seq_input)), class(seq_input));

%% 选择要显示的帧
frame_idx = 1;   % 可改成 1~16

gt_frame    = seq_GT(:, :, frame_idx);
input_frame = seq_input(:, :, frame_idx);

%% 显示 frame_type
disp('===== frame_type =====');
disp(frame_type);

%% 显示图像
figure;

subplot(1,2,1);
imshow(input_frame, []);
title(['seq\_input - frame ' num2str(frame_idx)]);

subplot(1,2,2);
imshow(gt_frame, []);
title(['seq\_GT - frame ' num2str(frame_idx)]);