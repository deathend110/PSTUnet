function img_linear = sar_inverse_normalize_modality(img_norm, v_max, v_min)
    % img_norm: 经过归一化、被 U-Net 预测出的 [0, 1] 矩阵
    % v_max: 该场景/模态下对应的全局 V_max (dB)，必须与正向归一化时使用的值完全一致
    
    % 2. 从 [0, 1] 逆向映射回 dB 域
    % 根据正向公式: img_norm = (img_db - v_min) / (v_max - v_min)
    % 逆向推导: img_db = img_norm * (v_max - v_min) + v_min
    img_db = img_norm * (v_max - v_min) + v_min;
    
    % 3. 从 dB 域还原到线性幅度域 (Linear Amplitude)
    % 根据正向公式: img_db = 20 * log10(abs(img_complex) + 1e-5)
    % 逆向推导: abs(img_complex) = 10^(img_db / 20) - 1e-5
    img_linear = 10.^(img_db / 20);
    
    % 4. 截断保护：防止因为浮点误差导致还原出极微小的负数幅度
    img_linear(img_linear < 0) = 0;
end