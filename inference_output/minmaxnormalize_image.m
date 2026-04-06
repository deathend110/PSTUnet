function img_norm = minmaxnormalize_image(x, v_max, v_min)
    % minmax归一化图像到[0,1]
    mag = abs(x);
    
    img_norm = (mag - v_min) / (v_max - v_min);
    img_norm(img_norm > 1) = 1;
    img_norm(img_norm < 0) = 0;
    img_norm = double(img_norm);
end
