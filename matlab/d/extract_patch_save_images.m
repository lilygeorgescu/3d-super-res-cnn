function [idx_image] = extract_patch_save_images(image, dim_patch_w, dim_patch_h, stride, resize_factor, folder_in, folder_gt, idx_image)
    [h, w] = size(image);
     
    for i=1:stride:h-dim_patch_h+1
        for j=1:stride:w-dim_patch_w+1
            
            gt_patch = image(i:i+dim_patch_h-1, j:j+dim_patch_w-1);
            [lines, cols] = size(gt_patch);
            sigma = rand(1,1) * 0.5;
            kernel = fspecial('gaussian', [3 3], sigma); 
            if rand(1,1) < 0.5
                in_patch = imresize(imfilter(gt_patch, kernel), [lines, round(cols/resize_factor)]); 
            else
                in_patch = imresize(gt_patch, [lines, round(cols/resize_factor)]); 
            end  
            
            if(max(in_patch(:)) < 5)
                continue;
            end
            idx_image = idx_image + 1;
%             subplot(1, 2, 1); imshow(gt_patch);
%             subplot(1, 2, 2); imshow(in_patch);
%             pause(1);
            imwrite(gt_patch, strcat(folder_gt, sprintf('/%d.png', idx_image)));
            imwrite(in_patch, strcat(folder_in, sprintf('/%d.png', idx_image)));
            
        end
    end
end

