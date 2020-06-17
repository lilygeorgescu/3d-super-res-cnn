function [idx_image] = extract_patch_save_images(image, dim_patch, stride, resize_factor, folder_in, folder_gt, idx_image)
    [h, w] = size(image); 
    
    for i=1:stride:h-dim_patch+1
        for j=1:stride:w-dim_patch+1
            idx_image = idx_image + 1;
            gt_patch = image(i:i+dim_patch-1, j:j+dim_patch-1);
            sigma = rand(1,1) * 0.5;
            kernel = fspecial('gaussian', [3 3], sigma); 
            if rand(1,1) < 0.5
                in_patch = imresize(imfilter(gt_patch, kernel), 1/resize_factor);
            else
                in_patch = imresize(gt_patch, 1/resize_factor);
            end
            
            if(sum(in_patch(:)) == 0)
                continue;
            end
%             subplot(1, 2, 1); imshow(gt_patch);
%             subplot(1, 2, 2); imshow(in_patch);
%             pause(1);
            imwrite(gt_patch, strcat(folder_gt, sprintf('/%d.png', idx_image)));
            imwrite(in_patch, strcat(folder_in, sprintf('/%d.png', idx_image))); 
        end
    end
end

