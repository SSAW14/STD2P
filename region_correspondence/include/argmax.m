function output = argmax(input,dim)
[~,output] = max(input,[],dim);
output = squeeze(output);
% [n,r,c] = size(input);
% output = zeros(r,c);
%     for i = 1:r
%         for j = 1:c
%             
%             output(i,j) = 
%         end
%     end
end