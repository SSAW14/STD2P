function out = countUnique(input)
x = input(:);
x = sort(x);
d = diff([x;max(x)+1]);
count = diff(find([1;d]));
out = [x(find(d)) count];
