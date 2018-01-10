function [success_rate_positive,success_rate_negative,success_rate_total] =...
    validate(data_positive,data_negative,w,b)


positive_length = size(data_positive,1);
tmp = data_positive(:,5).*(data_positive(:,1:4)*w - ones(positive_length,1)*b);
s=sign(tmp);
is_positive_correct = sum(s==1);
success_rate_positive = is_positive_correct/positive_length ;

negative_length = size(data_negative,1);
tmp = data_negative(:,5).*(data_negative(:,1:4)*w - ones(negative_length,1)*b);
s=sign(tmp);
is_negative_correct = sum(s==1);
success_rate_negative = is_negative_correct/negative_length;

success_rate_total = (is_positive_correct+is_negative_correct)/(positive_length+negative_length);
