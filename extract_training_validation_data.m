function [data_positive,data_negative,validation_positive,validation_negative] =...
    extract_training_validation_data(all_data,data_labels,reqested_data_length)

all_data_positive = all_data{all_data{:,5} == data_labels.POSITIVE,:};
all_data_negative = all_data{all_data{:,5} == data_labels.NEGATIVE,:};

all_data_positive_rows = size(all_data_positive,1);
all_data_negative_rows = size(all_data_negative,1);

step = ceil(all_data_positive_rows / reqested_data_length);
data_positive = all_data_positive(step:step:all_data_positive_rows,:);
step = ceil(all_data_negative_rows / reqested_data_length);
data_negative = all_data_negative(step:step:all_data_negative_rows,:);


step = ceil(all_data_positive_rows / reqested_data_length);
assert(step > 1);
validation_positive = all_data_positive(step-1:step:all_data_positive_rows,:);
step = ceil(all_data_negative_rows / reqested_data_length);
assert(step > 1);
validation_negative = all_data_negative(step-1:step:all_data_negative_rows,:);
