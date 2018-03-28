

%REGULATION PARAMETERS. MODIFY THE BELOW FOUR PARAMETERS TO INFLUENCE D-KSVD
%MODEL. 'number_of_atoms_in_dict' SHOULD BE MORE THAN 504. BECAUSE THE
%IMAGE VECTOR LENGTH IS 504 AND WE NEED THE DICTIONARY TO BE OVER-COMPLETE.

bounds = [1 32 64 96 128 160 192 224 255];

allvectors = containers.Map;
for i=1:8
    lower_bound = bounds(i);
    upper_bound = bounds(i+1);
    for vec=1:100
        vector = (upper_bound-lower_bound).*rand(504,1) + lower_bound;
        vecname = strcat(num2str(i),'_',num2str(vec));
        allvectors(vecname) = vector;
    end
end

vector_matrix = zeros(504,1000);
H_Labels_matrix = zeros(39,1000);
current_vector = 1;

currentdictsize = size(allvectors,1);
while currentdictsize > 2
    randomindex = randi(currentdictsize);
    allvectornames = keys(allvectors);
    pick = allvectornames{randomindex};
    pickedvector = allvectors(pick);
    H_label_vector = zeros(39,1);
    class_label_str = extractBetween(pick,1,1);
    index = str2double(class_label_str);
    H_label_vector(index) = 1;
    vector_matrix(:,current_vector) = pickedvector;
    H_Labels_matrix(:,current_vector) = H_label_vector;
    current_vector = current_vector+1;
    remove(allvectors,{pick});
    currentdictsize = currentdictsize-1;
end

currentdictsize = size(allvectors,1);
final_image_vector_matrix = vector_matrix(:,1:(current_vector-1));
final_H_Label_matrix = H_Labels_matrix(:,1:(current_vector-1));

currFolder = pwd;
ksvdboxpath = strcat(currFolder,'/ksvdbox13');
ompboxpath = strcat(currFolder,'/ompbox10');
addpath(ksvdboxpath);  
addpath(ompboxpath);

Training_Samples = final_image_vector_matrix(:,1:600);
H_Labels_Training = final_H_Label_matrix(:,1:600);
Testing_Samples = final_image_vector_matrix(:,600:790);
H_Labels_Testing = final_H_Label_matrix(:,600:790);

concatenated_Training_Data = [Training_Samples;gammafactor*H_Labels_Training];



params.data = concatenated_Training_Data;
params.dictsize = number_of_atoms_in_dict;
params.iternum = number_of_ksvd_iters;
params.Tdata = maximumsparsity;
params.memusage = 'high';

disp('Performing D-KSVD')
[Learned_Dictionary,alpha,Error] = ksvd(params,'');

Dictionary = Learned_Dictionary(1:504,:);
Weights = Learned_Dictionary(505:543,:);
denominators = sqrt(sum(Dictionary.*Dictionary,1)+eps);
Weights = Weights./repmat(denominators,size(Weights,1),1);
Dictionary = Dictionary./repmat(denominators,size(Dictionary,1),1);
Weights = Weights./gammafactor;


[predicted_classes,accuracy] = classification(Dictionary, Weights, Testing_Samples, H_Labels_Testing, maximumsparsity);
disp(strcat('Accuracy of classification is : ', num2str(accuracy*100)));




function [predictions_made, accuracy, Errors] = classification(Dictionary, Weights, TestSamples, H_Labels_Testing, maximumsparsity)
numberoferrors = 0;
G_Matrix = Dictionary'*Dictionary;
Gamma = omp(Dictionary'*TestSamples,G_Matrix,maximumsparsity);
Errors = [];
predictions_made = [];
for f_index=1:size(TestSamples,2)
    sparse_encoding = Gamma(:,f_index);
    estimations =  Weights * sparse_encoding;
    scores = H_Labels_Testing(:,f_index);
    [~, estimated_max_Index] = max(estimations);
    [~, max_index] = max(scores);
    predictions_made = [predictions_made estimated_max_Index];
    if(estimated_max_Index~=max_index)
        numberoferrors = numberoferrors + 1;
        Errors = [Errors;numberoferrors f_index max_index estimated_max_Index];
    end
end
accuracy = (size(TestSamples,2)-numberoferrors)/size(TestSamples,2);
end



