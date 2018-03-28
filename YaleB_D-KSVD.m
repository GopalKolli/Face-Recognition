


%REGULATION PARAMETERS. MODIFY THE BELOW FOUR PARAMETERS TO INFLUENCE D-KSVD
%MODEL. 'number_of_atoms_in_dict' SHOULD BE MORE THAN 504. BECAUSE THE
%IMAGE VECTOR LENGTH IS 504 AND WE NEED THE DICTIONARY TO BE OVER-COMPLETE.
maximumsparsity = 40;
gammafactor = 4;
number_of_atoms_in_dict = 580;
number_of_ksvd_iters = 64;
%========================================

allimagesdata = containers.Map;
image_vector_matrix = zeros(504,2496);
H_Labels_matrix = zeros(39,2496);
current_image_vector = 1;


for classnumber=1:39
    
    if classnumber<10
        foldername = strcat('0',num2str(classnumber));
    else
        foldername = num2str(classnumber);
    end
    foldername = strcat('yaleB',foldername);
    
    notif1 = strcat('================================= IN THE FOLDER ',foldername);
    disp(notif1);
    
    imagefiles = dir(fullfile('CroppedYale', 'CroppedYale', foldername,'*.pgm'));
    
    notif2 = strcat('=============================== Number of .pgm files = ',num2str(length(imagefiles)));
    disp(notif2)
    
    for k=1:length(imagefiles)
        imagefilenames=imagefiles(k).name;
        currentFolder = pwd;
        path = strcat(currentFolder,'/CroppedYale/CroppedYale/',foldername,'/',imagefilenames);
        A = imread(path);
        check = extractBetween(imagefilenames,'Am','nt');
        if check == 'bie'
            disp(' ');
        else
            allimagesdata(imagefilenames) = A;
        end
    end
    
end

currentdictsize = size(allimagesdata,1);

while currentdictsize > 2
    randomindex = randi(currentdictsize);
    allimagenames = keys(allimagesdata);
    pick = allimagenames{randomindex};
    pickedimage = allimagesdata(pick);
    
    
    resized_picked_image = imresize(pickedimage, 1/8);
    
    vectorized_pickedimage = resized_picked_image(:);
    H_label_vector = zeros(39,1);
    class_label_str = extractBetween(pick,6,7);
    
    index = str2double(class_label_str);
    
    H_label_vector(index) = 1;
    
    image_vector_matrix(:,current_image_vector) = vectorized_pickedimage;
    H_Labels_matrix(:,current_image_vector) = H_label_vector;
    current_image_vector = current_image_vector+1;
    
    remove(allimagesdata,{pick});
    currentdictsize = currentdictsize-1;
end

currentdictsize = size(allimagesdata,1);

for j=1:2
    remaining_imagenames = keys(allimagesdata);
    pick = remaining_imagenames{j};
    pickedimage = allimagesdata(pick);
    resized_picked_image = imresize(pickedimage, 1/8);
    vectorized_pickedimage = resized_picked_image(:);
    H_label_vector = zeros(39,1);
    class_label_str = extractBetween(pick,6,7);
    
    index = str2double(class_label_str);
    
    H_label_vector(index) = 1;
    
    image_vector_matrix(:,current_image_vector) = vectorized_pickedimage;
    H_Labels_matrix(:,current_image_vector) = H_label_vector;
    current_image_vector = current_image_vector+1;
end

disp(current_image_vector)

final_image_vector_matrix = image_vector_matrix(:,1:(current_image_vector-1));
final_H_Label_matrix = H_Labels_matrix(:,1:(current_image_vector-1));



%{
save imagevector_matrix.mat final_image_vector_matrix
save HLabel_matrix.mat final_H_Label_matrix
load imagevector_matrix.mat final_image_vector_matrix
load HLabel_matrix.mat final_H_Label_matrix
%}


disp('Matrices Ready!')
disp(size(final_image_vector_matrix))
disp(size(final_H_Label_matrix))



currFolder = pwd;
ksvdboxpath = strcat(currFolder,'/ksvdbox13');
ompboxpath = strcat(currFolder,'/ompbox10');
addpath(ksvdboxpath);  
addpath(ompboxpath);

Training_Samples = final_image_vector_matrix(:,1:1600);
H_Labels_Training = final_H_Label_matrix(:,1:1600);
Testing_Samples = final_image_vector_matrix(:,1601:2414);
H_Labels_Testing = final_H_Label_matrix(:,1601:2414);

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

