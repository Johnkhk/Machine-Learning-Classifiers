clear all; close all;
load data.mat
load label.mat
K=1;
A = reshape(imageTrain, 784,5000);
B = reshape(imageTest, 784,500);
a = reshape(A, 28, 28, 5000);
b = reshape(B, 28, 28, 500);

%% Getting New Predictions
NewLabel = zeros(1,500);
for i = 1:500
    difference_mat = B(:,i)-A;
    distances = sqrt(sum(difference_mat.^2,1));
    [M,Indices] = min(distances);
    NewLabel(i) = labelTrain(Indices);
end
%% Error by Class
error=zeros(1,10);
for i = 1:10
    class_index = find(labelTest==i-1);
    count = 0;
    for j = 1:length(class_index)
        if(NewLabel(class_index(j))~=labelTest(class_index(j)))
            count = count+1;
        end
    end
    error(i) = count/nnz(labelTest==i-1);
end
class = 0:1:9;
scatter(class, error, 'X');
xlim([-1,10])
ylim([-0.1,0.27])
title('Training Set vs Data set Euclidean distance KNN');
xlabel('Class (digit)')
ylabel('Error')
%% Total Error
wait = NewLabel.'-labelTest;
Totalerror = nnz(wait)/500;
    
%% 5 Missclassified Images

figure
k = find(wait);
for i = 1:5
    subplot(5,2,2*i-1)
    imshow(a(:,:,k(i)))
    title('Predicted Label')
    subplot(5,2,2*i)
    imshow(b(:,:,k(i)))
    title('Actual Label')
end

%% Confusion matrix
figure
C = confusionmat(labelTest,NewLabel);
cm = confusionchart(C);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

