
clear all;
load 'C:\Users\Steve\Documents\School\CS7495\attractiveness\Data\hotornot1\fem_picts.mat';
save_dir = 'C:\Users\Steve\Documents\Data\EigenHot\HotorNot_thirds\';

new_score = score - min(score);
new_score = new_score * 5 / max(new_score);

[sortedScore,IX]=sort(new_score);
sortedImgs=rectimgs(IX);

trainingScore=sortedScore;
trainingScore(1:10:end)=[];
trainingImg=sortedImgs;
trainingImg(1:10:end)=[];
testScore=sortedScore(1:10:end);
testImg=sortedImgs(1:10:end);

bottom3 = floor(length(trainingScore) * 0.33);
top3 = floor(length(trainingScore) * 0.66);

for i=1:length(trainingScore)
   if(i <= bottom3)
       bin = 0;
   elseif(i <= top3)
       bin = 1;
   else
       bin = 2;
   end
   save_loc = strcat(save_dir,'train_images/',num2str(bin),'/',num2str(i),'.bmp');
   imwrite(trainingImg{i},save_loc);
end

bottom3 = floor(length(testScore) * 0.33);
top3 = floor(length(testScore) * 0.66);
for i=1:length(testScore)
   if(i <= bottom3)
       bin = 0;
   elseif(i <= top3)
       bin = 1;
   else
       bin = 2;
   end
   save_loc = strcat(save_dir,'test_images/',num2str(bin),'_',num2str(i),'.bmp');
   imwrite(testImg{i},save_loc);    
end

% bottom25 = floor(length(trainingScore) * 0.25);
% top25 = floor(length(trainingScore) * 0.75);
% for i=1:bottom25
%    save_loc = strcat(save_dir,'train_images/0/',num2str(i),'.bmp');
%    imwrite(trainingImg{i},save_loc);    
% end
% for i=top25:length(trainingScore)
%    save_loc = strcat(save_dir,'train_images/1/',num2str(i),'.bmp');
%    imwrite(trainingImg{i},save_loc);
% end
% 
% bottom25 = floor(length(testScore) * 0.25);
% top25 = floor(length(testScore) * 0.75);
% for i=1:bottom25
%    save_loc = strcat(save_dir,'test_images/0_',num2str(i),'.bmp');
%    imwrite(testImg{i},save_loc);       
% end
% for i=top25:length(testScore)
%    save_loc = strcat(save_dir,'test_images/1_',num2str(i),'.bmp');
%    imwrite(testImg{i},save_loc);
% end

% for i=1:length(trainingScore)
%    bin = floor(trainingScore(i)); 
%    save_loc = strcat(save_dir,'train_images/',num2str(bin),'/',num2str(i),'.bmp');
%    imwrite(trainingImg{i},save_loc);    
% end
% 
% for i=1:length(testScore)
%    bin = floor(testScore(i)); 
%    save_loc = strcat(save_dir,'test_images/',num2str(bin),'_',num2str(i),'.bmp');
%    imwrite(testImg{i},save_loc);    
% end