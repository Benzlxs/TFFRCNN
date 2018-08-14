%% getting some example plotting, one sentence one branch,
%{
function: the function is created to plotting my data about how number of
anchor affect detection performance.
 the result format is rigid, 
     iteration number
        done
    car map:
    cyclist map:
    pedestrain map:
so the table rows should be integral number of 5,
arg: data_dir, the directory to generated results.
%}

function []=main(data_dir)

clc;
clear;

% checking the input data, the format should be 5n x 5
if nargin==0
    data_dir = '../../data/experiments_results/';
end

data_folder = dir(data_dir);
data_folder(ismember( {data_folder.name}, {'.', '..'})) = [];
data_num = length(data_folder);

start_model = 70000;

aspect_ratios = [3, 5, 7];
num_ar = length(aspect_ratios);

all_data = [];

for i = 1:num_ar
  all_data(i).ar = aspect_ratios(i);
  scale_temp = [];
  name_temp = [];
  i_temp = 1;
  %% collecting all scales
  for j=1:data_num
      data_name = data_folder(j).name;
      Ct = strsplit(data_name, '_');
      assert(length(Ct)==7,'the data name %s is not right',data_name)
      assert(  str2num(Ct{2}) == str2num(Ct{7})*str2num(Ct{4}) , 'the data name %s is not right', convertCharsToStrings(data_name))
      if str2num(Ct{7})== aspect_ratios(i)
          scale_temp(i_temp) = str2num(Ct{4});
          name_temp(i_temp) = j;
          i_temp = i_temp + 1;
      end
  end
  % change scale from small to large to make plotting easier
  [scale_temp, ind ]= sort(scale_temp);
  all_data(i).scale = scale_temp;
  all_data(i).name = name_temp(ind(:)); 
end


%% start to plotting
figure(1), hold on
axis([0, 30, 40, 100]);
for i=1:num_ar 
    %initilization
    scales = [];
    name_ind = [];
    mAP = [];
    ar_i = all_data(i).ar;
    scales = all_data(i).scale;
    num_scale=length(scales);
    name_ind = all_data(i).name;
    
    %read all file of aspect_ratios(as_i)
    for s_i=1:num_scale
        % calcluate mAP given aspect_ratio and scale_num
        files_path = dir(fullfile(data_folder(name_ind(s_i)).folder, data_folder(name_ind(s_i)).name,'*.txt'));
        assert(length(files_path)==1);
        txt_file_path = fullfile(files_path.folder, files_path.name);
        if exist(txt_file_path,'file') == 2
          map_i =  calculate_map(txt_file_path, start_model);
          assert(map_i>0)
          mAP(s_i) = map_i;        
        else
            error('file does not exist');
        end
        
        
    end
    
    % plotting
    plot(scales,mAP,'-o','LineWidth',2);
    
end

h=legend('3-aspect-ratio','5-aspect-ratio','7-aspect-ratio', 'location','northeast')
set(h,'FontSize',12);

xlabel('Number of scale', 'FontName', 'Times New Roman','FontSize',20);
ylabel('Detection mAP', 'FontName', 'Times New Roman','FontSize',20);

set(gcf,'unit','inches','position',[1 1 9 7]);

end

function map = calculate_map(file_path, start_model)
    % calculate the mAP of model
    map = 0;
    assert(exist(file_path,'file') == 2);
    assert(start_model>=5000);
    table = importfile(file_path);
    len = size(table,1);
    assert( rem(len,5)==0 , 'len %d is not right, check file %s', len, file_path);
    
    num_model =  len/5;
    assert(num_model>1);
    start_i = start_model/5000 - 1;
    %start_ind = (start_model/5000 -1)* 5 +1 ;
    sum_ap = 0;
    sum_ind=0;
    for i = start_i:num_model-1
        for j=1:1:3    %% j=1 for map of car, 2 for map of pedestrain, 3 for map of cyclist
            index = int16(i*5+j+2);
            assert( ~ isnan(table{ index, 'VarName3'}));
            assert( ~ isnan(table{ index, 'VarName4'}));
            assert( ~ isnan(table{ index, 'VarName5'}));
            sum_ap = sum_ap + table{ index, 'VarName3'} + table{ index, 'VarName4'} + table{ index, 'VarName5'} ;
            sum_ind = sum_ind + 3;
            %separate one
            %sum_ap = sum_ap + table{ index, 'VarName5'};
            %sum_ind = sum_ind + 1;
        end
    end
     
    map = sum_ap/sum_ind;

end



function y1=gaussmf_ee(x,a,c)

y1=exp(-(x-a).^2/(2*c^2));

end