%https://www.mathworks.com/matlabcentral/answers/13351-dialog-with-checkboxes-in-gui

function spectral_unmixing_gui(num_channels)
close all;

h.f = figure('units','pixels','position',[700,500,150,100],...
             'toolbar','none','menu','none');
         
h.t = uicontrol('style', 'text','units', 'pixels',...
    'position',[0, 60, 150, 30], 'string', 'Select an unmixing method');           
         
h.d = uicontrol('style', 'popupmenu', 'units', 'pixels',...
    'position',[10, 30, 90, 30], 'string', 'K-means|Other');
  
h.p = uicontrol('style','pushbutton','units','pixels',...
                'position',[40,5,70,20],'string','OK',...
                'callback',@d_call);

% Dropdown callback
function d_call(varargin)
    vals = get(h.d,'Value');
    switch vals
        case 1
            k_means_gui(num_channels) 
        case 2
            error('Other is not an option')     
    end
end

end



function k_means_gui(num_channels)
close all;

% Create figure
h.f = figure('units','pixels','position',[700,500,max(20+40*num_channels, 300),130],...
             'toolbar','none','menu','none');

%Prompt for # of fluorophores to unmix
h.t(1) = uicontrol('style', 'text','units', 'pixels',...
    'position',[0, 90, 300, 30], 'string', 'How many fluorophores do you want to unmix?');      
   
h.s = uicontrol('style', 'edit','units', 'pixels',...
    'position',[20, 70, 40, 30]);        
         
% Create yes/no checkboxes
h.t(2) = uicontrol('style', 'text','units', 'pixels',...
    'position',[0, 40, 200, 30], 'string', 'Select which channels to unmix');
for i = 1:num_channels
    %h.
    h.c(i) = uicontrol('style','checkbox','units','pixels',...
                    'position',[10+40*(i-1),30,50,15],'string',int2str(i));
end
% Create OK pushbutton   
h.p = uicontrol('style','pushbutton','units','pixels',...
                'position',[40,5,70,20],'string','OK',...
                'callback',@p_call);
% Pushbutton callback
function p_call(varargin)
    vals = get(h.c,'Value');
    checked = find([vals{:}]);
    if isempty(checked)
        checked = 'none';
    end
    disp(checked)
end



end