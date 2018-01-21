function data_scaled=stdscale(data)
% STDSCALE : Scaled data with standard deviation
%
% data_scaled=STDSCALE(data) scales data in order to have zero mean and
% unitary standard deviation. The variable data is a structure that follows
% the STPRtool nomenclature.
%
% César Teixeira

mu=mean(data.X,2);
s=std(data.X,[],2);

data_scaled = data;
data_scaled.X =(data.X-repmat(mu,1,data.num_data))./repmat(s,1,data.num_data);
end