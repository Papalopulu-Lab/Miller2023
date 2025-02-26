function [m,mr]=DetrendPoly(x,y,order,win)
warning off
% Input variables
% x is time
% y is gene expression
% order is the poly order
% win is the size of window in no of frames (optional)
% Output variable
% m is the poly fit order 2 when win is []
% m is the average of a local poly fit order 2 when win is specified
if isempty(win)|| (numel(y))<win
   % window is not specified
   p=polyfit(x,y,order);
   m=polyval(p,x);
   mr=m;
else
   flag=0;
   % run in a timewindow 
   mwin=[];
   for i=1:round(0.75*win):numel(y)
       if (i+win<numel(y))
           yy=y(i:i+round(win));
           xx=x(i:i+round(win));
           idx=[i:i+round(win)];
       else
%            yy=y(end-round(win):end);
%            xx=x(end-round(win):end);
%            idx=[numel(y)-round(win):numel(y)];
           yy=y(i:end);
           xx=x(i:end);
           idx=i:numel(y);
           tail=idx;
          if numel(xx)<round(win/2)
              flag=2; % too short
          else
              flag=1; % long enough for lin
          end
       end
      switch flag
          case 1
                p=polyfit(xx,yy,flag);
                m=polyval(p,xx);
           case 2
               % do not update polynomial
                m=polyval(p,xx);
           otherwise
                p=polyfit(xx,yy,order);
                m=polyval(p,xx);
       end
       %save tmp
       tmp=zeros(numel(y),1);
       tmp(idx)=m;
       mwin=[mwin; tmp'];
   end 
   % weighter average here

   m=sum(mwin)./(size(mwin,1)-sum(mwin==0));
      save temp
   mr=m; % save this
   % implement optimisation of smooth fit
    m0=smoothdata(m); % somthing to compare against
    err0=sum(sum((m0(:)-y).^2)); %sse
    n=[5:50];
    for k=1:numel(n) % try different poly
       % n(k)
        pk=polyfit(x,mr',n(k));
        mk=polyval(pk,x);
        err(k)=sum(sum((mk(:)-y).^2));%sse
        mstore{k}=mk;
        pstore{k}=pk;
    end
   % test agains smoothdata
   flag=sum(err<err0);
   if flag>0 % poly better than smoothdata
       thrange=max(err)-min(err);
       nidx=find(err<max(err)-0.9*thrange,1,'first'); % within less than 5% of min
       if ~isempty(nidx)
            % select that one to export
            m=mstore{nidx};
            p=pstore{nidx};
       end
   else % if it is same or worse
       m=m0; % use smoothdata
       p=[];
   end
   % refine the shape at the start of signal
   tail=1:round(numel(y)/10);
   m(tail)=smoothdata(m(tail));
   % refine the shape at the end of signal
   tail=numel(y)-round(numel(y)/10):numel(y);
   m(tail)=smoothdata(m(tail));
end  
m=m(:); % export as a vector
mr=mr(:);
