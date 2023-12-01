% %电基本振子的方向图
% clear;
% clc;
% theta=meshgrid(eps:pi/180:pi);
% phi=meshgrid(eps:2*pi/180:2*pi)';
% F=abs(sin(theta));
% [x,y,z]=sph2cart(phi,pi/2-theta,F);
% figure(1);
% mesh(x,y,z);
% title('电基本振子的立体方向图');
% xlabel('x'),ylabel('y'),zlabel('F(\theta,\phi)');

% close all; clear; clc;
% 
% Llambda = 1.5;
% Lr = 2*pi*Llambda;
% F = @(theta, phi) 2*((cos(Lr/2.*cos(theta)) - cos(Lr/2))./sin(theta)).^2;
% 
% Ntheta = 108; Nphi = 72;
% dtheta = pi/Ntheta; dphi = 2*pi/Nphi;
% theta = linspace(0,pi,Ntheta); phi = 0:dphi:2*pi-dphi;
% theta([1 end]) = [eps pi-eps];
% [THETA, PHI] = meshgrid(theta, phi);
% 
% R = F(THETA, PHI);
% Ravg = sum(sum(R.*sin(THETA)*dtheta*dphi/(4*pi)));
% Rm = max(R(:)); D = Rm/Ravg;
% R = R/Ravg; % R = Rn*D = R/Rm*D
% 
% 
% % for visual purposes, specify the segment of the pattern to be cut
% phiI1 = 1; phiI2 = Nphi/3+1;
% R(phiI1+1:phiI2-1,:) = NaN;
% 
% figure(1), hold on
% set(gcf, 'render','painter','color','w')
% view(150,8), axis image off, colormap jet
% % isotropic antenna pattern as reference
% [x,y,z] = sphere;
% surf(x,y,z,'facec',[.8 .8 .8],'facea',0.2,'edgea',0.2);
% % antenna pattern
% [x,y,z] = sph2cart([PHI;PHI(1,:)], pi/2-[THETA;THETA(1,:)], [R;R(1,:)]);
% surf(x,y,z,2*R/D-1,'facea',0.8,'edgea',0.5);
% % additional cross section
% [x,y,z] = sph2cart(PHI([phiI1 phiI2],:), pi/2-THETA([phiI1 phiI2],:),...
% R([phiI1 phiI2],:));
% patch(x',y',z','w','facea',0.6,'edgea',0.2);



