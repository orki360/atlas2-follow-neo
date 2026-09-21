#include "policy/tracking_policy.hpp"
#include "policy/spacing_controller.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
void cmd(dji::TrackingCommand c){std::cout<<' '<<c.yaw<<' '<<c.vertical<<' '<<c.roll<<' '<<c.forward<<'\n';}
int main(){
 std::cout<<std::setprecision(17);
 dji::DeterministicTrackingPolicy p;
 dji::TargetKinematics k; k.initialized=true;k.cx=320;k.cy=240;k.width=80;k.height=60;
 dji::TrackingCommand raw{.5,.25,.2,1.};
 double times[]={1,2,2.05,2.1,2.5,2.8,3,3.8,4.7,6.09,6.2};
 for(int i=0;i<11;++i){double t=times[i];k.cx=i==5?650:320;k.vx=i==5?150:0;
 std::optional<double> mt=i>=1&&i<=3?std::optional<double>(t):i>=4&&i<=5?std::optional<double>(2.1):std::nullopt;
 std::optional<uint64_t> id=i>=1&&i<=3?std::optional<uint64_t>(i):i>=4&&i<=5?std::optional<uint64_t>(3):std::nullopt;
 auto d=p.update(t,640,480,k,mt,id,i>=1&&i<=3?raw:dji::TrackingCommand{});
 std::cout<<"P "<<i<<' '<<dji::toString(d.state);cmd(d.command);}
 dji::SmartTrackingController smart;
 for(int i=0;i<100;++i){double cx=320+250*std::sin(i*.24),cy=240+150*std::cos(i*.19),w=60+i*5.;
 auto d=smart.compute({cx-w/2,cy-40,cx+w/2,cy+40},640,480,200*std::sin(i*.17),.05,i%17==0,1.,10+i*.05);
 std::cout<<"S "<<i;cmd(d.command);}
 dji::VisualSpacingController spacing;spacing.reset(true);spacing.setStopBboxWidthRatio(.20);
 for(int i=0;i<90;++i){dji::SpacingObservation o;o.nowSeconds=20+i*.05;o.trackActive=true;o.measurementValid=true;
 o.measurementId=i+1;o.measurementSeconds=o.nowSeconds-.02;o.measurementAgeSeconds=.02;
 o.rawWidthRatio=.05+i*.004;o.filteredWidthRatio=o.rawWidthRatio;o.trackCommand={.1,0,0,.3};o.forwardLimit=.3;
 auto d=spacing.update(o);std::cout<<"D "<<i<<' '<<dji::toString(d.phase);cmd(d.command);}
}
