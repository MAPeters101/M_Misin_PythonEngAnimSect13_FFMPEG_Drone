import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#print(matplotlib.__version__)

type=5

# Create the time array
t0=0
t_end=10
dt=0.02
t=np.arange(t0,t_end+dt,dt)

# Create array for x & y dimensions
if type==1:
    r=0.5+0*t
    f=.25+0*t
    x=0*t
    y=0*t
    # Create array for the Z dimension
    z=t
elif type==2:
    r=0.5+0*t
    f=.25+0*t
    x=0.25*t
    y=0*t
    # Create array for the Z dimension
    z=t
elif type==3:
    r=0.5+0*t
    f=.25+0*t
    x=0*t
    y=-0.25*t
    # Create array for the Z dimension
    z=t
elif type==4:
    r=0.5+0*t
    f=.25+0*t
    x=0*t
    y=0*t
    # Create array for the Z dimension
    z=t
elif type==5:
    r=0.25*t
    f=0.25*t
    x=0*t
    y=0*t
    # Create array for the Z dimension
    z=t

############################## ANIMATION ###########################
frame_amount=len(t)

def update_plot(num):
    # Trajectory
    # plane_trajectory.set_data(x[0:num],y[0:num])
    plane_trajectory.set_xdata(x[0:num])
    plane_trajectory.set_ydata(y[0:num])
    plane_trajectory.set_3d_properties(z[0:num])

    pos_x.set_data(t[0:num],x[0:num])
    pos_y.set_data(t[0:num],y[0:num])
    pos_z.set_data(t[0:num],z[0:num])

    if type==1:
        drone_body_x.set_xdata([x[num]-0.5,x[num]+0.5])
        drone_body_x.set_ydata([y[num],y[num]])
        drone_body_x.set_3d_properties([z[num],z[num]])
        drone_body_y.set_xdata([x[num],x[num]])
        drone_body_y.set_ydata([y[num]-0.5,y[num]+0.5])
        drone_body_y.set_3d_properties([z[num],z[num]])
    elif type==2:
        drone_body_x.set_xdata([x[num]-0.5*np.cos(np.pi/6),x[num]+0.5*np.cos(np.pi/6)])
        drone_body_x.set_ydata([y[num],y[num]])
        drone_body_x.set_3d_properties([z[num]+0.5*np.sin(np.pi/6),z[num]-0.5*np.sin(np.pi/6)])
        drone_body_y.set_xdata([x[num],x[num]])
        drone_body_y.set_ydata([y[num]-0.5,y[num]+0.5])
        drone_body_y.set_3d_properties([z[num],z[num]])
    elif type==3:
        drone_body_x.set_xdata([x[num]-0.5,x[num]+0.5])
        drone_body_x.set_ydata([y[num],y[num]])
        drone_body_x.set_3d_properties([z[num],z[num]])

        drone_body_y.set_xdata([x[num],x[num]])
        drone_body_y.set_ydata([y[num]-0.5*np.cos(np.pi/6),y[num]+0.5*np.cos(np.pi/6)])
        drone_body_y.set_3d_properties([z[num]-0.5*np.sin(np.pi/6),z[num]+0.5*np.sin(np.pi/6)])
    elif type==4:
        drone_body_x.set_xdata([x[num]-r[num]*np.cos(2*np.pi*(f[num])*t[num]),x[num]+r[num]*np.cos(2*np.pi*(f[num])*t[num])])
        drone_body_x.set_ydata([y[num]-r[num]*np.sin(2*np.pi*(f[num])*t[num]),y[num]+r[num]*np.sin(2*np.pi*(f[num])*t[num])])
        drone_body_x.set_3d_properties([z[num],z[num]])

        drone_body_y.set_xdata([x[num]-r[num]*np.cos(2*np.pi*(f[num])*t[num]+np.pi/2),x[num]+r[num]*np.cos(2*np.pi*(f[num])*t[num]+np.pi/2)])
        drone_body_y.set_ydata([y[num]-r[num]*np.sin(2*np.pi*(f[num])*t[num]+np.pi/2),y[num]+r[num]*np.sin(2*np.pi*(f[num])*t[num]+np.pi/2)])
        drone_body_y.set_3d_properties([z[num],z[num]])
    elif type==5:
        drone_body_x.set_xdata([x[num]-r[num]*np.cos(2*np.pi*(f[num])*t[num]),x[num]+r[num]*np.cos(2*np.pi*(f[num])*t[num])])
        drone_body_x.set_ydata([y[num]-r[num]*np.sin(2*np.pi*(f[num])*t[num]),y[num]+r[num]*np.sin(2*np.pi*(f[num])*t[num])])
        drone_body_x.set_3d_properties([z[num],z[num]])

        drone_body_y.set_xdata([x[num]-r[num]*np.cos(2*np.pi*(f[num])*t[num]+np.pi/2),x[num]+r[num]*np.cos(2*np.pi*(f[num])*t[num]+np.pi/2)])
        drone_body_y.set_ydata([y[num]-r[num]*np.sin(2*np.pi*(f[num])*t[num]+np.pi/2),y[num]+r[num]*np.sin(2*np.pi*(f[num])*t[num]+np.pi/2)])
        drone_body_y.set_3d_properties([z[num],z[num]])

    return plane_trajectory,pos_x,pos_y,pos_z,drone_body_x,drone_body_y

# Set up your figure properties
fig=plt.figure(figsize=(16,9),dpi=80,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(3,4)

# 3D motion
ax0=fig.add_subplot(gs[:,0:3],projection='3d',facecolor=(0.9,0.9,0.9))
plane_trajectory,=ax0.plot([],[],[],'r',linewidth=1,label='Flight trajectory')
drone_body_x,=ax0.plot([],[],[],'b',linewidth=5,label='drone_x')
drone_body_y,=ax0.plot([],[],[],'g',linewidth=5,label='drone_y')
ax0.set_xlim(-3,3)
ax0.set_ylim(-3,3)
# ax0.set_xlim(min(x),max(x))
# ax0.set_ylim(min(y),max(y))
ax0.set_zlim(min(z),max(z))
# ax0.set_xticks([0])
# ax0.set_yticks([0])
# ax0.set_zticks([5])
ax0.set_xlabel('position_x [m]',fontsize=12)
ax0.set_ylabel('position_y [m]',fontsize=12)
ax0.set_zlabel('position_z [m]',fontsize=12)
plt.grid(True)

ax1=fig.add_subplot(gs[0,3],facecolor=(0.9,0.9,0.9))
pos_x,=ax1.plot([],[],'b',linewidth=2,label='x = '+str(r)+'cos(2π'+str(f)+'t)')
plt.xlim(t0,t_end)
plt.ylim(-3,3)
# plt.ylim(min(x),max(x))
plt.ylabel('position_x [m]',fontsize=12)
plt.grid(True)

ax2=fig.add_subplot(gs[1,3],facecolor=(0.9,0.9,0.9))
pos_y,=ax2.plot([],[],'b',linewidth=2,label='y = '+str(r)+'sin(2π'+str(f)+'t)')
plt.xlim(t0,t_end)
plt.ylim(-3,3)
# plt.ylim(min(y),max(y))
plt.ylabel('position_y [m]',fontsize=12)
plt.grid(True)

ax3=fig.add_subplot(gs[2,3],facecolor=(0.9,0.9,0.9))
pos_z,=ax3.plot([],[],'b',linewidth=2,label='z = t')
plt.xlim(t0,t_end)
plt.ylim(-3,3)
# plt.ylim(min(z),max(z))
plt.xlabel('time [s]',fontsize=12)
plt.ylabel('position_z [m]',fontsize=12)
plt.grid(True)

plane_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=False,blit=True)
plt.show()



















###########################
