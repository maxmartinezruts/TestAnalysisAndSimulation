import scipy.io as sio
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

Tl = 0.2
TI = 1.0
tau_v = 0.35
tau_m = 0.1
kv =1
km = 1
wm = 10
cm = 0.25
w = np.linspace(0,20, 100)
y = np.absolute(kv*(1+1j*Tl*w)/(1+1j*TI*w)*np.exp(-1j*w*tau_v)*wm**2/((1j*w)**2+2*cm*wm*1j*w+wm**2))
plt.plot(w, y)
plt.show()



def H(kv,Tl,TI,tau, wm, cm,w):
    return kv*(1+1j*Tl)/(1+1j*TI)*np.exp(-1j*w*tau)*wm**2/((1j*w)**2+2*cm*wm*1j*w+wm**2)


conditions = ['C1','C2','C3','C4','C5','C6']
subject_ids = ['1','2','3','4','5','6',]

subjects = {}



for subject_id in subject_ids:
    subjects[subject_id] = {}
    file_name = 'ae2223I_measurement_data_subj' + subject_id + '.mat'
    subjects[subject_id]['file'] =file_name
    matlab_data = sio.loadmat(file_name)
    for condition in conditions:
        subjects[subject_id][condition] = {}
        subjects[subject_id][condition]['e'] = matlab_data['data_'+condition]['e'][0][0]
        subjects[subject_id][condition]['u'] = matlab_data['data_'+condition]['u'][0][0]
        subjects[subject_id][condition]['x'] = matlab_data['data_'+condition]['x'][0][0]
        subjects[subject_id][condition]['ft'] = matlab_data['data_'+condition]['ft'][0][0]
        subjects[subject_id][condition]['fd'] = matlab_data['data_'+condition]['fd'][0][0]
        subjects[subject_id][condition]['Hpe_FC'] = matlab_data['data_'+condition]['Hpe_FC'][0][0]
        subjects[subject_id][condition]['Hpxd_FC'] = matlab_data['data_'+condition]['Hpxd_FC'][0][0]
        subjects[subject_id][condition]['t'] = matlab_data['t'][0]
        subjects[subject_id][condition]['w_FC'] = matlab_data['w_FC']
        subjects[subject_id][condition]['H_magnitude_v'] = np.absolute(subjects[subject_id][condition]['Hpe_FC'][:,:])
        subjects[subject_id][condition]['H_angle_v'] = np.transpose(np.unwrap(np.transpose(np.angle(subjects[subject_id][condition]['Hpe_FC'][:,:])),discont=-np.pi))
        subjects[subject_id][condition]['H_magnitude_d'] = np.absolute(subjects[subject_id][condition]['Hpxd_FC'][:, :])
        subjects[subject_id][condition]['H_angle_d'] = np.transpose(np.unwrap(np.transpose(np.angle(subjects[subject_id][condition]['Hpxd_FC'][:, :])),discont=-np.pi))




        print(np.log(subjects[subject_id][condition]['Hpe_FC']))
        print('---------------')

#subject =  input('Select subject')
#condition = input('Select condition')
#x_axis = input('Select x axis')
#y_axis = input('Select y axis')
subject =  '1'
condition = 'C1'
x_axis = 'w_FC'
y_axis = 'H_angle_v'

visual = 'C1'
motion = 'C4'




f_log = np.log(subjects[subject_id][visual]['Hpe_FC'])
x_points_v = subjects[subject][visual][x_axis]
y_points_v = subjects[subject][visual][y_axis]
print(x_points_v)
x_points_m = subjects[subject][motion][x_axis]
y_points_m = subjects[subject][motion][y_axis]
phase_visual= subjects[subject_id][visual]['H_angle_v']
phase_motion = subjects[subject_id][motion]['H_angle_v']
magnitudes_visual=subjects[subject_id][visual]['H_magnitude_v']
magnitudes_motion=subjects[subject_id][motion]['H_magnitude_v']

plt.ion()
fig, ax = plt.subplots(nrows=2, ncols=2)
iteration=0
def error(v_0):
    global iteration
    iteration+=1
    print(v_0)
    kv = v_0[0]
    Tl = v_0[1]
    TI = v_0[2]
    km = v_0[3]
    tau_v = v_0[4]
    tau_m = v_0[5]
    wm = v_0[6]
    cm = v_0[7]

    Fmag_v = np.absolute(kv*(1+1j*Tl*x_points_v)/(1+1j*TI*x_points_v)*np.exp(-1j*x_points_v*tau_v)*wm**2/((1j*x_points_v)**2+2*cm*wm*1j*x_points_v+wm**2))
    Fang_v = np.transpose(np.unwrap(np.angle(np.transpose(kv*(1+1j*Tl*x_points_v)/(1+1j*TI*x_points_v)*np.exp(-1j*x_points_v*tau_v)*wm**2/((1j*x_points_v)**2+2*cm*wm*1j*x_points_v+wm**2)[0])), discont=-np.pi/2))

    Fmag_m = np.absolute(km*np.exp(-1j*x_points_m*tau_m)*wm**2/((1j*x_points_m)**2+2*cm*wm*1j*x_points_m+wm**2))
    Fang_m = np.transpose(np.unwrap(np.angle(np.transpose(km*np.exp(-1j*x_points_m*tau_m)*wm**2/((1j*x_points_m)**2+2*cm*wm*1j*x_points_m+wm**2)[0])), discont=-np.pi/2))

    Pmag_v = magnitudes_visual
    Pang_v = phase_visual
    Pmag_m = magnitudes_motion
    Pang_m = phase_motion

    error =np.sum(np.linalg.norm(Fmag_v-Pmag_v,axis=1))
    print(error)

    H_pe_magnitude = np.absolute(
        kv * (1 + 1j * Tl * w) / (1 + 1j * TI * w) * np.exp(-1j * w * tau_v) * wm ** 2 / (
                    (1j * w) ** 2 + 2 * cm * wm * 1j * w + wm ** 2))
    H_pe_angle = np.unwrap(np.angle(
        kv * (1 + 1j * Tl * w) / (1 + 1j * TI * w) * np.exp(-1j * w * tau_v) * wm ** 2 /
        ((1j * w) ** 2 + 2 * cm * wm * 1j * w + wm ** 2)[0]).flatten(), discont=-np.pi / 2)

    H_px_magnitude = np.absolute(
        km * np.exp(-1j * w * tau_m) * wm ** 2 / (
                (1j * w) ** 2 + 2 * cm * wm * 1j * w + wm ** 2))
    H_px_angle = np.unwrap(np.angle(
        km * np.exp(-1j * w * tau_m) * wm ** 2 /
        ((1j * w) ** 2 + 2 * cm * wm * 1j * w + wm ** 2)[0]).flatten(), discont=-np.pi / 2)

    ax[0][0].cla()
    ax[0][0].semilogx(w, H_pe_angle)
    ax[0][0].semilogx(x_points_v, phase_visual, 'rx')
    ax[0][0].set_ylim([-10,10])
    ax[0][0].set_xlim([0.1, max(w)])


    ax[0][1].cla()
    ax[0][1].loglog(w, H_pe_magnitude)
    ax[0][1].loglog(x_points_v, magnitudes_visual, 'rx')
    ax[0][1].set_ylim([0.1,10])
    ax[0][1].set_xlim([0.1, max(w)])
    ax[0][1].grid()

    # With motion
    ax[1][0].cla()
    ax[1][0].semilogx(w, H_px_angle)
    ax[1][0].semilogx(x_points_m, phase_motion, 'rx')
    ax[1][0].set_ylim([-10, 10])
    ax[1][0].set_xlim([0.1, max(w)])

    ax[1][1].cla()
    ax[1][1].loglog(w, H_px_magnitude)
    ax[1][1].loglog(x_points_m, magnitudes_motion, 'rx')
    ax[1][1].set_ylim([0.1, 10])
    ax[1][1].set_xlim([0.1, max(w)])
    ax[1][1].grid()

    # # Table from Ed Smith answer
    # clust_data = np.random.random((10, 3))
    # collabel = ("col 1", "col 2", "col 3")
    # ax[0][2].table(cellText=clust_data, colLabels=collabel, loc='center')
    # # Hide axes
    # ax[0][2].xaxis.set_visible(False)
    # ax[0][2].yaxis.set_visible(False)
    fig.suptitle('#Iteration: '+ str(iteration), fontsize=20)

    fig.savefig('Folded/temp.png', dpi=fig.dpi)
    # plt.pause(0.001)
    # plt.show()
    #

    return error
v_0 = np.array([kv,Tl,TI,km,tau_v, tau_m,wm,cm])
v_0 = optimize.fmin(error,v_0)

H_pe_magnitude = np.absolute(v_0[0]*(1+1j*v_0[1]*w)/(1+1j*v_0[2]*w)*np.exp(-1j*w*v_0[3])*v_0[4]**2/((1j*w)**2+2*v_0[5]*v_0[4]*1j*w+v_0[4]**2))
H_pe_angle      =np.unwrap(np.angle(v_0[0]*(1+1j*v_0[1]*w)/(1+1j*v_0[2]*w)*np.exp(-1j*w*v_0[3])*v_0[4]**2/((1j*w)**2+2*v_0[5]*v_0[4]*1j*w+v_0[4]**2)[0]).flatten(), discont=-np.pi/2)



plt.show()
y =np.absolute(v_0[0]*(1+1j*v_0[1]*w)/(1+1j*v_0[2]*w)*np.exp(-1j*w*v_0[3])*v_0[4]**2/((1j*w)**2+2*v_0[5]*v_0[4]*1j*w+v_0[4]**2))
plt.semilogx(w,y)

plt.show()


